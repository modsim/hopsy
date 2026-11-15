#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>

#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/gaussian_line_kernel.h>
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/target/gaussian.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/overrelaxed_line_kernel.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/draw.h>
#include <hopsy/sampling/chain.h>
#include <hopsy/target/uniform.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace {
    using State = hopsy::Vector<double>;
    using Matrix = hopsy::Matrix<double>;
    using Geometry = hopsy::LinearPolytope<double>;
    using Uniform = hopsy::UniformTarget;
    using Coordinates = hopsy::AffineCoordinates<double>;

    // The GIL is held throughout calls involving Python targets. Each evaluation
    // receives an owning copy: retaining or modifying it cannot corrupt the chain.
    struct PythonTarget {
        nb::object log_density;
        nb::object log_gradient;
        double operator()(const State &state) const {
            return nb::cast<double>(log_density(nb::cast(state, nb::rv_policy::copy)));
        }
    };
    struct TargetAdapter {
        const PythonTarget &target;
        double log_density(const State &state) const { return target(state); }
        State log_gradient(const State &state) const {
            if (!target.log_gradient || !PyCallable_Check(target.log_gradient.ptr()))
                throw nb::type_error("CSmMALA target.log_gradient must be callable");
            State gradient =
                nb::cast<State>(target.log_gradient(nb::cast(state, nb::rv_policy::copy)));
            if (gradient.size() != state.size())
                throw std::invalid_argument("target gradient dimension mismatch");
            return gradient;
        }
    };

    template <typename Function> auto with_target(nb::handle target, Function &&function) {
        if (target.is_none() || target.type().is(nb::type<Uniform>())) {
            nb::gil_scoped_release release;
            return function(Uniform{});
        }
        if (target.type().is(nb::type<hopsy::GaussianTarget>())) {
            const auto &density = nb::cast<const hopsy::GaussianTarget &>(target);
            nb::gil_scoped_release release;
            return function(density);
        }
        nb::object method = nb::getattr(target, "log_density");
        if (!PyCallable_Check(method.ptr()))
            throw nb::type_error("target.log_density must be callable");
        PythonTarget python_target{std::move(method), nb::hasattr(target, "log_gradient")
                                                          ? nb::getattr(target, "log_gradient")
                                                          : nb::object()};
        return function(TargetAdapter{python_target});
    }

    // The compatibility Python API chooses a target once per call. Bind a typed
    // context outside the native loop; no target identity checks occur per step.
    template <typename Kernel, typename Workspace, typename Function>
    auto with_chain(const Geometry &geometry, const std::shared_ptr<const Coordinates> &coordinates,
                    Kernel &kernel, State &state, Workspace &workspace, nb::handle target,
                    Function &&function) {
        return with_target(target, [&](const auto &density) {
            if (coordinates) {
                auto problem = hopsy::Problem(std::cref(geometry), std::cref(density),
                                              std::cref(*coordinates));
                auto chain = hopsy::detail::bind_existing_chain(problem, kernel, state, workspace);
                return function(chain);
            } else {
                auto problem = hopsy::Problem(std::cref(geometry), std::cref(density));
                auto chain = hopsy::detail::bind_existing_chain(problem, kernel, state, workspace);
                return function(chain);
            }
        });
    }

    // A Python kernel owns one chain's mutable storage, while geometries are shared
    // and immutable. No NumPy view is ever exported into state/workspace storage.
    template <typename DirectionGenerator, typename LineKernel> class PythonKernel {
        using Kernel = hopsy::HitAndRunKernel<State, DirectionGenerator, LineKernel>;
        struct Context {
            State state;
            Geometry::workspace_type workspace;
            Kernel kernel;
            Context(State initial, const Geometry &geometry, const LineKernel &line)
                : state(std::move(initial)), workspace(geometry.create_workspace(state)),
                  kernel(state, DirectionGenerator(state.size()), line) {}
        };

        std::shared_ptr<Geometry> geometry_;
        std::shared_ptr<const Coordinates> coordinates_;
        LineKernel line_;
        std::unique_ptr<Context> context_;
        hopsy::RandomNumberGenerator rng_;
        mutable std::mutex mutex_;

        void validate(const State &state) const {
            if (state.size() != geometry_->num_dimension())
                throw std::invalid_argument("state dimension must match the geometry");
            if (!state.allFinite() || !geometry_->is_feasible(state))
                throw std::invalid_argument("initial state must be finite and feasible");
        }

        // A target exception can leave a core proposal pending. Recover the kernel
        // and workspace from the last accepted state; completed transitions and RNG
        // consumption are not rolled back. Coordinate cycling restarts on recovery.
        template <typename Function> auto run(Function &&function) {
            try {
                return function(*context_);
            } catch (...) {
                context_ = std::make_unique<Context>(context_->state, *geometry_, line_);
                throw;
            }
        }

      public:
        PythonKernel(std::shared_ptr<Geometry> geometry, State initial, std::uint64_t seed,
                     LineKernel line, std::shared_ptr<const Coordinates> coordinates = {})
            : geometry_(std::move(geometry)), coordinates_(std::move(coordinates)),
              line_(std::move(line)), rng_(seed) {
            if (!geometry_)
                throw std::invalid_argument("geometry must not be None");
            if (coordinates_ && coordinates_->sampling_dimension() != geometry_->num_dimension())
                throw std::invalid_argument(
                    "coordinate map and sampling geometry dimensions differ");
            validate(initial);
            context_ = std::make_unique<Context>(std::move(initial), *geometry_, line_);
        }

        auto lock() const {
            std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
            if (!lock.owns_lock())
                throw std::runtime_error(
                    "kernel is already in use (concurrent or reentrant access)");
            return lock;
        }
        State state() const {
            auto guard = lock();
            return context_->state;
        }
        void reset(State state) {
            auto guard = lock();
            validate(state);
            context_ = std::make_unique<Context>(std::move(state), *geometry_, line_);
        }
        void seed(std::uint64_t value) {
            auto guard = lock();
            // Reset direction-distribution caches as well as the RNG.
            auto context = std::make_unique<Context>(context_->state, *geometry_, line_);
            rng_.seed(value);
            context_ = std::move(context);
        }
        hopsy::Index dimension() const { return geometry_->num_dimension(); }
        hopsy::Index target_dimension() const {
            return coordinates_ ? coordinates_->target_dimension() : dimension();
        }
        State target_state() const {
            auto guard = lock();
            if (!coordinates_)
                return context_->state;
            State output(target_dimension());
            coordinates_->to_target(context_->state, output);
            return output;
        }

        bool step(nb::handle target) {
            auto guard = lock();
            return run([&](Context &c) {
                return with_chain(*geometry_, coordinates_, c.kernel, c.state, c.workspace, target,
                                  [&](auto &chain) { return chain.step(rng_); });
            });
        }
        Matrix draw(hopsy::Index count, hopsy::Index thinning, nb::handle target) {
            if (count < 0 || thinning < 1)
                throw std::invalid_argument(
                    "sample count must be nonnegative and thinning positive");
            auto guard = lock();
            return run([&](Context &c) {
                return with_chain(*geometry_, coordinates_, c.kernel, c.state, c.workspace, target,
                                  [&](auto &chain) { return chain.draw(rng_, count, thinning); });
            });
        }
        Matrix draw_sampling(hopsy::Index count, hopsy::Index thinning, nb::handle target) {
            if (count < 0 || thinning < 1)
                throw std::invalid_argument(
                    "sample count must be nonnegative and thinning positive");
            auto guard = lock();
            return run([&](Context &c) {
                return with_chain(
                    *geometry_, coordinates_, c.kernel, c.state, c.workspace, target,
                    [&](auto &chain) { return chain.draw_sampling(rng_, count, thinning); });
            });
        }
        void draw_into(nb::DRef<Matrix> samples, hopsy::Index thinning, nb::handle target) {
            if (samples.rows() != target_dimension() || thinning < 1)
                throw std::invalid_argument("output dimension mismatch or invalid thinning");
            if (samples.size() &&
                (samples.innerStride() <= 0 || samples.outerStride() <= 0 ||
                 !(samples.outerStride() >= samples.rows() * samples.innerStride() ||
                   samples.innerStride() >= samples.cols() * samples.outerStride())))
                throw std::invalid_argument("output must have nonoverlapping positive strides");
            auto guard = lock();
            run([&](Context &c) {
                with_chain(*geometry_, coordinates_, c.kernel, c.state, c.workspace, target,
                           [&](auto &chain) { chain.draw_into(rng_, samples, thinning); });
            });
        }
    };

    template <typename Direction, typename Line>
    void bind_kernel(nb::module_ &m, const char *name) {
        using Chain = PythonKernel<Direction, Line>;
        auto cls = nb::class_<Chain>(m, name);
        if constexpr (std::is_same_v<Line, hopsy::GaussianLineKernel>) {
            cls.def(
                "__init__",
                [](Chain *self, std::shared_ptr<Geometry> geometry, State initial,
                   std::uint64_t seed, double sigma,
                   std::shared_ptr<const Coordinates> coordinates) {
                    new (self) Chain(std::move(geometry), std::move(initial), seed, Line(sigma),
                                     std::move(coordinates));
                },
                "geometry"_a, "initial_state"_a, "seed"_a = 0, "standard_deviation"_a = 1.0,
                "coordinates"_a = nb::none());
        } else {
            cls.def(
                "__init__",
                [](Chain *self, std::shared_ptr<Geometry> geometry, State initial,
                   std::uint64_t seed, std::shared_ptr<const Coordinates> coordinates) {
                    new (self) Chain(std::move(geometry), std::move(initial), seed, Line{},
                                     std::move(coordinates));
                },
                "geometry"_a, "initial_state"_a, "seed"_a = 0, "coordinates"_a = nb::none());
        }
        cls.def_prop_ro("state", &Chain::state, "A copy of the current accepted state.")
            .def_prop_ro("dimension", &Chain::dimension)
            .def_prop_ro("sampling_state", &Chain::state)
            .def_prop_ro("target_state", &Chain::target_state)
            .def_prop_ro("target_dimension", &Chain::target_dimension)
            .def("reset", &Chain::reset, "state"_a,
                 "Replace the accepted state and rebuild proposal/workspace caches; retain RNG "
                 "position.")
            .def("seed", &Chain::seed, "seed"_a,
                 "Reseed the RNG and reset direction-generator caches; retain accepted state.")
            .def("step", &Chain::step, "target"_a = nb::none(),
                 "Attempt one transition; return acceptance.")
            .def("draw", &Chain::draw, "num_samples"_a, "thinning"_a = 1, "target"_a = nb::none(),
                 "Return samples as columns of a (dimension, num_samples) NumPy array.")
            .def("draw_sampling", &Chain::draw_sampling, "num_samples"_a, "thinning"_a = 1,
                 "target"_a = nb::none())
            .def("draw_into", &Chain::draw_into, "samples"_a.noconvert(), "thinning"_a = 1,
                 "target"_a = nb::none(),
                 "Fill a writable float64 NumPy matrix without copying it.");

        m.def(
            "step", [](Chain &chain, nb::handle target) { return chain.step(target); }, "kernel"_a,
            "target"_a = nb::none());
        m.def(
            "draw",
            [](Chain &chain, hopsy::Index count, hopsy::Index thinning, nb::handle target) {
                return chain.draw(count, thinning, target);
            },
            "kernel"_a, "num_samples"_a, "thinning"_a = 1, "target"_a = nb::none());
        m.def(
            "draw_into",
            [](Chain &chain, nb::DRef<Matrix> samples, hopsy::Index thinning, nb::handle target) {
                chain.draw_into(samples, thinning, target);
            },
            "kernel"_a, "samples"_a.noconvert(), "thinning"_a = 1, "target"_a = nb::none());
    }
    // Gaussian-family chains use their own contexts; HR ownership is unchanged.
    template <typename Kernel> class PythonGaussianKernel {
        struct Context {
            State state;
            hopsy::EmptyWorkspace workspace;
            Kernel kernel;
            Context(State initial, double sigma)
                : state(std::move(initial)), kernel(state, {}, {}, sigma) {}
        };
        std::shared_ptr<Geometry> geometry_;
        std::shared_ptr<const Coordinates> coordinates_;
        std::unique_ptr<Context> context_;
        double sigma_;
        hopsy::RandomNumberGenerator rng_;
        mutable std::mutex mutex_;

        auto lock() const {
            std::unique_lock<std::mutex> guard(mutex_, std::try_to_lock);
            if (!guard.owns_lock())
                throw std::runtime_error(
                    "kernel is already in use (concurrent or reentrant access)");
            return guard;
        }
        void validate(const State &state) const {
            if (state.size() != geometry_->num_dimension() || !state.allFinite() ||
                !geometry_->is_feasible(state))
                throw std::invalid_argument(
                    "initial state must be finite, feasible and match the geometry dimension");
            if constexpr (!Kernel::symmetric_proposal) {
                // Dikin needs a strict interior and a nonsingular precision metric.
                hopsy::DikinMetric<double> metric;
                if (!metric.prepare(state, *geometry_, hopsy::EmptyWorkspace{}))
                    throw std::invalid_argument(
                        "Dikin metric requires a strict interior and full-rank constraints");
            }
        }
        template <typename Function> auto run(Function &&function) {
            try {
                return function(*context_);
            } catch (...) {
                context_ = std::make_unique<Context>(context_->state, sigma_);
                throw;
            }
        }

      public:
        PythonGaussianKernel(std::shared_ptr<Geometry> geometry, State initial, std::uint64_t seed,
                             double sigma, std::shared_ptr<const Coordinates> coordinates = {})
            : geometry_(std::move(geometry)), coordinates_(std::move(coordinates)), sigma_(sigma),
              rng_(seed) {
            if (!geometry_)
                throw std::invalid_argument("geometry must not be None");
            if (coordinates_ && coordinates_->sampling_dimension() != geometry_->num_dimension())
                throw std::invalid_argument(
                    "coordinate map and sampling geometry dimensions differ");
            validate(initial);
            context_ = std::make_unique<Context>(std::move(initial), sigma_);
        }
        State state() const {
            auto guard = lock();
            return context_->state;
        }
        hopsy::Index dimension() const { return geometry_->num_dimension(); }
        hopsy::Index target_dimension() const {
            return coordinates_ ? coordinates_->target_dimension() : dimension();
        }
        State target_state() const {
            auto guard = lock();
            if (!coordinates_)
                return context_->state;
            State output(target_dimension());
            coordinates_->to_target(context_->state, output);
            return output;
        }
        double step_size() const {
            auto guard = lock();
            return sigma_;
        }
        void set_step_size(double sigma) {
            auto guard = lock();
            context_->kernel.set_step_size(sigma);
            sigma_ = sigma;
        }
        void reset(State state) {
            auto guard = lock();
            validate(state);
            context_ = std::make_unique<Context>(std::move(state), sigma_);
        }
        void seed(std::uint64_t seed) {
            auto guard = lock();
            auto context = std::make_unique<Context>(context_->state, sigma_);
            rng_.seed(seed);
            context_ = std::move(context);
        }
        bool step(nb::handle target) {
            auto guard = lock();
            return run([&](Context &c) {
                return with_chain(*geometry_, coordinates_, c.kernel, c.state, c.workspace, target,
                                  [&](auto &chain) { return chain.step(rng_); });
            });
        }
        Matrix draw(hopsy::Index count, hopsy::Index thinning, nb::handle target) {
            if (count < 0 || thinning < 1)
                throw std::invalid_argument(
                    "sample count must be nonnegative and thinning positive");
            auto guard = lock();
            return run([&](Context &c) {
                return with_chain(*geometry_, coordinates_, c.kernel, c.state, c.workspace, target,
                                  [&](auto &chain) { return chain.draw(rng_, count, thinning); });
            });
        }
        Matrix draw_sampling(hopsy::Index count, hopsy::Index thinning, nb::handle target) {
            if (count < 0 || thinning < 1)
                throw std::invalid_argument(
                    "sample count must be nonnegative and thinning positive");
            auto guard = lock();
            return run([&](Context &c) {
                return with_chain(
                    *geometry_, coordinates_, c.kernel, c.state, c.workspace, target,
                    [&](auto &chain) { return chain.draw_sampling(rng_, count, thinning); });
            });
        }
        void draw_into(nb::DRef<Matrix> samples, hopsy::Index thinning, nb::handle target) {
            if (samples.rows() != target_dimension() || thinning < 1)
                throw std::invalid_argument("output dimension mismatch or invalid thinning");
            if (samples.size() &&
                (samples.innerStride() <= 0 || samples.outerStride() <= 0 ||
                 !(samples.outerStride() >= samples.rows() * samples.innerStride() ||
                   samples.innerStride() >= samples.cols() * samples.outerStride())))
                throw std::invalid_argument("output must have nonoverlapping positive strides");
            auto guard = lock();
            run([&](Context &c) {
                with_chain(*geometry_, coordinates_, c.kernel, c.state, c.workspace, target,
                           [&](auto &chain) { chain.draw_into(rng_, samples, thinning); });
            });
        }
    };

    template <typename Kernel> void bind_gaussian_kernel(nb::module_ &m, const char *name) {
        using Chain = PythonGaussianKernel<Kernel>;
        nb::class_<Chain>(m, name)
            .def(nb::init<std::shared_ptr<Geometry>, State, std::uint64_t, double,
                          std::shared_ptr<const Coordinates>>(),
                 "geometry"_a, "initial_state"_a, "seed"_a = 0, "step_size"_a = 1.,
                 "coordinates"_a = nb::none())
            .def_prop_ro("state", &Chain::state)
            .def_prop_ro("dimension", &Chain::dimension)
            .def_prop_ro("sampling_state", &Chain::state)
            .def_prop_ro("target_state", &Chain::target_state)
            .def_prop_ro("target_dimension", &Chain::target_dimension)
            .def_prop_rw("step_size", &Chain::step_size, &Chain::set_step_size,
                         "Standard-deviation scale; update between completed transitions.")
            .def("reset", &Chain::reset, "state"_a)
            .def("seed", &Chain::seed, "seed"_a)
            .def("step", &Chain::step, "target"_a = nb::none())
            .def("draw", &Chain::draw, "num_samples"_a, "thinning"_a = 1, "target"_a = nb::none())
            .def("draw_sampling", &Chain::draw_sampling, "num_samples"_a, "thinning"_a = 1,
                 "target"_a = nb::none())
            .def("draw_into", &Chain::draw_into, "samples"_a.noconvert(), "thinning"_a = 1,
                 "target"_a = nb::none());
    }
} // namespace

NB_MODULE(_hopsy_core, m) {
    m.doc() = "Experimental hopsy core: owning hit-and-run chains and batched sampling.";
    nb::class_<Geometry>(m, "LinearPolytope")
        .def(
            "__init__",
            [](Geometry *self, Matrix A, State b) {
                if (!A.allFinite() || !b.allFinite())
                    throw std::invalid_argument("A and b must contain finite values");
                new (self) Geometry(std::move(A), std::move(b));
            },
            "A"_a, "b"_a)
        .def_prop_ro("A", [](const Geometry &g) -> Matrix { return g.A(); })
        .def_prop_ro("b", [](const Geometry &g) -> State { return g.b(); })
        .def_prop_ro("dimension", &Geometry::num_dimension)
        .def_prop_ro("num_constraints", &Geometry::num_constraints)
        .def(
            "contains",
            [](const Geometry &g, const State &x) {
                if (x.size() != g.num_dimension())
                    throw std::invalid_argument("state dimension mismatch");
                return x.allFinite() && g.is_feasible(x);
            },
            "state"_a);
    nb::class_<Uniform>(m, "UniformTarget")
        .def(nb::init<>())
        .def("log_density", [](const Uniform &, const State &) { return 0.; }, "state"_a);

    nb::class_<Coordinates>(m, "AffineCoordinates")
        .def(nb::init<Matrix, State>(), "transformation"_a, "shift"_a)
        .def_prop_ro("matrix", [](const Coordinates &c) -> Matrix { return c.matrix(); })
        .def_prop_ro("shift", [](const Coordinates &c) -> State { return c.shift(); })
        .def_prop_ro("sampling_dimension", &Coordinates::sampling_dimension)
        .def_prop_ro("target_dimension", [](const Coordinates &c) { return c.target_dimension(); })
        .def(
            "to_target",
            [](const Coordinates &c, const State &z) {
                if (z.size() != c.sampling_dimension())
                    throw std::invalid_argument("sampling dimension mismatch");
                State x(c.target_dimension());
                c.to_target(z, x);
                return x;
            },
            "state"_a);
    nb::class_<hopsy::GaussianTarget>(m, "GaussianTarget")
        .def(nb::init<State, const Matrix &>(), "mean"_a, "covariance"_a)
        .def_prop_ro("dimension", &hopsy::GaussianTarget::dimension)
        .def("log_density", &hopsy::GaussianTarget::log_density, "state"_a)
        .def("log_gradient", &hopsy::GaussianTarget::log_gradient, "state"_a);
    bind_gaussian_kernel<hopsy::GaussianKernel<State, hopsy::ZeroDrift, hopsy::IdentityMetric>>(
        m, "GaussianRandomWalkKernel");
    bind_gaussian_kernel<hopsy::DikinWalkKernel<State>>(m, "DikinWalkKernel");
    bind_gaussian_kernel<hopsy::CSmMALAKernel<State>>(m, "CSmMALAKernel");

    using Dense = hopsy::RandomDirectionGenerator<double>;
    using Cycling = hopsy::CyclingCoordinateDirectionGenerator;
    using Random = hopsy::RandomCoordinateDirectionGenerator;
    using UniformLine = hopsy::UniformLineKernel;
    using GaussianLine = hopsy::GaussianLineKernel;
    using OverrelaxedLine = hopsy::OverRelaxedLineKernel;
    bind_kernel<Dense, UniformLine>(m, "UniformHitAndRunKernel");
    bind_kernel<Dense, GaussianLine>(m, "GaussianHitAndRunKernel");
    bind_kernel<Dense, OverrelaxedLine>(m, "OverRelaxedHitAndRunKernel");
    bind_kernel<Cycling, UniformLine>(m, "UniformCoordinateHitAndRunKernel");
    bind_kernel<Cycling, GaussianLine>(m, "GaussianCoordinateHitAndRunKernel");
    bind_kernel<Cycling, OverrelaxedLine>(m, "OverRelaxedCoordinateHitAndRunKernel");
    bind_kernel<Random, UniformLine>(m, "UniformRandomCoordinateHitAndRunKernel");
    bind_kernel<Random, GaussianLine>(m, "GaussianRandomCoordinateHitAndRunKernel");
    bind_kernel<Random, OverrelaxedLine>(m, "OverRelaxedRandomCoordinateHitAndRunKernel");
}
