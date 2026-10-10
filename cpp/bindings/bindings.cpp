#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>

#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/gaussian_line_kernel.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/overrelaxed_line_kernel.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/draw.h>
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

// The GIL is held throughout calls involving Python targets. Each evaluation
// receives an owning copy: retaining or modifying it cannot corrupt the chain.
struct PythonTarget {
    nb::object log_density;
    double operator()(const State &state) const {
        return nb::cast<double>(log_density(nb::cast(state, nb::rv_policy::copy)));
    }
};
struct TargetAdapter {
    const PythonTarget &target;
    double log_density(const State &state) const { return target(state); }
};

template<typename Function>
auto with_target(nb::handle target, Function &&function) {
    if (target.is_none() || target.type().is(nb::type<Uniform>())) {
        nb::gil_scoped_release release;
        return function(Uniform{});
    }
    nb::object method = nb::getattr(target, "log_density");
    if (!PyCallable_Check(method.ptr()))
        throw nb::type_error("target.log_density must be callable");
    PythonTarget python_target{std::move(method)};
    return function(TargetAdapter{python_target});
}

// A Python kernel owns one chain's mutable storage, while geometries are shared
// and immutable. No NumPy view is ever exported into state/workspace storage.
template<typename DirectionGenerator, typename LineKernel>
class PythonKernel {
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
    LineKernel line_;
    std::unique_ptr<Context> context_;
    hopsy::RandomNumberGenerator rng_;
    mutable std::mutex mutex_;

    void validate(const State &state) const {
        if (state.size() != geometry_->num_dimension())
            throw std::invalid_argument("state dimension must match the geometry");
        if (!state.allFinite() || !geometry_->contains(state))
            throw std::invalid_argument("initial state must be finite and feasible");
    }

    // A target exception can leave a core proposal pending. Recover the kernel
    // and workspace from the last accepted state; completed transitions and RNG
    // consumption are not rolled back. Coordinate cycling restarts on recovery.
    template<typename Function>
    auto run(Function &&function) {
        try {
            return function(*context_);
        } catch (...) {
            context_ = std::make_unique<Context>(context_->state, *geometry_, line_);
            throw;
        }
    }

public:
    PythonKernel(std::shared_ptr<Geometry> geometry, State initial,
                 std::uint64_t seed, LineKernel line)
        : geometry_(std::move(geometry)), line_(std::move(line)), rng_(seed) {
        if (!geometry_) throw std::invalid_argument("geometry must not be None");
        validate(initial);
        context_ = std::make_unique<Context>(std::move(initial), *geometry_, line_);
    }

    auto lock() const {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock())
            throw std::runtime_error("kernel is already in use (concurrent or reentrant access)");
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

    bool step(nb::handle target) {
        auto guard = lock();
        return with_target(target, [&](const auto &density) {
            return run([&](Context &c) {
                return hopsy::step(c.kernel, c.state, rng_, density, *geometry_, c.workspace);
            });
        });
    }
    Matrix draw(hopsy::Index count, hopsy::Index thinning, nb::handle target) {
        if (count < 0) throw std::invalid_argument("num_samples must be nonnegative");
        if (thinning < 1) throw std::invalid_argument("thinning must be positive");
        auto guard = lock();
        return with_target(target, [&](const auto &density) {
            return run([&](Context &c) {
                return hopsy::draw(c.kernel, c.state, rng_, density, *geometry_, c.workspace, count, thinning);
            });
        });
    }
    void draw_into(nb::DRef<Matrix> samples, hopsy::Index thinning, nb::handle target) {
        if (samples.rows() != dimension())
            throw std::invalid_argument("output rows must equal the state dimension");
        if (thinning < 1) throw std::invalid_argument("thinning must be positive");
        // Accept C/F-contiguous matrices and ordinary positive-stride slices,
        // while excluding overlapping writable layouts and negative strides.
        if (samples.size() &&
            (samples.innerStride() <= 0 || samples.outerStride() <= 0 ||
             !(samples.outerStride() >= samples.rows()*samples.innerStride() ||
               samples.innerStride() >= samples.cols()*samples.outerStride())))
            throw std::invalid_argument("output must have nonoverlapping positive strides");
        auto guard = lock();
        with_target(target, [&](const auto &density) {
            run([&](Context &c) {
                hopsy::draw_into(c.kernel, c.state, rng_, density, *geometry_, c.workspace, samples, thinning);
            });
        });
    }
};

template<typename Direction, typename Line>
void bind_kernel(nb::module_ &m, const char *name) {
    using Chain = PythonKernel<Direction, Line>;
    auto cls = nb::class_<Chain>(m, name);
    if constexpr (std::is_same_v<Line, hopsy::GaussianLineKernel>) {
        cls.def("__init__", [](Chain *self, std::shared_ptr<Geometry> geometry,
                               State initial, std::uint64_t seed, double sigma) {
            new (self) Chain(std::move(geometry), std::move(initial), seed, Line(sigma));
        }, "geometry"_a, "initial_state"_a, "seed"_a = 0, "standard_deviation"_a = 1.0);
    } else {
        cls.def("__init__", [](Chain *self, std::shared_ptr<Geometry> geometry,
                               State initial, std::uint64_t seed) {
            new (self) Chain(std::move(geometry), std::move(initial), seed, Line{});
        }, "geometry"_a, "initial_state"_a, "seed"_a = 0);
    }
    cls.def_prop_ro("state", &Chain::state, "A copy of the current accepted state.")
       .def_prop_ro("dimension", &Chain::dimension)
       .def("reset", &Chain::reset, "state"_a,
            "Replace the accepted state and rebuild proposal/workspace caches; retain RNG position.")
       .def("seed", &Chain::seed, "seed"_a,
            "Reseed the RNG and reset direction-generator caches; retain accepted state.")
       .def("step", &Chain::step, "target"_a = nb::none(), "Attempt one transition; return acceptance.")
       .def("draw", &Chain::draw, "num_samples"_a, "thinning"_a = 1, "target"_a = nb::none(),
            "Return samples as columns of a (dimension, num_samples) NumPy array.")
       .def("draw_into", &Chain::draw_into, "samples"_a.noconvert(), "thinning"_a = 1,
            "target"_a = nb::none(), "Fill a writable float64 NumPy matrix without copying it.");

    m.def("step", [](Chain &chain, nb::handle target) { return chain.step(target); },
          "kernel"_a, "target"_a = nb::none());
    m.def("draw", [](Chain &chain, hopsy::Index count, hopsy::Index thinning, nb::handle target) {
        return chain.draw(count, thinning, target);
    }, "kernel"_a, "num_samples"_a, "thinning"_a = 1, "target"_a = nb::none());
    m.def("draw_into", [](Chain &chain, nb::DRef<Matrix> samples, hopsy::Index thinning, nb::handle target) {
        chain.draw_into(samples, thinning, target);
    }, "kernel"_a, "samples"_a.noconvert(), "thinning"_a = 1, "target"_a = nb::none());
}
} // namespace

NB_MODULE(_hopsy_core, m) {
    m.doc() = "Experimental hopsy core: owning hit-and-run chains and batched sampling.";
    nb::class_<Geometry>(m, "LinearPolytope")
        .def("__init__", [](Geometry *self, Matrix A, State b) {
            if (!A.allFinite() || !b.allFinite())
                throw std::invalid_argument("A and b must contain finite values");
            new (self) Geometry(std::move(A), std::move(b));
        }, "A"_a, "b"_a)
        .def_prop_ro("A", [](const Geometry &g) -> Matrix { return g.A(); })
        .def_prop_ro("b", [](const Geometry &g) -> State { return g.b(); })
        .def_prop_ro("dimension", &Geometry::num_dimension)
        .def_prop_ro("num_constraints", &Geometry::num_constraints)
        .def("contains", [](const Geometry &g, const State &x) {
            if (x.size() != g.num_dimension()) throw std::invalid_argument("state dimension mismatch");
            return x.allFinite() && g.contains(x);
        }, "state"_a);
    nb::class_<Uniform>(m, "UniformTarget")
        .def(nb::init<>())
        .def("log_density", [](const Uniform &, const State &) { return 0.; }, "state"_a);

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
