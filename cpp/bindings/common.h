#ifndef HOPSY_BINDINGS_COMMON_H
#define HOPSY_BINDINGS_COMMON_H

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/kernel/gaussian_line_kernel.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/overrelaxed_line_kernel.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/owned_chain.h>
#include <hopsy/target/gaussian.h>
#include <hopsy/target/uniform.h>
#include <string>

namespace hopsy::bindings {
    namespace nb = nanobind;
    using namespace nb::literals;
    using State = Vector<double>;
    using DenseMatrix = Matrix<double>;
    using Geometry = LinearPolytope<double>;
    using Coordinates = AffineCoordinates<double>;
    using Uniform = UniformTarget;
    using Gaussian = GaussianTarget;

    // The sole Python-specific target adapter. Methods are captured at problem
    // construction, and inputs are copied so callbacks cannot mutate core state.
    struct PythonTarget {
        nb::object owner, density, gradient;
        double log_density(const State &x) const {
            return nb::cast<double>(density(nb::cast(x, nb::rv_policy::copy)));
        }
        State log_gradient(const State &x) const {
            if (!gradient || !PyCallable_Check(gradient.ptr()))
                throw nb::type_error("target.log_gradient must be callable");
            State value = nb::cast<State>(gradient(nb::cast(x, nb::rv_policy::copy)));
            if (value.size() != x.size())
                throw std::invalid_argument("target gradient dimension mismatch");
            return value;
        }
    };
    template <typename T, typename C = IdentityCoordinates>
    using P = Problem<std::reference_wrapper<const Geometry>,
                      std::conditional_t<std::same_as<T, PythonTarget>, PythonTarget,
                                         std::reference_wrapper<const T>>,
                      std::conditional_t<std::same_as<C, IdentityCoordinates>, C,
                                         std::reference_wrapper<const C>>>;

    template <typename P>
    constexpr bool native_target = !std::same_as<typename P::target_type, PythonTarget>;
    template <typename P, typename F> decltype(auto) evaluate(F &&f) {
        if constexpr (native_target<P>) {
            nb::gil_scoped_release release;
            return std::forward<F>(f)();
        } else
            return std::forward<F>(f)();
    }
    inline void valid_output(const nb::DRef<DenseMatrix> &out) {
        if (out.size() && (out.innerStride() <= 0 || out.outerStride() <= 0 ||
                           !(out.outerStride() >= out.rows() * out.innerStride() ||
                             out.innerStride() >= out.cols() * out.outerStride())))
            throw std::invalid_argument("output must have nonoverlapping positive strides");
    }
    template <typename Problem, typename Kernel>
    void bind_chain_type(nb::module_ &m, const std::string &name) {
        using Chain = OwnedStateChain<Problem, Kernel>;
        auto cls = nb::class_<Chain>(m, name.c_str(), nb::is_weak_referenceable());
        cls.def_prop_ro("sampling_state", &Chain::sampling_state)
            .def_prop_ro("target_state", &Chain::target_state)
            .def_prop_ro("sampling_dimension", &Chain::sampling_dimension)
            .def_prop_ro("target_dimension", &Chain::target_dimension)
            .def(
                "step",
                [](Chain &chain, RandomNumberGenerator &rng) {
                    return evaluate<Problem>([&] { return chain.step(rng); });
                },
                "rng"_a)
            .def(
                "draw",
                [](Chain &chain, RandomNumberGenerator &rng, Index count, Index thin) {
                    return evaluate<Problem>([&] { return chain.draw(rng, count, thin); });
                },
                "rng"_a, "num_samples"_a, "thinning"_a = 1)
            .def(
                "draw_sampling",
                [](Chain &chain, RandomNumberGenerator &rng, Index count, Index thin) {
                    return evaluate<Problem>([&] { return chain.draw_sampling(rng, count, thin); });
                },
                "rng"_a, "num_samples"_a, "thinning"_a = 1)
            .def(
                "draw_into",
                [](Chain &chain, RandomNumberGenerator &rng, nb::DRef<DenseMatrix> out,
                   Index thin) {
                    valid_output(out);
                    evaluate<Problem>([&] { chain.draw_into(rng, out, thin); });
                },
                "rng"_a, "samples"_a.noconvert(), "thinning"_a = 1)
            .def(
                "draw_sampling_into",
                [](Chain &chain, RandomNumberGenerator &rng, nb::DRef<DenseMatrix> out,
                   Index thin) {
                    valid_output(out);
                    evaluate<Problem>([&] { chain.draw_sampling_into(rng, out, thin); });
                },
                "rng"_a, "samples"_a.noconvert(), "thinning"_a = 1)
            .def(
                "rebind_problem",
                [](Chain &chain, std::shared_ptr<const Problem> p) {
                    chain.rebind_problem(std::move(p));
                },
                "problem"_a);
        if constexpr (requires(Kernel &k) { k.step_size(); }) {
            cls.def_prop_rw("step_size", &Chain::step_size, &Chain::set_step_size);
        }
        m.def(
            "bind_chain",
            [](std::shared_ptr<const Problem> p, Kernel &k, State state) {
                return hopsy::bind_chain(std::move(p), k, std::move(state));
            },
            "problem"_a, "kernel"_a, "initial_state"_a, nb::keep_alive<0, 2>());
    }
    template <typename Kernel> void bind_chains(nb::module_ &m, const std::string &name) {
        bind_chain_type<P<Uniform>, Kernel>(m, "_" + name + "UniformChain");
        bind_chain_type<P<Gaussian>, Kernel>(m, "_" + name + "GaussianChain");
        bind_chain_type<P<PythonTarget>, Kernel>(m, "_" + name + "PythonChain");
        bind_chain_type<P<Uniform, Coordinates>, Kernel>(m, "_" + name + "AffineUniformChain");
        bind_chain_type<P<Gaussian, Coordinates>, Kernel>(m, "_" + name + "AffineGaussianChain");
        bind_chain_type<P<PythonTarget, Coordinates>, Kernel>(m, "_" + name + "AffinePythonChain");
    }
    inline void valid_initial(const State &x) {
        if (!x.size() || !x.allFinite())
            throw std::invalid_argument("kernel initial state must be nonempty and finite");
    }
    template <typename Kernel> void bind_gaussian(nb::module_ &m, const char *name) {
        nb::class_<Kernel>(m, name)
            .def(nb::new_([](const State &initial, double sigma) {
                     valid_initial(initial);
                     return make_sampling_resource<Kernel>(Kernel(initial, {}, {}, sigma));
                 }),
                 "initial_state"_a, "step_size"_a = 1.)
            .def_prop_rw(
                "step_size",
                [](Kernel &k) {
                    ExclusiveLease<Kernel> lease(k);
                    return k.step_size();
                },
                [](Kernel &k, double sigma) {
                    ExclusiveLease<Kernel> lease(k);
                    k.set_step_size(sigma);
                });
        bind_chains<Kernel>(m, name);
    }
    template <typename Direction, typename Line>
    void bind_hit_and_run(nb::module_ &m, const char *name) {
        using K = HitAndRunKernel<State, Direction, Line>;
        auto cls = nb::class_<K>(m, name);
        cls.def(nb::new_([](const State &s, Direction direction, Line line) {
                    valid_initial(s);
                    return make_sampling_resource<K>(s, std::move(direction), std::move(line));
                }),
                "initial_state"_a, "direction_generator"_a, "line_kernel"_a);
        if constexpr (std::same_as<Line, GaussianLineKernel>) {
            cls.def(nb::new_([](const State &s, double sigma) {
                        valid_initial(s);
                        return make_sampling_resource<K>(s, Direction(s.size()), Line(sigma));
                    }),
                    "initial_state"_a, "standard_deviation"_a = 1.);
        } else {
            cls.def(nb::new_([](const State &s) {
                        valid_initial(s);
                        return make_sampling_resource<K>(s, Direction(s.size()), Line{});
                    }),
                    "initial_state"_a);
        }
        bind_chains<K>(m, name);
    }
    void bind_gaussian_types(nb::module_ &);
    void bind_dense_types(nb::module_ &);
    void bind_cycling_types(nb::module_ &);
    void bind_random_coordinate_types(nb::module_ &);
} // namespace hopsy::bindings
#endif
