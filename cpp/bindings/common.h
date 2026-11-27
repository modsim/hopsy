#ifndef HOPSY_BINDINGS_COMMON_H
#define HOPSY_BINDINGS_COMMON_H

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <limits>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/geometry/bilinear_polytope.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
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
    using Bilinear = BilinearPolytope<double>;
    using Coordinates = AffineCoordinates<double>;
    using Uniform = UniformTarget;
    using Gaussian = GaussianTarget;

}
#include "targets.h"
namespace hopsy::bindings {
    template <typename T, typename C = IdentityCoordinates, typename G = Geometry>
    using P = Problem<std::reference_wrapper<const G>,
                      std::conditional_t<runtime_target<T>, T,
                                         std::reference_wrapper<const T>>,
                      std::conditional_t<std::same_as<C, IdentityCoordinates>, C,
                                         std::reference_wrapper<const C>>>;

    template <typename F> decltype(auto) evaluate(F &&f) {
        nb::gil_scoped_release release;
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
            .def_prop_ro("state_log_density", &Chain::state_log_density)
            .def_prop_ro("proposal_log_density", &Chain::proposal_log_density)
            .def_prop_ro("sampling_dimension", &Chain::sampling_dimension)
            .def_prop_ro("target_dimension", &Chain::target_dimension)
            .def(
                "step",
                [](Chain &chain, RandomNumberGenerator &rng) {
                    return evaluate([&] { return chain.step(rng); });
                },
                "rng"_a)
            .def(
                "draw",
                [](Chain &chain, RandomNumberGenerator &rng, Index count, Index thin) {
                    return evaluate([&] { return chain.draw(rng, count, thin); });
                },
                "rng"_a, "num_samples"_a, "thinning"_a = 1)
            .def(
                "draw_traced",
                [](Chain &chain, RandomNumberGenerator &rng, Index count, Index thin) {
                    if (count < 0 || thin < 1)
                        throw std::invalid_argument("Invalid sample count or thinning");
                    const auto missing = std::numeric_limits<double>::quiet_NaN();
                    State accepted(count), proposed(count);
                    auto samples = evaluate([&] {
                        return chain.draw_observed(rng, count, thin,
                            [&](Index i, const auto &binding) {
                                accepted[i] = binding.state_log_density().value_or(missing);
                                proposed[i] = binding.proposal_log_density().value_or(missing);
                            });
                    });
                    return std::make_tuple(std::move(samples), std::move(accepted),
                                           std::move(proposed));
                },
                "rng"_a, "num_samples"_a, "thinning"_a = 1)
            .def(
                "draw_sampling",
                [](Chain &chain, RandomNumberGenerator &rng, Index count, Index thin) {
                    return evaluate([&] { return chain.draw_sampling(rng, count, thin); });
                },
                "rng"_a, "num_samples"_a, "thinning"_a = 1)
            .def(
                "draw_into",
                [](Chain &chain, RandomNumberGenerator &rng, nb::DRef<DenseMatrix> out,
                   Index thin) {
                    valid_output(out);
                    evaluate([&] { chain.draw_into(rng, out, thin); });
                },
                "rng"_a, "samples"_a.noconvert(), "thinning"_a = 1)
            .def(
                "draw_sampling_into",
                [](Chain &chain, RandomNumberGenerator &rng, nb::DRef<DenseMatrix> out,
                   Index thin) {
                    valid_output(out);
                    evaluate([&] { chain.draw_sampling_into(rng, out, thin); });
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
    template<class Kernel, class C, class G>
    void bind_runtime_chains(nb::module_ &m, const std::string &name) {
        if constexpr (!std::same_as<Kernel, CSmMALAKernel<State>>)
            bind_chain_type<P<RuntimeTarget<0>, C, G>, Kernel>(m, name + "0");
        else
            m.def("bind_chain", [](const P<RuntimeTarget<0>, C, G> &, Kernel &, const State &) {
                throw nb::type_error("kernel requires log_gradient");
            }, "problem"_a, "kernel"_a, "initial_state"_a);
        bind_chain_type<P<RuntimeTarget<1>, C, G>, Kernel>(m, name + "1");
        bind_chain_type<P<RuntimeTarget<2>, C, G>, Kernel>(m, name + "2");
    }
    template <typename Kernel> void bind_chains(nb::module_ &m, const std::string &name) {
        m.def(
            "supports_geometry",
            [](const Kernel &, const Geometry &) {
                return kernel_supports_geometry_v<Kernel, Geometry>;
            },
            "kernel"_a, "geometry"_a);
        m.def(
            "supports_geometry",
            [](const Kernel &, const Bilinear &) {
                return kernel_supports_geometry_v<Kernel, Bilinear>;
            },
            "kernel"_a, "geometry"_a);
        bind_chain_type<P<Uniform>, Kernel>(m, "_" + name + "UniformChain");
        bind_chain_type<P<Gaussian>, Kernel>(m, "_" + name + "GaussianChain");
        bind_runtime_chains<Kernel, IdentityCoordinates, Geometry>(m, "_" + name + "Runtime");
        bind_chain_type<P<Uniform, Coordinates>, Kernel>(m, "_" + name + "AffineUniformChain");
        bind_chain_type<P<Gaussian, Coordinates>, Kernel>(m, "_" + name + "AffineGaussianChain");
        bind_runtime_chains<Kernel, Coordinates, Geometry>(m, "_" + name + "AffineRuntime");
        if constexpr (kernel_supports_geometry_v<Kernel, Bilinear>) {
            bind_chain_type<P<Uniform, IdentityCoordinates, Bilinear>, Kernel>(
                m, "_" + name + "BilinearUniformChain");
            bind_chain_type<P<Gaussian, IdentityCoordinates, Bilinear>, Kernel>(
                m, "_" + name + "BilinearGaussianChain");
            bind_runtime_chains<Kernel, IdentityCoordinates, Bilinear>(m, "_" + name + "BilinearRuntime");
            bind_chain_type<P<Uniform, Coordinates, Bilinear>, Kernel>(
                m, "_" + name + "BilinearAffineUniformChain");
            bind_chain_type<P<Gaussian, Coordinates, Bilinear>, Kernel>(
                m, "_" + name + "BilinearAffineGaussianChain");
            bind_runtime_chains<Kernel, Coordinates, Bilinear>(m, "_" + name + "BilinearAffineRuntime");
        }
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
    void bind_tempering(nb::module_ &);
} // namespace hopsy::bindings
#endif
