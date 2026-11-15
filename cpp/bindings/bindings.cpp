#include "common.h"

namespace hopsy::bindings {
    template <typename Target, typename C, typename G = Geometry>
    void bind_problem(nb::module_ &m, const char *name) {
        using Problem = P<Target, C, G>;
        auto cls = nb::class_<Problem>(m, name);
        // These referents already have independent Python owners retained by
        // the problem factory. reference_internal would create a keep-alive
        // cycle when returning one of those existing objects.
        cls.def_prop_ro("geometry", &Problem::geometry, nb::rv_policy::reference)
            .def_prop_ro("target", [](const Problem &p) -> nb::object {
                if constexpr (std::same_as<Target, PythonTarget>)
                    return p.target().owner;
                else
                    return nb::cast(&p.target(), nb::rv_policy::reference);
            });
        if constexpr (std::same_as<C, Coordinates>)
            cls.def_prop_ro("coordinates", &Problem::coordinates, nb::rv_policy::reference);
        else
            cls.def_prop_ro("coordinates", [](const Problem &) { return IdentityCoordinates{}; });
    }
    template <typename Target, typename G>
    nb::object make_problem(const G &g, const Target &t, nb::handle coordinates) {
        auto target = [&] {
            if constexpr (std::same_as<Target, PythonTarget>)
                return t;
            else
                return std::cref(t);
        }();
        if (coordinates.is_none() || nb::isinstance<IdentityCoordinates>(coordinates))
            return nb::cast(
                std::make_unique<P<Target, IdentityCoordinates, G>>(std::cref(g), target));
        const auto &map = nb::cast<const Coordinates &>(coordinates);
        return nb::cast(
            std::make_unique<P<Target, Coordinates, G>>(std::cref(g), target, std::cref(map)));
    }
    template <typename Problem> void unsupported_bilinear(nb::module_ &m) {
        m.def(
            "bind_chain",
            [](const Problem &, nb::handle kernel, const State &) -> nb::object {
                throw nb::type_error(
                    (nb::cast<std::string>(nb::getattr(kernel.type(), "__name__")) +
                     " does not support BilinearPolytope. Use uniform or Gaussian-displacement "
                     "Hit-and-Run; Gaussian random walk, Dikin, CSmMALA, and overrelaxed "
                     "Hit-and-Run are unsupported.")
                        .c_str());
            },
            "problem"_a, "kernel"_a, "initial_state"_a);
    }

} // namespace hopsy::bindings

NB_MODULE(_hopsy_core, m) {
    using namespace hopsy;
    using namespace hopsy::bindings;
    m.doc() = "Direct C++ problems, kernels, explicit RNGs, and persistent chain bindings.";
    nb::class_<Geometry>(m, "LinearPolytope")
        .def(
            "__init__",
            [](Geometry *self, DenseMatrix A, State b) {
                if (!A.allFinite() || !b.allFinite())
                    throw std::invalid_argument("A and b must be finite");
                new (self) Geometry(std::move(A), std::move(b));
            },
            "A"_a, "b"_a)
        .def_prop_ro("A", [](const Geometry &g) -> DenseMatrix { return g.A(); })
        .def_prop_ro("b", [](const Geometry &g) -> State { return g.b(); })
        .def("num_dimension", &Geometry::num_dimension)
        .def("num_constraints", &Geometry::num_constraints)
        .def(
            "is_feasible",
            [](const Geometry &g, const State &x, double tolerance) {
                if (x.size() != g.num_dimension())
                    throw std::invalid_argument("state dimension mismatch");
                return x.allFinite() && g.is_feasible(x, tolerance);
            },
            "state"_a, "tolerance"_a = 0.);
    nb::class_<Bilinear>(m, "BilinearPolytope")
        .def(
            "__init__",
            [](Bilinear *self, DenseMatrix A, State b, Index split, std::vector<DenseMatrix> cross,
               State rhs, nb::handle affine) {
                DenseMatrix terms =
                    affine.is_none() ? DenseMatrix{} : nb::cast<DenseMatrix>(affine);
                new (self) Bilinear(std::move(A), std::move(b), split, std::move(cross),
                                    std::move(rhs), std::move(terms));
            },
            "A"_a, "b"_a, "split"_a, "cross_terms"_a, "bilinear_bounds"_a, "affine"_a = nb::none())
        .def_prop_ro("A", [](const Bilinear &g) -> DenseMatrix { return g.A(); })
        .def_prop_ro("b", [](const Bilinear &g) -> State { return g.b(); })
        .def_prop_ro("cross_terms", [](const Bilinear &g) { return g.cross_terms(); })
        .def_prop_ro("affine", [](const Bilinear &g) -> DenseMatrix { return g.affine(); })
        .def_prop_ro("bilinear_bounds",
                     [](const Bilinear &g) -> State { return g.bilinear_bounds(); })
        .def_prop_ro("split", &Bilinear::split)
        .def("num_dimension", &Bilinear::num_dimension)
        .def("num_constraints", &Bilinear::num_constraints)
        .def("is_feasible", &Bilinear::is_feasible, "state"_a, "tolerance"_a = 0.)
        .def(
            "line_domain",
            [](const Bilinear &g, const State &z, const State &d) {
                auto workspace = g.create_workspace(z);
                const auto &domain = g.line_domain(z, d, workspace);
                DenseMatrix out(domain.size(), 2);
                for (Index i = 0; i < out.rows(); ++i) {
                    out(i, 0) = domain[i].lower;
                    out(i, 1) = domain[i].upper;
                }
                return out;
            },
            "origin"_a, "direction"_a);
    nb::class_<Uniform>(m, "UniformTarget")
        .def(nb::init<>())
        .def(
            "log_density", [](const Uniform &t, const State &x) { return t.log_density(x); },
            "state"_a);
    nb::class_<Gaussian>(m, "GaussianTarget")
        .def(nb::init<State, const DenseMatrix &>(), "mean"_a, "covariance"_a)
        .def_prop_ro("dimension", &Gaussian::dimension)
        .def("log_density", &Gaussian::log_density, "state"_a)
        .def("log_gradient", &Gaussian::log_gradient, "state"_a);
    nb::class_<IdentityCoordinates>(m, "IdentityCoordinates").def(nb::init<>());
    nb::class_<Coordinates>(m, "AffineCoordinates")
        .def(nb::init<DenseMatrix, State>(), "transformation"_a, "shift"_a)
        .def_prop_ro("matrix", [](const Coordinates &c) -> DenseMatrix { return c.matrix(); })
        .def_prop_ro("shift", [](const Coordinates &c) -> State { return c.shift(); })
        .def_prop_ro("sampling_dimension", &Coordinates::sampling_dimension)
        .def_prop_ro("target_dimension", [](const Coordinates &c) { return c.target_dimension(); })
        .def(
            "to_target",
            [](const Coordinates &c, const State &s) {
                if (s.size() != c.sampling_dimension())
                    throw std::invalid_argument("sampling dimension mismatch");
                State out(c.target_dimension());
                c.to_target(s, out);
                return out;
            },
            "state"_a);
    bind_problem<Uniform, IdentityCoordinates>(m, "_UniformProblem");
    bind_problem<Gaussian, IdentityCoordinates>(m, "_GaussianProblem");
    bind_problem<PythonTarget, IdentityCoordinates>(m, "_PythonProblem");
    bind_problem<Uniform, Coordinates>(m, "_AffineUniformProblem");
    bind_problem<Gaussian, Coordinates>(m, "_AffineGaussianProblem");
    bind_problem<PythonTarget, Coordinates>(m, "_AffinePythonProblem");
    bind_problem<Uniform, IdentityCoordinates, Bilinear>(m, "_BilinearUniformProblem");
    bind_problem<Gaussian, IdentityCoordinates, Bilinear>(m, "_BilinearGaussianProblem");
    bind_problem<PythonTarget, IdentityCoordinates, Bilinear>(m, "_BilinearPythonProblem");
    bind_problem<Uniform, Coordinates, Bilinear>(m, "_BilinearAffineUniformProblem");
    bind_problem<Gaussian, Coordinates, Bilinear>(m, "_BilinearAffineGaussianProblem");
    bind_problem<PythonTarget, Coordinates, Bilinear>(m, "_BilinearAffinePythonProblem");
    m.def(
        "Problem",
        [](const Geometry &g, nb::handle t, nb::handle c) {
            if (t.type().is(nb::type<Uniform>()))
                return make_problem(g, nb::cast<const Uniform &>(t), c);
            if (t.type().is(nb::type<Gaussian>()))
                return make_problem(g, nb::cast<const Gaussian &>(t), c);
            nb::object method = nb::getattr(t, "log_density");
            if (!PyCallable_Check(method.ptr()))
                throw nb::type_error("target.log_density must be callable");
            PythonTarget target{nb::borrow<nb::object>(t), std::move(method),
                                nb::hasattr(t, "log_gradient") ? nb::getattr(t, "log_gradient")
                                                               : nb::object()};
            return make_problem(g, target, c);
        },
        "geometry"_a, "target"_a, "coordinates"_a = nb::none(), nb::keep_alive<0, 1>(),
        nb::keep_alive<0, 2>(), nb::keep_alive<0, 3>());
    m.def(
        "Problem",
        [](const Bilinear &g, nb::handle t, nb::handle c) {
            if (t.type().is(nb::type<Uniform>()))
                return make_problem(g, nb::cast<const Uniform &>(t), c);
            if (t.type().is(nb::type<Gaussian>()))
                return make_problem(g, nb::cast<const Gaussian &>(t), c);
            nb::object method = nb::getattr(t, "log_density");
            if (!PyCallable_Check(method.ptr()))
                throw nb::type_error("target.log_density must be callable");
            PythonTarget target{nb::borrow<nb::object>(t), std::move(method),
                                nb::hasattr(t, "log_gradient") ? nb::getattr(t, "log_gradient")
                                                               : nb::object()};
            return make_problem(g, target, c);
        },
        "geometry"_a, "target"_a, "coordinates"_a = nb::none(), nb::keep_alive<0, 1>(),
        nb::keep_alive<0, 2>(), nb::keep_alive<0, 3>());
    nb::class_<RandomNumberGenerator>(m, "RandomNumberGenerator")
        .def(nb::new_([](std::uint64_t seed) {
                 return make_sampling_resource<RandomNumberGenerator>(seed);
             }),
             "seed"_a = 0)
        .def(
            "seed",
            [](RandomNumberGenerator &rng, std::uint64_t value) {
                ExclusiveLease<RandomNumberGenerator> lease(rng);
                rng.seed(value);
            },
            "seed"_a);
    nb::class_<CyclingCoordinateDirectionGenerator>(m, "CyclingCoordinateDirectionGenerator")
        .def(nb::init<Index>(), "num_dimensions"_a);
    nb::class_<RandomCoordinateDirectionGenerator>(m, "RandomCoordinateDirectionGenerator")
        .def(nb::init<Index>(), "num_dimensions"_a);
    nb::class_<RandomDirectionGenerator<double>>(m, "RandomDirectionGenerator")
        .def(nb::init<Index>(), "num_dimensions"_a);
    nb::class_<UniformLineKernel>(m, "UniformLineKernel").def(nb::init<>());
    nb::class_<GaussianLineKernel>(m, "GaussianLineKernel")
        .def(nb::init<double>(), "standard_deviation"_a);
    nb::class_<OverRelaxedLineKernel>(m, "OverRelaxedLineKernel").def(nb::init<>());
    bind_gaussian_types(m);
    bind_dense_types(m);
    bind_cycling_types(m);
    bind_random_coordinate_types(m);
    unsupported_bilinear<P<Uniform, IdentityCoordinates, Bilinear>>(m);
    unsupported_bilinear<P<Gaussian, IdentityCoordinates, Bilinear>>(m);
    unsupported_bilinear<P<PythonTarget, IdentityCoordinates, Bilinear>>(m);
    unsupported_bilinear<P<Uniform, Coordinates, Bilinear>>(m);
    unsupported_bilinear<P<Gaussian, Coordinates, Bilinear>>(m);
    unsupported_bilinear<P<PythonTarget, Coordinates, Bilinear>>(m);
}
