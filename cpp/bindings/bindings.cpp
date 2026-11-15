#include "common.h"

namespace hopsy::bindings {
    template <typename Target, typename C> void bind_problem(nb::module_ &m, const char *name) {
        using Problem = P<Target, C>;
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
    template <typename Target>
    nb::object make_problem(const Geometry &g, const Target &t, nb::handle coordinates) {
        auto target = [&] {
            if constexpr (std::same_as<Target, PythonTarget>)
                return t;
            else
                return std::cref(t);
        }();
        if (coordinates.is_none() || nb::isinstance<IdentityCoordinates>(coordinates))
            return nb::cast(std::make_unique<P<Target>>(std::cref(g), target));
        const auto &map = nb::cast<const Coordinates &>(coordinates);
        return nb::cast(
            std::make_unique<P<Target, Coordinates>>(std::cref(g), target, std::cref(map)));
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
}
