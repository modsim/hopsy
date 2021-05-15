#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>

#include <Eigen/Core>

#include "../extern/hops/include/hops/hops.hpp"

#include <string>
#include <memory>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hopsy {
    std::unique_ptr<py::object> polyRoundApi = nullptr;
    std::unique_ptr<py::object> polyRoundChebyshevFinder = nullptr;
    std::unique_ptr<py::object> polyRoundPolytope = nullptr;

    template<typename T>
    hops::Problem<T> round(const hops::Problem<T>& problem) {
        //py::scoped_interpreter guard{};

        if (!polyRoundApi) {
            polyRoundApi = std::make_unique<py::object>(py::module_::import("PolyRound.api").attr("PolyRoundApi"));
        }

        if (!polyRoundPolytope) {
            polyRoundPolytope = std::make_unique<py::object>(py::module_::import("PolyRound.mutable_classes.polytope"));
        }

        py::object polytope = polyRoundPolytope->attr("Polytope")(problem.getA(), problem.getB());

        py::object roundedPolytope = polyRoundApi->attr("simplify_transform_and_round")(polytope);

        hops::Problem<T> roundedProblem(roundedPolytope.attr("A.values").cast<Eigen::MatrixXd>(), 
                                        roundedPolytope.attr("b.values").cast<Eigen::VectorXd>(), 
                                        problem.getModel());

        roundedProblem.setUnroundingTransformation(roundedPolytope.attr("S.values").cast<Eigen::MatrixXd>());
        roundedProblem.setUnroundingShift(roundedPolytope.attr("h.values").cast<Eigen::VectorXd>());       

        return roundedProblem;
    }

    template<typename T>
    Eigen::VectorXd computeChebyshevCenter(const hops::Problem<T>& problem) {
        //py::scoped_interpreter guard{};

        if (!polyRoundChebyshevFinder) {
            polyRoundChebyshevFinder = std::make_unique<py::object>(py::module_::import("PolyRound.static_classes.lp_utils").attr("ChebyshevFinder"));
        }

        if (!polyRoundPolytope) {
            polyRoundPolytope = std::make_unique<py::object>(py::module_::import("PolyRound.mutable_classes.polytope"));
        }

        py::object polytope = polyRoundPolytope->attr("Polytope")(problem.getA(), problem.getB());

        return polyRoundChebyshevFinder->attr("chebyshev_center")(polytope).attr("__getitem__")(0).cast<Eigen::VectorXd>();
    }
}

