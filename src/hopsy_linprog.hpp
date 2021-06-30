#ifndef HOPSY_LINPROG_HPP
#define HOPSY_LINPROG_HPP

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
    //std::unique_ptr<py::object> polyRoundApi = nullptr;
    //std::unique_ptr<py::object> polyRoundChebyshevFinder = nullptr;
    //std::unique_ptr<py::object> polyRoundPolytope = nullptr;

    //template<typename T>
    //hops::Problem<T> round(const hops::Problem<T>& problem) {
    //    //py::scoped_interpreter guard{};

    //    if (!polyRoundApi) {
    //        polyRoundApi = std::make_unique<py::object>(py::module_::import("PolyRound.api").attr("PolyRoundApi"));
    //    }

    //    if (!polyRoundPolytope) {
    //        polyRoundPolytope = std::make_unique<py::object>(py::module_::import("PolyRound.mutable_classes.polytope"));
    //    }

    //    py::object polytope = polyRoundPolytope->attr("Polytope")(problem.getA(), problem.getB());

    //    py::object roundedPolytope = polyRoundApi->attr("simplify_transform_and_round")(polytope);

    //    hops::Problem<T> roundedProblem(roundedPolytope.attr("A.values").cast<Eigen::MatrixXd>(), 
    //                                    roundedPolytope.attr("b.values").cast<Eigen::VectorXd>(), 
    //                                    problem.getModel());

    //    roundedProblem.setUnroundingTransformation(roundedPolytope.attr("S.values").cast<Eigen::MatrixXd>());
    //    roundedProblem.setUnroundingShift(roundedPolytope.attr("h.values").cast<Eigen::VectorXd>());       

    //    return roundedProblem;
    //}

    //template<typename T>
    //Eigen::VectorXd computeChebyshevCenter(const hops::Problem<T>& problem) {
    //    //py::scoped_interpreter guard{};

    //    if (!polyRoundChebyshevFinder) {
    //        polyRoundChebyshevFinder = std::make_unique<py::object>(py::module_::import("PolyRound.static_classes.lp_utils").attr("ChebyshevFinder"));
    //    }

    //    if (!polyRoundPolytope) {
    //        polyRoundPolytope = std::make_unique<py::object>(py::module_::import("PolyRound.mutable_classes.polytope"));
    //    }

    //    py::object polytope = polyRoundPolytope->attr("Polytope")(problem.getA(), problem.getB());

    //    return polyRoundChebyshevFinder->attr("chebyshev_center")(polytope).attr("__getitem__")(0).cast<Eigen::VectorXd>();
    //}
    
    template<typename T>
    hops::Problem<T> round(const hops::Problem<T>& problem) {
        py::object polyRoundApi = py::module_::import("PolyRound.api").attr("PolyRoundApi");
        py::object polyRoundPolytope = py::module_::import("PolyRound.mutable_classes.polytope");

        py::object polytope = polyRoundPolytope.attr("Polytope")(problem.getA(), problem.getB());

        py::object simplifiedPolytope = polyRoundApi.attr("simplify_polytope")(polytope);

        py::dict local;
        local["p"] = simplifiedPolytope;

        py::exec(R"(
            import numpy as np
            import pandas as pd
            number_of_reactions = p.A.shape[1]
            p.transformation = pd.DataFrame(np.identity(number_of_reactions))
            p.transformation.index = [str(i) for i in range(p.transformation.to_numpy().shape[0])]
            p.transformation.columns = [str(i) for i in range(p.transformation.to_numpy().shape[1])]
            p.shift = pd.Series(np.zeros(number_of_reactions))
        )", local);

        py::object roundedPolytope = polyRoundApi.attr("round_polytope")(simplifiedPolytope);
        //py::object roundedPolytope = polyRoundApi.attr("round_polytope")(local["p"]);

        //local["rounded_polytope"] = roundedPolytope;

        //py::exec(R"(
        //    print("A", rounded_polytope.A)
        //    print("b", rounded_polytope.b)
        //    print("S", rounded_polytope.S)
        //    print("h", rounded_polytope.h)
        //)", local);

        hops::Problem<T> roundedProblem(roundedPolytope.attr("A").attr("values").cast<Eigen::MatrixXd>(), 
                                        roundedPolytope.attr("b").attr("values").cast<Eigen::VectorXd>(), 
                                        problem.getModel());

        roundedProblem.setUnroundingTransformation(roundedPolytope.attr("transformation").attr("values").cast<Eigen::MatrixXd>());
        roundedProblem.setUnroundingShift(roundedPolytope.attr("shift").attr("values").cast<Eigen::VectorXd>());       

        return roundedProblem;
    }

    template<typename T>
    Eigen::VectorXd computeChebyshevCenter(const hops::Problem<T>& problem) {
        py::object polyRoundChebyshevFinder = py::module_::import("PolyRound.static_classes.lp_utils").attr("ChebyshevFinder");
        py::object polyRoundPolytope = py::module_::import("PolyRound.mutable_classes.polytope");

        py::object polytope = polyRoundPolytope.attr("Polytope")(problem.getA(), problem.getB());
        py::object settings = py::module_::import("PolyRound.settings").attr("PolyRoundSettings")();

        return polyRoundChebyshevFinder.attr("chebyshev_center")(polytope, settings).attr("__getitem__")(0).cast<Eigen::VectorXd>();
    }
}

#endif // HOPSY_LINPROG
