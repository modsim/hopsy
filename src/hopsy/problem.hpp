#ifndef HOPSY_PROBLEM_HPP
#define HOPSY_PROBLEM_HPP

#include <memory>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>

#include "misc.hpp"
#include "model.hpp"

namespace hopsy {
    struct Problem {
        MatrixType A;
        VectorType b;
        std::shared_ptr<Model> model;
        std::optional<VectorType> startingPoint;
        std::optional<MatrixType> transformation;
        std::optional<VectorType> shift;

        Problem(MatrixType A, 
                VectorType b, 
                std::shared_ptr<Model> model, 
                std::optional<VectorType> startingPoint, 
                std::optional<MatrixType> transformation, 
                std::optional<VectorType> shift) : 
                A(A),
                b(b),
                model(model),
                startingPoint(startingPoint),
                transformation(transformation),
                shift(shift) { 
        }

        std::shared_ptr<Model> getModel() const { return model; }
        void setModel(std::shared_ptr<Model> model) { this->model = model; }

        std::string __repr__() const {
            std::string repr = "hopsy.Problem(A=" + get__repr__(A) + ", ";
            repr += "b=" + get__repr__(b);
            repr += ( model ? ", model=" + get__repr__(model) : "" );
            repr += ( startingPoint ? ", starting_point=" + get__repr__(startingPoint) : "" );
            repr += ( transformation ? ", transformation=" + get__repr__(transformation) : "" );
            repr += ( shift ? ", shift=" + get__repr__(shift) : "" );
            repr += ")";
            return repr;
            ;
        }
    };

    MatrixType computeSqrtMaximumVolumeEllipsoid(const Problem& problem) {
        MatrixType sqrtMve;

        py::dict local;
        local["A"] = problem.A;
        local["b"] = problem.b;
        local["sqrt_mve"] = sqrtMve;

        py::exec(R"(
            from numpy import identity, zeros
            from pandas import DataFrame, Series

            import os
            import sys

            try:
                from PolyRound.api import PolyRoundApi as prapi
                from PolyRound.mutable_classes.polytope import Polytope
                from PolyRound.static_classes.rounding.maximum_volume_ellipsoid import MaximumVolumeEllipsoidFinder

                from PolyRound.settings import PolyRoundSettings

                polytope = Polytope(A, b)

                polytope = prapi.simplify_polytope(polytope)

                number_of_reactions = polytope.A.shape[1]
                polytope.transformation = DataFrame(identity(number_of_reactions))
                polytope.transformation.index = [str(i) for i in range(polytope.transformation.to_numpy().shape[0])]
                polytope.transformation.columns = [str(i) for i in range(polytope.transformation.to_numpy().shape[1])]
                polytope.shift = Series(zeros(number_of_reactions))

                MaximumVolumeEllipsoidFinder.iterative_solve(polytope, PolyRoundSettings())
                sqrt_mve = polytope.transformation.values
            except:
                sqrt_mve = identity(A.shape[1])
        )", local);

        return sqrtMve;
    }


    void addProblem(py::module& m) {
        py::class_<Problem>(m, "Problem")
            .def(py::init<MatrixType, 
                          VectorType, 
                          std::shared_ptr<Model>, 
                          std::optional<VectorType>, 
                          std::optional<MatrixType>, 
                          std::optional<VectorType>>(),
                    py::arg("A"), 
                    py::arg("b"), 
                    py::arg("model") = py::none(), 
                    py::arg("starting_point") = py::none(), 
                    py::arg("transformation") = py::none(), 
                    py::arg("shift") = py::none())
            .def_readwrite("A", &Problem::A)
            .def_readwrite("b", &Problem::A)
            .def_readwrite("starting_point", &Problem::startingPoint)
            .def_property("model", &Problem::getModel, &Problem::setModel)
            .def_readwrite("transformation", &Problem::transformation)
            .def_readwrite("shift", &Problem::shift)
            .def("__repr__", &Problem::__repr__)
        ;
    }
}

#endif // HOPSY_PROBLEM_HPP
