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
        std::unique_ptr<Model> model;
        std::optional<VectorType> startingPoint;
        std::optional<MatrixType> transformation;
        std::optional<VectorType> shift;

        Problem(const MatrixType& A, 
                const VectorType& b, 
                const std::unique_ptr<Model>& model, 
                const std::optional<VectorType> startingPoint, 
                const std::optional<MatrixType> transformation, 
                const std::optional<VectorType> shift) : 
                A(A),
                b(b),
                startingPoint(startingPoint),
                transformation(transformation),
                shift(shift) { 
            if(model) this->model = std::move(model->copyModel());
        }

        std::unique_ptr<Model>& getModel() { return model; }
        void setModel(std::unique_ptr<Model> model) { if(model) this->model = std::move(model->copyModel()); }

        std::string __repr__() const {
            Model* _model = static_cast<Model*>(model.get());
            std::string repr = "hopsy.Problem(A=" + get__repr__(A) + ", ";
            repr += "b=" + get__repr__(b);
            repr += ( model ? ", model=" + get__repr__(_model) : "" );
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
            .def(py::init<const MatrixType&, 
                          const VectorType&, 
                          const std::unique_ptr<Model>&, 
                          const std::optional<VectorType>, 
                          const std::optional<MatrixType>, 
                          const std::optional<VectorType>>(),
                    py::arg("A"), 
                    py::arg("b"), 
                    py::arg("model") = py::none(), 
                    py::arg("starting_point") = py::none(), 
                    py::arg("transformation") = py::none(), 
                    py::arg("shift") = py::none())
            .def_readwrite("A", &Problem::A)
            .def_readwrite("b", &Problem::b)
            .def_readwrite("starting_point", &Problem::startingPoint)
            .def_property("model", &Problem::getModel, &Problem::setModel)
            .def_readwrite("transformation", &Problem::transformation)
            .def_readwrite("shift", &Problem::shift)
            .def("__repr__", &Problem::__repr__)
            .def(py::pickle([] (const hopsy::Problem& self) { // __getstate__
                                /* Return a tuple that fully encodes the state of the object */
                                return py::make_tuple(self.A, 
                                                      self.b, 
                                                      static_cast<hopsy::Model*>(self.model.get()), 
                                                      self.startingPoint, 
                                                      self.transformation, 
                                                      self.shift);
                            },
                            [] (py::tuple t) { // __setstate__
                                if (t.size() != 6) throw std::runtime_error("Tried to build hopsy.Model with invalid state.");

                                /* Create a new C++ instance */
                                hopsy::Problem p(t[0].cast<Eigen::MatrixXd>(),
                                                 t[1].cast<Eigen::VectorXd>(),
                                                 t[2].cast<std::unique_ptr<hopsy::Model>>(),
                                                 t[3].cast<std::optional<VectorType>>(),
                                                 t[4].cast<std::optional<MatrixType>>(),
                                                 t[5].cast<std::optional<VectorType>>());

                                return p;
                            }
                    ))
        ;
    }
}

#endif // HOPSY_PROBLEM_HPP
