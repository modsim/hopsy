#ifndef HOPSY_PROBLEM_HPP
#define HOPSY_PROBLEM_HPP

#include <memory>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
//#include <pybind11/smart_holder.h>
#include <pybind11/pybind11.h>
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

        Problem(const Problem& other) :
                A(other.A),
                b(other.b),
                //model(std::move(other.model->copyModel())),
                startingPoint(other.startingPoint),
                transformation(other.transformation),
                shift(other.shift) {
            if(other.model) this->model = std::move(other.model->copyModel());
        }

        Problem(const MatrixType& A,
                const VectorType& b,
                const Model* model,
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

        Problem(const MatrixType& A,
                const VectorType& b,
                const std::unique_ptr<Model> model,
                const std::optional<VectorType> startingPoint,
                const std::optional<MatrixType> transformation,
                const std::optional<VectorType> shift) :
                A(A),
                b(b),
                startingPoint(startingPoint),
                transformation(transformation),
                shift(shift) {
            if(model) this->model = model->copyModel();
        }

          std::variant<py::object, std::unique_ptr<Model>> getModel() {
            if(model) {
                std::shared_ptr<Model> sharedModelPtr = std::move(model->copyModel());
                std::shared_ptr<PyModel> pyModelPtr = std::dynamic_pointer_cast<PyModel>(sharedModelPtr);
                if(pyModelPtr) {
                    return pyModelPtr->pyObj;
                }
                else {
                    return model->copyModel();
                }
            }
            else return nullptr;
        }

        void setModel(std::variant<py::object, std::unique_ptr<Model>> model) {
            std::unique_ptr<Model> modelPtr;
            try {
                py::object object = std::get<py::object>(model);
                this->model = std::make_unique<PyModel>(PyModel(object));
            }
            catch(std::bad_variant_access&) {
                if(!std::get<std::unique_ptr<Model>>(model)) {
                    throw std::runtime_error("Pointer to model is null.");
                }
                this->model = std::get<std::unique_ptr<Model>>(model)->copyModel();
            }
        }

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
        }
    };

    MatrixType computeSqrtMaximumVolumeEllipsoid(const Problem& problem) {
        MatrixType sqrtMve;

        py::dict local;
        local["A"] = problem.A;
        local["b"] = problem.b;

        py::exec(R"(
            from numpy import identity, zeros
            from pandas import DataFrame, Series

            import os
            import sys

            try:
                from PolyRound.api import PolyRoundApi as prapi
                from PolyRound.mutable_classes.polytope import Polytope
                from PolyRound.static_classes.rounding.maximum_volume_ellipsoid import MaximumVolumeEllipsoidFinder

                polytope = Polytope(A, b)
                polytope = prapi.simplify_polytope(polytope, LP().settings)

                if polytope.S is not None:
                    polytope = prapi.transform_polytope(polytope, LP().settings)
                else:
                    number_of_reactions = polytope.A.shape[1]
                    polytope.transformation = DataFrame(identity(number_of_reactions))
                    polytope.transformation.index = [str(i) for i in range(polytope.transformation.to_numpy().shape[0])]
                    polytope.transformation.columns = [str(i) for i in range(polytope.transformation.to_numpy().shape[1])]
                    polytope.shift = Series(zeros(number_of_reactions))

                MaximumVolumeEllipsoidFinder.iterative_solve(polytope, LP().settings)
                sqrt_mve = polytope.transformation.values
            except:
                sqrt_mve = identity(A.shape[1])
        )", local);

        sqrtMve = local["sqrt_mve"].cast<MatrixType>();
        return sqrtMve;
    }

    void addProblem(py::module& m) {
        py::class_<Problem>(m, "Problem", doc::Problem::base)
            .def(py::init<const MatrixType&,
                          const VectorType&,
                          const Model*,
                          const std::optional<VectorType>,
                          const std::optional<MatrixType>,
                          const std::optional<VectorType>>(),
                    doc::Problem::__init__,
                    py::arg("A"),
                    py::arg("b"),
                    py::arg("model") = static_cast<Model*>(nullptr),
                    py::arg("starting_point") = py::none(),
                    py::arg("transformation") = py::none(),
                    py::arg("shift") = py::none()
            )
            .def(py::init([] (const MatrixType& A,
                          const VectorType& b,
                          const py::object model,
                          const std::optional<VectorType> startingPoint,
                          const std::optional<MatrixType> transformation,
                          const std::optional<VectorType> shift) {
                        return Problem(A, b, std::make_unique<PyModel>(PyModel(model)), startingPoint, transformation, shift);
                    }),
                    doc::Problem::__init__,
                    py::arg("A"),
                    py::arg("b"),
                    py::arg("model").none(false),
                    py::arg("starting_point") = py::none(),
                    py::arg("transformation") = py::none(),
                    py::arg("shift") = py::none()
            )
            .def("slacks", [] (const Problem& self, const VectorType& point) {
               return self.b - self.A * point;
            }, doc::Problem::slacks)
            .def_readwrite("A", &Problem::A, doc::Problem::A)
            .def_readwrite("b", &Problem::b, doc::Problem::b)
            .def_readwrite("starting_point", &Problem::startingPoint, doc::Problem::startingPoint)
            .def_property("model", &Problem::getModel, &Problem::setModel, doc::Problem::model)
            .def_readwrite("transformation", &Problem::transformation, doc::Problem::transformation)
            .def_readwrite("shift", &Problem::shift, doc::Problem::shift)
            .def("__repr__", &Problem::__repr__)
            .def(py::pickle([] (const Problem& self) { // __getstate__
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
                                Problem p(t[0].cast<MatrixType>(),
                                          t[1].cast<VectorType>(),
                                          t[2].cast<Model*>(),
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
