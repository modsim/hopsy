#ifndef HOPSY_PROBLEM_HPP
#define HOPSY_PROBLEM_HPP

#include <memory>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>

#include "misc.hpp"
#include "model.hpp"

namespace hopsy {
    struct Problem {
        MatrixType A;
        VectorType b;
        Model* model;
        std::optional<VectorType> startingPoint;
        std::optional<MatrixType> transformation;
        std::optional<VectorType> shift;

        Problem(MatrixType A, 
                VectorType b, 
                Model* model, 
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

        Model* getModel() const { return model; }
        void setModel(Model* model) { this->model = model; }

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

    void addProblem(py::module& m) {
        py::class_<Problem>(m, "Problem")
            .def(py::init<MatrixType, 
                          VectorType, 
                          Model*, 
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
