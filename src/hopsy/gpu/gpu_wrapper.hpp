#ifndef HOPSY_PROBLEM_HPP
#define HOPSY_PROBLEM_HPP


#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace hopsy {
    namespace GPU {
    // TODO Harris: Add code for GPU CHAR

        template <typename Real>
        HMatrix<Real> CoordinatedHitAndRun(const HMatrix<Real>& A, const HVector<Real>& b, const HVector<Real>& x0, const Config& config) {}



        // TODO Harris: add function for adding GPU support
        void addGPUSupport(py::module& m) {

            m.def("gpu_uniform_coordinate_hit_and_run", &CoordinateHitAndRun)



            // TODO: replace Problem by CHAR
            py::class_<CONFIG>(m, "CHARgpuConfig")
                .def()
                ;
            py::class_<Problem>(m, "Problem")
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
                .def_readwrite("original_A", &Problem::original_A, doc::Problem::original_A)
                .def_readwrite("original_b", &Problem::original_b, doc::Problem::original_b)
                .def("__repr__", &Problem::__repr__)
                .def(py::pickle([] (const Problem& self) { // __getstate__
                                    /* Return a tuple that fully encodes the state of the object */
                                    return py::make_tuple(self.A,
                                                          self.b,
                                                          static_cast<hopsy::Model*>(self.model.get()),
                                                          self.startingPoint,
                                                          self.transformation,
                                                          self.shift,
                                                          self.original_A,
                                                          self.original_b
                                                          );
                                },
                                [] (py::tuple t) { // __setstate__
                                    if (t.size() != 8) throw std::runtime_error("Tried to build hopsy.Model with invalid state.");

                                    /* Create a new C++ instance */
                                    Problem p(t[0].cast<MatrixType>(),
                                              t[1].cast<VectorType>(),
                                              t[2].cast<Model*>(),
                                              t[3].cast<std::optional<VectorType>>(),
                                              t[4].cast<std::optional<MatrixType>>(),
                                              t[5].cast<std::optional<VectorType>>());
                                    p.original_A = t[6].cast<MatrixType>();
                                    p.original_b = t[7].cast<VectorType>();

                                    return p;
                                }
                        ))
            ;
        }
    }
}

#endif // HOPSY_PROBLEM_HPP
