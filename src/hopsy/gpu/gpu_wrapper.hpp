#ifndef GPU_WRAPPER_HPP
#define GPU_WRAPPER_HPP


#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include "dvector.h"
#include "dmatrix.h"
#include "gpusamplers.h"

namespace hopsy {

    void addGPUSupport(py::module& m) {

        /*****Expose CUDA device selection with available devices*******/
        // TODO: Implement device selection logic

        /*****Expose GPU Vector Class*******/

        py::class_<hopsy::GPU::DVector<double>, std::shared_ptr<hopsy::GPU::DVector<double>>>(m, "DVector", "GPU vector of doubles")
        
        // DVector(int len) construct from length
        .def(py::init<int>(), py::arg("length"))

        // DVector(int len, double value) construct from length and value
        .def(py::init<int, double>(), py::arg("length"), py::arg("value"))

        // DVector(const HVector& v) construct from Eigen vector
        .def(py::init<const hopsy::GPU::HVector<double>&>(), py::arg("host_vector"))

        // toHost() method to copy data back to host
        .def("toHost", &hopsy::GPU::DVector<double>::toHost)

        // length as read-only attribute
        .def_readonly("len", &hopsy::GPU::DVector<double>::len)

        // Expose the copy constructor explicitly (copying the DVector object)
        .def(py::init<const hopsy::GPU::DVector<double>&>(), "Copy constructor");


        /*****Expose GPU Matrix Class*******/

        py::class_<hopsy::GPU::DMatrix<double>, std::shared_ptr<hopsy::GPU::DMatrix<double>>>(m, "DMatrix", "GPU matrix of doubles")

        // DMatrix(const HMatrix& m) construct from Eigen matrix
        .def(py::init<const hopsy::GPU::HMatrix<double>&>(), py::arg("host_matrix"))

        // DMatrix(int rows, int cols) construct from rows and columns
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))

        // DMatrix(const HVector& v, int cols) construct from Eigen vector and number of columns
        .def(py::init<const hopsy::GPU::HVector<double>&, int>(), py::arg("host_vector"), py::arg("cols"))

        // DMatrix(const DVector& v, int cols) construct from GPU vector and number of columns
        .def(py::init<const hopsy::GPU::DVector<double>&, int>(), py::arg("device_vector"), py::arg("cols"))

        // Copy constructor
        .def(py::init<const hopsy::GPU::DMatrix<double>&>(), "Copy constructor")

        // toHost() method
        .def("toHost", &hopsy::GPU::DMatrix<double>::toHost)

        // Read-only attributes
        .def_readonly("rows", &hopsy::GPU::DMatrix<double>::rows)
        .def_readonly("cols", &hopsy::GPU::DMatrix<double>::cols);


        /*****Expose samplers******/

        m.def(  "GPUWarmUp",
                &hopsy::GPU::WarumUp,
                py::return_value_policy::move,
                py::arg("A_d"),
                py::arg("b_d"),
                py::arg("x0_d"),
                py::arg("nwarmup"),
                py::arg("nchains"),
                py::arg("tpb_rd") = -1,
                py::arg("tpb_ss") = -1);

        m.def(  "GPUCoordinateHitAndRun",
                &hopsy::GPU::CoordinateHitAndRun,
                py::return_value_policy::move,
                py::arg("A_d"),
                py::arg("b_d"),
                py::arg("X_d"),
                py::arg("nspc"),
                py::arg("thinningfactor"),
                py::arg("nchains"),
                py::arg("tpb_ss") = -1);

    }
}

#endif // GPU_WRAPPER_HPP