#ifndef HOPSY_GPU_DEVICE_BINDINGS_HPP
#define HOPSY_GPU_DEVICE_BINDINGS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "device.hpp"

namespace py = pybind11;

namespace hopsy {
    namespace gpu {
        inline void addDevice(py::module &m) {
            py::class_<DeviceProperties>(m, "DeviceProperties",
                                         "Properties of a CUDA device.")
                    .def_readonly("index", &DeviceProperties::index)
                    .def_readonly("name", &DeviceProperties::name)
                    .def_readonly("major", &DeviceProperties::major,
                                  "Compute capability, major version.")
                    .def_readonly("minor", &DeviceProperties::minor,
                                  "Compute capability, minor version.")
                    .def_readonly("total_memory", &DeviceProperties::totalMemory,
                                  "Global memory in bytes.")
                    .def("__repr__", [](const DeviceProperties &p) {
                        return "DeviceProperties(index=" + std::to_string(p.index) +
                               ", name='" + p.name +
                               "', compute_capability=" + std::to_string(p.major) + "." +
                               std::to_string(p.minor) +
                               ", total_memory=" + std::to_string(p.totalMemory) + ")";
                    });

            m.def("device_count", &deviceCount,
                  "Number of visible CUDA devices (0 if there is no usable device or "
                  "driver).");
            m.def("is_available", &isAvailable,
                  "True if at least one CUDA device can be used.");
            m.def("runtime_version", &runtimeVersion,
                  "CUDA runtime version hopsy.gpu was built against, e.g. 12020 for "
                  "12.2.");
            m.def("driver_version", &driverVersion,
                  "Latest CUDA version supported by the installed driver, 0 if there is "
                  "no driver.");
            m.def("device_properties", &deviceProperties, py::arg("index") = 0,
                  "Properties of CUDA device `index`. Raises IndexError for an invalid "
                  "index.");
        }
    }// namespace gpu
}// namespace hopsy

#endif// HOPSY_GPU_DEVICE_BINDINGS_HPP
