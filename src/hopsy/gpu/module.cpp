#include <pybind11/pybind11.h>

#include "device_bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Compiled part of hopsy.gpu (CUDA backend, experimental).";

    hopsy::gpu::addDevice(m);
}
