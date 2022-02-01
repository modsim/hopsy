#ifndef HOPSY_MISC_HPP
#define HOPSY_MISC_HPP

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "../../extern/hops/src/hops/hops.hpp"

namespace py = pybind11;

namespace hopsy {
    using VectorType = hops::VectorType;
    using MatrixType = hops::MatrixType;

    std::string get__repr__(const py::object& obj) {
        return py::cast<std::string>(obj.attr("__repr__")());
    }

    template<typename T>
    std::string get__repr__(const std::shared_ptr<T>& t) {
        return py::cast<std::string>(py::cast(static_cast<T*>(t.get())).attr("__repr__")());
    }

    template<typename T>
    std::string get__repr__(T& t) {
        return py::cast<std::string>(py::cast(t).attr("__repr__")());
    }
} // namespace hopsy

#endif // HOPSY_MISC_HPP

