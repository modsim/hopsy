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

    std::string removeTrailingZeros(double number) {
        std::string str = std::to_string(number);
        str.erase(str.find_last_not_of('0') + 1, std::string::npos); // remove trailing zeros
        return str;
    }

} // namespace hopsy

#endif // HOPSY_MISC_HPP
