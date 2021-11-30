#ifndef HOPSY_TUNING_HPP
#define HOPSY_TUNING_HPP

#include <memory>
#include <tuple>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>
#include <pybind11/trampoline_self_life_support.h>

#include "../../extern/hops/src/hops/hops.hpp"

#include "misc.hpp"

namespace py = pybind11;

namespace hopsy {
    class PyTuningTarget {
    public:
        PyTuningTarget() = default;

		PyTuningTarget(py::object pyObj) : pyObj(std::move(pyObj)) {};

        std::tuple<double, double> operator() (const VectorType x) {
            return pyObj.attr("__call__")(x).cast<std::tuple<double, double>>();
        }

        std::string getName() {
            try {
                return pyObj.attr("get_name")().cast<std::string>();
            } catch (...) {
                return "";
            }
        }

        std::vector<std::shared_ptr<hops::MarkovChain>> markovChain;
        std::vector<hops::RandomNumberGenerator>* randomNumberGenerator;
        unsigned long numberOfTestSamples;

	private:
		py::object pyObj;
    };
}

#endif // HOPSY_TUNING_HPP
