#ifndef HOPSY_RANDOM_HPP
#define HOPSY_RANDOM_HPP

#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>

#include <Eigen/Core>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>
#include <pybind11/trampoline_self_life_support.h>

#include "../../extern/hops/src/hops/hops.hpp"

namespace py = pybind11;

namespace hopsy {
    struct RandomNumberGenerator {
        std::optional<unsigned int> seed;
        std::optional<unsigned int> stream;
        hops::RandomNumberGenerator rng;

        RandomNumberGenerator() {
            // 
        }

        RandomNumberGenerator(unsigned int seed) :
            seed(seed),
            rng(seed) {
            // 
        }

        RandomNumberGenerator(unsigned int seed, unsigned int stream) :
            seed(seed),
            stream(stream),
            rng(seed, stream) {
            // 
        }

        unsigned int operator()() {
            return rng();
        }

        std::string __repr__() const {
            std::string repr = "hopsy.RandomNumberGenerator("; 
            repr += (seed ? "seed=" + std::to_string(*seed) : "");
            repr += (stream ? ", stream=" + std::to_string(*stream) : "");
            repr += ")";
            return repr;
        }
    };

    using Normal = std::normal_distribution<double>;
    using Uniform = std::uniform_real_distribution<double>;

    void addRandom(py::module& m) {
        py::class_<RandomNumberGenerator>(m, "RandomNumberGenerator")
            .def(py::init<>())
            .def(py::init<unsigned int>(), py::arg("seed"))
            .def(py::init<unsigned int, unsigned int>(), py::arg("seed"), py::arg("stream"))
            .def("__call__", [] (RandomNumberGenerator& self) { return self(); })
            .def("__repr__", &RandomNumberGenerator::__repr__)
        ;

        py::class_<Uniform>(m, "Uniform")
            .def(py::init<double, double>(), py::arg("a") = 0, py::arg("b") = 1)
            .def("__call__", [] (Uniform& self, RandomNumberGenerator& rng) -> double { 
                    return self(rng.rng); 
                })
            .def("__repr__", [] (Uniform& self) -> std::string {
                    std::string repr = "hopsy.Uniform(";
                    repr += "a=" + std::to_string(self.a()) + ", ";
                    repr += "b=" + std::to_string(self.b()) + ")";
                    return repr;
                })
        ;

        py::class_<Normal>(m, "Normal")
            .def(py::init<double, double>(), py::arg("mean") = 0, py::arg("stddev") = 1)
            .def("__call__", [] (Normal& self, RandomNumberGenerator& rng) -> double { 
                    return self(rng.rng); 
                })
            .def("__repr__", [] (hopsy::Normal& self) -> std::string {
                    std::string repr = "hopsy.Normal(";
                    repr += "mean=" + std::to_string(self.mean()) + ", ";
                    repr += "stddev=" + std::to_string(self.stddev()) + ")";
                    return repr;
                })
        ;
    }

} // namespace hopsy

#endif // HOPSY_RANDOM_HPP

