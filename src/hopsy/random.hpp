#ifndef HOPSY_RANDOM_HPP
#define HOPSY_RANDOM_HPP

#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include <Eigen/Core>

#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>
#include <pybind11/trampoline_self_life_support.h>

#include "../../extern/hops/src/hops/hops.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"

namespace py = pybind11;

namespace hopsy {
    namespace {
      std::string stateToString(hops::RandomNumberGenerator::state_type state) {
        std::string str;
        do {
            int digit = state % 10;
            str = std::to_string(digit) + str;
            state = (state - digit) / 10;
        } while (state != 0);
        return str;
      }

      hops::RandomNumberGenerator::state_type stringToState(std::string str) {
        hops::RandomNumberGenerator::state_type state = 0;

        int place = 0;
        for(auto it = str.rbegin(); it!=str.rend(); it++) {
          decltype(state) digit = std::stoi(std::string(1, *it))*std::pow(10, place);
          state += digit;
          place++;
        }
        return state;
      }
    }

    struct RandomNumberGenerator {
        unsigned int seed;
        unsigned int stream;
        hops::RandomNumberGenerator rng;

        RandomNumberGenerator(unsigned int seed = 0, unsigned int stream = 0) :
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
            repr += (seed ? "seed=" + std::to_string(seed) : "");
            repr += (seed && stream ? ", " : "");
            repr += (stream ? "stream=" + std::to_string(stream) : "");
            repr += ")";
            return repr;
        }
    };

    using Normal = std::normal_distribution<double>;
    using Uniform = std::uniform_real_distribution<double>;

    void addRandom(py::module& m) {
        py::class_<RandomNumberGenerator>(m, "RandomNumberGenerator", doc::RandomNumberGenerator::base)
            .def(py::init<>(), doc::RandomNumberGenerator::__init__)
            .def(py::init<unsigned int>(), 
                    doc::RandomNumberGenerator::__init__, 
                    py::arg("seed"))
            .def(py::init<unsigned int, unsigned int>(), 
                    doc::RandomNumberGenerator::__init__, 
                    py::arg("seed"), 
                    py::arg("stream"))
            .def("__call__", [] (RandomNumberGenerator& self) { return self(); }, 
                    doc::RandomNumberGenerator::__call__)
            .def("__repr__", &RandomNumberGenerator::__repr__)
            .def(py::pickle([] (const RandomNumberGenerator& self) {
                        hops::RandomNumberGenerator rng(self.seed, self.stream);
                        auto state = stateToString(self.rng - rng);
                        return py::make_tuple(self.seed, self.stream, state);
                    },
                    [] (py::tuple t) {
                        if (t.size() != 3) throw std::runtime_error("Tried to build hopsy.RandomNumberGenerator with invalid state.");

                        RandomNumberGenerator rng(t[0].cast<unsigned int>(), t[1].cast<unsigned int>());
                        rng.rng.advance(stringToState(t[2].cast<std::string>()));
                        return rng;
                    }))
        ;

        py::class_<Uniform>(m, "Uniform", doc::Uniform::base)
            .def(py::init<double, double>(), 
                    doc::Uniform::__init__,
                    py::arg("a") = 0, 
                    py::arg("b") = 1)
            .def("__call__", [] (Uniform& self, RandomNumberGenerator& rng) -> double { 
                        return self(rng.rng); 
                    },
                    doc::Uniform::__call__
                )
            .def_property("a", &Uniform::a, [] (Uniform& self, double a) {
                        self = Uniform(a, self.b());
                    })
            .def_property("b", &Uniform::a, [] (Uniform& self, double b) {
                        self = Uniform(self.a(), b);
                    })
            .def("__repr__", [] (Uniform& self) -> std::string {
                        std::string repr = "hopsy.Uniform(";
                        repr += "a=" + std::to_string(self.a()) + ", ";
                        repr += "b=" + std::to_string(self.b()) + ")";
                        return repr;
                    })
        ;

        py::class_<Normal>(m, "Normal", doc::Normal::base)
            .def(py::init<double, double>(), py::arg("mean") = 0, py::arg("stddev") = 1)
            .def("__call__", [] (Normal& self, RandomNumberGenerator& rng) -> double { 
                        return self(rng.rng); 
                    },
                    doc::Normal::__call__)
            .def_property("mean", &Normal::mean, [] (Normal& self, double mean) {
                        self = Normal(mean, self.stddev());
                    })
            .def_property("stddev", &Normal::stddev, [] (Normal& self, double stddev) {
                        self = Normal(self.mean(), stddev);
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

