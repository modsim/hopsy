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

    struct RandomNumberGenerator {
        hops::RandomNumberGenerator rng;

        RandomNumberGenerator(unsigned int seed = 0, unsigned int stream = 0) :
            rng(seed, stream) {
            //
        }

      RandomNumberGenerator(hops::RandomNumberGenerator rng_) {
        rng = rng_;
      }

        unsigned int getSeed() const {
            return rng.getSeed();
        }

        unsigned int getStream() const {
            return rng.getStream();
        }

        std::array<char, 16> getState() const {
            return rng.getStateInBytes();
        }

        void setSeed(unsigned int seed) {
            rng.setSeed(seed);
        }

        void setStream(unsigned int stream) {
            rng.setStream(stream);
        }

        void setState(const std::array<char, 16> &bytes) {
            rng.setState(bytes);
        }

        unsigned int operator()() {
            return rng();
        }

        std::string __repr__() const {
            std::string repr = "hopsy.RandomNumberGenerator(";
            auto seed = rng.getSeed();
            repr += (seed ? "seed=" + hops::RandomNumberGenerator::stringRepresentation(seed) : "");
            auto stream = rng.getStream();
            repr += (seed && stream ? ", " : "");
            repr += (stream ? "stream=" + hops::RandomNumberGenerator::stringRepresentation(stream) : "");
            repr += ")";
            return repr;
        }
    };

    using Normal = std::normal_distribution<double>;
    using Uniform = std::uniform_real_distribution<double>;
    using UniformInt = std::uniform_int_distribution<long>;

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
            .def_property("seed", &RandomNumberGenerator::getSeed, &RandomNumberGenerator::setSeed)
            .def_property("stream", &RandomNumberGenerator::getStream, &RandomNumberGenerator::setStream)
            .def_property("state", &RandomNumberGenerator::getState, &RandomNumberGenerator::setState)
            .def("__call__", [] (RandomNumberGenerator& self) { return self(); },
                    doc::RandomNumberGenerator::__call__)
            .def("__repr__", &RandomNumberGenerator::__repr__)
            .def(py::pickle([] (const RandomNumberGenerator& self) {
                        return py::make_tuple(self.rng.getSeed(), self.getStream(), self.getState());
                    },
                    [] (py::tuple t) {
                        if (t.size() != 3) throw std::runtime_error("Tried to build hopsy.RandomNumberGenerator with invalid state.");

                        RandomNumberGenerator rng(t[0].cast<unsigned int>(), t[1].cast<unsigned int>());
                        rng.setState(t[2].cast<std::array<char, 16>>());
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
                    });

        py::class_<UniformInt>(m, "UniformInt", doc::UniformInt::base)
            .def(py::init<long, long>(),
                    doc::UniformInt::__init__,
                    py::arg("a") = 0,
                    py::arg("b") = 1)
            .def("__call__", [] (UniformInt& self, RandomNumberGenerator& rng) -> long {
                        return self(rng.rng);
                    },
                    doc::UniformInt::__call__
                )
            .def_property("a", &UniformInt::a, [] (UniformInt& self, long a) {
                        self = UniformInt(a, self.b());
                    })
            .def_property("b", &UniformInt::a, [] (UniformInt& self, long b) {
                        self = UniformInt(self.a(), b);
                    })
            .def("__repr__", [] (Uniform& self) -> std::string {
                        std::string repr = "hopsy.UniformInt(";
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
