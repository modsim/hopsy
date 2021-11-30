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
} // namespace hopsy

#endif // HOPSY_RANDOM_HPP

