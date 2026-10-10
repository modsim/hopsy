#ifndef DIRECTION_GENERATOR_H
#define DIRECTION_GENERATOR_H

#include <stdexcept>
#include <random>


#include <hopsy/kernel/direction.h>
#include <hopsy/linalg.h>
#include <hopsy/random_number_generator.h>

namespace hopsy {
    class CyclingCoordinateDirectionGenerator {
    public:
        using direction_type = CoordinateDirection;

        explicit CyclingCoordinateDirectionGenerator(const Index num_dimensions) {
            if (num_dimensions <= 0) {
                throw std::invalid_argument("CyclingCoordinateDirectionGenerator: num_dimensions must be positive");
            }
            num_dimensions_ = num_dimensions;
            direction_.coordinate = num_dimensions - Index{1};
        }

        [[nodiscard]] const direction_type &generate(RandomNumberGenerator &) {
            ++direction_.coordinate;
            direction_.coordinate %= num_dimensions_;
            return direction_;
        }

    private:
        direction_type direction_{};
        Index num_dimensions_;
    };


    class RandomCoordinateDirectionGenerator {
    public:
        using direction_type = CoordinateDirection;

        explicit RandomCoordinateDirectionGenerator(const Index num_dimensions) {
            if (num_dimensions <= 0) {
                throw std::invalid_argument("RandomCoordinateDirectionGenerator: num_dimensions must be positive");
            }
            uniform_distribution_ = std::uniform_int_distribution<Index>(0, num_dimensions - 1);
        }

        [[nodiscard]] const direction_type &generate(RandomNumberGenerator &rng) {
            direction_.coordinate = uniform_distribution_(rng);
            return direction_;
        }

    private:
        direction_type direction_{};
        std::uniform_int_distribution<Index> uniform_distribution_;
    };

    template<typename Scalar>
    class RandomDirectionGenerator {
    public:
        using direction_type = Vector<Scalar>;

        explicit RandomDirectionGenerator(const Index num_dimensions) {
            if (num_dimensions <= 0) {
                throw std::invalid_argument("RandomDirectionGenerator: num_dimensions must be positive");
            }
            direction_.resize(num_dimensions);
        }

        [[nodiscard]] const direction_type &generate(RandomNumberGenerator &rng) {
            Scalar norm;
            do {
                for (Index i = 0; i < direction_.rows(); ++i) {
                    direction_(i) = normal_distribution_(rng);
                }
                norm = direction_.norm();
            } while (norm == Scalar{0});
            direction_ /= norm;
            return direction_;
        }

    private:
        direction_type direction_;
        std::normal_distribution<Scalar> normal_distribution_;
    };
}

#endif // DIRECTION_GENERATOR_H
