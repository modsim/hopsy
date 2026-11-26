#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

#include <hopsy/kernel/direction_generator.h>


TEMPLATE_TEST_CASE(
    "Generators reject nonpositive dimensions",
    "[direction_generator]",
    hopsy::CyclingCoordinateDirectionGenerator,
    hopsy::RandomCoordinateDirectionGenerator
) {
    REQUIRE_THROWS_AS(TestType(0), std::invalid_argument);
    REQUIRE_THROWS_AS(TestType(-1), std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "Coordinate generators return valid coordinate indices",
    "[direction_generator]",
    // hopsy::RandomCoordinateDirectionGenerator,
    hopsy::CyclingCoordinateDirectionGenerator
) {
    for (const hopsy::Index dimension: {1, 2, 3, 10}) {
        CAPTURE(dimension);

        TestType generator(dimension);
        hopsy::RandomNumberGenerator rng(42);

        for (int draw = 0; draw < 100; ++draw) {
            CAPTURE(draw);

            const auto coordinate = generator.generate(rng).coordinate;

            REQUIRE(coordinate >= 0);
            REQUIRE(coordinate < dimension);
        }
    }
}

TEST_CASE(
    "Cycling coordinate generator advances and wraps around",
    "[direction_generator]"
) {
    for (const hopsy::Index dimension: {1, 2, 3, 10}) {
        CAPTURE(dimension);

        hopsy::CyclingCoordinateDirectionGenerator generator(dimension);
        hopsy::RandomNumberGenerator rng(42);

        const auto first = generator.generate(rng).coordinate;
        REQUIRE(first >= 0);
        REQUIRE(first < dimension);

        for (hopsy::Index draw = 1; draw < 4 * dimension; ++draw) {
            CAPTURE(draw);

            const auto coordinate = generator.generate(rng).coordinate;

            REQUIRE(coordinate == (first + draw) % dimension);
        }
    }
}

TEST_CASE(
    "Random coordinate generator explores every coordinate",
    "[direction_generator]"
) {
    constexpr hopsy::Index dimension = 5;

    hopsy::RandomCoordinateDirectionGenerator generator(dimension);
    hopsy::RandomNumberGenerator rng(42);

    std::vector<bool> visited(dimension, false);
    bool repeated_coordinate = false;
    hopsy::Index previous = -1;

    for (int draw = 0; draw < 1000; ++draw) {
        const auto coordinate = generator.generate(rng).coordinate;

        REQUIRE(coordinate >= 0);
        REQUIRE(coordinate < dimension);

        visited[coordinate] = true;
        repeated_coordinate |= coordinate == previous;
        previous = coordinate;
    }

    for (hopsy::Index i = 0; i < dimension; ++i) {
        CAPTURE(i);
        REQUIRE(visited[i]);
    }

    REQUIRE(repeated_coordinate);
}

TEMPLATE_TEST_CASE(
    "RandomDirectionGenerator rejects nonpositive dimensions",
    "[direction_generator]",
    float, double
) {
    using Generator = hopsy::RandomDirectionGenerator<TestType>;

    REQUIRE_THROWS_AS(Generator(0), std::invalid_argument);
    REQUIRE_THROWS_AS(Generator(-1), std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "RandomDirectionGenerator produces finite unit vectors",
    "[direction_generator]",
    float, double
) {
    using Scalar = TestType;
    const Scalar tolerance =
            Scalar{100} * std::numeric_limits<Scalar>::epsilon();

    for (const hopsy::Index dimension: {1, 2, 3, 10, 100}) {
        CAPTURE(dimension);

        hopsy::RandomDirectionGenerator<Scalar> generator(dimension);
        hopsy::RandomNumberGenerator rng(42);

        for (int draw = 0; draw < 100; ++draw) {
            CAPTURE(draw);

            const auto &direction = generator.generate(rng);

            REQUIRE(direction.size() == dimension);
            REQUIRE(direction.allFinite());
            REQUIRE(std::abs(direction.norm() - Scalar{1}) < tolerance);
        }
    }
}

TEST_CASE(
    "RandomDirectionGenerator reuses its vector storage",
    "[direction_generator]"
) {
    hopsy::RandomDirectionGenerator<double> generator(5);
    hopsy::RandomNumberGenerator rng(42);

    const auto &first = generator.generate(rng);
    const auto *vector_address = &first;
    const auto *data_address = first.data();

    for (int draw = 0; draw < 100; ++draw) {
        const auto &direction = generator.generate(rng);

        REQUIRE(&direction == vector_address);
        REQUIRE(direction.data() == data_address);
    }
}

TEMPLATE_TEST_CASE(
    "Direction stepping reuses proposal storage",
    "[direction]",
    float,
    double
) {
    using Vector = hopsy::Vector<TestType>;
    const Vector state = (Vector(3) << 1, -2, 3).finished();
    const Vector original = state;
    Vector output(3);
    const auto *storage = output.data();
    SECTION("Dense direction") {
        const Vector direction = (Vector(3) << -2, 4, 1).finished();
        for (const TestType distance: {TestType{0}, TestType{0.5}, TestType{-2}}) {
            hopsy::step_direction(direction, state, distance, output);
            CHECK(output.isApprox(state + distance * direction));
            CHECK(output.data() == storage);
            CHECK(state == original);
        }
    }
    SECTION("Coordinate direction") {
        for (hopsy::Index i = 0; i < 3; ++i) {
            const auto direction = hopsy::CoordinateDirection{i};
            constexpr TestType distance = TestType{-0.5};

            Vector expected = state;
            expected(i) += distance;

            hopsy::step_direction(direction, state, distance, output);

            // Only the affected coordinate is materialized.
            CHECK(output(i) == expected(i));
            CHECK(output.data() == storage);
            CHECK(state == original);

            // Accepting the proposal must produce the complete state.
            Vector accepted = state;
            hopsy::accept_direction(direction, accepted, output);

            CHECK(accepted == expected);
            CHECK(output.data() == storage);
            CHECK(state == original);
        }
    }
}
