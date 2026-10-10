#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <hopsy/geometry/line_interval.h>
#include <hopsy/kernel/gaussian_line_kernel.h>
#include <hopsy/kernel/overrelaxed_line_kernel.h>
#include <hopsy/kernel/uniform_line_kernel.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
using Interval = hopsy::LineInterval<double>;
using Domain = std::array<Interval, 1>;
double cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
}

TEST_CASE("Uniform line sampling weights disjoint intervals by length", "[line_kernel][uniform]") {
    hopsy::UniformLineKernel kernel;
    hopsy::RandomNumberGenerator rng(42);
    const std::array<Interval, 2> domain{{{3, 6}, {0, 1}}}; // Unsorted is valid.
    int first = 0;
    constexpr int n = 10000;
    for (int i = 0; i < n; ++i) {
        const auto t = kernel.propose(rng, domain);
        REQUIRE(((t >= 0 && t <= 1) || (t >= 3 && t <= 6)));
        first += t <= 1;
    }
    CHECK(double(first) / n == Catch::Approx(0.25).margin(0.025));
}

TEST_CASE("Uniform line sampling handles a single chord and degenerate domains", "[line_kernel][uniform]") {
    hopsy::UniformLineKernel kernel;
    hopsy::RandomNumberGenerator rng(42);
    const Domain domain{{{-2, 4}}};
    double sum = 0;
    for (int i = 0; i < 10000; ++i) {
        const auto t = kernel.propose(rng, domain);
        REQUIRE(t >= -2);
        REQUIRE(t <= 4);
        sum += t;
    }
    CHECK(sum / 10000 == Catch::Approx(1).margin(0.08));
    const std::array<Interval, 0> empty{};
    const Domain reversed{{{1, 0}}};
    const Domain point{{{0, 0}}};
    const Domain infinite{{{0, std::numeric_limits<double>::infinity()}}};
    CHECK_THROWS_AS(kernel.propose(rng, empty), std::invalid_argument);
    CHECK_THROWS_AS(kernel.propose(rng, reversed), std::invalid_argument);
    CHECK_THROWS_AS(kernel.propose(rng, point), std::invalid_argument);
    CHECK_THROWS_AS(kernel.propose(rng, infinite), std::invalid_argument);
}

TEST_CASE("Gaussian proposal correction matches normalization and reverses sign", "[line_kernel][gaussian]") {
    hopsy::GaussianLineKernel kernel(1.5);
    const Domain domain{{{-1, 2}}};
    const double step = 0.6;
    const Domain reverse{{{-1 - step, 2 - step}}};
    const double z_forward = cdf(2 / 1.5) - cdf(-1 / 1.5);
    const double z_reverse = cdf((2 - step) / 1.5) - cdf((-1 - step) / 1.5);
    const double ratio = kernel.log_proposal_ratio(domain, step);
    CHECK(ratio == Catch::Approx(std::log(z_forward / z_reverse)).margin(1e-13));
    CHECK(kernel.log_proposal_ratio(reverse, -step) == Catch::Approx(-ratio).margin(1e-13));
    CHECK(kernel.log_proposal_ratio(domain, 0.0) == 0);
}

TEST_CASE("Gaussian sampling uses Gaussian interval masses", "[line_kernel][gaussian]") {
    hopsy::GaussianLineKernel kernel;
    hopsy::RandomNumberGenerator rng(43);
    const std::vector<Interval> domain{{2, 3}, {-1, 0}};
    const double expected = (cdf(0) - cdf(-1)) /
        ((cdf(0) - cdf(-1)) + (cdf(3) - cdf(2)));
    int negative = 0;
    constexpr int n = 10000;
    for (int i = 0; i < n; ++i) {
        const double t = kernel.propose(rng, domain);
        REQUIRE(((t >= -1 && t <= 0) || (t >= 2 && t <= 3)));
        negative += t <= 0;
    }
    CHECK(double(negative) / n == Catch::Approx(expected).margin(0.02));
}

TEST_CASE("Gaussian sampling supports unbounded domains and the configured scale", "[line_kernel][gaussian]") {
    hopsy::GaussianLineKernel kernel(2);
    hopsy::RandomNumberGenerator rng(44);
    const double inf = std::numeric_limits<double>::infinity();
    const Domain domain{{{-inf, inf}}};
    double sum = 0, squares = 0;
    constexpr int n = 10000;
    for (int i = 0; i < n; ++i) {
        const double t = kernel.propose(rng, domain);
        REQUIRE(std::isfinite(t));
        sum += t;
        squares += t * t;
    }
    CHECK(sum / n == Catch::Approx(0).margin(0.1));
    CHECK(squares / n == Catch::Approx(4).margin(0.3));
    CHECK(kernel.log_proposal_ratio(domain, 0.7) == 0);
}

TEST_CASE("Gaussian tail intervals remain sampleable", "[line_kernel][gaussian]") {
    hopsy::GaussianLineKernel kernel;
    hopsy::RandomNumberGenerator rng(45);
    for (const Domain domain : {Domain{{{8, 9}}}, Domain{{{-9, -8}}}}) {
        double sum = 0;
        for (int i = 0; i < 1000; ++i) {
            const double t = kernel.propose(rng, domain);
            REQUIRE(t >= domain[0].lower);
            REQUIRE(t <= domain[0].upper);
            sum += std::abs(t);
        }
        CHECK(sum / 1000 > 8.05);
        CHECK(sum / 1000 < 8.25);
    }
}

TEST_CASE("Gaussian rejects invalid scales and domains", "[line_kernel][gaussian]") {
    CHECK_THROWS_AS(hopsy::GaussianLineKernel(0), std::invalid_argument);
    CHECK_THROWS_AS(hopsy::GaussianLineKernel(-1), std::invalid_argument);
    CHECK_THROWS_AS(hopsy::GaussianLineKernel(std::numeric_limits<double>::infinity()), std::invalid_argument);
    hopsy::GaussianLineKernel kernel;
    hopsy::RandomNumberGenerator rng(46);
    const std::array<Interval, 0> empty{};
    const Domain reversed{{{1, 0}}};
    const Domain point{{{0, 0}}};
    CHECK_THROWS_AS(kernel.propose(rng, empty), std::invalid_argument);
    CHECK_THROWS_AS(kernel.propose(rng, reversed), std::invalid_argument);
    CHECK_THROWS_AS(kernel.propose(rng, point), std::invalid_argument);
    kernel.set_standard_deviation(0.5);
    CHECK(kernel.standard_deviation() == 0.5);
}

TEST_CASE("Overrelaxation has the expected conditional mean", "[line_kernel][overrelaxed]") {
    hopsy::OverRelaxedLineKernel kernel;
    hopsy::RandomNumberGenerator rng(47);
    // u is the current normalized coordinate. E[v|u] = (2-u)/3.
    for (const double u : {0.0, 0.1, 0.5, 0.9, 1.0}) {
        const Domain domain{{{-u, 1-u}}};
        double sum = 0;
        constexpr int n = 10000;
        for (int i = 0; i < n; ++i) {
            const double step = kernel.propose(rng, domain);
            REQUIRE(step >= -u);
            REQUIRE(step <= 1-u);
            sum += step + u;
        }
        CHECK(sum / n == Catch::Approx((2-u)/3).margin(0.02));
        CHECK(kernel.log_proposal_ratio(domain, 0.0) == 0);
    }
}

TEST_CASE("Overrelaxation preserves uniform chord moments", "[line_kernel][overrelaxed]") {
    hopsy::OverRelaxedLineKernel kernel;
    hopsy::RandomNumberGenerator rng(48);
    double state = 0.5, sum = 0, squares = 0;
    constexpr int n = 10000;
    for (int i = 0; i < n; ++i) {
        const Domain domain{{{-state, 1-state}}};
        state += kernel.propose(rng, domain);
        sum += state;
        squares += state * state;
    }
    CHECK(sum / n == Catch::Approx(0.5).margin(0.02));
    CHECK(squares / n == Catch::Approx(1.0/3).margin(0.02));
}

TEST_CASE("Overrelaxation rejects unsupported domains", "[line_kernel][overrelaxed]") {
    hopsy::OverRelaxedLineKernel kernel;
    hopsy::RandomNumberGenerator rng(49);
    const std::array<Interval, 0> empty{};
    const std::array<Interval, 2> multiple{{{-1, 1}, {2, 3}}};
    const Domain outside{{{1, 2}}};
    const Domain point{{{0, 0}}};
    const Domain infinite{{{-1, std::numeric_limits<double>::infinity()}}};
    CHECK_THROWS_AS(kernel.propose(rng, empty), std::invalid_argument);
    CHECK_THROWS_AS(kernel.propose(rng, multiple), std::invalid_argument);
    CHECK_THROWS_AS(kernel.propose(rng, outside), std::invalid_argument);
    CHECK_THROWS_AS(kernel.propose(rng, point), std::invalid_argument);
    CHECK_THROWS_AS(kernel.propose(rng, infinite), std::invalid_argument);
}

TEST_CASE("Line kernels derive return types from the domain", "[line_kernel]") {
    hopsy::RandomNumberGenerator rng(50);
    const std::array<hopsy::LineInterval<float>, 1> domain{{{-1, 1}}};
    hopsy::UniformLineKernel uniform;
    hopsy::GaussianLineKernel gaussian;
    hopsy::OverRelaxedLineKernel overrelaxed;
    STATIC_REQUIRE(std::is_same_v<decltype(uniform.propose(rng, domain)), float>);
    STATIC_REQUIRE(std::is_same_v<decltype(gaussian.propose(rng, domain)), float>);
    STATIC_REQUIRE(std::is_same_v<decltype(overrelaxed.propose(rng, domain)), float>);
}
