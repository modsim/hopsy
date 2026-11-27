#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/parallel_tempering/non_reversible_parallel_tempering.h>

namespace {
    using State = hopsy::Vector<double>;
    using hopsy::RandomNumberGenerator;
    using hopsy::TemperatureLadder;

    auto interval(double radius) {
        hopsy::Matrix<double> A(2, 1);
        A << 1., -1.;
        return hopsy::LinearPolytope<double>(A, State::Constant(2, radius));
    }

    auto uniform_coordinate(const State &state) {
        return hopsy::HitAndRunKernel(state, hopsy::RandomCoordinateDirectionGenerator(1),
                                      hopsy::UniformLineKernel{});
    }

    template <typename Problem, typename Factory>
    auto make_sampler(const Problem &problem, const TemperatureLadder &ladder, const State &initial,
                      Factory make_kernel) {
        using Kernel = decltype(make_kernel(initial));
        std::vector<hopsy::TemperingReplica<Kernel>> replicas;
        for (std::size_t i = 0; i < ladder.size(); ++i)
            replicas.push_back({initial, make_kernel(initial), RandomNumberGenerator(841, i)});
        return hopsy::NonReversibleParallelTempering(problem, ladder, std::move(replicas),
                                                     RandomNumberGenerator(51, 1000));
    }

    double expected_variance(double beta, double radius) {
        if (beta == 0.)
            return radius * radius / 3.;
        const double a = radius * std::sqrt(beta);
        const double phi = std::exp(-0.5 * a * a) / std::sqrt(2. * std::numbers::pi);
        return (1. - 2. * a * phi / std::erf(a / std::sqrt(2.))) / beta;
    }

    // Between-batch variance includes local/exchange autocorrelation. The
    // long batches and six-standard-error cutoff make false failures unlikely.
    template <std::size_t Batches>
    void check_batch_means(const std::array<double, Batches> &batches, double expected,
                           double max_error) {
        const double mean = std::accumulate(batches.begin(), batches.end(), 0.) / Batches;
        double sum_squares = 0.;
        for (double batch : batches)
            sum_squares += (batch - mean) * (batch - mean);
        const double error = std::sqrt(sum_squares / (Batches * (Batches - 1)));
        CAPTURE(expected, mean, error);
        CHECK(std::abs(mean - expected) < 6. * error);
        CHECK(std::abs(mean - expected) < max_error);
    }
} // namespace

TEST_CASE("NRPT samples the product of constrained Gaussian powers", "[nrpt][statistical]") {
    struct Normal {
        double log_density(const State &x) const { return -0.5 * x.squaredNorm(); }
    };
    constexpr double radius = 4.;
    const auto geometry = interval(radius);
    const auto problem = hopsy::Problem(std::cref(geometry), Normal{});
    const TemperatureLadder ladder({0., 0.15, 0.5, 1.});
    auto sampler = make_sampler(problem, ladder, State::Zero(1), uniform_coordinate);
    for (int scan = 0; scan < 500; ++scan)
        sampler.scan(3, 1);
    sampler.reset_statistics();

    constexpr int batch_size = 2500;
    constexpr std::size_t batches = 24;
    std::array<std::array<double, batches>, 4> means{}, second_moments{};
    std::array<double, batches> hot_cold_products{}, adjacent_products{};
    bool feasible = true;
    for (std::size_t batch = 0; batch < batches; ++batch) {
        for (int i = 0; i < batch_size; ++i) {
            sampler.scan(3, 1);
            for (std::size_t slot = 0; slot < 4; ++slot) {
                const double x = sampler.sampling_state(slot)[0];
                feasible &= std::abs(x) <= radius;
                means[slot][batch] += x / batch_size;
                second_moments[slot][batch] += x * x / batch_size;
            }
            hot_cold_products[batch] +=
                sampler.sampling_state(0)[0] * sampler.sampling_state(3)[0] / batch_size;
            adjacent_products[batch] +=
                sampler.sampling_state(1)[0] * sampler.sampling_state(2)[0] / batch_size;
        }
    }
    REQUIRE(feasible);
    for (std::size_t slot = 0; slot < 4; ++slot) {
        CAPTURE(slot);
        check_batch_means(means[slot], 0., 0.055);
        check_batch_means(second_moments[slot], expected_variance(ladder[slot], radius), 0.09);
    }
    check_batch_means(hot_cold_products, 0., 0.06);
    check_batch_means(adjacent_products, 0., 0.06);
    for (const auto &edge : sampler.exchange_statistics()) {
        CHECK(edge.accepts > edge.attempts / 4);
        CHECK(edge.accepts < edge.attempts);
    }
    for (const auto trips : sampler.round_trips())
        CHECK(trips > 100);
}

TEST_CASE("Only the reference replica explores zero-density regions of the polytope",
          "[nrpt][statistical]") {
    struct HalfSupport {
        double log_density(const State &x) const {
            return x[0] < 0. ? 0. : -std::numeric_limits<double>::infinity();
        }
    };
    auto problem = hopsy::Problem(interval(1.), HalfSupport{});
    const TemperatureLadder ladder({0., 1e-12, 1.});
    auto sampler = make_sampler(problem, ladder, State::Constant(1, -0.5), uniform_coordinate);
    int hot_outside = 0;
    bool warm_in_support = true, feasible = true;
    for (int i = 0; i < 5000; ++i) {
        sampler.scan(1, 1);
        hot_outside += sampler.sampling_state(0)[0] >= 0.;
        warm_in_support &= sampler.sampling_state(1)[0] < 0. && sampler.sampling_state(2)[0] < 0.;
        for (std::size_t slot = 0; slot < 3; ++slot)
            feasible &= std::abs(sampler.sampling_state(slot)[0]) <= 1.;
    }
    CHECK(feasible);
    CHECK(warm_in_support);
    CHECK(hot_outside > 2200);
    CHECK(hot_outside < 2800);
}

TEST_CASE("Reference communication brings both isolated modes to the cold replica",
          "[nrpt][statistical]") {
    struct TwoWells {
        double log_density(const State &x) const {
            return std::abs(std::abs(x[0]) - 5.) < 0.4 ? 0. : -30.;
        }
    };
    const auto geometry = interval(6.);
    auto problem = hopsy::Problem(std::cref(geometry), TwoWells{});
    const TemperatureLadder ladder({0., 0.02, 0.06, 0.15, 0.35, 0.65, 1.});
    auto walk = [](const State &x) {
        return hopsy::GaussianKernel<State, hopsy::ZeroDrift, hopsy::IdentityMetric>(x, {}, {},
                                                                                     0.5);
    };
    const State initial = State::Constant(1, -5.);
    auto sampler = make_sampler(problem, ladder, initial, walk);
    for (int i = 0; i < 500; ++i)
        sampler.scan(2, 20);
    sampler.reset_statistics();

    constexpr int scans = 18000;
    const auto samples = sampler.draw(scans, 2, 20);
    const auto right = (samples.row(0).array() > 4.6).count();
    const auto left = (samples.row(0).array() < -4.6).count();
    CAPTURE(left, right);
    CHECK((samples.cwiseAbs().array() <= 6.).all());
    CHECK(left + right == scans);
    CHECK(right > scans * 0.4);
    CHECK(right < scans * 0.6);
    const auto trips = sampler.round_trips();
    CHECK(std::accumulate(trips.begin(), trips.end(), std::size_t{0}) > 100);

    // Same cold local kernel and number of local steps, without communication.
    State isolated = initial;
    auto kernel = walk(initial);
    auto chain = hopsy::bind_chain(problem, kernel, isolated);
    RandomNumberGenerator rng(841, 6);
    CHECK((chain.draw(rng, scans, 2).array() < -4.6).all());
}
