#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/parallel.h>
#include <hopsy/sampling/parallel_tempering/non_reversible_parallel_tempering.h>

namespace {
    using State = hopsy::Vector<double>;
    using Matrix = hopsy::Matrix<double>;
    using Catch::Approx;
    using hopsy::RandomNumberGenerator;
    using hopsy::TemperatureLadder;

    auto box(int dimension = 1, double radius = 5.) {
        Matrix A(2 * dimension, dimension);
        A.topRows(dimension).setIdentity();
        A.bottomRows(dimension) = -A.topRows(dimension);
        return hopsy::LinearPolytope<double>(A, State::Constant(2 * dimension, radius));
    }

    auto hit_and_run(const State &state) {
        return hopsy::HitAndRunKernel(state,
                                      hopsy::RandomCoordinateDirectionGenerator(state.size()),
                                      hopsy::UniformLineKernel{});
    }

    template <typename Factory>
    auto replicas(const std::vector<State> &initial, Factory make_kernel) {
        using Kernel = decltype(make_kernel(initial.front()));
        std::vector<hopsy::TemperingReplica<Kernel>> result;
        for (std::size_t i = 0; i < initial.size(); ++i)
            result.push_back({initial[i], make_kernel(initial[i]), RandomNumberGenerator(217, i)});
        return result;
    }

    struct LinearDensity {
        double log_density(const State &x) const { return x[0]; }
    };

    struct HalfSupport {
        double log_density(const State &x) const {
            return x[0] < 0. ? 0. : -std::numeric_limits<double>::infinity();
        }
    };

    struct Normal {
        double log_density(const State &x) const { return -0.5 * x.squaredNorm(); }
        auto log_gradient(const State &x) const { return -x; }
    };
} // namespace

TEST_CASE("Inverse temperature ladders validate endpoints and order", "[nrpt]") {
    const auto ladder = TemperatureLadder::linear(5);
    REQUIRE(std::ranges::equal(ladder.inverse_temperatures(), std::array{0., 0.25, 0.5, 0.75, 1.}));
    REQUIRE(TemperatureLadder::linear(2)[1] == 1.);
    REQUIRE_THROWS_AS(TemperatureLadder::linear(1), std::invalid_argument);
    REQUIRE_THROWS_AS(TemperatureLadder({0.}), std::invalid_argument);
    REQUIRE_THROWS_AS(TemperatureLadder({0.1, 1.}), std::invalid_argument);
    REQUIRE_THROWS_AS(TemperatureLadder({0., 0.9}), std::invalid_argument);
    REQUIRE_THROWS_AS(TemperatureLadder({0., 0.3, 0.3, 1.}), std::invalid_argument);
    REQUIRE_THROWS_AS(TemperatureLadder({0., 0.8, 0.3, 1.}), std::invalid_argument);
    REQUIRE_THROWS_AS(TemperatureLadder({0., std::numeric_limits<double>::quiet_NaN(), 1.}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(TemperatureLadder({0., std::numeric_limits<double>::infinity(), 1.}),
                      std::invalid_argument);
}

TEST_CASE("Ladder adaptation spaces the inverse PCHIP barrier over all temperature gaps",
          "[nrpt]") {
    const TemperatureLadder ladder({0., 0.1, 0.3, 0.55, 1.});
    const auto tuned = ladder.adapted(std::array{0.2, 0.05, 0.4, 0.1});
    // SciPy PchipInterpolator as in annealing.py, evaluated on i/(N-1).
    REQUIRE(tuned[0] == 0.);
    REQUIRE(tuned[1] == Approx(0.0873123972039474).margin(1e-14));
    REQUIRE(tuned[2] == Approx(0.4020325408584770).margin(1e-14));
    REQUIRE(tuned[3] == Approx(0.4702240691787896).margin(1e-14));
    REQUIRE(tuned[4] == 1.);
    const auto flat = ladder.adapted(std::array{0., 0., 0., 0.});
    REQUIRE(std::ranges::equal(flat.inverse_temperatures(), ladder.inverse_temperatures()));
    const auto uniform = TemperatureLadder::linear(3).adapted(std::array{0.4, 0.4});
    REQUIRE(uniform[1] == 0.5);
    const auto skewed = TemperatureLadder::linear(3).adapted(std::array{0., 1.});
    REQUIRE(skewed[1] > 0.5);
    REQUIRE(skewed[1] < 1.);
    REQUIRE(TemperatureLadder::linear(2).adapted(std::array{1.})[1] == 1.);
    REQUIRE_THROWS_AS(ladder.adapted(std::array{0.4}), std::invalid_argument);
    REQUIRE_THROWS_AS(ladder.adapted(std::array{0., 0., 0., -0.1}), std::invalid_argument);
    REQUIRE_THROWS_AS(ladder.adapted(std::array{0., 0., 0., 1.1}), std::invalid_argument);
}

TEST_CASE("Exchange acceptance respects support and remains stable for extreme densities",
          "[nrpt]") {
    using hopsy::replica_exchange_log_acceptance;
    constexpr auto inf = std::numeric_limits<double>::infinity();
    constexpr auto max = std::numeric_limits<double>::max();
    REQUIRE(replica_exchange_log_acceptance(0.25, 0.75, -4., -2.) == -1.);
    REQUIRE(std::exp(replica_exchange_log_acceptance(0.25, 0.75, -4., -2.)) ==
            Approx(1. / std::exp(1.)));
    REQUIRE(replica_exchange_log_acceptance(0.25, 0.75, -2., -4.) == 0.);
    REQUIRE(replica_exchange_log_acceptance(0., 1., -max, -max) == 0.);
    REQUIRE(replica_exchange_log_acceptance(0., 0.25, -max, max) == -max / 2.);
    REQUIRE(replica_exchange_log_acceptance(0., 1., -max, max) == -inf);
    REQUIRE(replica_exchange_log_acceptance(0., 1e-20, -1., 0.) == -1e-20);
    REQUIRE(replica_exchange_log_acceptance(0., 1., -inf, 0.) == -inf);
    REQUIRE(replica_exchange_log_acceptance(0., 1e-12, -inf, 0.) == -inf);
    REQUIRE_THROWS_AS(replica_exchange_log_acceptance(0.1, 1., -inf, 0.), std::domain_error);
    REQUIRE_THROWS_AS(replica_exchange_log_acceptance(0., 1., -inf, -inf), std::domain_error);
    REQUIRE_THROWS_AS(replica_exchange_log_acceptance(0., 1., inf, 0.), std::domain_error);
    REQUIRE_THROWS_AS(
        replica_exchange_log_acceptance(0., 1., 0., std::numeric_limits<double>::quiet_NaN()),
        std::domain_error);
    REQUIRE_THROWS_AS(replica_exchange_log_acceptance(1., 0., 0., 0.), std::invalid_argument);
}

TEST_CASE("NRPT matches Python exchange decisions with the same initial states and variates",
          "[nrpt][reference]") {
    const std::vector<State> initial{State::Constant(1, -4.), State::Constant(1, -1.),
                                     State::Constant(1, -2.), State::Constant(1, 3.),
                                     State::Constant(1, 0.)};
    auto problem = hopsy::Problem(box(), LinearDensity{});
    auto sampler = hopsy::NonReversibleParallelTempering(
        problem, TemperatureLadder({0., 0.1, 0.3, 0.55, 1.}), replicas(initial, hit_and_run),
        RandomNumberGenerator(431, 1000));
    // Generated by annealing.py::_swap using the 12 uniform variates from
    // std::uniform_real_distribution<double>(0,1) on PCG64(seed=431,stream=1000).
    const std::array<std::array<std::size_t, 5>, 6> python_labels{{{0, 2, 1, 4, 3},
                                                                   {2, 0, 4, 1, 3},
                                                                   {2, 0, 4, 1, 3},
                                                                   {0, 2, 1, 4, 3},
                                                                   {0, 1, 2, 4, 3},
                                                                   {1, 0, 4, 2, 3}}};
    std::array<const State *, 5> physical_states{};
    for (std::size_t i = 0; i < physical_states.size(); ++i)
        physical_states[i] = &sampler.sampling_state(i);
    for (const auto &labels : python_labels) {
        sampler.communicate();
        REQUIRE(std::ranges::equal(sampler.replica_indices(), labels));
        for (std::size_t slot = 0; slot < labels.size(); ++slot)
            REQUIRE(&sampler.sampling_state(slot) == physical_states[labels[slot]]);
    }

    // Remove Python's 1e-10 floor from the pre-swap mean rejection estimates.
    const auto mean = sampler.mean_exchange_rejections();
    REQUIRE(mean[0] == Approx(0.1468170087801001 - 2e-10 / 6).margin(1e-14));
    REQUIRE(mean[1] == Approx(0.2439800943015989 - 2e-10 / 6).margin(1e-14));
    REQUIRE(mean[2] == Approx(0.2582271629140612 - 2e-10 / 6).margin(1e-14));
    REQUIRE(mean[3] == Approx(0.6486135736198587 - 1e-10 / 6).margin(1e-14));
    const auto stats = sampler.exchange_statistics();
    const std::array<std::size_t, 4> accepts{3, 2, 3, 1};
    for (std::size_t i = 0; i < stats.size(); ++i) {
        REQUIRE(stats[i].attempts == 3);
        REQUIRE(stats[i].accepts == accepts[i]);
    }
}

TEST_CASE("Alternating exchange moves replicas ballistically and turns at ladder boundaries",
          "[nrpt]") {
    for (std::size_t n : {2, 4, 5}) {
        CAPTURE(n);
        auto sampler = hopsy::NonReversibleParallelTempering(
            hopsy::Problem(box(), hopsy::UniformTarget{}), TemperatureLadder::linear(n),
            replicas(std::vector<State>(n, State::Zero(1)), hit_and_run),
            RandomNumberGenerator(71, 1000));
        for (std::size_t scan = 1; scan <= 2 * n; ++scan) {
            CAPTURE(scan);
            sampler.communicate();
            const auto position = scan <= n ? scan - 1 : 2 * n - scan;
            REQUIRE(sampler.replica_indices()[position] == 0);
            REQUIRE(sampler.round_trips()[0] == (scan == 2 * n ? 1 : 0));
        }
        for (const auto &edge : sampler.exchange_statistics()) {
            REQUIRE(edge.attempts == n);
            REQUIRE(edge.accepts == n);
        }
        sampler.communicate(); // odd edges
        sampler.reset_statistics();
        sampler.communicate(); // even edges, including edge 0
        REQUIRE(sampler.exchange_statistics()[0].attempts == 1);
    }
}

TEST_CASE("NRPT validates the ensemble before exploration", "[nrpt]") {
    auto problem = hopsy::Problem(box(), HalfSupport{});
    const auto ladder = TemperatureLadder::linear(2);
    const auto rng = RandomNumberGenerator(431, 1000);
    auto initial = std::vector<State>{State::Constant(1, 1.), State::Constant(1, -1.)};
    REQUIRE_THROWS_AS(hopsy::NonReversibleParallelTempering(
                          problem, ladder, replicas({initial.front()}, hit_and_run), rng),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(hopsy::NonReversibleParallelTempering(problem, ladder,
                                                            replicas(initial, hit_and_run),
                                                            RandomNumberGenerator(1, 0)),
                      std::invalid_argument);
    auto mismatched = replicas(initial, hit_and_run);
    mismatched[1].state[0] = -2.;
    REQUIRE_THROWS_AS(
        hopsy::NonReversibleParallelTempering(problem, ladder, std::move(mismatched), rng),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        hopsy::NonReversibleParallelTempering(
            problem, ladder, replicas({initial.back(), initial.front()}, hit_and_run), rng),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        hopsy::NonReversibleParallelTempering(
            problem, ladder, replicas({State::Constant(1, 6.), initial.back()}, hit_and_run), rng),
        std::invalid_argument);
    auto sampler =
        hopsy::NonReversibleParallelTempering(problem, ladder, replicas(initial, hit_and_run), rng);
    REQUIRE_THROWS_AS(sampler.explore(0, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(sampler.explore(2), std::out_of_range);
    REQUIRE_THROWS_AS(sampler.draw(-1), std::invalid_argument);
    REQUIRE_THROWS_AS(sampler.scan(1, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(sampler.mean_exchange_rejections(), std::logic_error);
    REQUIRE_THROWS_AS(sampler.set_ladder(TemperatureLadder::linear(3)), std::invalid_argument);
    REQUIRE(sampler.draw(0).cols() == 0);
    sampler.communicate();
    sampler.communicate();
    REQUIRE(sampler.exchange_statistics()[0].attempts == 1);
    REQUIRE(sampler.exchange_statistics()[0].accepts == 0);
    REQUIRE(sampler.mean_exchange_rejections()[0] == 1.);
}

TEST_CASE("NRPT draw observes the explored cold state before communication", "[nrpt]") {
    const auto problem = hopsy::Problem(box(2), hopsy::UniformTarget{});
    const auto ladder = TemperatureLadder::linear(3);
    auto make_sampler = [&] {
        return hopsy::NonReversibleParallelTempering(
            problem, ladder, replicas(std::vector<State>(3, State::Zero(2)), hit_and_run),
            RandomNumberGenerator(431, 1000));
    };
    auto sampler = make_sampler();
    auto manual = make_sampler();
    const auto samples = sampler.draw(20, 3, 4);
    for (hopsy::Index i = 0; i < samples.cols(); ++i) {
        for (std::size_t slot = 0; slot < manual.size(); ++slot)
            manual.explore(slot, slot == 0 ? 4 : 3);
        REQUIRE(samples.col(i) == manual.sampling_state(2));
        manual.communicate();
        if (i % 2 == 0)
            REQUIRE(samples.col(i) != manual.sampling_state(2));
    }
    REQUIRE(sampler.sampling_state(2) == manual.sampling_state(2));
}

TEST_CASE("Communication waits for a valid density after a failed reference evaluation", "[nrpt]") {
    struct IntermittentTarget {
        const bool &fail;
        double log_density(const State &x) const {
            return fail ? std::numeric_limits<double>::quiet_NaN() : -x.squaredNorm();
        }
    };
    bool fail = false;
    auto problem = hopsy::Problem(box(), IntermittentTarget{fail});
    auto sampler = hopsy::NonReversibleParallelTempering(
        problem, TemperatureLadder::linear(3),
        replicas(std::vector<State>(3, State::Zero(1)), hit_and_run),
        RandomNumberGenerator(431, 1000));
    fail = true;
    REQUIRE_THROWS_AS(sampler.explore(0), std::domain_error);
    REQUIRE_THROWS_AS(sampler.communicate(), std::domain_error);
    REQUIRE_THROWS_AS(sampler.mean_exchange_rejections(), std::logic_error);
    REQUIRE(std::ranges::equal(sampler.replica_indices(), std::array<std::size_t, 3>{0, 1, 2}));
    fail = false;
    REQUIRE(sampler.explore(0) == 1);
    sampler.communicate();
    REQUIRE(sampler.exchange_statistics()[0].attempts == 0);
    REQUIRE(sampler.exchange_statistics()[1].attempts == 1);
}

TEST_CASE("NRPT keeps Gaussian state caches and refreshes powered density and drift", "[nrpt]") {
    using Kernel = hopsy::CSmMALAKernel<State>;
    const auto geometry = box();
    // Raw reference in sampling coordinates: target x = 2*z + 0.3.
    struct MappedPower {
        double beta;
        double log_density(const State &z) const {
            return beta * (-0.5 * std::pow(2. * z[0] + 0.3, 2) + std::log(2.));
        }
        State log_gradient(const State &z) const {
            return State::Constant(1, -2. * beta * (2. * z[0] + 0.3));
        }
    };
    auto problem = hopsy::Problem(
        std::cref(geometry), Normal{},
        hopsy::AffineCoordinates(Matrix::Constant(1, 1, 2.), State::Constant(1, 0.3)));
    auto factory = [](const State &state) { return Kernel(state, {}, {}, 0.35); };
    const std::vector<State> initial{State::Constant(1, -0.5), State::Constant(1, 0.2),
                                     State::Constant(1, 0.8)};
    auto direct = replicas(initial, factory);
    auto sampler = hopsy::NonReversibleParallelTempering(problem, TemperatureLadder::linear(3),
                                                         replicas(initial, factory),
                                                         RandomNumberGenerator(431, 1000));
    hopsy::EmptyWorkspace workspace;
    std::size_t exchanged = 0;
    for (int scan = 0; scan < 250; ++scan) {
        if (scan == 51) {
            exchanged =
                sampler.exchange_statistics()[0].accepts + sampler.exchange_statistics()[1].accepts;
            sampler.set_ladder(TemperatureLadder({0., 0.125, 1.}));
            REQUIRE(sampler.exchange_statistics()[0].attempts == 0);
        }
        for (std::size_t slot = 0; slot < sampler.size(); ++slot) {
            auto &replica = direct[sampler.replica_indices()[slot]];
            auto fresh = factory(replica.state);
            const auto beta = sampler.ladder()[slot];
            std::size_t expected_accepts = 0;
            for (int step = 0; step < 3; ++step) {
                expected_accepts += beta == 0.
                                        ? hopsy::step(fresh, replica.state, replica.rng,
                                                      hopsy::UniformTarget{}, geometry, workspace)
                                        : hopsy::step(fresh, replica.state, replica.rng,
                                                      MappedPower{beta}, geometry, workspace);
            }
            REQUIRE(sampler.explore(slot, 3) == expected_accepts);
            REQUIRE(sampler.sampling_state(slot).isApprox(replica.state, 1e-12));
            REQUIRE(sampler.target_state(slot)[0] ==
                    Approx(2. * replica.state[0] + 0.3).margin(1e-12));
        }
        sampler.communicate();
        if (scan == 51)
            REQUIRE(sampler.exchange_statistics()[0].attempts == 1);
    }
    REQUIRE(exchanged > 10);
}

TEST_CASE("NRPT split exploration is reproducible across independent slot workers", "[nrpt]") {
    const Matrix scale = 2. * Matrix::Identity(2, 2);
    const auto problem =
        hopsy::Problem(box(2), Normal{}, hopsy::AffineCoordinates(scale, State::Constant(2, 0.3)));
    auto make_sampler = [&] {
        return hopsy::NonReversibleParallelTempering(
            problem, TemperatureLadder::linear(4),
            replicas(std::vector<State>(4, State::Zero(2)), hit_and_run),
            RandomNumberGenerator(431, 1000));
    };
    auto serial = make_sampler();
    auto parallel = make_sampler();
    for (int scan = 0; scan < 32; ++scan) {
        hopsy::parallel_for(parallel.size(), 2, [&](std::size_t slot) {
            parallel.explore(slot, slot == 0 ? 5 : 3);
        });
        parallel.communicate();
        serial.scan(3, 5);
        REQUIRE(std::ranges::equal(parallel.replica_indices(), serial.replica_indices()));
        for (std::size_t slot = 0; slot < parallel.size(); ++slot)
            REQUIRE(serial.sampling_state(slot) == parallel.sampling_state(slot));
    }
}
