#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <string_view>
#include <utility>
#include <vector>
#include <boost/math/special_functions/beta.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/kernel/gaussian_line_kernel.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/overrelaxed_line_kernel.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/non_reversible_parallel_tempering.h>

namespace {
    using State = hopsy::Vector<double>;
    using hopsy::RandomNumberGenerator;
    using SimplexPoint = Eigen::Vector3d;
    constexpr std::array<double, 3> alpha{8., 2., 2.};
    constexpr std::array<double, 4> betas{0., 0.25, 0.6, 1.};
    constexpr std::size_t seed = 537;
    constexpr std::size_t runs = 4;
    constexpr int warmup = 2048;
    constexpr int batches_per_run = 12;
    constexpr int batch_length = 1024;
    constexpr hopsy::Index local_steps = 2; // A full sweep for cycling coordinates.
    constexpr int reference_draws = 65536;
    constexpr double standard_error_multiplier = 6.;
    constexpr double max_scaled_error = 0.08;
    constexpr double max_scaled_mcse = 0.022;

    struct DirichletTarget {
        double log_density(const State &x) const {
            if ((x.array() <= 0.).any())
                return -std::numeric_limits<double>::infinity();
            return (alpha[0] - 1.) * std::log(x[0]) + (alpha[1] - 1.) * std::log(x[1]) +
                   (alpha[2] - 1.) * std::log(x[2]);
        }
        void log_gradient(const State &x, State &out) const {
            for (hopsy::Index i = 0; i < x.size(); ++i)
                out[i] = (alpha[i] - 1.) / x[i];
        }
        State log_gradient(const State &x) const {
            State out(x.size());
            log_gradient(x, out);
            return out;
        }
    };

    auto dirichlet_problem() {
        hopsy::Matrix<double> A(3, 2), map(3, 2);
        A << -1., 0., 0., -1., 1., 1.;
        map << 1., 0., 0., 1., -1., -1.;
        SimplexPoint b(0., 0., 1.);
        return hopsy::Problem(hopsy::LinearPolytope<double>(A, b), DirichletTarget{},
                              hopsy::AffineCoordinates(map, b));
    }

    State initial_state(std::size_t run) {
        constexpr std::array<std::array<double, 2>, runs> starts{
            {{0.72, 0.14}, {0.14, 0.72}, {0.14, 0.14}, {0.34, 0.33}}};
        return Eigen::Vector2d(starts[run][0], starts[run][1]);
    }

    template <typename Direction, typename Line> struct HitAndRun {
        auto operator()(const State &state) const {
            if constexpr (std::same_as<Line, hopsy::GaussianLineKernel>)
                return hopsy::HitAndRunKernel(state, Direction(state.size()), Line(0.2));
            else
                return hopsy::HitAndRunKernel(state, Direction(state.size()), Line{});
        }
    };

    using RandomCoordinate = hopsy::RandomCoordinateDirectionGenerator;
    using CyclingCoordinate = hopsy::CyclingCoordinateDirectionGenerator;
    using DenseDirection = hopsy::RandomDirectionGenerator<double>;
    using UniformRandomCoordinate = HitAndRun<RandomCoordinate, hopsy::UniformLineKernel>;
    using UniformCyclingCoordinate = HitAndRun<CyclingCoordinate, hopsy::UniformLineKernel>;
    using UniformDense = HitAndRun<DenseDirection, hopsy::UniformLineKernel>;
    using GaussianRandomCoordinate = HitAndRun<RandomCoordinate, hopsy::GaussianLineKernel>;
    using GaussianCyclingCoordinate = HitAndRun<CyclingCoordinate, hopsy::GaussianLineKernel>;
    using GaussianDense = HitAndRun<DenseDirection, hopsy::GaussianLineKernel>;
    using OverRelaxedRandomCoordinate = HitAndRun<RandomCoordinate, hopsy::OverRelaxedLineKernel>;
    using OverRelaxedCyclingCoordinate = HitAndRun<CyclingCoordinate, hopsy::OverRelaxedLineKernel>;
    using OverRelaxedDense = HitAndRun<DenseDirection, hopsy::OverRelaxedLineKernel>;

    template <typename Drift, typename Metric> struct GaussianWalk {
        auto operator()(const State &state) const {
            constexpr double scale = std::same_as<Metric, hopsy::IdentityMetric> ? 0.16 : 0.65;
            return hopsy::GaussianKernel<State, Drift, Metric>(state, {}, {}, scale);
        }
    };

    using GaussianRandomWalk = GaussianWalk<hopsy::ZeroDrift, hopsy::IdentityMetric>;
    using IdentityMetricMALA = GaussianWalk<hopsy::ClippedGradientDrift, hopsy::IdentityMetric>;
    using DikinWalk = GaussianWalk<hopsy::ZeroDrift, hopsy::DikinMetric<double>>;
    using CSmMALA = GaussianWalk<hopsy::ClippedGradientDrift, hopsy::DikinMetric<double>>;

    constexpr std::array<std::string_view, 13> feature_names{
        "mean x1",
        "mean x2",
        "mean x3",
        "variance x1",
        "variance x2",
        "variance x3",
        "covariance x1 x2",
        "covariance x1 x3",
        "covariance x2 x3",
        "lower quartile x1",
        "lower quartile x2",
        "lower quartile x3",
        "upper quartile x1 and lower quartile x2"};
    using Features = Eigen::Array<double, feature_names.size(), 1>;
    constexpr hopsy::Index joint_tail = feature_names.size() - 1;

    bool in_simplex(const SimplexPoint &x) {
        return x.allFinite() && x.minCoeff() >= -1e-12 && std::abs(x.sum() - 1.) <= 1e-12;
    }

    struct DirichletReference {
        explicit DirichletReference(std::size_t slot) {
            SimplexPoint shapes;
            for (std::size_t i = 0; i < alpha.size(); ++i)
                shapes[i] = 1. + betas[slot] * (alpha[i] - 1.);
            const double total = shapes.sum();
            mean = shapes / total;
            for (hopsy::Index i = 0; i < shapes.size(); ++i)
                lower_quartile[i] = boost::math::ibeta_inv(shapes[i], total - shapes[i], 0.25);
            upper_quartile_x1 = boost::math::ibeta_inv(shapes[0], total - shapes[0], 0.75);

            const auto cov = [&](hopsy::Index i, hopsy::Index j) {
                return mean[i] * ((i == j ? 1. : 0.) - mean[j]) / (total + 1.);
            };
            expected << mean[0], mean[1], mean[2], cov(0, 0), cov(1, 1), cov(2, 2), cov(0, 1),
                cov(0, 2), cov(1, 2), 0.25, 0.25, 0.25, 0.;

            RandomNumberGenerator rng(1203, slot);
            std::array<std::gamma_distribution<double>, 3> gamma{
                std::gamma_distribution<double>(shapes[0]),
                std::gamma_distribution<double>(shapes[1]),
                std::gamma_distribution<double>(shapes[2])};
            Features sum = Features::Zero(), squares = Features::Zero();
            for (int i = 0; i < reference_draws; ++i) {
                SimplexPoint x;
                for (std::size_t j = 0; j < gamma.size(); ++j)
                    x[j] = gamma[j](rng);
                x /= x.sum();
                feasible &= in_simplex(x);
                const auto values = observe(x);
                sum += values;
                squares += values.square();
            }
            iid_mean = sum / reference_draws;
            iid_sd = ((squares - sum.square() / reference_draws) / (reference_draws - 1)).sqrt();
            expected[joint_tail] = iid_mean[joint_tail];
        }

        Features observe(const SimplexPoint &x) const {
            const SimplexPoint centered = x - mean;
            Features result;
            result << x[0], x[1], x[2], centered[0] * centered[0], centered[1] * centered[1],
                centered[2] * centered[2], centered[0] * centered[1], centered[0] * centered[2],
                centered[1] * centered[2], x[0] <= lower_quartile[0], x[1] <= lower_quartile[1],
                x[2] <= lower_quartile[2], x[0] >= upper_quartile_x1 && x[1] <= lower_quartile[1];
            return result;
        }

        SimplexPoint mean, lower_quartile;
        double upper_quartile_x1;
        Features expected, iid_mean, iid_sd;
        bool feasible = true;
    };

    // Batches never span different seeds. Their means account for serial
    // correlation and are pooled across the dispersed starting points.
    struct BatchMeans {
        std::array<Features, runs * batches_per_run> values;
        bool feasible = true;

        Features observe(const DirichletReference &reference, const SimplexPoint &x) {
            feasible &= in_simplex(x);
            return reference.observe(x);
        }

        void check(const DirichletReference &reference) const {
            REQUIRE(feasible);
            Features mean = Features::Zero();
            for (const auto &batch : values)
                mean += batch;
            mean /= values.size();
            Features squared_errors = Features::Zero();
            for (const auto &batch : values)
                squared_errors += (batch - mean).square();
            const Features mcse = (squared_errors / (values.size() * (values.size() - 1))).sqrt();

            for (hopsy::Index i = 0; i < mean.size(); ++i) {
                const auto label = feature_names[i];
                const double expected = reference.expected[i];
                const double observed = mean[i];
                const double scaled_mcse = mcse[i] / reference.iid_sd[i];
                const double scaled_error = std::abs(observed - expected) / reference.iid_sd[i];
                const double reference_mcse =
                    i == joint_tail ? reference.iid_sd[i] / std::sqrt(reference_draws) : 0.;
                CAPTURE(label, observed, expected, scaled_mcse, scaled_error);
                CHECK(scaled_mcse < max_scaled_mcse);
                CHECK(scaled_error < max_scaled_error);
                CHECK(std::abs(observed - expected) <=
                      standard_error_multiplier * std::hypot(mcse[i], reference_mcse));
            }
        }
    };
} // namespace

TEST_CASE("Dirichlet IID reference agrees with exact simplex statistics",
          "[integration][statistical][dirichlet]") {
    for (std::size_t slot = 0; slot < betas.size(); ++slot) {
        const DirichletReference reference(slot);
        CAPTURE(slot);
        REQUIRE(reference.feasible);
        REQUIRE(reference.iid_sd.allFinite());
        REQUIRE(reference.iid_sd.minCoeff() > 0.);
        for (hopsy::Index i = 0; i < joint_tail; ++i) {
            CAPTURE(feature_names[i], reference.iid_mean[i], reference.expected[i]);
            CHECK(std::abs(reference.iid_mean[i] - reference.expected[i]) <
                  standard_error_multiplier * reference.iid_sd[i] / std::sqrt(reference_draws));
        }
    }
}

#define HOPSY_DIRICHLET_KERNELS                                                                    \
    UniformRandomCoordinate, UniformCyclingCoordinate, UniformDense, GaussianRandomCoordinate,     \
        GaussianCyclingCoordinate, GaussianDense, OverRelaxedRandomCoordinate,                     \
        OverRelaxedCyclingCoordinate, OverRelaxedDense, GaussianRandomWalk, IdentityMetricMALA,    \
        DikinWalk, CSmMALA

TEMPLATE_TEST_CASE("Dirichlet stationary distribution with a standalone chain",
                   "[integration][statistical][dirichlet]", HOPSY_DIRICHLET_KERNELS) {
    const auto problem = dirichlet_problem();
    const DirichletReference reference(betas.size() - 1);
    BatchMeans batches;
    hopsy::Matrix<double> samples(alpha.size(), batch_length);
    for (std::size_t run = 0; run < runs; ++run) {
        State x = initial_state(run);
        auto kernel = TestType{}(x);
        auto chain = hopsy::bind_chain(problem, kernel, x);
        RandomNumberGenerator rng(seed, run);
        for (int i = 0; i < warmup * local_steps; ++i)
            chain.step(rng);
        for (int batch = 0; batch < batches_per_run; ++batch) {
            chain.draw_into(rng, samples, local_steps);
            Features sum = Features::Zero();
            for (hopsy::Index i = 0; i < samples.cols(); ++i)
                sum += batches.observe(reference, samples.col(i));
            batches.values[run * batches_per_run + batch] = sum / batch_length;
        }
    }
    batches.check(reference);
}

TEMPLATE_TEST_CASE("Dirichlet stationary distribution with NRPT",
                   "[integration][statistical][dirichlet][nrpt]", HOPSY_DIRICHLET_KERNELS) {
    const auto problem = dirichlet_problem();
    const auto ladder = hopsy::TemperatureLadder({betas.begin(), betas.end()});
    const std::array references{DirichletReference(0), DirichletReference(1), DirichletReference(2),
                                DirichletReference(3)};
    std::array<BatchMeans, betas.size()> batches;
    for (std::size_t run = 0; run < runs; ++run) {
        using Kernel = decltype(TestType{}(initial_state(0)));
        const std::size_t first_stream = runs + run * (betas.size() + 1);
        std::vector<hopsy::TemperingReplica<Kernel>> replicas;
        for (std::size_t slot = 0; slot < betas.size(); ++slot) {
            const State x = initial_state((run + slot) % runs);
            replicas.push_back(
                {x, TestType{}(x), RandomNumberGenerator(seed, first_stream + slot)});
        }
        auto sampler = hopsy::NonReversibleParallelTempering(
            problem, ladder, std::move(replicas),
            RandomNumberGenerator(seed, first_stream + betas.size()));
        for (int i = 0; i < warmup; ++i)
            sampler.scan(local_steps, local_steps);
        sampler.reset_statistics();
        for (int batch = 0; batch < batches_per_run; ++batch) {
            std::array<Features, betas.size()> sums;
            for (auto &sum : sums)
                sum.setZero();
            for (int i = 0; i < batch_length; ++i) {
                for (std::size_t slot = 0; slot < betas.size(); ++slot) {
                    sampler.explore(slot, local_steps);
                    SimplexPoint x;
                    problem.coordinates().to_target(sampler.sampling_state(slot), x);
                    sums[slot] += batches[slot].observe(references[slot], x);
                }
                sampler.communicate();
            }
            for (std::size_t slot = 0; slot < betas.size(); ++slot)
                batches[slot].values[run * batches_per_run + batch] = sums[slot] / batch_length;
        }
        CAPTURE(run);
        std::size_t edge = 0;
        for (const auto &statistics : sampler.exchange_statistics()) {
            CAPTURE(edge);
            CHECK(statistics.accepts > 0);
            ++edge;
        }
        const auto trips = sampler.round_trips();
        CHECK(std::accumulate(trips.begin(), trips.end(), std::size_t{0}) > 0);
    }
    for (std::size_t slot = 0; slot < betas.size(); ++slot) {
        CAPTURE(slot, betas[slot]);
        batches[slot].check(references[slot]);
    }
}
#undef HOPSY_DIRICHLET_KERNELS
