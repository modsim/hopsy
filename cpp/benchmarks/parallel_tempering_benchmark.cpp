#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/kernel/gaussian_line_kernel.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/overrelaxed_line_kernel.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/non_reversible_parallel_tempering.h>
#include "batch_means.h"

namespace {
    using namespace hopsy;
    using State = Vector<double>;
    using Clock = std::chrono::steady_clock;
    constexpr Index dimension = 8;
    constexpr double radius = 4.;
    constexpr Index warmup_scans = 1000;
    constexpr Index samples_per_run = 30000;
    constexpr Index batch_size = 240;
    constexpr std::size_t runs = 5;
    static_assert(samples_per_run % batch_size == 0 && batch_size % dimension == 0 &&
                  batch_size % 2 == 0);

    struct Normal {
        double log_density(const State &x) const { return -0.5 * x.squaredNorm(); }
        auto log_gradient(const State &x) const { return -x; }
    };

    auto make_problem() {
        Matrix<double> A(2 * dimension, dimension);
        A.topRows(dimension).setIdentity();
        A.bottomRows(dimension) = -A.topRows(dimension);
        return Problem(LinearPolytope<double>(A, State::Constant(2 * dimension, radius)), Normal{});
    }

    template <typename Factory>
    auto make_sampler(std::size_t count, std::size_t run, const Factory &make_kernel) {
        using Kernel = decltype(make_kernel(std::declval<const State &>()));
        std::vector<TemperingReplica<Kernel>> replicas;
        replicas.reserve(count);
        const auto first_stream = run * (count + 1);
        for (std::size_t i = 0; i < count; ++i) {
            const State state = State::Constant(dimension, 2. * static_cast<double>(i) / count);
            replicas.push_back(
                {state, make_kernel(state), RandomNumberGenerator(70 + run, first_stream + i)});
        }
        return NonReversibleParallelTempering(
            make_problem(), TemperatureLadder::linear(count), std::move(replicas),
            RandomNumberGenerator(80 + run, first_stream + count));
    }

    double median(std::array<double, runs> timings) {
        std::sort(timings.begin(), timings.end());
        return timings[timings.size() / 2];
    }

    void benchmark_communication(std::size_t count) {
        constexpr Index scans = 100000;
        std::array<double, runs> timings{};
        std::size_t trips = 0;
        for (std::size_t run = 0; run < runs; ++run) {
            auto sampler = make_sampler(count, run, [](const State &x) {
                return HitAndRunKernel(x, RandomCoordinateDirectionGenerator(x.size()),
                                       UniformLineKernel{});
            });
            for (Index scan = 0; scan < warmup_scans; ++scan)
                sampler.communicate();
            sampler.reset_statistics();
            auto start = Clock::now();
            for (Index scan = 0; scan < scans; ++scan)
                sampler.communicate();
            timings[run] = std::chrono::duration<double, std::nano>(Clock::now() - start).count();
            const auto completed = sampler.round_trips();
            trips += std::accumulate(completed.begin(), completed.end(), std::size_t{0});
        }
        std::cout << count << " replicas, communication only, " << std::fixed
                  << std::setprecision(1) << median(timings) / scans << " median ns/scan, " << trips
                  << " round trips across runs\n";
    }

    template <typename Factory>
    void benchmark_kernel(std::string_view name, std::size_t count, const Factory &make_kernel) {
        std::array<double, runs> elapsed_ns{};
        std::vector<ReplicaExchangeStatistics> exchanges(count - 1);
        std::vector<double> rejection_sums(count - 1);
        std::vector<std::size_t> round_trips(count);
        benchmark::BatchMeans batch_means(dimension, batch_size);

        for (std::size_t run = 0; run < runs; ++run) {
            auto sampler = make_sampler(count, run, make_kernel);
            for (Index scan = 0; scan < warmup_scans; ++scan)
                sampler.scan(1, 1);
            sampler.reset_statistics();
            const auto start = Clock::now();
            const auto samples = sampler.draw(samples_per_run, 1, 1);
            elapsed_ns[run] =
                std::chrono::duration<double, std::nano>(Clock::now() - start).count();

            if (!samples.allFinite() || (samples.array().abs() > radius + 1e-10).any())
                throw std::runtime_error(std::string(name) + " produced an invalid cold sample");
            batch_means.add(samples);

            const auto stats = sampler.exchange_statistics();
            const auto mean_rejections = sampler.mean_exchange_rejections();
            for (std::size_t i = 0; i < stats.size(); ++i) {
                exchanges[i].accepts += stats[i].accepts;
                exchanges[i].attempts += stats[i].attempts;
                rejection_sums[i] += mean_rejections[i];
            }
            const auto trips = sampler.round_trips();
            for (std::size_t i = 0; i < trips.size(); ++i)
                round_trips[i] += trips[i];
        }

        const State ess = batch_means.ess();
        const double min_ess =
            ess.allFinite() ? ess.minCoeff() : std::numeric_limits<double>::quiet_NaN();
        const double total_seconds =
            1e-9 * std::accumulate(elapsed_ns.begin(), elapsed_ns.end(), 0.);
        const auto trips = std::accumulate(round_trips.begin(), round_trips.end(), std::size_t{0});
        const double barrier =
            std::accumulate(rejection_sums.begin(), rejection_sums.end(), 0.) / runs;
        std::cout << '\n'
                  << name << ", " << count << " replicas, " << std::setprecision(1)
                  << median(elapsed_ns) / samples_per_run << " median ns/scan, "
                  << batch_means.sample_count() << " cold samples, min ESS " << min_ess
                  << ", min ESS/s " << min_ess / total_seconds << '\n';
        std::cout << "  ESS by coordinate             " << ess.transpose() << '\n';
        std::cout << "  Exchanges (accepted/attempted)";
        for (const auto &edge : exchanges)
            std::cout << ' ' << edge.accepts << '/' << edge.attempts;
        std::cout << "\n  Mean rejection by edge        " << std::setprecision(3);
        for (double sum : rejection_sums)
            std::cout << ' ' << sum / runs;
        std::cout << "\n  Global communication barrier  " << barrier;
        std::cout << "\n  Round trips by replica        ";
        for (auto completed : round_trips)
            std::cout << ' ' << completed;
        std::cout << "\n  Total round trips             " << trips << '\n';
    }

    template <typename Direction>
    void benchmark_hit_and_run(std::string_view direction, std::size_t count) {
        const auto family = [&](std::string_view line_name, auto line) {
            const auto name =
                std::string(line_name) + " Hit-and-Run (" + std::string(direction) + ')';
            benchmark_kernel(name, count, [line](const State &x) {
                return HitAndRunKernel(x, Direction(x.size()), line);
            });
        };
        family("Uniform", UniformLineKernel{});
        family("Gaussian", GaussianLineKernel{});
        family("Overrelaxed", OverRelaxedLineKernel{});
    }
} // namespace

int main() {
    for (std::size_t count : {8, 32}) {
        benchmark_communication(count);
        benchmark_hit_and_run<RandomCoordinateDirectionGenerator>("random coordinate", count);
        benchmark_hit_and_run<CyclingCoordinateDirectionGenerator>("cycling coordinate", count);
        benchmark_hit_and_run<RandomDirectionGenerator<double>>("dense", count);
        benchmark_kernel("Gaussian random walk", count, [](const State &x) {
            return GaussianKernel<State, ZeroDrift, IdentityMetric>(x, {}, {}, 0.7);
        });
        benchmark_kernel("MALA (identity metric)", count, [](const State &x) {
            return GaussianKernel<State, ClippedGradientDrift, IdentityMetric>(x, {}, {}, 0.7);
        });
        benchmark_kernel("Dikin walk", count,
                         [](const State &x) { return DikinWalkKernel<State>(x, {}, {}, 0.3); });
        benchmark_kernel("CSmMALA", count,
                         [](const State &x) { return CSmMALAKernel<State>(x, {}, {}, 0.3); });
        std::cout << '\n' << std::flush;
    }
}
