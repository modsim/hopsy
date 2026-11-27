#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/parallel.h>
#include <hopsy/sampling/parallel_tempering/non_reversible_parallel_tempering.h>
#include <hopsy/target/gaussian.h>
#include "batch_means.h"

namespace {
    using namespace hopsy;
    using State = Vector<double>;
    using Samples = Matrix<double>;
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t chains = 8;
    constexpr std::size_t runs = 3;
    constexpr Index dimension = 8;

    struct Workload {
        double radius;
        Index warmup_steps, steps, batch_steps;
    };

    struct Result {
        std::vector<Samples> traces;
        double seconds = 0.;
        double barrier = 0.;
        std::size_t round_trips = 0;
    };

    struct Normal {
        double log_density(const State &x) const { return -0.5 * x.squaredNorm(); }
    };

    auto box(double radius) {
        Matrix<double> A(2 * dimension, dimension);
        A.topRows(dimension).setIdentity();
        A.bottomRows(dimension) = -A.topRows(dimension);
        return LinearPolytope<double>(A, State::Constant(2 * dimension, radius));
    }

    State initial_state(double radius, std::size_t slot) {
        return State::Constant(dimension, 0.8 * radius * (2. * slot / (chains - 1) - 1.));
    }

    RandomNumberGenerator local_rng(std::size_t run, std::size_t slot) {
        return RandomNumberGenerator(70 + run, run * (chains + 1) + slot);
    }

    template <typename Problem, typename Factory>
    Result independent(const Problem &problem, const Factory &make_kernel, const Workload &workload,
                       std::size_t run) {
        Result result;
        result.traces.resize(chains);
        Clock::time_point start, stop;
        std::array<std::exception_ptr, chains> failures{};
        std::barrier ready(chains, [&]() noexcept { start = Clock::now(); });
        std::barrier done(chains, [&]() noexcept { stop = Clock::now(); });
        hopsy::detail::run_workers(chains, [&](std::size_t id) {
            bool warmed_up = false;
            try {
                auto &samples = result.traces[id];
                samples.resize(dimension, workload.steps);
                State state = initial_state(workload.radius, id);
                auto kernel = make_kernel(state);
                auto chain = bind_chain(problem, kernel, state);
                auto rng = local_rng(run, id);
                for (Index n = 0; n < workload.warmup_steps; ++n)
                    chain.step(rng);
                ready.arrive_and_wait();
                warmed_up = true;
                chain.draw_sampling_into(rng, samples);
            } catch (...) {
                failures[id] = std::current_exception();
                if (!warmed_up)
                    ready.arrive_and_wait();
            }
            done.arrive_and_wait();
        });
        for (const auto &error : failures)
            if (error)
                std::rethrow_exception(error);
        result.seconds = std::chrono::duration<double>(stop - start).count();
        return result;
    }

    template <typename Sampler>
    double timed_scans(Sampler &sampler, Samples &samples, Index warmup_scans, Index block,
                       std::optional<std::size_t> workers) {
        const Index scans = warmup_scans + samples.cols();
        const std::vector<Index> steps(chains, block);
        auto start = Clock::now();
        Clock::time_point stop;
        sampler.run_scans(scans, steps, workers,
            [&](Index scan) {
                if (scan >= warmup_scans)
                    samples.col(scan - warmup_scans) = sampler.sampling_state(chains - 1);
            },
            [&](Index scan) {
                if (scan + 1 == warmup_scans) {
                    sampler.reset_statistics();
                    start = Clock::now();
                }
                if (scan + 1 == scans)
                    stop = Clock::now();
            });
        return std::chrono::duration<double>(stop - start).count();
    }

    template <typename Problem, typename Factory>
    Result tempering(const Problem &problem, const Factory &make_kernel, const Workload &workload,
                     std::size_t run, Index block, bool threaded) {
        using Kernel = decltype(make_kernel(std::declval<const State &>()));
        std::vector<TemperingReplica<Kernel>> replicas;
        replicas.reserve(chains);
        for (std::size_t slot = 0; slot < chains; ++slot) {
            const State state = initial_state(workload.radius, slot);
            replicas.push_back({state, make_kernel(state), local_rng(run, slot)});
        }
        auto sampler = NonReversibleParallelTempering(
            problem, TemperatureLadder::linear(chains), std::move(replicas),
            RandomNumberGenerator(80 + run, run * (chains + 1) + chains));
        const Index warmup_scans = workload.warmup_steps / block;
        Result result;
        auto &samples = result.traces.emplace_back(dimension, workload.steps / block);
        result.seconds = timed_scans(sampler, samples, warmup_scans, block,
                                     threaded ? std::optional{chains} : std::nullopt);
        const auto rejections = sampler.mean_exchange_rejections();
        const auto trips = sampler.round_trips();
        result.barrier = std::accumulate(rejections.begin(), rejections.end(), 0.);
        result.round_trips = std::accumulate(trips.begin(), trips.end(), std::size_t{0});
        return result;
    }

    struct Summary {
        explicit Summary(Index batch_size) : ess(dimension + 1, batch_size) {}

        void add(const Result &result, double radius) {
            for (const auto &samples : result.traces) {
                if (!samples.allFinite() || (samples.array().abs() > radius + 1e-10).any())
                    throw std::runtime_error("Invalid constrained sample");
                Samples observations(dimension + 1, samples.cols());
                observations.topRows(dimension) = samples;
                observations.row(dimension) = samples.colwise().mean();
                ess.add(observations);
            }
            seconds += result.seconds;
            barrier += result.barrier;
            round_trips += result.round_trips;
        }

        void print(std::string_view target, std::string_view kernel, std::string_view mode,
                   Index block) const {
            const auto estimates = ess.ess();
            if (!estimates.allFinite())
                throw std::runtime_error("Benchmark produced a degenerate ESS estimate");
            const double minimum = estimates.head(dimension).minCoeff();
            const double common = estimates[dimension];
            std::cout << target << ',' << kernel << ',' << mode << ',' << block << ','
                      << ess.sample_count() << ',' << seconds << ',' << minimum << ','
                      << minimum / seconds << ',' << common << ',' << common / seconds << ',';
            if (block > 0)
                std::cout << barrier / runs << ',' << round_trips;
            else
                std::cout << ',';
            std::cout << '\n' << std::flush;
        }

        benchmark::BatchMeans ess;
        double seconds = 0., barrier = 0.;
        std::size_t round_trips = 0;
    };

    template <typename Problem, typename Factory>
    void compare(std::string_view target, std::string_view kernel, const Problem &problem,
                 const Factory &make_kernel, const Workload &workload) {
        Summary cold(workload.batch_steps);
        for (std::size_t run = 0; run < runs; ++run)
            cold.add(independent(problem, make_kernel, workload, run), workload.radius);
        cold.print(target, kernel, "8 independent workers", 0);

        for (Index block : {1, 2, 4, 8, 16, 32}) {
            Summary serial(workload.batch_steps / block), threaded(workload.batch_steps / block);
            for (std::size_t run = 0; run < runs; ++run) {
                const auto a = tempering(problem, make_kernel, workload, run, block, false);
                const auto b = tempering(problem, make_kernel, workload, run, block, true);
                if (a.traces.front() != b.traces.front() || a.barrier != b.barrier ||
                    a.round_trips != b.round_trips)
                    throw std::runtime_error("Serial and threaded NRPT diverged with equal seeds");
                serial.add(a, workload.radius);
                threaded.add(b, workload.radius);
            }
            serial.print(target, kernel, "serial NRPT", block);
            threaded.print(target, kernel, "8-worker NRPT", block);
        }
    }
} // namespace

int main() {
    std::cout
        << "target,kernel,mode,steps_per_exchange,cold_samples,wall_seconds,min_coordinate_ess,"
           "min_coordinate_ess_per_second,common_mode_ess,common_mode_ess_per_second,"
           "communication_barrier,round_trips\n"
        << std::fixed << std::setprecision(6);
    constexpr Workload standard{4., 8192, 131072, 512};
    compare(
        "standard Gaussian", "uniform cycling-coordinate Hit-and-Run",
        Problem(box(standard.radius), Normal{}),
        [](const State &x) {
            return HitAndRunKernel(x, CyclingCoordinateDirectionGenerator(x.size()),
                                   UniformLineKernel{});
        },
        standard);

    constexpr Workload correlated{1., 32768, 262144, 8192};
    constexpr double rho = 0.975;
    Matrix<double> covariance = Matrix<double>::Constant(dimension, dimension, rho);
    covariance.diagonal().setOnes();
    const auto problem =
        Problem(box(correlated.radius), GaussianTarget(State::Zero(dimension), covariance));
    compare(
        "correlated Gaussian (rho 0.975)", "CSmMALA", problem,
        [](const State &x) { return CSmMALAKernel<State>(x, {}, {}, 0.25); }, correlated);
    compare(
        "correlated Gaussian (rho 0.975)", "uniform dense Hit-and-Run", problem,
        [](const State &x) {
            return HitAndRunKernel(x, RandomDirectionGenerator<double>(x.size()),
                                   UniformLineKernel{});
        },
        correlated);
}
