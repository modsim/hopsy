// Release benchmark for the public native draw paths in either source layout.
#include <bit>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/chain.h>
#if __has_include(<hopsy/sampling/parallel_tempering/non_reversible_parallel_tempering.h>)
#include <hopsy/sampling/parallel_tempering/non_reversible_parallel_tempering.h>
#else
#include <hopsy/sampling/non_reversible_parallel_tempering.h>
#endif
#include <hopsy/target/gaussian.h>
#include <hopsy/target/uniform.h>

namespace {
    using namespace hopsy;
    using State = Vector<double>;
    using Samples = Matrix<double>;
    using Clock = std::chrono::steady_clock;

    struct Normal {
        double log_density(const State &x) const { return -0.5 * x.squaredNorm(); }
    };

    auto box(Index dimension) {
        Samples A(2 * dimension, dimension);
        A.topRows(dimension).setIdentity();
        A.bottomRows(dimension) = -A.topRows(dimension);
        return LinearPolytope<double>(A, State::Constant(2 * dimension, 3.));
    }

    std::uint64_t digest(const Samples &samples) {
        std::uint64_t hash = 14695981039346656037ULL;
        for (Index i = 0; i < samples.size(); ++i) {
            hash ^= std::bit_cast<std::uint64_t>(samples.data()[i]);
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    template <typename Draw>
    void measure(double setup, Index loops, const Draw &draw) {
        auto samples = draw();
        const auto cpu = std::clock();
        const auto start = Clock::now();
        for (Index i = 0; i < loops; ++i)
            samples = draw();
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        const double cpu_seconds = double(std::clock() - cpu) / CLOCKS_PER_SEC;
        std::cout << std::setprecision(17) << "{\"seconds\":" << seconds
                  << ",\"setup_seconds\":" << setup << ",\"cpu_seconds\":" << cpu_seconds
                  << ",\"digest\":\"" << digest(samples) << "\"}\n";
    }

    void regular(Index dimension, Index batch, Index thinning, Index loops, unsigned long seed) {
        const auto start = Clock::now();
        const auto problem = Problem(box(dimension), UniformTarget{});
        State state = State::Zero(dimension);
        auto kernel = HitAndRunKernel(state, CyclingCoordinateDirectionGenerator(dimension),
                                      UniformLineKernel{});
        auto chain = bind_chain(problem, kernel, state);
        RandomNumberGenerator rng(seed);
        const double setup = std::chrono::duration<double>(Clock::now() - start).count();
        for (Index i = 0; i < 4096; ++i)
            chain.step(rng);
        measure(setup, loops, [&] { return chain.draw(rng, batch, thinning); });
    }

    template <typename Direction, typename Problem>
    void nrpt(Problem problem, Index dimension, Index batch, Index thinning, Index loops,
              unsigned long seed, Clock::time_point start) {
        constexpr std::size_t slots = 8;
        const auto state = State::Zero(dimension).eval();
        using Kernel = decltype(HitAndRunKernel(state, Direction(dimension), UniformLineKernel{}));
        std::vector<TemperingReplica<Kernel>> replicas;
        for (std::size_t i = 0; i < slots; ++i)
            replicas.push_back({state, {state, Direction(dimension), {}},
                                RandomNumberGenerator(seed, i)});
        auto sampler = NonReversibleParallelTempering(
            std::move(problem), TemperatureLadder::linear(slots),
            std::move(replicas), RandomNumberGenerator(seed, slots));
        const double setup = std::chrono::duration<double>(Clock::now() - start).count();
        for (Index i = 0; i < 512; ++i)
            sampler.scan(thinning, thinning);
        measure(setup, loops, [&] { return sampler.draw(batch, thinning, thinning); });
    }
}

int main(int argc, char **argv) {
    if (argc != 7)
        throw std::invalid_argument("Expected mode dimension batch thinning loops seed");
    const std::string_view mode = argv[1];
    const auto dimension = std::stol(argv[2]);
    const auto batch = std::stol(argv[3]);
    const auto thinning = std::stol(argv[4]);
    const auto loops = std::stol(argv[5]);
    const auto seed = std::stoul(argv[6]);
    const auto start = Clock::now();
    if (mode == "regular") {
        regular(dimension, batch, thinning, loops, seed);
    } else if (mode == "nrpt") {
        nrpt<RandomCoordinateDirectionGenerator>(Problem(box(dimension), Normal{}), dimension,
                                                batch, thinning, loops, seed, start);
    } else if (mode == "nrpt-gaussian") {
        const auto geometry = box(dimension);
        const auto target =
            GaussianTarget(State::Zero(dimension), Samples::Identity(dimension, dimension));
        nrpt<CyclingCoordinateDirectionGenerator>(Problem(std::cref(geometry), std::cref(target)),
                                                dimension, batch, thinning, loops, seed, start);
    } else {
        throw std::invalid_argument("Unknown mode");
    }
}
