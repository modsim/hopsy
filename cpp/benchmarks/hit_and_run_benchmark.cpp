// A small reproducible microbenchmark, not a comparison against old HOPS.
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/step.h>
#include <hopsy/target/uniform.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

using Vector = hopsy::Vector<double>;

// Make results observable without adding vector reductions to measured loops.
template<typename T> void observe(const T &value) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(&value) : "memory");
#else
    volatile const T *sink = &value;
    (void)sink;
#endif
}

template<typename Function> double measure(int iterations, Function function) {
    std::array<double, 5> timings{};
    for (auto &timing : timings) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) function();
        timing = std::chrono::duration<double, std::nano>(
            std::chrono::steady_clock::now()-start).count()/iterations;
    }
    std::sort(timings.begin(), timings.end());
    return timings[2];
}

template<typename Generator> void run(int n, const char *name) {
    const int iterations = n <= 100'000'000;
    hopsy::Matrix<double> A(2*n, n);
    A.topRows(n).setIdentity();
    A.bottomRows(n) = -hopsy::Matrix<double>::Identity(n,n);
    hopsy::LinearPolytope<double> geometry(A, Vector::Ones(2*n));
    Vector state = Vector::Zero(n), output(n);
    auto workspace = geometry.create_workspace(state);
    hopsy::UniformTarget target;
    hopsy::UniformLineKernel line;
    hopsy::RandomNumberGenerator rng(42);
    hopsy::HitAndRunKernel kernel(state, Generator(n), line);
    hopsy::step(kernel, state, rng, target, geometry, workspace);
    const auto total = measure(iterations, [&] {
        hopsy::step(kernel, state, rng, target, geometry, workspace);
        observe(state);
    });

    // Fused in-place baseline uses the same geometry, RNG and line kernel,
    // but has no candidate buffer and cannot reject after materialization.
    state.setZero();
    geometry.reset_workspace(state, workspace);
    Generator generator(n);
    const auto fused = measure(iterations, [&] {
        const auto &direction = generator.generate(rng);
        const auto domain = geometry.line_domain(state, direction, workspace);
        const auto distance = line.propose(rng, domain);
        if constexpr (std::is_same_v<typename Generator::direction_type, hopsy::CoordinateDirection>)
            state(direction.coordinate) += distance;
        else
            state += distance * direction;
        geometry.advance_workspace(distance, direction, workspace);
        observe(state);
    });
    state.setZero();
    geometry.reset_workspace(state, workspace);
    const auto direction_time = measure(iterations, [&] { observe(generator.generate(rng)); });
    const auto &direction = generator.direction();
    const auto geometry_time = measure(iterations, [&] {
        const auto domain = geometry.line_domain(state, direction, workspace);
        observe(domain);
    });
    const auto domain = geometry.line_domain(state, direction, workspace);
    const auto line_time = measure(iterations, [&] { const auto t = line.propose(rng, domain); observe(t); });
    const auto materialization = measure(iterations, [&] {
        hopsy::step_direction(direction, state, 0.1, output);
        observe(output);
    });
    const auto target_time = measure(iterations, [&] {
        const double ratio = -0.5*output.squaredNorm()+0.5*state.squaredNorm();
        observe(ratio);
    });
    std::uniform_real_distribution<double> uniform(0,1);
    const auto mh_time = measure(iterations, [&] {
        const bool accepted = std::log(uniform(rng)) < -0.5;
        observe(accepted);
    });
    std::cout << name << ',' << n << ',' << total << ',' << fused << ','
              << direction_time << ',' << geometry_time << ',' << line_time << ','
              << materialization << ',' << target_time << ',' << mh_time << '\n';
}

int main() {
    std::cout << "direction,dimension,total_ns,fused_baseline_ns,direction_ns,geometry_ns,"
                 "line_draw_ns,materialization_ns,quadratic_target_ratio_ns,negative_MH_decision_ns\n";
    std::cout << std::fixed << std::setprecision(1);
    for (const int n : {8, 64, 256}) {
        run<hopsy::CyclingCoordinateDirectionGenerator>(n, "coordinate");
        run<hopsy::RandomDirectionGenerator<double>>(n, "general");
    }
}
