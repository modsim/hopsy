#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/target/uniform.h>
#include <hopsy/sampling/step.h>
#ifdef HOPSY_BOUND_CONTRACT
#include <hopsy/sampling/chain.h>
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>

using State = hopsy::Vector<double>;
struct Normal {
    double log_density(const State &x) const { return -0.5 * x.squaredNorm(); }
    State log_gradient(const State &x) const { return -x; }
};
void observe(const State &x) { asm volatile("" : : "g"(&x) : "memory"); }

template <typename Target, typename Factory, bool Cached>
void run(const char *name, int n, const Target &target, Factory factory) {
    hopsy::Matrix<double> A(2 * n, n);
    A.topRows(n).setIdentity();
    A.bottomRows(n) = -A.topRows(n);
    hopsy::LinearPolytope<double> geometry(A, State::Constant(2 * n, 3.));
    std::array<double, 5> timings{};
    double checksum = 0.;
    for (int repeat = 0; repeat < 5; ++repeat) {
        State state = State::Zero(n);
        auto kernel = factory(state, n);
        auto workspace = [&] {
            if constexpr (Cached)
                return geometry.create_workspace(state);
            else
                return hopsy::EmptyWorkspace{};
        }();
        hopsy::RandomNumberGenerator rng(42 + repeat);
#ifdef HOPSY_BOUND_CONTRACT
        auto problem = hopsy::Problem(std::cref(geometry), std::cref(target));
        auto chain = hopsy::bind_chain(problem, kernel, state, workspace);
        auto transition = [&] { return chain.step(rng); };
#else
        auto transition = [&] {
            return hopsy::step(kernel, state, rng, target, geometry, workspace);
        };
#endif
        for (int i = 0; i < 1000; ++i)
            transition();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 100000; ++i) {
            transition();
            observe(state);
        }
        timings[repeat] =
            std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
                .count() /
            100000;
        checksum += state.sum();
        if (!state.allFinite() || !geometry.is_feasible(state, -1e-10))
            throw std::runtime_error("invalid state");
    }
    std::sort(timings.begin(), timings.end());
    std::cout << name << ',' << n << ',' << timings[2] << ',' << 1e9 / timings[2] << ',' << checksum
              << '\n';
}
int main() {
    std::cout << "kernel,dimension,median_ns,steps_per_second,checksum\n" << std::setprecision(17);
    for (int n : {10, 100}) {
        run<hopsy::UniformTarget, decltype([](const State &s, int n) {
                return hopsy::HitAndRunKernel(s, hopsy::CyclingCoordinateDirectionGenerator(n),
                                              hopsy::UniformLineKernel{});
            }),
            true>("coordinate_hr", n, {}, {});
        run<hopsy::UniformTarget, decltype([](const State &s, int n) {
                return hopsy::HitAndRunKernel(s, hopsy::RandomDirectionGenerator<double>(n),
                                              hopsy::UniformLineKernel{});
            }),
            true>("dense_hr", n, {}, {});
    }
    run<Normal, decltype([](const State &s, int) {
            return hopsy::HitAndRunKernel(s, hopsy::CyclingCoordinateDirectionGenerator(10),
                                          hopsy::UniformLineKernel{});
        }),
        true>("coordinate_hr_normal", 10, {}, {});
    run<Normal, decltype([](const State &s, int) {
            return hopsy::GaussianKernel<State, hopsy::ZeroDrift, hopsy::IdentityMetric>(s, {}, {},
                                                                                         0.7);
        }),
        false>("grw", 10, {}, {});
    run<Normal,
        decltype([](const State &s, int) { return hopsy::DikinWalkKernel<State>(s, {}, {}, 0.3); }),
        false>("dikin", 10, {}, {});
    run<Normal,
        decltype([](const State &s, int) { return hopsy::CSmMALAKernel<State>(s, {}, {}, 0.3); }),
        false>("csmala", 10, {}, {});
}
