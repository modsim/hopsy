// Isolate arithmetic choices without changing production sampler contracts.
#include <hopsy/kernel/metric.h>
#include <hopsy/kernel/drift.h>
#include <hopsy/geometry/linear_polytope.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>

using Matrix = hopsy::Matrix<double>;
using State = hopsy::Vector<double>;
void observe(const auto &x) { asm volatile("" : : "g"(&x) : "memory"); }
double measure(auto function) {
    std::array<double, 5> times;
    for (auto &ns : times) {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 200000; ++i)
            function();
        ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
                 .count() /
             200000;
    }
    std::sort(times.begin(), times.end());
    return times[2];
}
int main() {
    Matrix A = Matrix::Random(20, 10), weighted(20, 10);
    State slacks = State::LinSpaced(20, 1., 3.), inverse(20),
          gradient = State::LinSpaced(10, -1., 1.);
    auto row_divide = [&] {
        weighted = A;
        for (Eigen::Index row = 0; row < weighted.rows(); ++row)
            weighted.row(row) /= slacks(row);
        observe(weighted);
    };
    auto reciprocal_scale = [&] {
        inverse = slacks.cwiseInverse();
        weighted.noalias() = inverse.asDiagonal() * A;
        observe(weighted);
    };
    row_divide();
    const Matrix expected = weighted;
    reciprocal_scale();
    if (!weighted.isApprox(expected, 1e-14))
        return 1;
    Matrix box_A(20, 10);
    box_A.topRows(10).setIdentity();
    box_A.bottomRows(10) = -box_A.topRows(10);
    const State b = State::Constant(20, 3.), point = State::Constant(10, 0.5);
    hopsy::LinearPolytope<double> geometry(box_A, b);
    hopsy::DikinMetric<double> metric;
    hopsy::EmptyWorkspace workspace;
    if (!metric.prepare(point, geometry, workspace))
        return 1;
    struct Target {
        State log_gradient(const State &x) const { return -x; }
    } target;
    State displacement(10);
    std::cout << "stage,ns\nrow_divide," << measure(row_divide) << "\nreciprocal_scale,"
              << measure(reciprocal_scale) << "\ngradient_norm," << measure([&] {
                     const double norm = gradient.norm();
                     observe(norm);
                 })
              << "\ngradient_stable_norm," << measure([&] {
                     const double norm = gradient.stableNorm();
                     observe(norm);
                 })
              << "\nrow_feasibility," << measure([&] {
                     const bool valid = geometry.is_feasible(point);
                     observe(valid);
                 })
              << "\neigen_feasibility," << measure([&] {
                     const bool valid = ((b - box_A * point).array() >= 0.).all();
                     observe(valid);
                 })
              << "\nclipped_preconditioned_drift," << measure([&] {
                     hopsy::ClippedGradientDrift{}.compute(point, 0.3, target, metric, geometry,
                                                           workspace, displacement);
                     observe(displacement);
                 })
              << '\n';
}
