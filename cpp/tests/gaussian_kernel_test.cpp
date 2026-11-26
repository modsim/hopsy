#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/kernel/metric.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/sampling/draw.h>
#include <hopsy/target/uniform.h>
#include <array>
#include <limits>

namespace {
    using State = hopsy::Vector<double>;
    using Kernel = hopsy::GaussianKernel<State, hopsy::ZeroDrift, hopsy::IdentityMetric>;
    using Catch::Approx;
    hopsy::LinearPolytope<double> box(int n, double radius = 1.) {
        hopsy::Matrix<double> A(2 * n, n);
        A.topRows(n).setIdentity();
        A.bottomRows(n) = -A.topRows(n);
        return {A, State::Constant(2 * n, radius)};
    }
    struct GaussianTarget {
        double log_density(const State &x) const { return -0.5 * x.squaredNorm(); }
    };
    struct CountingTarget {
        mutable int calls = 0;
        double value = 0.;
        double log_density(const State &) const {
            ++calls;
            return value;
        }
    };
    struct NoEigenAllocation {
        bool previous = Eigen::internal::is_malloc_allowed();
        NoEigenAllocation() { Eigen::internal::set_is_malloc_allowed(false); }
        ~NoEigenAllocation() { Eigen::internal::set_is_malloc_allowed(previous); }
    };
} // namespace

TEST_CASE("Gaussian parameters validate transactionally", "[gaussian]") {
    State x = State::Zero(3);
    Kernel kernel(x, {}, {}, 0.4);
    const std::array invalid{0., -1., std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()};
    for (double h : invalid) {
        REQUIRE_THROWS_AS(kernel.set_step_size(h), std::invalid_argument);
        REQUIRE(kernel.step_size() == 0.4);
        REQUIRE_THROWS_AS((Kernel(x, {}, {}, h)), std::invalid_argument);
    }
    kernel.set_step_size(2.);
    REQUIRE(kernel.step_size() == 2.);
}

TEST_CASE("Gaussian increments have covariance sigma squared I and zero mean",
          "[gaussian][statistics]") {
    State x = State::Constant(3, 1.25);
    auto geometry = box(3, 100.);
    hopsy::EmptyWorkspace workspace;
    hopsy::UniformTarget target;
    hopsy::RandomNumberGenerator rng(157);
    Kernel kernel(x, {}, {}, 0.2);
    for (double sigma : {0.2, 1.3}) {
        kernel.set_step_size(sigma);
        State sum = State::Zero(3);
        hopsy::Matrix<double> second = hopsy::Matrix<double>::Zero(3, 3);
        constexpr int count = 100000;
        for (int i = 0; i < count; ++i) {
            kernel.propose(x, rng, target, geometry, workspace);
            State delta = kernel.proposal().state - x;
            sum += delta;
            second.noalias() += delta * delta.transpose();
            kernel.reject(x);
        }
        State mean = sum / count;
        hopsy::Matrix<double> covariance = second / count - mean * mean.transpose();
        // Fixed-seed check of standardized proposal moments, not chain mixing.
        INFO("sigma = " << sigma);
        REQUIRE(mean.cwiseAbs().maxCoeff() < 0.015 * sigma);
        REQUIRE((covariance / (sigma * sigma) - hopsy::Matrix<double>::Identity(3, 3))
                    .cwiseAbs()
                    .maxCoeff() < 0.025);
        REQUIRE(kernel.proposal().state.size() == x.size());
        REQUIRE(kernel.proposal().step_size == sigma);
    }
}

TEST_CASE("Gaussian full MH ratio counts the target exactly once", "[gaussian]") {
    State x = State::Constant(3, 0.4);
    auto geometry = box(3, 100.);
    auto workspace = geometry.create_workspace(x);
    hopsy::RandomNumberGenerator rng(27);
    GaussianTarget target;
    Kernel kernel(x, {}, {}, 0.3);
    kernel.propose(x, rng, target, geometry, workspace);
    const State y = kernel.proposal().state;
    REQUIRE(kernel.proposal().step_size == 0.3);
    REQUIRE(kernel.log_acceptance_ratio(x, target, geometry, workspace) ==
            Approx(target.log_density(y) - target.log_density(x)));
    kernel.accept(x, geometry, workspace);
    kernel.set_step_size(7.);
    REQUIRE(kernel.proposal().step_size == 0.3);
    kernel.propose(x, rng, target, geometry, workspace);
    REQUIRE(kernel.proposal().step_size == 7.);
    kernel.reject(x);
}

TEST_CASE("Infeasible proposals reject without target evaluation or acceptance RNG", "[gaussian]") {
    State x = State::Zero(3);
    const State old = x;
    auto geometry = box(3, 0.);
    auto workspace = geometry.create_workspace(x);
    workspace.direction_product.setZero();
    const State slacks = workspace.slacks;
    hopsy::RandomNumberGenerator rng(12), reference(12);
    CountingTarget target;
    Kernel kernel(x, {}, {}, 1.), other(x, {}, {}, 1.);
    other.propose(x, reference, target, geometry, workspace);
    REQUIRE_FALSE(hopsy::step(kernel, x, rng, target, geometry, workspace));
    REQUIRE(target.calls == 0);
    REQUIRE(x == old);
    REQUIRE(workspace.slacks == slacks);
    REQUIRE(workspace.direction_product.isZero());
    REQUIRE(rng() == reference());
}

TEST_CASE("Uniform feasible transitions avoid acceptance RNG", "[gaussian]") {
    State x = State::Zero(3);
    auto geometry = box(3, 100.);
    hopsy::EmptyWorkspace workspace;
    hopsy::UniformTarget target;
    hopsy::RandomNumberGenerator rng(12), reference(12);
    Kernel kernel(x, {}, {}, 0.1), other(x, {}, {}, 0.1);
    other.propose(x, reference, target, geometry, workspace);
    REQUIRE(hopsy::step(kernel, x, rng, target, geometry, workspace));
    REQUIRE(rng() == reference());
}

TEST_CASE("Gaussian invalid numerical proposals and targets reject", "[gaussian]") {
    auto geometry = box(2, 100.);
    hopsy::EmptyWorkspace workspace;
    hopsy::RandomNumberGenerator rng(7);
    for (double bad :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        State x = State::Constant(2, bad);
        Kernel kernel(x, {}, {}, 1.);
        CountingTarget target;
        REQUIRE_FALSE(hopsy::step(kernel, x, rng, target, geometry, workspace));
        REQUIRE(target.calls == 0);
    }
    State x = State::Zero(2);
    Kernel kernel(x, {}, {}, 0.1);
    CountingTarget target{0, std::numeric_limits<double>::quiet_NaN()};
    REQUIRE_FALSE(hopsy::step(kernel, x, rng, target, geometry, workspace));
    REQUIRE(x.isZero());
}

TEST_CASE("Gaussian rejects dimension changes", "[gaussian]") {
    State x = State::Zero(2), wrong = State::Zero(3);
    Kernel kernel(x, {}, {}, 1.);
    auto geometry = box(2);
    hopsy::EmptyWorkspace workspace;
    hopsy::RandomNumberGenerator rng(7);
    REQUIRE_THROWS_AS(kernel.propose(wrong, rng, hopsy::UniformTarget{}, geometry, workspace),
                      std::invalid_argument);
}

TEST_CASE("Gaussian target stationary moments", "[gaussian][statistics]") {
    State x = State::Zero(2);
    // Unbounded domain, including the zero-constraint workspace case.
    hopsy::LinearPolytope<double> geometry(hopsy::Matrix<double>(0, 2), State(0));
    auto workspace = geometry.create_workspace(x);
    hopsy::RandomNumberGenerator rng(416);
    GaussianTarget target;
    Kernel kernel(x, {}, {}, 1.);
    for (int i = 0; i < 5000; ++i)
        hopsy::step(kernel, x, rng, target, geometry, workspace);
    auto samples = hopsy::draw(kernel, x, rng, target, geometry, workspace, 100000, 3);
    REQUIRE(samples.rowwise().mean().cwiseAbs().maxCoeff() < 0.035);
    REQUIRE((samples.array().square().rowwise().mean() - 1.).abs().maxCoeff() < 0.045);
    REQUIRE(std::abs(samples.row(0).dot(samples.row(1)) / samples.cols()) < 0.035);
}

TEST_CASE("Gaussian uniform box marginals", "[gaussian][statistics]") {
    State x = State::Zero(2);
    auto geometry = box(2);
    auto workspace = geometry.create_workspace(x);
    hopsy::RandomNumberGenerator rng(948);
    hopsy::UniformTarget target;
    Kernel kernel(x, {}, {}, 0.5);
    for (int i = 0; i < 2000; ++i)
        hopsy::step(kernel, x, rng, target, geometry, workspace);
    auto samples = hopsy::draw(kernel, x, rng, target, geometry, workspace, 100000, 3);
    REQUIRE(samples.cwiseAbs().maxCoeff() <= 1.);
    REQUIRE(samples.rowwise().mean().cwiseAbs().maxCoeff() < 0.02);
    REQUIRE((samples.array().square().rowwise().mean() - 1. / 3.).abs().maxCoeff() < 0.015);
    for (int row = 0; row < 2; ++row) {
        std::array<int, 10> bins{};
        for (int j = 0; j < samples.cols(); ++j)
            ++bins[std::min(9, int(5 * (samples(row, j) + 1)))];
        for (int count : bins)
            REQUIRE(std::abs(count / 100000. - 0.1) < 0.008);
    }
}

TEST_CASE("Gaussian transitions allocate no Eigen buffers", "[gaussian][allocation]") {
    for (int n : {3, 64, 256}) {
        State x = State::Zero(n);
        auto geometry = box(n, 100.);
        auto workspace = geometry.create_workspace(x);
        hopsy::EmptyWorkspace empty;
        Kernel kernel(x, {}, {}, 0.01);
        hopsy::RandomNumberGenerator rng(123);
        hopsy::UniformTarget target;
        NoEigenAllocation guard;
        for (int i = 0; i < 10; ++i) {
            kernel.propose(x, rng, target, geometry, workspace);
            REQUIRE(kernel.log_acceptance_ratio(x, target, geometry, workspace) == 0.);
            kernel.accept(x, geometry, workspace);
            kernel.propose(x, rng, target, geometry, workspace);
            kernel.reject(x);
            REQUIRE(hopsy::step(kernel, x, rng, target, geometry, empty));
            // Rebuild before using a cached workspace again after EmptyWorkspace.
            geometry.reset_workspace(x, workspace);
        }
    }
}

TEST_CASE("Gaussian float and fixed-size states", "[gaussian]") {
    using FloatState = hopsy::Vector<float>;
    FloatState x = FloatState::Zero(3);
    hopsy::LinearPolytope<float> geometry(hopsy::Matrix<float>::Identity(3, 3),
                                          FloatState::Constant(3, 100.f));
    auto workspace = geometry.create_workspace(x);
    hopsy::RandomNumberGenerator rng(123);
    hopsy::GaussianKernel<FloatState, hopsy::ZeroDrift, hopsy::IdentityMetric> kernel(x, {}, {},
                                                                                      0.1f);
    REQUIRE(hopsy::step(kernel, x, rng, hopsy::UniformTarget{}, geometry, workspace));
    REQUIRE(workspace.slacks.isApprox(geometry.b() - geometry.A() * x));
    Eigen::Vector3d fixed = Eigen::Vector3d::Zero();
    auto double_geometry = box(3, 100.);
    hopsy::EmptyWorkspace empty;
    hopsy::GaussianKernel<Eigen::Vector3d, hopsy::ZeroDrift, hopsy::IdentityMetric> fixed_kernel(
        fixed, {}, {}, 0.1);
    REQUIRE(hopsy::step(fixed_kernel, fixed, rng, hopsy::UniformTarget{}, double_geometry, empty));
}

TEST_CASE("Gaussian scale is linear and updates after accept or reject", "[gaussian]") {
    State x = State::Constant(3, 0.25);
    auto geometry = box(3, 100.);
    auto workspace = geometry.create_workspace(x);
    hopsy::UniformTarget target;
    Kernel kernel(x, {}, {}, 0.25);

    for (bool accept : {true, false}) {
        const State before = x;
        hopsy::RandomNumberGenerator rng(81);
        kernel.propose(x, rng, target, geometry, workspace);
        const State delta = kernel.proposal().state - before;
        REQUIRE(delta.norm() > 0.);
        REQUIRE(kernel.log_acceptance_ratio(x, target, geometry, workspace) == 0.);
        if (accept)
            kernel.accept(x, geometry, workspace);
        else
            kernel.reject(x);
        const State accepted_state = x;
        const State slacks = workspace.slacks;
        const double old_scale = kernel.step_size();

        kernel.set_step_size(2. * old_scale);
        REQUIRE(kernel.step_size() == 2. * old_scale);
        REQUIRE(kernel.proposal().step_size == old_scale);
        REQUIRE(x == accepted_state);
        REQUIRE(workspace.slacks == slacks);

        // Replaying the same noise isolates scale from RNG and state changes.
        hopsy::RandomNumberGenerator replay(81);
        kernel.propose(x, replay, target, geometry, workspace);
        REQUIRE((kernel.proposal().state - x).isApprox(2. * delta, 1e-13));
        REQUIRE(kernel.proposal().step_size == 2. * old_scale);
        REQUIRE(kernel.log_acceptance_ratio(x, target, geometry, workspace) == 0.);
        kernel.reject(x);
    }
}

TEST_CASE("Gaussian accepted and rejected transitions preserve geometry caches", "[gaussian]") {
    State x = State::Zero(3);
    auto geometry = box(3, 100.);
    auto workspace = geometry.create_workspace(x);
    // A previous line proposal's cached product must not drive a Gaussian commit.
    workspace.direction_product.setConstant(123.);
    hopsy::RandomNumberGenerator rng(57);
    hopsy::UniformTarget target;
    Kernel kernel(x, {}, {}, 0.1);
    for (int i = 0; i < 100; ++i) {
        kernel.set_step_size(i % 2 == 0 ? 0.1 : 0.4);
        const State before = x;
        const State slacks = workspace.slacks;
        kernel.propose(x, rng, target, geometry, workspace);
        const State candidate = kernel.proposal().state;
        REQUIRE(x == before);
        REQUIRE(workspace.slacks == slacks);
        REQUIRE(kernel.log_acceptance_ratio(x, target, geometry, workspace) == 0.);
        if (i % 2 == 0) {
            kernel.accept(x, geometry, workspace);
            REQUIRE(x == candidate);
            REQUIRE(kernel.proposal().state == before);
        } else {
            kernel.reject(x);
            REQUIRE(x == before);
            REQUIRE(workspace.slacks == slacks);
        }
        REQUIRE(workspace.slacks.isApprox(geometry.b() - geometry.A() * x, 1e-13));
        REQUIRE(workspace.direction_product == State::Constant(6, 123.));
    }
}

TEST_CASE("Gaussian and dense hit-and-run share a geometry workspace", "[gaussian]") {
    State x = State::Zero(3);
    auto geometry = box(3);
    auto workspace = geometry.create_workspace(x);
    hopsy::RandomNumberGenerator rng(92);
    hopsy::UniformTarget target;
    Kernel gaussian(x, {}, {}, 0.25);
    hopsy::HitAndRunKernel hit_and_run(x, hopsy::RandomDirectionGenerator<double>(3),
                                       hopsy::UniformLineKernel{});
    int accepted = 0, rejected = 0;
    for (int i = 0; i < 200; ++i) {
        REQUIRE(hopsy::step(hit_and_run, x, rng, target, geometry, workspace));
        REQUIRE(workspace.slacks.isApprox(geometry.b() - geometry.A() * x, 1e-12));
        gaussian.set_step_size(i % 2 == 0 ? 0.05 : 10.);
        const State before = x;
        const State slacks = workspace.slacks;
        if (hopsy::step(gaussian, x, rng, target, geometry, workspace))
            ++accepted;
        else {
            ++rejected;
            REQUIRE(x == before);
            REQUIRE(workspace.slacks == slacks);
        }
        REQUIRE(geometry.is_feasible(x, -1e-12));
        REQUIRE(workspace.slacks.isApprox(geometry.b() - geometry.A() * x, 1e-12));
    }
    REQUIRE(accepted > 0);
    REQUIRE(rejected > 0);
}
