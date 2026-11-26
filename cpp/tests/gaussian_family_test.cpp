#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/sampling/draw.h>
#include <hopsy/target/uniform.h>
#include <hopsy/target/gaussian.h>

namespace {
    using State = hopsy::Vector<double>;
    using Matrix = hopsy::Matrix<double>;
    using Catch::Approx;
    struct Normal {
        double log_density(const State &x) const { return -0.5 * x.squaredNorm(); }
        State log_gradient(const State &x) const { return -x; }
    };
    auto skew_geometry() {
        Matrix A(4, 2);
        A << 1., 0.5, -1., -0.5, 0., 1., 0., -1.;
        return hopsy::LinearPolytope<double>(A, State::Constant(4, 2.));
    }
    auto box(int n, double radius) {
        Matrix A(2 * n, n);
        A.topRows(n).setIdentity();
        A.bottomRows(n) = -A.topRows(n);
        return hopsy::LinearPolytope<double>(A, State::Constant(2 * n, radius));
    }
    Matrix precision(const State &x, const auto &geometry) {
        const State slacks = geometry.b() - geometry.A() * x;
        return geometry.A().transpose() * slacks.array().square().inverse().matrix().asDiagonal() *
               geometry.A();
    }
    State clipped(const State &x) {
        State g = -x;
        return g / std::max(1., g.norm());
    }
    // Independent dense-matrix oracle, without production triangular solves.
    double log_q(const State &from, const State &to, const auto &geometry, double sigma,
                 bool drift) {
        Matrix G = precision(from, geometry);
        State mean = from;
        if (drift)
            mean += 0.5 * sigma * sigma * G.inverse() * clipped(from);
        State delta = to - mean;
        return 0.5 * std::log(G.determinant()) - 0.5 * delta.dot(G * delta) / (sigma * sigma);
    }
} // namespace

TEST_CASE("Gaussian symmetry is independent of always acceptance", "[gaussian_family]") {
    using RandomWalk = hopsy::GaussianKernel<State, hopsy::ZeroDrift, hopsy::IdentityMetric>;
    STATIC_REQUIRE(RandomWalk::symmetric_proposal);
    STATIC_REQUIRE_FALSE(RandomWalk::always_accept<hopsy::UniformTarget>);
    STATIC_REQUIRE_FALSE(hopsy::DikinWalkKernel<State>::symmetric_proposal);
    STATIC_REQUIRE_FALSE(hopsy::CSmMALAKernel<State>::symmetric_proposal);
}

TEST_CASE("Dikin metric uses inverse precision and correct Cholesky orientation",
          "[gaussian_family]") {
    const auto geometry = skew_geometry();
    State x(2);
    x << 0.3, -0.2;
    auto workspace = geometry.create_workspace(x);
    hopsy::DikinMetric<double> metric;
    REQUIRE(metric.prepare(x, geometry, workspace));
    const Matrix G = precision(x, geometry);
    REQUIRE(metric.precision().isApprox(G, 1e-13));
    REQUIRE(std::abs(G(0, 1)) > 0.1);
    REQUIRE(metric.log_sqrt_determinant() == Approx(0.5 * std::log(G.determinant())));
    State rhs(2);
    rhs << 0.7, -0.4;
    State solved = rhs;
    metric.solve_in_place(solved);
    REQUIRE(solved.isApprox(G.inverse() * rhs, 1e-13));
    REQUIRE(metric.squared_norm(rhs) == Approx(rhs.dot(G * rhs)));
    hopsy::RandomNumberGenerator rng(739);
    State noise = State::Zero(2), sum = State::Zero(2);
    Matrix second = Matrix::Zero(2, 2);
    constexpr int count = 100000;
    for (int i = 0; i < count; ++i) {
        metric.sample(rng, noise);
        sum += noise;
        second.noalias() += noise * noise.transpose();
    }
    // Fixed-seed covariance test detects L^-1 versus L^-T sampling.
    REQUIRE(sum.cwiseAbs().maxCoeff() / count < 0.02);
    REQUIRE((second / count - G.inverse()).cwiseAbs().maxCoeff() < 0.03);
}

TEST_CASE("Dikin metrics reject boundary and singular states", "[gaussian_family]") {
    auto geometry = box(2, 1.);
    hopsy::EmptyWorkspace workspace;
    hopsy::DikinMetric<double> metric;
    State x = State::Ones(2);
    REQUIRE_FALSE(metric.prepare(x, geometry, workspace));
    x.setConstant(2.);
    REQUIRE_FALSE(metric.prepare(x, geometry, workspace));
    Matrix A = Matrix::Zero(1, 2);
    A(0, 0) = 1.;
    hopsy::LinearPolytope<double> singular(A, State::Ones(1));
    x.setZero();
    REQUIRE_FALSE(metric.prepare(x, singular, workspace));
    hopsy::DikinWalkKernel<State> kernel(x, {}, {}, 0.1);
    hopsy::RandomNumberGenerator rng(7);
    REQUIRE_THROWS_AS(kernel.propose(x, rng, hopsy::UniformTarget{}, singular, workspace),
                      std::domain_error);
}

TEST_CASE("Clipped CSmMALA drift is continuous at zero and clips only above one",
          "[gaussian_family]") {
    hopsy::ClippedGradientDrift drift;
    hopsy::IdentityMetric metric;
    auto geometry = box(2, 10.);
    hopsy::EmptyWorkspace workspace;
    Normal target;
    State x = State::Zero(2), displacement = x;
    for (double magnitude : {0., 1e-12, 0.25, 1., 1.000001, 2.}) {
        x << magnitude, 0.;
        drift.compute(x, 0.4, target, metric, geometry, workspace, displacement);
        REQUIRE(displacement(0) == Approx(-0.08 * std::min(1., magnitude)).margin(1e-15));
        REQUIRE(displacement(1) == 0.);
    }
    x << 3., 4.;
    drift.compute(x, 0.4, target, metric, geometry, workspace, displacement);
    REQUIRE(displacement.isApprox(-0.08 * x / 5., 1e-13));
}

TEST_CASE("Asymmetric Gaussian MH ratios match independent densities", "[gaussian_family]") {
    auto geometry = skew_geometry();
    auto run = [&](auto kernel, bool drift) {
        State x(2);
        x << 0.3, -0.2;
        auto workspace = geometry.create_workspace(x);
        hopsy::RandomNumberGenerator rng(43);
        Normal target;
        for (int i = 0; i < 20; ++i) {
            const double sigma = i % 2 ? 0.1 : 0.25;
            kernel.set_step_size(sigma);
            const State before = x, slacks = workspace.slacks;
            kernel.propose(x, rng, target, geometry, workspace);
            const State y = kernel.proposal().state;
            REQUIRE(geometry.is_feasible(y));
            const double correction =
                log_q(y, x, geometry, sigma, drift) - log_q(x, y, geometry, sigma, drift);
            REQUIRE(
                kernel.log_acceptance_ratio(x, target, geometry, workspace) ==
                Approx(target.log_density(y) - target.log_density(x) + correction).margin(1e-12));
            if (i % 2) {
                kernel.reject(x);
                REQUIRE(x == before);
                REQUIRE(workspace.slacks == slacks);
            } else {
                kernel.accept(x, geometry, workspace);
                REQUIRE(x == y);
            }
            REQUIRE(workspace.slacks.isApprox(geometry.b() - geometry.A() * x, 1e-12));
        }
    };
    State initial(2);
    initial << 0.3, -0.2;
    run(hopsy::DikinWalkKernel<State>(initial, {}, {}, 0.25), false);
    run(hopsy::CSmMALAKernel<State>(initial, {}, {}, 0.25), true);
}

TEST_CASE("Asymmetric Gaussian support checks precede target evaluation", "[gaussian_family]") {
    struct Guarded {
        mutable int calls = 0;
        double log_density(const State &) const {
            ++calls;
            return 0.;
        }
        State log_gradient(const State &x) const {
            ++calls;
            return State::Zero(x.size());
        }
    } target;
    auto geometry = box(2, 1.);
    State x = State::Zero(2);
    auto workspace = geometry.create_workspace(x);
    hopsy::CSmMALAKernel<State> kernel(x, {}, {}, 1e6);
    hopsy::RandomNumberGenerator rng(27);
    kernel.propose(x, rng, target, geometry, workspace);
    REQUIRE_FALSE(geometry.is_feasible(kernel.proposal().state));
    const int forward_calls = target.calls;
    REQUIRE(kernel.log_acceptance_ratio(x, target, geometry, workspace) ==
            -std::numeric_limits<double>::infinity());
    REQUIRE(target.calls == forward_calls);
    kernel.reject(x);
    REQUIRE(x.isZero());
    REQUIRE(workspace.slacks == geometry.b());
}

TEST_CASE("Dikin and CSmMALA recover truncated Gaussian and uniform moments",
          "[gaussian_family][statistics]") {
    auto geometry = box(2, 2.);
    auto run = [&](auto kernel, const auto &target, double expected_second) {
        State x = State::Zero(2);
        auto workspace = geometry.create_workspace(x);
        hopsy::RandomNumberGenerator rng(831);
        for (int i = 0; i < 10000; ++i)
            hopsy::step(kernel, x, rng, target, geometry, workspace);
        auto samples = hopsy::draw(kernel, x, rng, target, geometry, workspace, 150000, 3);
        // Fixed-seed moment checks supplement exact density-ratio tests.
        REQUIRE(samples.cwiseAbs().maxCoeff() < 2.);
        REQUIRE(samples.rowwise().mean().cwiseAbs().maxCoeff() < 0.05);
        REQUIRE((samples.array().square().rowwise().mean() - expected_second).abs().maxCoeff() <
                0.06);
    };
    State initial = State::Zero(2);
    // E[X^2 | |X|<2] for N(0,1).
    const double second =
        1. - 4. * std::exp(-2.) / std::sqrt(2. * std::acos(-1.)) / std::erf(std::sqrt(2.));
    run(hopsy::DikinWalkKernel<State>(initial, {}, {}, 0.5), Normal{}, second);
    run(hopsy::CSmMALAKernel<State>(initial, {}, {}, 0.5), Normal{}, second);
    run(hopsy::DikinWalkKernel<State>(initial, {}, {}, 0.5), hopsy::UniformTarget{}, 4. / 3.);
    run(hopsy::CSmMALAKernel<State>(initial, {}, {}, 0.5), hopsy::UniformTarget{}, 4. / 3.);
}

TEST_CASE("Dikin scale updates reuse unscaled factors after accept and reject",
          "[gaussian_family]") {
    struct CountingMetric : hopsy::DikinMetric<double> {
        int *preparations;
        explicit CountingMetric(int &count) : preparations(&count) {}
        bool prepare(const State &x, const hopsy::LinearPolytope<double> &geometry,
                     const hopsy::EmptyWorkspace &workspace) {
            ++*preparations;
            return hopsy::DikinMetric<double>::prepare(x, geometry, workspace);
        }
    };
    auto geometry = skew_geometry();
    State x = State::Zero(2);
    hopsy::EmptyWorkspace workspace;
    int preparations = 0;
    using Kernel = hopsy::GaussianKernel<State, hopsy::ClippedGradientDrift, CountingMetric>;
    Kernel kernel(x, {}, CountingMetric(preparations), 0.1);
    hopsy::RandomNumberGenerator rng(19);
    Normal target;
    for (int i = 0; i < 4; ++i) {
        kernel.set_step_size(i % 2 ? 0.2 : 0.1);
        const int previous = preparations;
        kernel.propose(x, rng, target, geometry, workspace);
        REQUIRE(preparations == previous + (i == 0 ? 1 : 0));
        REQUIRE(std::isfinite(kernel.log_acceptance_ratio(x, target, geometry, workspace)));
        REQUIRE(preparations == previous + (i == 0 ? 2 : 1));
        if (i % 2)
            kernel.reject(x);
        else
            kernel.accept(x, geometry, workspace);
    }
}

TEST_CASE("Dikin-metric kernels support float and fixed-size state storage", "[gaussian_family]") {
    hopsy::Matrix<float> A(4, 2);
    A << 1, 0, -1, 0, 0, 1, 0, -1;
    hopsy::LinearPolytope<float> geometry(A, hopsy::Vector<float>::Constant(4, 2));
    Eigen::Vector2f x = Eigen::Vector2f::Zero();
    hopsy::CSmMALAKernel<Eigen::Vector2f> kernel(x, {}, {}, 0.1f);
    auto workspace = geometry.create_workspace(x);
    hopsy::RandomNumberGenerator rng(64);
    for (int i = 0; i < 10; ++i)
        hopsy::step(kernel, x, rng, hopsy::UniformTarget{}, geometry, workspace);
    REQUIRE(workspace.slacks.isApprox(geometry.b() - geometry.A() * x));
    REQUIRE(geometry.is_feasible(x));
}

TEST_CASE("Warmed Dikin transitions reuse Eigen buffers", "[gaussian_family][allocation]") {
    auto geometry = box(3, 3.);
    State x = State::Zero(3);
    auto workspace = geometry.create_workspace(x);
    hopsy::DikinWalkKernel<State> kernel(x, {}, {}, 0.1);
    hopsy::RandomNumberGenerator rng(89);
    hopsy::UniformTarget target;
    // Allocate both factors and their work buffers before enforcing the guard.
    for (int i = 0; i < 3; ++i)
        hopsy::step(kernel, x, rng, target, geometry, workspace);
    struct Guard {
        bool previous = Eigen::internal::is_malloc_allowed();
        Guard() { Eigen::internal::set_is_malloc_allowed(false); }
        ~Guard() { Eigen::internal::set_is_malloc_allowed(previous); }
    } guard;
    for (int i = 0; i < 10; ++i) {
        kernel.propose(x, rng, target, geometry, workspace);
        REQUIRE(std::isfinite(kernel.log_acceptance_ratio(x, target, geometry, workspace)));
        if (i % 2)
            kernel.accept(x, geometry, workspace);
        else
            kernel.reject(x);
    }
}

TEST_CASE("Native Gaussian target matches a correlated analytic density",
          "[gaussian_family][target]") {
    State mean(2);
    mean << 0.5, -0.25;
    Matrix covariance(2, 2);
    covariance << 2., 0.5, 0.5, 1.;
    hopsy::GaussianTarget target(mean, covariance);
    State x(2);
    x << 1., 0.75;
    State delta = x - mean;
    const double expected = -std::log(2. * std::acos(-1.)) -
                            0.5 * std::log(covariance.determinant()) -
                            0.5 * delta.dot(covariance.inverse() * delta);
    REQUIRE(target.log_density(x) == Approx(expected));
    REQUIRE(target.log_gradient(x).isApprox(-covariance.inverse() * delta));
    REQUIRE_THROWS_AS(target.log_density(State::Zero(3)), std::invalid_argument);
    REQUIRE_THROWS_AS((hopsy::GaussianTarget(mean, Matrix::Zero(2, 2))), std::invalid_argument);
    REQUIRE_THROWS_AS((hopsy::GaussianTarget(mean, Matrix::Identity(3, 3))), std::invalid_argument);
}

TEST_CASE("Uniform targets retain the asymmetric Gaussian correction", "[gaussian_family]") {
    auto geometry = skew_geometry();
    auto run = [&](auto kernel) {
        State x(2);
        x << 0.3, -0.2;
        hopsy::EmptyWorkspace workspace;
        hopsy::UniformTarget target;
        hopsy::RandomNumberGenerator rng(18);
        kernel.propose(x, rng, target, geometry, workspace);
        const State y = kernel.proposal().state;
        REQUIRE(geometry.is_feasible(y));
        const double expected =
            log_q(y, x, geometry, 0.2, false) - log_q(x, y, geometry, 0.2, false);
        REQUIRE(std::abs(expected) > 1e-10);
        REQUIRE(kernel.log_acceptance_ratio(x, target, geometry, workspace) ==
                Catch::Approx(expected).margin(1e-12));
        kernel.accept(x, geometry, workspace);
        REQUIRE(x == y);
    };
    State initial(2);
    initial << 0.3, -0.2;
    run(hopsy::DikinWalkKernel<State>(initial, {}, {}, 0.2));
    run(hopsy::CSmMALAKernel<State>(initial, {}, {}, 0.2));
}
