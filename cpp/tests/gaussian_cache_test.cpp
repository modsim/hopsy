#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/sampling/draw.h>
#include <hopsy/target/uniform.h>

namespace {
    using State = hopsy::Vector<double>;
    using Geometry = hopsy::LinearPolytope<double>;
    using Kernel = hopsy::CSmMALAKernel<State>;
    Geometry box() {
        hopsy::Matrix<double> A(4, 2);
        A << 1, 0, -1, 0, 0, 1, 0, -1;
        return {A, State::Constant(4, 3.)};
    }
    struct Target {
        mutable int densities = 0, gradients = 0;
        double mean = 0.;
        double upper_support = std::numeric_limits<double>::infinity();
        bool fail_density = false, fail_gradient = false;
        double log_density(const State &x) const {
            ++densities;
            if (fail_density)
                throw std::runtime_error("density failure");
            if (x(0) > upper_support)
                return -std::numeric_limits<double>::infinity();
            return -0.5 * (x.array() - mean).matrix().squaredNorm();
        }
        State log_gradient(const State &x) const {
            ++gradients;
            if (fail_gradient || x(0) > upper_support)
                throw std::runtime_error("gradient outside support or failure");
            return (mean - x.array()).matrix();
        }
    };
    template <typename K>
    concept StoresCorrection = requires(const K &k) { k.proposal().log_proposal_ratio; };
} // namespace

TEST_CASE("Gaussian proposals store state and scale rather than a pending correction",
          "[gaussian_cache]") {
    STATIC_REQUIRE_FALSE(StoresCorrection<Kernel>);
    STATIC_REQUIRE_FALSE(
        StoresCorrection<hopsy::GaussianKernel<State, hopsy::ZeroDrift, hopsy::IdentityMetric>>);
}

TEST_CASE("CSmMALA caches accepted data through rejection and scale changes", "[gaussian_cache]") {
    auto geometry = box();
    State x = State::Constant(2, 0.2);
    hopsy::EmptyWorkspace workspace;
    Kernel kernel(x, {}, {}, 0.1);
    Target target;
    hopsy::RandomNumberGenerator rng(519);
    for (int i = 0; i < 8; ++i) {
        kernel.set_step_size(i % 2 ? 0.05 : 0.15);
        const int old_g = target.gradients, old_d = target.densities;
        kernel.propose(x, rng, target, geometry, workspace);
        REQUIRE(target.gradients == old_g + (i == 0));
        REQUIRE(target.densities == old_d);
        const double ratio = kernel.log_acceptance_ratio(x, target, geometry, workspace);
        REQUIRE(std::isfinite(ratio));
        REQUIRE(target.gradients == old_g + 1 + (i == 0));
        REQUIRE(target.densities == old_d + 1 + (i == 0));
        REQUIRE(kernel.log_acceptance_ratio(x, target, geometry, workspace) == ratio);
        REQUIRE(target.gradients == old_g + 1 + (i == 0));
        REQUIRE(target.densities == old_d + 1 + (i == 0));
        if (i % 2)
            kernel.reject(x);
        else
            kernel.accept(x, geometry, workspace);
    }
}

TEST_CASE("Explicit target invalidation preserves geometry factors", "[gaussian_cache]") {
    auto geometry = box();
    State x = State::Constant(2, 0.2);
    hopsy::EmptyWorkspace workspace;
    Kernel kernel(x, {}, {}, 0.15);
    Target target;
    hopsy::RandomNumberGenerator rng(76);
    kernel.propose(x, rng, target, geometry, workspace);
    REQUIRE(std::isfinite(kernel.log_acceptance_ratio(x, target, geometry, workspace)));
    kernel.accept(x, geometry, workspace);

    auto compare_fresh = [&](Target &density) {
        Kernel fresh(x, {}, {}, kernel.step_size());
        hopsy::RandomNumberGenerator a(51), b(51);
        kernel.propose(x, a, density, geometry, workspace);
        fresh.propose(x, b, density, geometry, workspace);
        REQUIRE(kernel.proposal().state.isApprox(fresh.proposal().state, 1e-13));
        REQUIRE(kernel.log_acceptance_ratio(x, density, geometry, workspace) ==
                Catch::Approx(fresh.log_acceptance_ratio(x, density, geometry, workspace))
                    .margin(1e-12));
        kernel.reject(x);
        fresh.reject(x);
    };
    // The low-level interface assumes a fixed target. A new bound context
    // performs this invalidation once; raw calls must do it explicitly.
    target.mean = 0.8;
    kernel.invalidate_target_cache();
    compare_fresh(target);
    Target replacement;
    replacement.mean = -0.7;
    kernel.invalidate_target_cache();
    compare_fresh(replacement);
}

TEST_CASE("Batch sampling refreshes an in-place changed target", "[gaussian_cache]") {
    auto geometry = box();
    State x = State::Constant(2, 0.1);
    hopsy::EmptyWorkspace workspace;
    Kernel kernel(x, {}, {}, 0.15);
    Target target;
    hopsy::RandomNumberGenerator rng(743);
    const auto initial_samples = hopsy::draw(kernel, x, rng, target, geometry, workspace, 10);
    REQUIRE(initial_samples.allFinite());
    target.mean = 0.9;
    State reference_state = x;
    auto reference_rng = rng;
    Kernel reference(x, {}, {}, 0.15);
    const auto expected =
        hopsy::draw(reference, reference_state, reference_rng, target, geometry, workspace, 50);
    const auto actual = hopsy::draw(kernel, x, rng, target, geometry, workspace, 50);
    REQUIRE(actual.isApprox(expected, 1e-13));
}

TEST_CASE("Gaussian pending lifecycle prevents stale commits and mid-transition changes",
          "[gaussian_cache]") {
    auto geometry = box();
    State x = State::Zero(2);
    hopsy::EmptyWorkspace workspace;
    Kernel kernel(x, {}, {}, 0.1);
    Target target;
    hopsy::RandomNumberGenerator rng(619);
    REQUIRE_FALSE(kernel.has_pending_proposal());
    REQUIRE_THROWS_AS(kernel.log_acceptance_ratio(x, target, geometry, workspace),
                      std::logic_error);
    REQUIRE_THROWS_AS(kernel.accept(x, geometry, workspace), std::logic_error);
    kernel.propose(x, rng, target, geometry, workspace);
    REQUIRE(kernel.has_pending_proposal());
    REQUIRE_THROWS_AS(kernel.propose(x, rng, target, geometry, workspace), std::logic_error);
    REQUIRE_THROWS_AS(kernel.set_step_size(0.2), std::logic_error);
    REQUIRE(kernel.step_size() == 0.1);
    REQUIRE_THROWS_AS(kernel.invalidate_target_cache(), std::logic_error);
    REQUIRE_THROWS_AS(kernel.accept(x, geometry, workspace), std::logic_error);
    REQUIRE(std::isfinite(kernel.log_acceptance_ratio(x, target, geometry, workspace)));
    kernel.accept(x, geometry, workspace);
    REQUIRE_FALSE(kernel.has_pending_proposal());
    REQUIRE_THROWS_AS(kernel.accept(x, geometry, workspace), std::logic_error);
    kernel.reject(x); // Cancellation is idempotent.
    kernel.invalidate_target_cache();
}

TEST_CASE("Target zero mass rejects before reverse gradients and metrics", "[gaussian_cache]") {
    struct Metric : hopsy::DikinMetric<double> {
        int *calls;
        explicit Metric(int &n) : calls(&n) {}
        bool prepare(const State &x, const Geometry &g, const hopsy::EmptyWorkspace &w) {
            ++*calls;
            return hopsy::DikinMetric<double>::prepare(x, g, w);
        }
    };
    auto geometry = box();
    State x = State::Zero(2);
    hopsy::EmptyWorkspace workspace;
    int preparations = 0;
    hopsy::GaussianKernel<State, hopsy::ClippedGradientDrift, Metric> kernel(
        x, {}, Metric(preparations), 0.3);
    Target target;
    target.upper_support = 0.;
    hopsy::RandomNumberGenerator rng(55);
    bool found = false;
    for (int i = 0; i < 100; ++i) {
        kernel.propose(x, rng, target, geometry, workspace);
        if (kernel.proposal().state(0) > 0. && geometry.is_feasible(kernel.proposal().state)) {
            found = true;
            break;
        }
        kernel.reject(x);
    }
    REQUIRE(found);
    REQUIRE(preparations == 1);
    REQUIRE(target.gradients == 1);
    REQUIRE(kernel.log_acceptance_ratio(x, target, geometry, workspace) ==
            -std::numeric_limits<double>::infinity());
    REQUIRE(preparations == 1);
    REQUIRE(target.gradients == 1);
    REQUIRE_THROWS_AS(kernel.accept(x, geometry, workspace), std::logic_error);
    REQUIRE(x.isZero());
    kernel.reject(x);
    REQUIRE_FALSE(kernel.has_pending_proposal());
}

TEST_CASE("Gaussian target exceptions leave cancellable or idle proposals", "[gaussian_cache]") {
    auto geometry = box();
    State x = State::Zero(2);
    hopsy::EmptyWorkspace workspace;
    Kernel kernel(x, {}, {}, 0.1);
    Target target;
    target.fail_gradient = true;
    hopsy::RandomNumberGenerator rng(63);
    REQUIRE_THROWS_AS(kernel.propose(x, rng, target, geometry, workspace), std::runtime_error);
    REQUIRE_FALSE(kernel.has_pending_proposal());
    target.fail_gradient = false;
    target.fail_density = true;
    kernel.propose(x, rng, target, geometry, workspace);
    REQUIRE_THROWS_AS(kernel.log_acceptance_ratio(x, target, geometry, workspace),
                      std::runtime_error);
    REQUIRE(kernel.has_pending_proposal());
    REQUIRE(x.isZero());
    kernel.reject(x);
    target.fail_density = false;
    kernel.invalidate_target_cache();
    kernel.propose(x, rng, target, geometry, workspace);
    REQUIRE(std::isfinite(kernel.log_acceptance_ratio(x, target, geometry, workspace)));
    kernel.accept(x, geometry, workspace);
}

TEST_CASE("Nonquadratic drift policies recompute at each new scale", "[gaussian_cache]") {
    struct LinearDrift {
        void compute(const State &x, double sigma, const Target &t, const hopsy::IdentityMetric &,
                     const Geometry &, const hopsy::EmptyWorkspace &, State &out) const {
            out = sigma * t.log_gradient(x);
        }
    };
    auto geometry = box();
    State x = State::Constant(2, 0.2);
    hopsy::EmptyWorkspace workspace;
    Target target;
    hopsy::GaussianKernel<State, LinearDrift, hopsy::IdentityMetric> kernel(x, {}, {}, 0.1);
    hopsy::RandomNumberGenerator rng(917);
    for (double sigma : {0.1, 0.2, 0.05}) {
        kernel.set_step_size(sigma);
        const int before = target.gradients;
        kernel.propose(x, rng, target, geometry, workspace);
        REQUIRE(target.gradients == before + 1);
        const State y = kernel.proposal().state;
        const double correction =
            ((y - (1. - sigma) * x).squaredNorm() - (x - (1. - sigma) * y).squaredNorm()) /
            (2. * sigma * sigma);
        const double expected = -0.5 * y.squaredNorm() + 0.5 * x.squaredNorm() + correction;
        REQUIRE(kernel.log_acceptance_ratio(x, target, geometry, workspace) ==
                Catch::Approx(expected).margin(1e-12));
        REQUIRE(target.gradients == before + 2);
        kernel.accept(x, geometry, workspace);
    }
}

TEST_CASE("Copied evaluated proposals own independent caches and buffers", "[gaussian_cache]") {
    auto geometry = box();
    State x = State::Constant(2, 0.2), copy_state = x;
    hopsy::EmptyWorkspace workspace;
    Kernel kernel(x, {}, {}, 0.1);
    Target target;
    hopsy::RandomNumberGenerator rng(137);
    kernel.propose(x, rng, target, geometry, workspace);
    const double ratio = kernel.log_acceptance_ratio(x, target, geometry, workspace);
    REQUIRE(std::isfinite(ratio));
    Kernel copy = kernel;
    REQUIRE(copy.proposal().state.data() != kernel.proposal().state.data());
    copy.accept(copy_state, geometry, workspace);
    REQUIRE(kernel.has_pending_proposal());
    REQUIRE(kernel.log_acceptance_ratio(x, target, geometry, workspace) == ratio);
    kernel.accept(x, geometry, workspace);
    REQUIRE(x == copy_state);
    copy.invalidate_target_cache();
    copy.set_step_size(0.2);
    REQUIRE(kernel.step_size() == 0.1);
    const int before = target.gradients;
    kernel.propose(x, rng, target, geometry, workspace);
    REQUIRE(target.gradients == before);
    kernel.reject(x);
}
