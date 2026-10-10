#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/gaussian_line_kernel.h>
#include <hopsy/kernel/overrelaxed_line_kernel.h>
#include <hopsy/sampling/draw.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/target/uniform.h>

namespace {
using Vector = hopsy::Vector<double>;
hopsy::LinearPolytope<double> box(hopsy::Index n) {
    hopsy::Matrix<double> A(2*n, n);
    A.topRows(n).setIdentity();
    A.bottomRows(n) = -hopsy::Matrix<double>::Identity(n, n);
    return {A, Vector::Ones(2*n)};
}
}

TEMPLATE_TEST_CASE("Hit-and-run keeps a reusable candidate and preserves accepted state",
                   "[hit_and_run]", hopsy::RandomDirectionGenerator<double>,
                   hopsy::CyclingCoordinateDirectionGenerator,
                   hopsy::RandomCoordinateDirectionGenerator) {
    Vector state = Vector::Zero(3);
    const auto geometry = box(3);
    auto workspace = geometry.create_workspace(state);
    hopsy::HitAndRunKernel kernel(state, TestType(3), hopsy::UniformLineKernel{});
    hopsy::RandomNumberGenerator rng(42);
    // Proposal generation must compile without any target operations.
    struct UnusedTarget {};
    const auto *storage = kernel.proposal().state.data();
    for (int i = 0; i < 100; ++i) {
        kernel.propose(state, rng, UnusedTarget{}, geometry, workspace);
        const auto &proposal = kernel.proposal();
        CHECK(proposal.state.data() == storage);
        CHECK(geometry.contains(proposal.state, -1e-14));
        CHECK(state.isZero());
        CHECK(workspace.slacks == Vector::Ones(6));
        CHECK(proposal.log_proposal_ratio == 0.);
        CHECK(kernel.log_acceptance_ratio(state, hopsy::UniformTarget{}, geometry, workspace) == 0.);
        kernel.reject(state, UnusedTarget{}, geometry, workspace);
    }
}

TEMPLATE_TEST_CASE("Hit-and-run commits accepted displacements without workspace drift",
                   "[hit_and_run]", hopsy::RandomDirectionGenerator<double>,
                   hopsy::CyclingCoordinateDirectionGenerator,
                   hopsy::RandomCoordinateDirectionGenerator) {
    Vector state = Vector::Zero(5);
    const auto geometry = box(5);
    auto workspace = geometry.create_workspace(state);
    hopsy::UniformTarget target;
    hopsy::HitAndRunKernel kernel(state, TestType(5), hopsy::UniformLineKernel{});
    hopsy::RandomNumberGenerator rng(53);
    const auto *buffer_a = state.data();
    const auto *buffer_b = kernel.proposal().state.data();
    for (int i = 0; i < 10000; ++i) {
        const Vector before = state;
        const Vector slacks = workspace.slacks;
        kernel.propose(state, rng, target, geometry, workspace);
        const Vector candidate = kernel.proposal().state;
        CHECK(state == before);
        CHECK(workspace.slacks == slacks);
        if (i % 3 == 0) {
            kernel.reject(state, target, geometry, workspace);
            CHECK(state == before);
            CHECK(workspace.slacks == slacks);
        } else {
            kernel.accept(state, target, geometry, workspace);
            CHECK(state == candidate);
        }
        CHECK((workspace.slacks - (geometry.b() - geometry.A()*state)).cwiseAbs().maxCoeff() < 1e-11);
        CHECK(geometry.contains(state, -1e-11));
        CHECK((state.data() == buffer_a || state.data() == buffer_b));
        CHECK((kernel.proposal().state.data() == buffer_a || kernel.proposal().state.data() == buffer_b));
        CHECK(state.data() != kernel.proposal().state.data());
    }
}

TEMPLATE_TEST_CASE("Hit-and-run supports uncached geometry workspaces",
                   "[hit_and_run]", hopsy::RandomDirectionGenerator<double>,
                   hopsy::CyclingCoordinateDirectionGenerator) {
    Vector state = Vector::Zero(2);
    const auto geometry = box(2);
    hopsy::EmptyWorkspace workspace;
    hopsy::UniformTarget target;
    hopsy::HitAndRunKernel kernel(state, TestType(2), hopsy::UniformLineKernel{});
    hopsy::RandomNumberGenerator rng(54);
    for (int i = 0; i < 200; ++i) {
        kernel.propose(state, rng, target, geometry, workspace);
        kernel.accept(state, target, geometry, workspace);
        CHECK(geometry.contains(state, -1e-13));
    }
}


namespace {
struct TiltedTarget {
    double log_density(const Vector &state) const { return state(0); }
};
}

TEST_CASE("Hit-and-run adds the correctly signed Gaussian Hastings correction", "[hit_and_run]") {
    Vector state(1);
    state << -0.8;
    const auto geometry = box(1);
    auto workspace = geometry.create_workspace(state);
    hopsy::GaussianLineKernel line(0.7);
    hopsy::HitAndRunKernel kernel(state, hopsy::CyclingCoordinateDirectionGenerator(1), line);
    STATIC_REQUIRE_FALSE(decltype(kernel)::always_accept<hopsy::UniformTarget>);
    hopsy::RandomNumberGenerator rng(55);
    kernel.propose(state, rng, TiltedTarget{}, geometry, workspace);
    const auto &proposal = kernel.proposal();
    const auto cdf = [](double x) { return 0.5 * std::erfc(-x/std::sqrt(2.)); };
    const auto z = [&](double x) { return cdf((1-x)/0.7) - cdf((-1-x)/0.7); };
    const double expected = std::log(z(state(0))) - std::log(z(proposal.state(0)));
    CHECK(proposal.log_proposal_ratio == Catch::Approx(expected).margin(1e-13));
    CHECK(kernel.log_acceptance_ratio(state, TiltedTarget{}, geometry, workspace) ==
          Catch::Approx(proposal.state(0)-state(0)+expected).margin(1e-13));
    const std::array<hopsy::LineInterval<double>, 1> reverse{{{-1-proposal.state(0), 1-proposal.state(0)}}};
    CHECK(line.log_proposal_ratio(reverse, -proposal.step) == Catch::Approx(-expected).margin(1e-13));
}

TEMPLATE_TEST_CASE("Hit-and-run recovers uniform and exponential box moments",
                   "[hit_and_run][distribution]", hopsy::UniformLineKernel,
                   hopsy::GaussianLineKernel, hopsy::OverRelaxedLineKernel) {
    // p(x0) proportional to exp(x0) on [-1,1]; x1 remains independent uniform.
    // Broad moment tolerances allow Monte Carlo correlation and platform RNG differences.
    const auto run = []<typename Generator, typename Target>(Generator generator, Target target) {
        Vector state = Vector::Zero(2);
        const auto geometry = box(2);
        auto workspace = geometry.create_workspace(state);
        hopsy::HitAndRunKernel kernel(state, std::move(generator), TestType{});
        hopsy::RandomNumberGenerator rng(56);
        for (int i = 0; i < 1000; ++i) hopsy::step(kernel, state, rng, target, geometry, workspace);
        const auto samples = hopsy::draw(kernel, state, rng, target, geometry, workspace, 40000, 2);
        const bool uniform = hopsy::target_is_uniform_v<Target>;
        const double expected_mean = uniform ? 0. : 1./std::tanh(1.)-1.;
        const double expected_second = uniform ? 1./3. : 3.-2./std::tanh(1.);
        CHECK(samples.row(0).mean() == Catch::Approx(expected_mean).margin(0.025));
        CHECK(samples.row(0).array().square().mean() == Catch::Approx(expected_second).margin(0.025));
        CHECK(samples.row(1).mean() == Catch::Approx(0).margin(0.025));
        CHECK(samples.row(1).array().square().mean() == Catch::Approx(1./3.).margin(0.025));
        CHECK((samples.row(0).array()*samples.row(1).array()).mean() == Catch::Approx(0).margin(0.025));
        CHECK(samples.cwiseAbs().maxCoeff() <= 1.+1e-11);
    };
    SECTION("General uniform") { run(hopsy::RandomDirectionGenerator<double>(2), hopsy::UniformTarget{}); }
    SECTION("Coordinate uniform") { run(hopsy::CyclingCoordinateDirectionGenerator(2), hopsy::UniformTarget{}); }
    SECTION("General tilted") { run(hopsy::RandomDirectionGenerator<double>(2), TiltedTarget{}); }
    SECTION("Coordinate tilted") { run(hopsy::CyclingCoordinateDirectionGenerator(2), TiltedTarget{}); }
}

TEST_CASE("Hit-and-run skips the acceptance draw for uniform chord sampling", "[hit_and_run]") {
    Vector state = Vector::Zero(3), reference = state;
    const auto geometry = box(3);
    auto workspace = geometry.create_workspace(state);
    auto reference_workspace = workspace;
    hopsy::HitAndRunKernel kernel(state, hopsy::RandomDirectionGenerator<double>(3), hopsy::UniformLineKernel{});
    auto reference_kernel = kernel;
    STATIC_REQUIRE(decltype(kernel)::always_accept<hopsy::UniformTarget>);
    STATIC_REQUIRE_FALSE(decltype(kernel)::always_accept<TiltedTarget>);
    hopsy::UniformTarget target;
    hopsy::RandomNumberGenerator rng(57), reference_rng = rng;
    for (int i = 0; i < 100; ++i) {
        CHECK(hopsy::step(kernel, state, rng, target, geometry, workspace));
        reference_kernel.propose(reference, reference_rng, target, geometry, reference_workspace);
        reference_kernel.accept(reference, target, geometry, reference_workspace);
        CHECK(state == reference);
        CHECK(rng == reference_rng);
    }
}

TEMPLATE_TEST_CASE("Cached hit-and-run performs no Eigen heap allocation in the hot loop",
                   "[hit_and_run][allocation]", hopsy::UniformLineKernel,
                   hopsy::GaussianLineKernel, hopsy::OverRelaxedLineKernel) {
    const auto run = []<typename Generator>(Generator generator) {
        Vector state = Vector::Zero(20);
        const auto geometry = box(20);
        auto workspace = geometry.create_workspace(state);
        hopsy::HitAndRunKernel kernel(state, std::move(generator), TestType{});
        hopsy::RandomNumberGenerator rng(58);
        const TiltedTarget target;
        hopsy::step(kernel, state, rng, target, geometry, workspace);
        struct NoMalloc {
            NoMalloc() { Eigen::internal::set_is_malloc_allowed(false); }
            ~NoMalloc() { Eigen::internal::set_is_malloc_allowed(true); }
        };
        {
            NoMalloc guard;
            for (int i = 0; i < 2000; ++i)
                hopsy::step(kernel, state, rng, target, geometry, workspace);
        }
        CHECK(geometry.contains(state, -1e-11));
    };
    SECTION("General") { run(hopsy::RandomDirectionGenerator<double>(20)); }
    SECTION("Coordinate") { run(hopsy::CyclingCoordinateDirectionGenerator(20)); }
}

TEST_CASE("A copied pending kernel owns an independent direction and proposal", "[hit_and_run]") {
    Vector state = Vector::Zero(3);
    const auto geometry = box(3);
    auto workspace = geometry.create_workspace(state);
    hopsy::HitAndRunKernel kernel(state, hopsy::RandomDirectionGenerator<double>(3), hopsy::UniformLineKernel{});
    hopsy::RandomNumberGenerator rng(59);
    hopsy::UniformTarget target;
    kernel.propose(state, rng, target, geometry, workspace);
    auto copied = kernel;
    auto copied_workspace = workspace;
    Vector copied_state = state;
    const Vector candidate = copied.proposal().state;
    kernel.reject(state, target, geometry, workspace);
    kernel.propose(state, rng, target, geometry, workspace);
    copied.accept(copied_state, target, geometry, copied_workspace);
    CHECK(copied_state == candidate);
    CHECK(copied_workspace.slacks.isApprox(geometry.create_workspace(copied_state).slacks, 1e-13));
}

TEMPLATE_TEST_CASE("Hit-and-run supports float and double states with double MH arithmetic",
                   "[hit_and_run]", float, double) {
    using State = hopsy::Vector<TestType>;
    State state = State::Zero(2);
    hopsy::Matrix<TestType> A(4, 2);
    A << 1, 0, -1, 0, 0, 1, 0, -1;
    hopsy::LinearPolytope<TestType> geometry(A, State::Ones(4));
    auto workspace = geometry.create_workspace(state);
    hopsy::HitAndRunKernel kernel(state, hopsy::RandomDirectionGenerator<TestType>(2),
                                 hopsy::GaussianLineKernel{});
    hopsy::UniformTarget target;
    hopsy::RandomNumberGenerator rng(60);
    STATIC_REQUIRE(std::is_same_v<decltype(kernel.log_acceptance_ratio(state, target, geometry, workspace)), double>);
    for (int i = 0; i < 500; ++i) hopsy::step(kernel, state, rng, target, geometry, workspace);
    CHECK(geometry.contains(state, -TestType{100}*std::numeric_limits<TestType>::epsilon()));
}
