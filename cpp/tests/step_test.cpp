#include <catch2/catch_test_macros.hpp>
#include <hopsy/sampling/step.h>

#include <limits>
#include <stdexcept>

namespace {
struct Target {};
struct Geometry {};
struct Workspace {};

// A non-Eigen state verifies that step does not inspect position storage.
struct State { int position = 0; };

struct AlwaysAcceptKernel {
    template <typename> static constexpr bool always_accept = true;
    int proposals = 0;
    int acceptances = 0;
    int candidate = 0;

    void propose(const State& state, hopsy::RandomNumberGenerator&,
                 const Target&, const Geometry&, Workspace&) {
        ++proposals;
        candidate = state.position + 1;
    }
    void accept(State& state, const Target&, const Geometry&, Workspace&) {
        ++acceptances;
        state.position = candidate;
    }
    // Deliberately no log_acceptance_ratio or reject: the discarded
    // if constexpr branch must not require either operation.
};

struct MHKernel : AlwaysAcceptKernel {
    template <typename> static constexpr bool always_accept = false;
    double ratio = 0;
    int evaluations = 0;
    int rejections = 0;

    double log_acceptance_ratio(const State&, const Target&,
                                const Geometry&, Workspace&) {
        ++evaluations;
        return ratio;
    }
    void reject(const State&, const Target&, const Geometry&, Workspace&) {
        ++rejections;
    }
};
} // namespace

TEST_CASE("step supports an always-accept kernel without MH methods", "[sampling][step]") {
    AlwaysAcceptKernel kernel;
    State state;
    Target target;
    Geometry geometry;
    Workspace workspace;
    hopsy::RandomNumberGenerator rng(42);
    auto untouched_rng = rng;

    CHECK(hopsy::step(kernel, state, rng, target, geometry, workspace));
    CHECK(state.position == 1);
    CHECK(kernel.proposals == 1);
    CHECK(kernel.acceptances == 1);
    CHECK(rng == untouched_rng);
}

TEST_CASE("step handles deterministic MH outcomes", "[sampling][step]") {
    MHKernel kernel;
    State state{7};
    Target target;
    Geometry geometry;
    Workspace workspace;
    hopsy::RandomNumberGenerator rng(42);
    auto untouched_rng = rng;
    bool expected_acceptance = true;

    SECTION("zero ratio") { kernel.ratio = 0; }
    SECTION("positive ratio") { kernel.ratio = 2; }
    SECTION("positive infinity") {
        kernel.ratio = std::numeric_limits<double>::infinity();
    }
    SECTION("negative infinity") {
        kernel.ratio = -std::numeric_limits<double>::infinity();
        expected_acceptance = false;
    }

    CHECK(hopsy::step(kernel, state, rng, target, geometry, workspace)
          == expected_acceptance);
    CHECK(state.position == (expected_acceptance ? 8 : 7));
    CHECK(kernel.proposals == 1);
    CHECK(kernel.evaluations == 1);
    CHECK(kernel.acceptances == (expected_acceptance ? 1 : 0));
    CHECK(kernel.rejections == (expected_acceptance ? 0 : 1));
    CHECK(rng == untouched_rng);
}

TEST_CASE("step rejects a pending proposal when log_ratio is NaN", "[sampling][step]") {
    MHKernel kernel;
    kernel.ratio = std::numeric_limits<double>::quiet_NaN();
    State state{7};
    Target target;
    Geometry geometry;
    Workspace workspace;
    hopsy::RandomNumberGenerator rng(42);

    const auto untouched_rng = rng;
    hopsy::step(kernel, state, rng, target, geometry, workspace);
    CHECK(rng == untouched_rng);
    CHECK(state.position == 7);
    CHECK(kernel.proposals == 1);
    CHECK(kernel.evaluations == 1);
    CHECK(kernel.acceptances == 0);
    CHECK(kernel.rejections == 1);
}

TEST_CASE("step applies a finite negative log ratio probabilistically", "[sampling][step]") {
    MHKernel kernel;
    kernel.ratio = -0.69314718055994530942; // log(1/2)
    State state;
    Target target;
    Geometry geometry;
    Workspace workspace;
    hopsy::RandomNumberGenerator rng(42);

    constexpr int attempts = 4096;
    for (int i = 0; i < attempts; ++i) {
        const int previous = state.position;
        const bool accepted = hopsy::step(kernel, state, rng, target, geometry, workspace);
        REQUIRE(state.position == previous + (accepted ? 1 : 0));
    }
    // Broad bounds avoid dependence on a particular standard library's
    // uniform_real_distribution implementation or exact RNG sequence.
    CHECK(kernel.acceptances > 1600);
    CHECK(kernel.acceptances < 2500);
    CHECK(kernel.acceptances + kernel.rejections == attempts);
    CHECK(kernel.evaluations == attempts);
}
