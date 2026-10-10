#include <catch2/catch_test_macros.hpp>
#include <hopsy/sampling/draw.h>

#include <Eigen/Core>
#include <limits>
#include <stdexcept>

namespace {
struct Target {};
struct Geometry {};
struct Workspace {};

struct IncrementKernel {
    template <typename> static constexpr bool always_accept = true;
    int proposals = 0;
    int acceptances = 0;

    void propose(const Eigen::VectorXd&, hopsy::RandomNumberGenerator&,
                 const Target&, const Geometry&, Workspace&) { ++proposals; }
    void accept(Eigen::VectorXd& state, const Target&, const Geometry&, Workspace&) {
        state.array() += 1;
        ++acceptances;
    }
};

struct AlternatingKernel : IncrementKernel {
    template <typename> static constexpr bool always_accept = false;
    int rejections = 0;
    double log_acceptance_ratio(const Eigen::VectorXd&, const Target&,
                                const Geometry&, Workspace&) {
        return proposals % 2 == 1 ? 0.0 : -std::numeric_limits<double>::infinity();
    }
    void reject(const Eigen::VectorXd&, const Target&, const Geometry&, Workspace&) {
        ++rejections;
    }
};

struct Fixture {
    IncrementKernel kernel;
    Eigen::VectorXd state = Eigen::VectorXd::Zero(2);
    hopsy::RandomNumberGenerator rng{42};
    Target target;
    Geometry geometry;
    Workspace workspace;
};
} // namespace

TEST_CASE_METHOD(Fixture, "draw allocates columns and honors thinning", "[sampling][draw]") {
    const auto samples = hopsy::draw(kernel, state, rng, target, geometry, workspace, 3, 2);
    REQUIRE(samples.rows() == 2);
    REQUIRE(samples.cols() == 3);
    for (Eigen::Index col = 0; col < samples.cols(); ++col) {
        CHECK((samples.col(col).array() == 2.0 * (col + 1)).all());
    }
    CHECK((state.array() == 6).all());
    CHECK(kernel.proposals == 6);
    CHECK(kernel.acceptances == 6);
}

TEST_CASE_METHOD(Fixture, "draw_into reuses an existing matrix", "[sampling][draw]") {
    Eigen::MatrixXd samples = Eigen::MatrixXd::Constant(2, 3, -1);
    const auto* storage = samples.data();
    hopsy::draw_into(kernel, state, rng, target, geometry, workspace, samples);
    CHECK(samples.data() == storage);
    for (Eigen::Index col = 0; col < samples.cols(); ++col) {
        CHECK((samples.col(col).array() == double(col + 1)).all());
    }
}

TEST_CASE_METHOD(Fixture, "draw_into supports batches in matrix blocks", "[sampling][draw]") {
    Eigen::MatrixXd samples = Eigen::MatrixXd::Constant(2, 6, -1);
    auto first = samples.middleCols(1, 2);
    hopsy::draw_into(kernel, state, rng, target, geometry, workspace, first);
    auto second = samples.middleCols(3, 2);
    hopsy::draw_into(kernel, state, rng, target, geometry, workspace, second);

    CHECK((samples.col(0).array() == -1).all());
    CHECK((samples.col(5).array() == -1).all());
    for (Eigen::Index col = 1; col <= 4; ++col) {
        CHECK((samples.col(col).array() == double(col)).all());
    }
    CHECK((state.array() == 4).all());
    CHECK(kernel.proposals == 4);
}

TEST_CASE_METHOD(Fixture, "draw records repeated states after rejection", "[sampling][draw]") {
    AlternatingKernel alternating;
    const auto samples = hopsy::draw(alternating, state, rng, target, geometry, workspace, 4);
    REQUIRE(samples.cols() == 4);
    const double expected[] = {1, 1, 2, 2};
    for (Eigen::Index col = 0; col < 4; ++col) {
        CHECK((samples.col(col).array() == expected[col]).all());
    }
    CHECK(alternating.acceptances == 2);
    CHECK(alternating.rejections == 2);
    CHECK((state.array() == 2).all());
}

TEST_CASE_METHOD(Fixture, "drawing zero samples leaves the chain unchanged", "[sampling][draw]") {
    SECTION("allocating draw") {
        const auto samples = hopsy::draw(kernel, state, rng, target, geometry, workspace, 0);
        CHECK(samples.rows() == 2);
        CHECK(samples.cols() == 0);
    }
    SECTION("preallocated draw") {
        Eigen::MatrixXd samples(2, 0);
        hopsy::draw_into(kernel, state, rng, target, geometry, workspace, samples);
    }
    CHECK(kernel.proposals == 0);
    CHECK((state.array() == 0).all());
}

TEST_CASE_METHOD(Fixture, "draw validates arguments before taking steps", "[sampling][draw]") {
    SECTION("negative sample count") {
        CHECK_THROWS_AS(hopsy::draw(kernel, state, rng, target, geometry, workspace, -1),
                        std::invalid_argument);
    }
    SECTION("zero thinning") {
        CHECK_THROWS_AS(hopsy::draw(kernel, state, rng, target, geometry, workspace, 3, 0),
                        std::invalid_argument);
    }
    SECTION("negative thinning") {
        Eigen::MatrixXd samples(2, 3);
        CHECK_THROWS_AS(hopsy::draw_into(kernel, state, rng, target, geometry, workspace, samples, -1),
                        std::invalid_argument);
    }
    SECTION("wrong output dimension") {
        Eigen::MatrixXd samples(3, 2);
        CHECK_THROWS_AS(hopsy::draw_into(kernel, state, rng, target, geometry, workspace, samples),
                        std::invalid_argument);
    }
    CHECK(kernel.proposals == 0);
    CHECK((state.array() == 0).all());
}

TEST_CASE_METHOD(Fixture, "draw thinning counts rejected attempts too", "[sampling][draw]") {
    AlternatingKernel alternating;
    const auto samples = hopsy::draw(
        alternating, state, rng, target, geometry, workspace, 3, 2
    );
    REQUIRE(samples.cols() == 3);
    for (Eigen::Index col = 0; col < samples.cols(); ++col) {
        CHECK((samples.col(col).array() == double(col + 1)).all());
    }
    CHECK(alternating.proposals == 6);
    CHECK(alternating.acceptances == 3);
    CHECK(alternating.rejections == 3);
}

TEST_CASE_METHOD(Fixture, "draw_into writes into mapped storage with padding", "[sampling][draw]") {
    double storage[15];
    for (auto &value : storage) value = -1.;
    using Map = Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned, Eigen::OuterStride<>>;
    Map samples(storage, 2, 3, Eigen::OuterStride<>(5));
    hopsy::draw_into(kernel, state, rng, target, geometry, workspace, samples);
    for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < 5; ++row) {
            CHECK(storage[5*col + row] == (row < 2 ? double(col + 1) : -1.));
        }
    }
    CHECK(kernel.proposals == 3);
    CHECK((state.array() == 3).all());
}
