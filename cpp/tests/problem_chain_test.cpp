#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <hopsy/sampling/chain.h>
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/target/gaussian.h>
#include <hopsy/target/uniform.h>
#include <array>

namespace {
    using State = hopsy::Vector<double>;
    using Matrix = hopsy::Matrix<double>;
    using Catch::Approx;
    auto box(int n, double radius = 3.) {
        Matrix A(2 * n, n);
        A.topRows(n).setIdentity();
        A.bottomRows(n) = -A.topRows(n);
        return hopsy::LinearPolytope<double>(A, State::Constant(2 * n, radius));
    }
    struct Normal {
        double log_density(const State &x) const { return -0.5 * x.squaredNorm(); }
        auto log_gradient(const State &x) const { return -x; }
    };
    template <typename Chain>
    concept TakesTargetPerStep = requires(Chain &c, hopsy::RandomNumberGenerator &rng,
                                          Normal &target) { c.step(rng, target); };
    template <typename P, typename K>
    concept BindsTemporary =
        requires(P &&p, K &k, State &s) { hopsy::bind_chain(std::move(p), k, s); };

    // A geometry deliberately without A/b, with two disconnected line intervals.
    // This checks the capability boundary needed by future nonconvex geometries.
    struct Disconnected {
        hopsy::Index num_dimension() const { return 1; }
        hopsy::EmptyWorkspace create_workspace(const State &) const { return {}; }
        bool is_feasible(const State &x) const {
            return (x(0) >= -2. && x(0) <= -1.) || (x(0) >= 1. && x(0) <= 2.);
        }
        auto line_domain(const State &x, const hopsy::CoordinateDirection &,
                         hopsy::EmptyWorkspace &) const {
            return std::array{hopsy::LineInterval<double>{-2. - x(0), -1. - x(0)},
                              hopsy::LineInterval<double>{1. - x(0), 2. - x(0)}};
        }
        void reset_workspace(const State &, hopsy::EmptyWorkspace &) const {}
        void advance_workspace(double, const hopsy::CoordinateDirection &,
                               hopsy::EmptyWorkspace &) const {}
    };
    template <typename G>
    concept DikinGeometry = requires(G &g, State &s, hopsy::EmptyWorkspace &w) {
        hopsy::DikinMetric<double>{}.prepare(s, g, w);
    };
    struct ExpCoordinates {
        static constexpr bool constant_volume = false;
        hopsy::Index sampling_dimension() const { return 1; }
        hopsy::Index target_dimension(hopsy::Index) const { return 1; }
        template <typename S, typename Out> void to_target(const S &z, Out &out) const {
            out(0) = std::exp(z(0));
        }
        double log_volume(const State &z) const { return z(0); }
        template <typename Gradient>
        void pullback_gradient(const State &z, const Gradient &g, State &out) const {
            out(0) = std::exp(z(0)) * g(0);
        }
        void log_volume_gradient(const State &, State &out) const { out.setOnes(); }
    };
} // namespace

TEST_CASE("Bound identity contracts preserve exact transitions and target type",
          "[problem][chain]") {
    auto geometry = box(3);
    auto run = [&](auto factory, const auto &target) {
        State direct = State::Zero(3), bound = direct;
        auto a = factory(direct), b = factory(bound);
        auto wa = decltype(a)::make_workspace(geometry, direct);
        auto problem = hopsy::Problem(std::cref(geometry), std::cref(target));
        auto chain = hopsy::bind_chain(problem, b, bound);
        STATIC_REQUIRE_FALSE(TakesTargetPerStep<decltype(chain)>);
        STATIC_REQUIRE_FALSE(BindsTemporary<decltype(problem), decltype(b)>);
        STATIC_REQUIRE(std::same_as<decltype(chain.sampling_target()),
                                    const std::remove_cvref_t<decltype(target)> &>);
        hopsy::RandomNumberGenerator ra(15), rb(15);
        for (int i = 0; i < 300; ++i) {
            REQUIRE(hopsy::step(a, direct, ra, target, geometry, wa) == chain.step(rb));
            REQUIRE(direct == bound);
        }
        REQUIRE(ra == rb);
        REQUIRE(chain.sampling_state() == bound);
        REQUIRE(chain.target_state() == bound);
    };
    auto coordinate = [](const State &s) {
        return hopsy::HitAndRunKernel(s, hopsy::CyclingCoordinateDirectionGenerator(3),
                                      hopsy::UniformLineKernel{});
    };
    auto dense = [](const State &s) {
        return hopsy::HitAndRunKernel(s, hopsy::RandomDirectionGenerator<double>(3),
                                      hopsy::UniformLineKernel{});
    };
    run(coordinate, hopsy::UniformTarget{});
    run(coordinate, Normal{});
    run(dense, hopsy::UniformTarget{});
    run(
        [](const State &s) {
            return hopsy::GaussianKernel<State, hopsy::ZeroDrift, hopsy::IdentityMetric>(s, {}, {},
                                                                                         0.5);
        },
        Normal{});
    run([](const State &s) { return hopsy::DikinWalkKernel<State>(s, {}, {}, 0.2); }, Normal{});
    run([](const State &s) { return hopsy::CSmMALAKernel<State>(s, {}, {}, 0.2); }, Normal{});
}

TEST_CASE("Affine problems include volume and pull back gradients", "[problem][coordinates]") {
    auto run = [&](const Matrix &T, const State &shift) {
        State z(2);
        z << 0.2, -0.4;
        auto geometry = box(2);
        hopsy::GaussianTarget target(State::Zero(T.rows()), Matrix::Identity(T.rows(), T.rows()));
        hopsy::AffineCoordinates coordinates(T, shift);
        auto problem =
            hopsy::Problem(std::cref(geometry), std::cref(target), std::cref(coordinates));
        hopsy::CSmMALAKernel<State> kernel(z, {}, {}, 0.1);
        auto chain = hopsy::bind_chain(problem, kernel, z);
        const State x = T * z + shift;
        const double volume = 0.5 * std::log((T.transpose() * T).determinant());
        REQUIRE(chain.target_state().isApprox(x));
        REQUIRE(chain.sampling_target().log_density(z) == Approx(target.log_density(x) + volume));
        REQUIRE(chain.sampling_target().log_gradient(z).isApprox(-T.transpose() * x));
        // Finite differences also check which side of the map is differentiated.
        State finite(2);
        for (int i = 0; i < 2; ++i) {
            State plus = z, minus = z;
            plus(i) += 1e-6;
            minus(i) -= 1e-6;
            finite(i) = (chain.sampling_target().log_density(plus) -
                         chain.sampling_target().log_density(minus)) /
                        (2e-6);
        }
        REQUIRE(chain.sampling_target().log_gradient(z).isApprox(finite, 1e-8));
        REQUIRE(chain.sampling_dimension() == 2);
        REQUIRE(chain.target_dimension() == T.rows());
    };
    Matrix square(2, 2);
    square << -2., 0.5, 0., 1.5;
    run(square, State::Constant(2, 0.3));
    Matrix embedding(3, 2);
    embedding << 2., 0., 0.5, 1., 1., -1.;
    run(embedding, State::Constant(3, -0.2));
}

TEST_CASE("Coordinate maps and bindings validate dimensions and rank once",
          "[problem][coordinates]") {
    REQUIRE_THROWS_AS((hopsy::AffineCoordinates(Matrix::Zero(2, 2), State::Zero(2))),
                      std::invalid_argument);
    REQUIRE_THROWS_AS((hopsy::AffineCoordinates(Matrix::Identity(2, 2), State::Zero(3))),
                      std::invalid_argument);
    auto geometry = box(2);
    hopsy::AffineCoordinates wrong(Matrix::Identity(3, 3), State::Zero(3));
    REQUIRE_THROWS_AS((hopsy::Problem(std::cref(geometry), Normal{}, wrong)),
                      std::invalid_argument);
    State z = State::Zero(2), bad = State::Zero(3);
    auto problem = hopsy::Problem(std::cref(geometry), Normal{});
    hopsy::CSmMALAKernel<State> kernel(z, {}, {}, 0.1);
    REQUIRE_THROWS_AS(hopsy::bind_chain(problem, kernel, bad), std::invalid_argument);
}

TEST_CASE("Bound caches persist between batches and rebind explicitly", "[problem][chain]") {
    struct Counted {
        mutable int densities = 0, gradients = 0;
        double mean = 0.;
        double log_density(const State &x) const {
            ++densities;
            return -0.5 * (x.array() - mean).matrix().squaredNorm();
        }
        State log_gradient(const State &x) const {
            ++gradients;
            return (mean - x.array()).matrix();
        }
    } original;
    auto geometry = box(2);
    auto problem = hopsy::Problem(std::cref(geometry), original); // Own target snapshot.
    original.mean = 2.;
    REQUIRE(problem.target().mean == 0.);
    State z = State::Constant(2, 0.2);
    hopsy::CSmMALAKernel<State> kernel(z, {}, {}, 0.05);
    hopsy::RandomNumberGenerator rng(51);
    {
        auto chain = hopsy::bind_chain(problem, kernel, z);
        const auto first = chain.draw(rng, 4);
        REQUIRE(first.allFinite());
        REQUIRE(problem.target().densities == 5);
        REQUIRE(problem.target().gradients == 5);
        chain.set_step_size(0.1);
        const auto second = chain.draw(rng, 4);
        REQUIRE(second.allFinite());
        REQUIRE(problem.target().densities == 9);
        REQUIRE(problem.target().gradients == 9);
    }
    Counted changed;
    changed.mean = 0.8;
    auto other = hopsy::Problem(std::cref(geometry), changed);
    auto chain = hopsy::bind_chain(other, kernel, z);
    chain.step(rng);
    REQUIRE(other.target().densities == 2);
    REQUIRE(other.target().gradients == 2);
}

TEST_CASE("Bound step cancels target exceptions", "[problem][chain]") {
    struct Throwing {
        bool fail = true;
        double log_density(const State &x) const {
            if (fail)
                throw std::runtime_error("target failure");
            return -0.5 * x.squaredNorm();
        }
        auto log_gradient(const State &x) const { return -x; }
    } target;
    auto geometry = box(2);
    auto problem = hopsy::Problem(std::cref(geometry), std::cref(target));
    State z = State::Zero(2);
    hopsy::CSmMALAKernel<State> kernel(z, {}, {}, 0.05);
    auto chain = hopsy::bind_chain(problem, kernel, z);
    hopsy::RandomNumberGenerator rng(51);
    REQUIRE_THROWS_AS(chain.step(rng), std::runtime_error);
    REQUIRE_FALSE(kernel.has_pending_proposal());
    REQUIRE(z.isZero());
    target.fail = false; // Resolve an exception; the mathematical density is unchanged.
    chain.step(rng);
    REQUIRE_FALSE(kernel.has_pending_proposal());
}

TEST_CASE("Hit-and-run and random walk accept disconnected geometry without linear constraints",
          "[problem][geometry]") {
    STATIC_REQUIRE_FALSE(DikinGeometry<Disconnected>);
    auto problem = hopsy::Problem(Disconnected{}, hopsy::UniformTarget{});
    auto run = [&](auto factory, int thinning) {
        State z = State::Constant(1, -1.5);
        auto kernel = factory(z);
        auto chain = hopsy::bind_chain(problem, kernel, z);
        hopsy::RandomNumberGenerator rng(51);
        const auto samples = chain.draw(rng, 50000, thinning);
        REQUIRE(std::abs(samples.mean()) < 0.06);
        const double positive = (samples.array() > 0.).template cast<double>().mean();
        REQUIRE(positive == Approx(0.5).margin(0.025));
        for (int i = 0; i < samples.cols(); ++i) {
            const double x = samples(0, i);
            REQUIRE(((x >= -2. && x <= -1.) || (x >= 1. && x <= 2.)));
        }
    };
    run(
        [](const State &z) {
            return hopsy::HitAndRunKernel(z, hopsy::CyclingCoordinateDirectionGenerator(1),
                                          hopsy::UniformLineKernel{});
        },
        1);
    run(
        [](const State &z) {
            return hopsy::GaussianKernel<State, hopsy::ZeroDrift, hopsy::IdentityMetric>(z, {}, {},
                                                                                         2.);
        },
        3);
}

TEST_CASE("Nonconstant map volume changes density and gradient even for a uniform target",
          "[problem][coordinates][statistics]") {
    auto problem = hopsy::Problem(box(1, 1.), hopsy::UniformTarget{}, ExpCoordinates{});
    State z = State::Zero(1);
    hopsy::GaussianKernel<State, hopsy::ClippedGradientDrift, hopsy::IdentityMetric> kernel(
        z, {}, {}, 0.5);
    auto chain = hopsy::bind_chain(problem, kernel, z);
    STATIC_REQUIRE_FALSE(hopsy::target_is_uniform_v<decltype(chain.sampling_target())>);
    const State query = State::Constant(1, 0.2);
    REQUIRE(chain.sampling_target().log_density(query) == Approx(0.2));
    REQUIRE(chain.sampling_target().log_gradient(query)(0) == Approx(1.));
    hopsy::RandomNumberGenerator rng(99);
    for (int i = 0; i < 3000; ++i)
        chain.step(rng);
    const auto samples = chain.draw(rng, 80000, 2);
    REQUIRE(samples.minCoeff() >= std::exp(-1.));
    REQUIRE(samples.maxCoeff() <= std::exp(1.));
    REQUIRE(samples.mean() == Approx(std::cosh(1.)).margin(0.025));
}

TEST_CASE("Rounded Gaussian sampling returns physical Gaussian moments",
          "[problem][coordinates][statistics]") {
    Matrix T(2, 2);
    T << 1.3, 0.2, -0.4, 0.8;
    State shift(2);
    shift << 0.3, -0.25;
    auto geometry = hopsy::LinearPolytope<double>(Matrix(0, 2), State(0));
    auto problem =
        hopsy::Problem(std::cref(geometry), Normal{}, hopsy::AffineCoordinates(T, shift));
    State z = -T.inverse() * shift;
    hopsy::GaussianKernel<State, hopsy::ZeroDrift, hopsy::IdentityMetric> kernel(z, {}, {}, 0.8);
    auto chain = hopsy::bind_chain(problem, kernel, z);
    hopsy::RandomNumberGenerator rng(716);
    for (int i = 0; i < 5000; ++i)
        chain.step(rng);
    const auto samples = chain.draw(rng, 80000, 3);
    REQUIRE(samples.rowwise().mean().cwiseAbs().maxCoeff() < 0.04);
    REQUIRE((samples.array().square().rowwise().mean() - 1.).abs().maxCoeff() < 0.06);
    REQUIRE(std::abs(samples.row(0).dot(samples.row(1)) / samples.cols()) < 0.05);
    REQUIRE(samples.col(samples.cols() - 1).isApprox(chain.target_state()));
}

TEST_CASE("Identity and affine bindings allocate no Eigen buffers while drawing",
          "[problem][allocation]") {
    auto run = [&](const auto &problem) {
        State z = State::Zero(2);
        auto kernel = hopsy::HitAndRunKernel(z, hopsy::CyclingCoordinateDirectionGenerator(2),
                                             hopsy::UniformLineKernel{});
        auto chain = hopsy::bind_chain(problem, kernel, z);
        STATIC_REQUIRE(decltype(kernel)::template always_accept<
                       std::remove_cvref_t<decltype(chain.sampling_target())>>);
        Matrix output(chain.target_dimension(), 100);
        hopsy::RandomNumberGenerator rng(51);
        const bool previous = Eigen::internal::is_malloc_allowed();
        struct Restore {
            bool value;
            ~Restore() { Eigen::internal::set_is_malloc_allowed(value); }
        } restore{previous};
        Eigen::internal::set_is_malloc_allowed(false);
        chain.draw_into(rng, output, 3);
        REQUIRE(output.allFinite());
    };
    auto geometry = box(2);
    auto identity = hopsy::Problem(std::cref(geometry), hopsy::UniformTarget{});
    run(identity);
    Matrix T(3, 2);
    T << 1, 0, 0, 2, 1, 1;
    auto mapped = hopsy::Problem(std::cref(geometry), hopsy::UniformTarget{},
                                 hopsy::AffineCoordinates(T, State::Ones(3)));
    run(mapped);
}
