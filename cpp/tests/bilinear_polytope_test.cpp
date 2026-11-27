#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <hopsy/geometry/bilinear_polytope.h>
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/kernel/gaussian_line_kernel.h>
#include <hopsy/kernel/overrelaxed_line_kernel.h>
#include <hopsy/sampling/chain.h>
#include <hopsy/sampling/parallel_tempering/non_reversible_parallel_tempering.h>
#include <hopsy/target/uniform.h>
#include <numeric>
#include <random>

namespace {
    using V = hopsy::Vector<double>;
    using M = hopsy::Matrix<double>;
    using G = hopsy::BilinearPolytope<>;
    using Catch::Approx;
    std::pair<M, V> box(int n, double lower, double upper) {
        M A(2 * n, n);
        A.topRows(n).setIdentity();
        A.bottomRows(n) = -A.topRows(n);
        V b(2 * n);
        b.head(n).setConstant(upper);
        b.tail(n).setConstant(-lower);
        return {A, b};
    }
    G product(double bound, double lower = 0, double upper = 1, double sign = 1) {
        auto [A, b] = box(2, lower, upper);
        return G(A, b, 1, {M::Constant(1, 1, sign)}, V::Constant(1, bound));
    }
    bool in_domain(const G::line_domain_type &domain, double t) {
        for (const auto &i : domain)
            if (i.lower <= t && t <= i.upper)
                return true;
        return false;
    }
} // namespace

TEST_CASE("NRPT preserves bilinear supports while exchanging cached workspaces", "[bilinear][nrpt]") {
    const auto geometry = product(0.2);
    struct Target {
        double log_density(const V &x) const { return -2. * x.squaredNorm(); }
    };
    const auto problem = hopsy::Problem(std::cref(geometry), Target{});
    const V initial = V::Constant(2, 0.2);
    const auto kernel = [&] {
        return hopsy::HitAndRunKernel(initial, hopsy::RandomDirectionGenerator<double>(2),
                                      hopsy::UniformLineKernel{});
    };
    std::vector<hopsy::TemperingReplica<decltype(kernel())>> replicas;
    for (std::size_t i = 0; i < 4; ++i)
        replicas.push_back({initial, kernel(), hopsy::RandomNumberGenerator(201, i)});
    auto sampler = hopsy::NonReversibleParallelTempering(
        problem, hopsy::TemperatureLadder::linear(4), std::move(replicas),
        hopsy::RandomNumberGenerator(631, 4));
    for (int scan = 0; scan < 300; ++scan) {
        sampler.scan(2, 3);
        for (std::size_t slot = 0; slot < sampler.size(); ++slot)
            REQUIRE(geometry.is_feasible(sampler.sampling_state(slot)));
    }
    for (auto trips : sampler.round_trips())
        REQUIRE(trips > 0);
}

TEST_CASE("Quadratic slices retain all components and nearly linear roots", "[bilinear][numeric]") {
    auto q = hopsy::detail::quadratic_slice(1, 0, -1);
    REQUIRE(q.size == 1);
    REQUIRE(q.intervals[0].lower == -1);
    REQUIRE(q.intervals[0].upper == 1);
    q = hopsy::detail::quadratic_slice(-1, 0, 1);
    REQUIRE(q.size == 2);
    REQUIRE(q.intervals[0].upper == -1);
    REQUIRE(q.intervals[1].lower == 1);
    q = hopsy::detail::quadratic_slice(0, 0, -1);
    REQUIRE(q.size == 1);
    REQUIRE(std::isinf(q.intervals[0].lower));
    REQUIRE(hopsy::detail::quadratic_slice(0, 0, 1).size == 0);
    q = hopsy::detail::quadratic_slice(0, 2, -4);
    REQUIRE(q.intervals[0].upper == 2);
    q = hopsy::detail::quadratic_slice(0, -2, 4);
    REQUIRE(q.intervals[0].lower == 2);
    q = hopsy::detail::quadratic_slice(1, -2, 1);
    REQUIRE(q.size == 1);
    REQUIRE(q.intervals[0].lower == 1);
    REQUIRE(q.intervals[0].upper == 1);
    q = hopsy::detail::quadratic_slice(9, -6, 1);
    REQUIRE(q.size == 1);
    REQUIRE(double(q.intervals[0].lower) == Approx(1. / 3));
    REQUIRE(hopsy::detail::quadratic_slice(-1, 2, -1).size == 1);
    REQUIRE(hopsy::detail::quadratic_slice(1, -2, std::nextafter(1.L, 2.L)).size == 0);
    q = hopsy::detail::quadratic_slice(-1e-30L, 1, -1);
    REQUIRE(q.size == 2);
    REQUIRE(double(q.intervals[0].upper) == Approx(1));
    REQUIRE(double(q.intervals[1].lower) == Approx(1e30));
    q = hopsy::detail::quadratic_slice(1, 1e16L, 1);
    REQUIRE(double(q.intervals[0].lower) == Approx(-1e16));
    REQUIRE(double(q.intervals[0].upper) == Approx(-1e-16));
    for (long double scale : {1e-300L, 1e300L}) {
        q = hopsy::detail::quadratic_slice(scale, -3 * scale, 2 * scale);
        REQUIRE(double(q.intervals[0].lower) == Approx(1));
        REQUIRE(double(q.intervals[0].upper) == Approx(2));
    }
    REQUIRE_THROWS_AS(hopsy::detail::quadratic_slice(INFINITY, 1, 1), std::overflow_error);
}

TEST_CASE("Bilinear partition, feasibility, and disconnected x+y=1 slice", "[bilinear]") {
    auto g = product(.2);
    V z(2), d(2);
    z << 0, 1;
    d << 1, -1;
    auto w = g.create_workspace(z);
    REQUIRE(reinterpret_cast<std::uintptr_t>(w.domain.data()) % 64 == 0);
    REQUIRE(reinterpret_cast<std::uintptr_t>(w.scratch.data()) % 64 == 0);
    const auto domain = g.line_domain(z, d, w);
    REQUIRE(domain.size() == 2);
    const double root = (1 - std::sqrt(.2)) / 2;
    REQUIRE(domain[0].lower == 0);
    REQUIRE(domain[0].upper == Approx(root));
    REQUIRE(domain[1].lower == Approx(1 - root));
    REQUIRE(domain[1].upper == 1);
    REQUIRE(in_domain(domain, .1));
    REQUIRE_FALSE(in_domain(domain, .5));
    REQUIRE(in_domain(domain, .9));
    z << .2, 1;
    REQUIRE(g.is_feasible(z));
    REQUIRE_FALSE(g.is_feasible(z, 1e-10));
    z << .2 + 1e-12, 1;
    REQUIRE_FALSE(g.is_feasible(z));
    REQUIRE(g.is_feasible(z, -1e-10));
    z << .1, .1;
    d.setZero();
    auto all = g.line_domain(z, d, w = g.create_workspace(z));
    REQUIRE(all.size() == 1);
    REQUIRE(std::isinf(all[0].lower));
    REQUIRE(std::isinf(all[0].upper));
    hopsy::RandomNumberGenerator rng(4);
    REQUIRE_THROWS_AS(hopsy::UniformLineKernel{}.propose(rng, all), std::invalid_argument);
    auto [A, b] = box(2, 0, 1);
    REQUIRE_THROWS_AS(G(A, b, 0, {M::Ones(1, 1)}, V::Ones(1)), std::invalid_argument);
    REQUIRE_THROWS_AS(G(A, b, 1, {M::Ones(2, 2)}, V::Ones(1)), std::invalid_argument);
    REQUIRE_THROWS_AS(G(A, b, 1, {M::Ones(1, 1)}, V::Ones(2)), std::invalid_argument);
    auto huge = G(M::Constant(1, 2, 1e308), V::Ones(1), 1, {}, V(0));
    REQUIRE_THROWS_AS(huge.create_workspace(V::Constant(2, 1e308)), std::overflow_error);
}

TEST_CASE("Bilinear intersection has three intervals, tangencies, and empty domains",
          "[bilinear]") {
    auto [A, b] = box(2, -1, 1);
    M affine = M::Zero(2, 2);
    affine(0, 0) = -1;
    affine(1, 0) = 1;
    G g(A, b, 1, {M::Constant(1, 1, -1), M::Constant(1, 1, -1)}, V::Constant(2, .24), affine);
    V z = V::Zero(2), d = V::Ones(2);
    auto w = g.create_workspace(z);
    auto domain = g.line_domain(z, d, w);
    REQUIRE(domain.size() == 3);
    REQUIRE(domain[0].upper == Approx(-.6));
    REQUIRE(domain[1].lower == Approx(-.4));
    REQUIRE(domain[1].upper == Approx(.4));
    REQUIRE(domain[2].lower == Approx(.6));
    auto tangent = product(0, -1, 1);
    auto tw = tangent.create_workspace(z);
    domain = tangent.line_domain(z, d, tw);
    REQUIRE(domain.size() == 1);
    REQUIRE(domain[0].lower == 0);
    REQUIRE(domain[0].upper == 0);
    hopsy::RandomNumberGenerator rng(4);
    REQUIRE_THROWS_AS(hopsy::UniformLineKernel{}.propose(rng, domain), std::invalid_argument);
    auto empty = product(-2, -1, 1);
    auto ew = empty.create_workspace(z);
    REQUIRE(empty.line_domain(z, d, ew).empty());
    auto unbounded = G(M(0, 2), V(0), 1, {M::Constant(1, 1, -1)}, V::Constant(1, -1));
    auto uw = unbounded.create_workspace(z);
    domain = unbounded.line_domain(z, d, uw);
    REQUIRE(domain.size() == 2);
    REQUIRE(std::isinf(domain[0].lower));
    REQUIRE(std::isinf(domain[1].upper));
    REQUIRE_THROWS_AS(hopsy::UniformLineKernel{}.propose(rng, domain), std::invalid_argument);
    REQUIRE(std::isfinite(hopsy::GaussianLineKernel(.5).propose(rng, domain)));
}

TEST_CASE("Bilinear line sets agree with direct evaluation and reverse translation",
          "[bilinear][property]") {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> random(-1, 1);
    auto [A, b] = box(4, -1, 1);
    for (int repeat = 0; repeat < 100; ++repeat) {
        M B(2, 2), affine(2, 4);
        for (auto &v : B.reshaped())
            v = random(rng);
        for (auto &v : affine.reshaped())
            v = random(rng);
        V rhs = V::Constant(2, .2), z(4), d(4);
        for (auto &v : z)
            v = .8 * random(rng);
        for (auto &v : d)
            v = random(rng);
        G g(A, b, 2, {B, -B}, rhs, affine);
        auto w = g.create_workspace(z);
        const auto intervals = g.line_domain(z, d, w);
        for (int sample = 0; sample <= 300; ++sample) {
            double t = -3. + 6. * sample / 300;
            V point = z + t * d;
            double q1 = (point.head(2).transpose() * B * point.tail(2)).value() +
                        affine.row(0).dot(point) - .2;
            double q2 = -(point.head(2).transpose() * B * point.tail(2)).value() +
                        affine.row(1).dot(point) - .2;
            if (std::min(std::abs(q1), std::abs(q2)) < 1e-10 ||
                (point.array().abs() - 1).abs().minCoeff() < 1e-10)
                continue;
            bool expected = (point.array().abs() <= 1).all() && q1 <= 0 && q2 <= 0;
            REQUIRE(in_domain(intervals, t) == expected);
        }
        if (!intervals.empty() && intervals[0].upper > intervals[0].lower) {
            double shift = std::midpoint(intervals[0].lower, intervals[0].upper);
            V next = z + shift * d;
            auto nw = g.create_workspace(next);
            auto translated = g.line_domain(next, d, nw);
            REQUIRE(translated.size() == intervals.size());
            for (std::size_t i = 0; i < intervals.size(); ++i) {
                REQUIRE(translated[i].lower + shift == Approx(intervals[i].lower).margin(1e-10));
                REQUIRE(translated[i].upper + shift == Approx(intervals[i].upper).margin(1e-10));
            }
        }
    }
}

TEST_CASE("Bilinear uniform sampling matches analytic area, marginals, and moments",
          "[bilinear][statistics]") {
    constexpr double c = .2;
    double Z = c * (1 - std::log(c)), mean = (c - c * c / 2) / Z;
    auto g = product(c);
    auto p = hopsy::Problem(std::cref(g), hopsy::UniformTarget{});
    V z = V::Constant(2, .1);
    auto k = hopsy::HitAndRunKernel(z, hopsy::RandomDirectionGenerator<double>(2),
                                    hopsy::UniformLineKernel{});
    auto chain = hopsy::bind_chain(p, k, z);
    hopsy::RandomNumberGenerator rng(42);
    chain.draw(rng, 3000);
    auto samples = chain.draw(rng, 100000, 2);
    for (int axis = 0; axis < 2; ++axis) {
        REQUIRE(samples.row(axis).mean() == Approx(mean).margin(.012));
        for (double s : {.05, .2, .5, .8}) {
            double expected = s <= c ? s / Z : (c + c * std::log(s / c)) / Z;
            double fraction = (samples.row(axis).array() <= s).cast<double>().mean();
            REQUIRE(fraction == Approx(expected).margin(.015));
        }
    }
    REQUIRE((samples.row(0).array() * samples.row(1).array()).maxCoeff() <= c + 1e-11);
    REQUIRE(samples.minCoeff() >= -1e-11);
    REQUIRE(samples.maxCoeff() <= 1 + 1e-11);
}

TEST_CASE("Dense bilinear sampling visits both opposite-sign components",
          "[bilinear][statistics]") {
    auto g = product(0, -1, 1);
    auto p = hopsy::Problem(std::cref(g), hopsy::UniformTarget{});
    V z(2);
    z << -.5, .5;
    auto k = hopsy::HitAndRunKernel(z, hopsy::RandomDirectionGenerator<double>(2),
                                    hopsy::UniformLineKernel{});
    auto chain = hopsy::bind_chain(p, k, z);
    hopsy::RandomNumberGenerator rng(24);
    chain.draw(rng, 3000);
    auto samples = chain.draw(rng, 80000, 2);
    REQUIRE(samples.rowwise().mean().cwiseAbs().maxCoeff() < .02);
    REQUIRE((samples.row(0).array() > 0).cast<double>().mean() == Approx(.5).margin(.02));
    REQUIRE((samples.row(0).array() * samples.row(1).array()).mean() == Approx(-.25).margin(.015));
    REQUIRE((samples.row(0).array() * samples.row(1).array()).maxCoeff() <= 1e-11);
    for (int axis = 0; axis < 2; ++axis) {
        REQUIRE(samples.row(axis).array().square().mean() == Approx(1. / 3).margin(.015));
        for (double s : {-.5, 0., .5})
            REQUIRE((samples.row(axis).array() <= s).cast<double>().mean() ==
                    Approx((s + 1) / 2).margin(.02));
    }
}

TEST_CASE("Bilinear support traits and Gaussian line MH use full union", "[bilinear][mh]") {
    using UniformK = hopsy::HitAndRunKernel<V, hopsy::RandomDirectionGenerator<double>,
                                            hopsy::UniformLineKernel>;
    using OverK = hopsy::HitAndRunKernel<V, hopsy::RandomDirectionGenerator<double>,
                                         hopsy::OverRelaxedLineKernel>;
    STATIC_REQUIRE(hopsy::kernel_supports_geometry_v<UniformK, G>);
    STATIC_REQUIRE_FALSE(hopsy::kernel_supports_geometry_v<hopsy::DikinWalkKernel<V>, G>);
    STATIC_REQUIRE_FALSE(hopsy::kernel_supports_geometry_v<
                         hopsy::GaussianKernel<V, hopsy::ZeroDrift, hopsy::IdentityMetric>, G>);
    STATIC_REQUIRE_FALSE(hopsy::kernel_supports_geometry_v<OverK, G>);
    struct Fixed {
        V d;
        const V &generate(hopsy::RandomNumberGenerator &) { return d; }
        const V &direction() const { return d; }
    };
    auto g = product(0, -1, 1);
    V z(2), d(2);
    z << -.5, .5;
    d << 1, -2;
    auto w = g.create_workspace(z);
    hopsy::HitAndRunKernel k(z, Fixed{d}, hopsy::GaussianLineKernel(.4));
    hopsy::UniformTarget target;
    hopsy::RandomNumberGenerator rng(7);
    auto normal_cdf = [](double v) { return .5 * std::erfc(-v / std::sqrt(2.)); };
    for (int i = 0; i < 100; ++i) {
        auto domain = g.line_domain(z, d, w);
        REQUIRE(domain.size() == 2);
        k.propose(z, rng, target, g, w);
        double step = k.proposal().step, old_mass = 0, new_mass = 0;
        for (auto interval : domain) {
            old_mass += normal_cdf(interval.upper / .4) - normal_cdf(interval.lower / .4);
            new_mass +=
                normal_cdf((interval.upper - step) / .4) - normal_cdf((interval.lower - step) / .4);
        }
        REQUIRE(k.log_acceptance_ratio(z, target, g, w) ==
                Approx(std::log(old_mass / new_mass)).margin(1e-11));
        REQUIRE(g.is_feasible(k.proposal().state, -1e-12));
        k.reject(z);
    }
}

TEST_CASE("Exact fallback resolves coefficient cancellation and true tangencies",
          "[bilinear][numeric]") {
    auto [A, b] = box(4, -1, 1);
    M B(2, 2);
    B << 1e30, 1, -1e30, 0;
    G g(A, b, 2, {B}, V::Constant(1, .25));
    V z = V::Zero(4), d = V::Ones(4);
    auto w = g.create_workspace(z);
    auto domain = g.line_domain(z, d, w);
    REQUIRE(domain.size() == 1);
    REQUIRE(domain[0].lower == -.5);
    REQUIRE(domain[0].upper == .5);
    auto tangent = G(A, b, 2, {B}, V::Zero(1));
    z.setConstant(.1);
    auto tw = tangent.create_workspace(z);
    domain = tangent.line_domain(z, d, tw);
    REQUIRE(domain.size() == 1);
    REQUIRE(domain[0].lower == -.1);
    REQUIRE(domain[0].upper == -.1);
    REQUIRE_FALSE(tangent.is_feasible(z));
    M affine(1, 4);
    affine << 1e30, 1, -1e30, 0;
    G linear(A, b, 2, {M::Zero(2, 2)}, V::Zero(1), affine);
    auto lw = linear.create_workspace(z);
    domain = linear.line_domain(z, d, lw);
    REQUIRE(domain.size() == 1);
    REQUIRE(domain[0].upper == Approx(-.1));
    REQUIRE_FALSE(linear.is_feasible(z));
    auto none = G(A, b, 2, {B}, V::Constant(1, std::nextafter(0., -1.)));
    z.setZero();
    auto nw = none.create_workspace(z);
    REQUIRE(none.line_domain(z, d, nw).empty());
    struct Derived : G {
        using G::G;
    };
    STATIC_REQUIRE_FALSE(hopsy::kernel_supports_geometry_v<hopsy::DikinWalkKernel<V>, Derived>);
}

TEST_CASE("Numerically unrepresentable positive-width slices fail explicitly",
          "[bilinear][numeric]") {
    if constexpr (std::numeric_limits<long double>::max_exponent >
                  3 * std::numeric_limits<double>::max_exponent) {
        auto [A, b] = box(2, -1, 1);
        G tiny(A, b, 1, {M::Constant(1, 1, 1e300)}, V::Ones(1));
        V z = V::Zero(2), d = V::Constant(2, 1e300);
        auto w = tiny.create_workspace(z);
        REQUIRE_THROWS_AS(tiny.line_domain(z, d, w), std::range_error);
        G huge(M(0, 2), V(0), 1, {M::Constant(1, 1, 1e-300)}, V::Ones(1));
        d.setConstant(1e-300);
        auto hw = huge.create_workspace(z);
        REQUIRE_THROWS_AS(huge.line_domain(z, d, hw), std::overflow_error);
    }
}

TEST_CASE("Factored signs, float geometry, and allocation-free ordinary slices",
          "[bilinear][numeric][allocation]") {
    auto g = product(0, -1, 1, -1);
    V z(2), d(2);
    z << -.5, .5;
    d << 1, -2;
    auto w = g.create_workspace(z);
    auto domain = g.line_domain(z, d, w);
    REQUIRE(domain.size() == 1);
    REQUIRE(domain[0].lower == .25);
    REQUIRE(domain[0].upper == .5);
    auto [A, b] = box(2, 0, 1);
    hopsy::BilinearPolytope<float> f(A.cast<float>(), b.cast<float>(), 1,
                                     {M::Ones(1, 1).cast<float>()},
                                     V::Constant(1, .2).cast<float>());
    hopsy::Vector<float> origin(2), direction(2);
    origin << 0, 1;
    direction << 1, -1;
    auto fw = f.create_workspace(origin);
    const auto &slice = f.line_domain(origin, direction, fw);
    REQUIRE(slice.size() == 2);
    REQUIRE(slice[0].upper == Approx((1 - std::sqrt(.2)) / 2).margin(1e-6));
    auto regular = product(.2);
    z.setConstant(.1);
    auto p = hopsy::Problem(std::cref(regular), hopsy::UniformTarget{});
    auto kernel = hopsy::HitAndRunKernel(z, hopsy::RandomDirectionGenerator<double>(2),
                                         hopsy::UniformLineKernel{});
    auto chain = hopsy::bind_chain(p, kernel, z);
    hopsy::RandomNumberGenerator rng(11);
    M out(2, 2000);
    Eigen::internal::set_is_malloc_allowed(false);
    chain.draw_into(rng, out, 2);
    Eigen::internal::set_is_malloc_allowed(true);
    REQUIRE(out.allFinite());
}

TEST_CASE("Dynamic singleton sampling matches the fixed linear path bit for bit",
          "[bilinear][line-kernel]") {
    std::array<hopsy::LineInterval<double>, 1> fixed{{{-2., 3.}}};
    std::vector<hopsy::LineInterval<double>> dynamic(fixed.begin(), fixed.end());
    hopsy::UniformLineKernel kernel;
    hopsy::RandomNumberGenerator a(9), b(9);
    for (int i = 0; i < 1000; ++i)
        REQUIRE(kernel.propose(a, fixed) == kernel.propose(b, dynamic));
    REQUIRE(a == b);
}

TEST_CASE("Positive-product complement has independently known area and marginals",
          "[bilinear][statistics]") {
    constexpr double c = .2;
    const double lower_area = c * (1 - std::log(c)), Z = 1 - lower_area,
                 mean = (.5 - c + c * c / 2) / Z;
    auto g = product(-c, 0, 1, -1);
    V z = V::Constant(2, .7);
    auto p = hopsy::Problem(std::cref(g), hopsy::UniformTarget{});
    auto kernel = hopsy::HitAndRunKernel(z, hopsy::RandomDirectionGenerator<double>(2),
                                         hopsy::UniformLineKernel{});
    auto chain = hopsy::bind_chain(p, kernel, z);
    hopsy::RandomNumberGenerator rng(17);
    chain.draw(rng, 3000);
    auto samples = chain.draw(rng, 80000, 2);
    for (int axis = 0; axis < 2; ++axis) {
        REQUIRE(samples.row(axis).mean() == Approx(mean).margin(.012));
        for (double t : {.1, .3, .5, .8}) {
            double expected = t <= c ? 0 : (t - c - c * std::log(t / c)) / Z;
            REQUIRE((samples.row(axis).array() <= t).cast<double>().mean() ==
                    Approx(expected).margin(.015));
        }
    }
    REQUIRE((samples.row(0).array() * samples.row(1).array()).minCoeff() >= c - 1e-11);
    // Independent box Monte Carlo checks the area, not another Markov chain.
    std::mt19937 random(123);
    std::uniform_real_distribution<double> uniform(0, 1);
    int inside = 0;
    for (int i = 0; i < 50000; ++i) {
        V point(2);
        point << uniform(random), uniform(random);
        inside += g.is_feasible(point);
    }
    REQUIRE(double(inside) / 50000 == Approx(Z).margin(.008));
}

TEST_CASE("Coordinate bilinear slices match dense unit directions and analytic means",
          "[bilinear][coordinate]") {
    auto [A, b] = box(4, -1, 1);
    M B(2, 2);
    B << 1, .2, -.4, .7;
    M affine = M::Zero(1, 4);
    affine(0, 2) = .1;
    G geometry(A, b, 2, {B}, V::Constant(1, .3), affine);
    V z = V::Constant(4, .1);
    for (int axis = 0; axis < 4; ++axis) {
        V d = V::Zero(4);
        d(axis) = 1;
        auto a = geometry.create_workspace(z), c = geometry.create_workspace(z);
        auto expected = geometry.line_domain(z, d, a);
        auto actual = geometry.line_domain(z, hopsy::CoordinateDirection{axis}, c);
        REQUIRE(actual.size() == expected.size());
        for (std::size_t j = 0; j < actual.size(); ++j) {
            REQUIRE(actual[j].lower == expected[j].lower);
            REQUIRE(actual[j].upper == expected[j].upper);
        }
    }
    auto g = product(.2);
    z = V::Constant(2, .1);
    auto p = hopsy::Problem(std::cref(g), hopsy::UniformTarget{});
    auto kernel = hopsy::HitAndRunKernel(z, hopsy::CyclingCoordinateDirectionGenerator(2),
                                         hopsy::UniformLineKernel{});
    auto chain = hopsy::bind_chain(p, kernel, z);
    hopsy::RandomNumberGenerator rng(17);
    chain.draw(rng, 3000);
    auto samples = chain.draw(rng, 80000, 2);
    double mean = (.2 - .2 * .2 / 2) / (.2 * (1 - std::log(.2)));
    REQUIRE(samples.row(0).mean() == Approx(mean).margin(.012));
    REQUIRE(samples.row(1).mean() == Approx(mean).margin(.012));
}
