#include <catch2/catch_test_macros.hpp>
#include <hopsy/sampling/owned_chain.h>
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/target/uniform.h>
#include <type_traits>

namespace {
    using State = hopsy::Vector<double>;
    using Matrix = hopsy::Matrix<double>;
    auto box(int n, double bound = 3.) {
        Matrix A(2 * n, n);
        A.topRows(n).setIdentity();
        A.bottomRows(n) = -A.topRows(n);
        return hopsy::LinearPolytope<double>(A, State::Constant(2 * n, bound));
    }
    template <typename P, typename K>
    concept BindsTemporaryOwner =
        requires(P &&p, K &k, State s) { hopsy::bind_chain(std::move(p), k, std::move(s)); };
    struct Target {
        mutable int densities = 0, gradients = 0;
        double mean = 0.;
        bool fail = false;
        double log_density(const State &s) const {
            if (fail)
                throw std::runtime_error("target failed");
            ++densities;
            return -.5 * (s.array() - mean).matrix().squaredNorm();
        }
        State log_gradient(const State &s) const {
            ++gradients;
            return (mean - s.array()).matrix();
        }
    };
} // namespace

TEST_CASE("Owning and borrowed bindings follow identical transitions", "[chain][ownership]") {
    auto geometry = box(3);
    auto problem = hopsy::Problem(std::cref(geometry), Target{});
    auto test = [&](auto factory) {
        State a = State::Zero(3);
        auto first = factory(a), second = factory(a);
        auto view = hopsy::bind_chain(problem, first, a);
        auto owner = hopsy::bind_chain(problem, second, State(State::Zero(3)));
        STATIC_REQUIRE_FALSE(BindsTemporaryOwner<decltype(problem), decltype(second)>);
        STATIC_REQUIRE_FALSE(std::is_move_constructible_v<typename decltype(owner)::element_type>);
        hopsy::RandomNumberGenerator r1(17), r2(17);
        for (int i = 0; i < 200; ++i) {
            REQUIRE(view.step(r1) == owner->step(r2));
            REQUIRE(a == owner->sampling_state());
        }
        REQUIRE(r1 == r2);
        REQUIRE_THROWS_AS(hopsy::bind_chain(problem, second, owner->sampling_state()),
                          std::runtime_error);
    };
    test([](const State &s) {
        return hopsy::HitAndRunKernel(s, hopsy::CyclingCoordinateDirectionGenerator(3),
                                      hopsy::UniformLineKernel{});
    });
    test([](const State &s) { return hopsy::CSmMALAKernel<State>(s, {}, {}, .3); });
}

TEST_CASE("Owning chain retains exactly its current shared problem", "[chain][ownership][cache]") {
    auto geometry = box(2, 100.);
    using P = hopsy::Problem<std::reference_wrapper<const decltype(geometry)>, Target>;
    auto p = std::make_shared<const P>(std::cref(geometry), Target{});
    std::weak_ptr<const P> old = p;
    State initial = State::Zero(2);
    hopsy::CSmMALAKernel<State> kernel(initial, {}, {}, .001);
    auto chain = hopsy::bind_chain(p, kernel, State(initial));
    p.reset();
    hopsy::RandomNumberGenerator rng(5);
    chain->draw(rng, 4);
    chain->draw(rng, 4);
    REQUIRE(old.lock()->target().densities == 9);
    REQUIRE(old.lock()->target().gradients == 9);
    auto next = std::make_shared<const P>(std::cref(geometry), Target{.mean = .5});
    chain->rebind_problem(next);
    REQUIRE(old.expired());
    chain->step(rng);
    REQUIRE(next->target().densities == 2);
    REQUIRE(next->target().gradients == 2);
    auto other_geometry = box(2, 100.);
    auto invalid = std::make_shared<const P>(std::cref(other_geometry), Target{});
    REQUIRE_THROWS_AS(chain->rebind_problem(invalid), std::invalid_argument);
    chain->step(rng);
    REQUIRE(next->target().densities == 3);
}

TEST_CASE("Owner cancels exceptions and releases temporary RNG leases", "[chain][ownership]") {
    auto geometry = box(2);
    Target target;
    target.fail = true;
    auto problem = hopsy::Problem(std::cref(geometry), std::cref(target));
    State state = State::Zero(2);
    hopsy::CSmMALAKernel<State> kernel(state, {}, {}, .2);
    auto chain = hopsy::bind_chain(problem, kernel, State(state));
    hopsy::RandomNumberGenerator rng(4);
    REQUIRE_THROWS_AS(chain->step(rng), std::runtime_error);
    REQUIRE_FALSE(kernel.has_pending_proposal());
    REQUIRE(chain->sampling_state().isZero());
    {
        hopsy::ExclusiveLease<hopsy::RandomNumberGenerator> lease(rng);
    }
    target.fail = false;
    chain->step(rng);
    chain.reset();
    hopsy::ExclusiveLease<decltype(kernel)> lease(kernel);
}

TEST_CASE("A fresh owner invalidates old geometry metrics", "[chain][cache]") {
    auto geometry = box(2), other_geometry = box(2, 5.);
    auto first = hopsy::Problem(std::cref(geometry), Target{});
    auto second = hopsy::Problem(std::cref(other_geometry), Target{});
    State state = State::Zero(2);
    hopsy::CSmMALAKernel<State> kernel(state, {}, {}, .2);
    {
        auto chain = hopsy::bind_chain(first, kernel, State(state));
        hopsy::RandomNumberGenerator warmup(7);
        chain->draw(warmup, 100);
    }
    // Explicitly choose the kernel's candidate as a fresh matched initial pair.
    State next = kernel.proposal().state;
    hopsy::CSmMALAKernel<State> fresh(next, {}, {}, .2);
    auto reused = hopsy::bind_chain(second, kernel, State(next));
    auto expected = hopsy::bind_chain(second, fresh, State(next));
    hopsy::RandomNumberGenerator a(17), b(17);
    REQUIRE(reused->draw(a, 100) == expected->draw(b, 100));
}

TEST_CASE("Owner validates initial pairs and does not allocate Eigen buffers while drawing",
          "[chain][ownership][allocation]") {
    auto geometry = box(2);
    auto problem = hopsy::Problem(std::cref(geometry), hopsy::UniformTarget{});
    State initial = State::Zero(2);
    auto kernel = hopsy::HitAndRunKernel(initial, hopsy::CyclingCoordinateDirectionGenerator(2),
                                         hopsy::UniformLineKernel{});
    REQUIRE_THROWS_AS(hopsy::bind_chain(problem, kernel, State(State::Ones(2))),
                      std::invalid_argument);
    auto chain = hopsy::bind_chain(problem, kernel, State(initial));
    hopsy::RandomNumberGenerator rng(6);
    Matrix out(2, 100);
    Eigen::internal::set_is_malloc_allowed(false);
    chain->draw_into(rng, out, 2);
    Eigen::internal::set_is_malloc_allowed(true);
    REQUIRE(out.allFinite());
}

TEST_CASE("Sampling resource allocation preserves type, ownership, and alignment",
          "[chain][ownership]") {
    auto rng = hopsy::make_sampling_resource<hopsy::RandomNumberGenerator>(42);
    hopsy::RandomNumberGenerator expected(42);
    STATIC_REQUIRE(std::same_as<decltype(rng), std::shared_ptr<hopsy::RandomNumberGenerator>>);
    REQUIRE(reinterpret_cast<std::uintptr_t>(rng.get()) % 64 == 0);
    REQUIRE((*rng)() == expected());
    struct Counted {
        int *alive;
        explicit Counted(int &count) : alive(&count) { ++*alive; }
        ~Counted() { --*alive; }
    };
    int alive = 0;
    {
        auto owner = hopsy::make_sampling_resource<Counted>(alive);
        auto alias = owner;
        owner.reset();
        REQUIRE(alive == 1);
        alias.reset();
        REQUIRE(alive == 0);
    }
}
