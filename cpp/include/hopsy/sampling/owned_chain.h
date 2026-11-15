#ifndef HOPSY_OWNED_CHAIN_H
#define HOPSY_OWNED_CHAIN_H

#include <memory>
#include <mutex>
#include <unordered_set>
#include <hopsy/sampling/chain.h>

namespace hopsy {
    namespace detail {
        // Keep independently mutated resources off each other's cache lines.
        // This changes allocation only, not the kernel/RNG type or hot loop.
        template <typename T> struct alignas(alignof(T) > 64 ? alignof(T) : 64) SamplingResource {
            template <typename... Args>
            explicit SamplingResource(Args &&...args) : value(std::forward<Args>(args)...) {}
            T value;
        };
    } // namespace detail
    template <typename T, typename... Args>
    std::shared_ptr<T> make_sampling_resource(Args &&...args) {
        auto owner = std::make_shared<detail::SamplingResource<T>>(std::forward<Args>(args)...);
        return std::shared_ptr<T>(owner, &owner->value);
    }
    // Boundary-only exclusion for borrowed mutable resources. No registry lookup
    // occurs inside a transition loop. Raw core APIs retain their caller-managed
    // exclusivity contract; this guard is used by the owning frontend.
    template <typename Resource> class ExclusiveLease {
      public:
        explicit ExclusiveLease(Resource &resource) : resource_(&resource) {
            std::lock_guard guard(mutex_);
            if (!active_.insert(resource_).second)
                throw std::runtime_error("resource is already in use or bound to a chain");
        }
        ~ExclusiveLease() {
            std::lock_guard guard(mutex_);
            active_.erase(resource_);
        }
        ExclusiveLease(const ExclusiveLease &) = delete;
        ExclusiveLease &operator=(const ExclusiveLease &) = delete;

      private:
        Resource *resource_;
        inline static std::mutex mutex_;
        inline static std::unordered_set<Resource *> active_;
    };

    // Own accepted-state storage and its workspace, borrow kernel/problem.
    // Nonmovable: the persistent binding references these exact member objects.
    // The borrowed objects must outlive this owner. An rvalue state selects this
    // owning overload; the original bind_chain(lvalue state) remains a view.
    template <typename Problem, typename Kernel> class OwnedStateChain {
      public:
        using state_type = typename Kernel::state_type;
        using scalar_type = typename Kernel::scalar_type;
        using workspace_type = decltype(Kernel::make_workspace(
            std::declval<const Problem &>().geometry(), std::declval<const state_type &>()));
        using binding_type = BoundChain<Problem, Kernel, std::reference_wrapper<workspace_type>>;
        OwnedStateChain(const Problem &problem, Kernel &kernel, state_type initial)
            : lease_(kernel), problem_(&problem), kernel_(kernel),
              state_(validated(problem, kernel, std::move(initial))),
              sampling_dimension_(state_.size()),
              target_dimension_(problem.coordinates().target_dimension(state_.size())),
              workspace_(Kernel::make_workspace(problem.geometry(), state_)),
              binding_(initial_binding(problem)) {}
        OwnedStateChain(std::shared_ptr<const Problem> problem, Kernel &kernel, state_type initial)
            : OwnedStateChain(require_problem(problem), kernel, std::move(initial)) {
            problem_owner_ = std::move(problem);
        }
        OwnedStateChain(const OwnedStateChain &) = delete;
        OwnedStateChain(const Problem &&, Kernel &, state_type) = delete;
        OwnedStateChain &operator=(const OwnedStateChain &) = delete;
        OwnedStateChain(OwnedStateChain &&) = delete;
        OwnedStateChain &operator=(OwnedStateChain &&) = delete;

        bool step(RandomNumberGenerator &rng) {
            return run(rng, [&] { return binding_->step(rng); });
        }
        Matrix<scalar_type> draw(RandomNumberGenerator &rng, Index count, Index thinning = 1) {
            return run(rng, [&] { return binding_->draw(rng, count, thinning); });
        }
        Matrix<scalar_type> draw_sampling(RandomNumberGenerator &rng, Index count,
                                          Index thinning = 1) {
            return run(rng, [&] { return binding_->draw_sampling(rng, count, thinning); });
        }
        template <typename Derived>
        void draw_into(RandomNumberGenerator &rng, Eigen::MatrixBase<Derived> &out,
                       Index thinning = 1) {
            run(rng, [&] { binding_->draw_into(rng, out, thinning); });
        }
        template <typename Derived>
        void draw_sampling_into(RandomNumberGenerator &rng, Eigen::MatrixBase<Derived> &out,
                                Index thinning = 1) {
            run(rng, [&] { binding_->draw_sampling_into(rng, out, thinning); });
        }
        state_type sampling_state() const {
            auto guard = lock();
            return state_;
        }
        Vector<scalar_type> target_state() const {
            auto guard = lock();
            return binding_->target_state();
        }
        Index sampling_dimension() const noexcept { return sampling_dimension_; }
        Index target_dimension() const noexcept { return target_dimension_; }
        scalar_type step_size() const
            requires requires(const Kernel &k) { k.step_size(); }
        {
            auto guard = lock();
            return kernel_.step_size();
        }
        void set_step_size(scalar_type sigma)
            requires requires(Kernel &k) { k.set_step_size(sigma); }
        {
            auto guard = lock();
            kernel_.set_step_size(sigma);
        }
        // A same-type target replacement is cold and explicit. Geometry and maps
        // remain identical objects, not merely dimensionally compatible ones.
        void rebind_problem(const Problem &problem) {
            auto guard = lock();
            rebind(problem);
            if (problem_owner_.get() != &problem)
                problem_owner_.reset();
        }
        void rebind_problem(std::shared_ptr<const Problem> problem) {
            auto guard = lock();
            rebind(require_problem(problem));
            problem_owner_ = std::move(problem);
        }

      private:
        static const Problem &require_problem(const std::shared_ptr<const Problem> &p) {
            if (!p)
                throw std::invalid_argument("problem must not be null");
            return *p;
        }
        void rebind(const Problem &problem) {
            if (&problem.geometry() != &problem_->geometry())
                throw std::invalid_argument("rebinding requires the same geometry object");
            if constexpr (!Problem::identity_coordinates) {
                if (&problem.coordinates() != &problem_->coordinates())
                    throw std::invalid_argument(
                        "rebinding requires the same coordinate map object");
            }
            auto replacement = make_binding(problem);
            binding_ = std::move(replacement);
            problem_ = &problem;
        }
        static state_type validated(const Problem &p, const Kernel &k, state_type state) {
            detail::validate_binding(p, k, state);
            if (state != k.proposal().state)
                throw std::invalid_argument("initial state must match the kernel candidate; "
                                            "construct a fresh kernel for a new state");
            return state;
        }
        std::unique_ptr<binding_type> initial_binding(const Problem &p) {
            if constexpr (requires { kernel_.invalidate_state_cache(); })
                kernel_.invalidate_state_cache();
            return make_binding(p);
        }
        std::unique_ptr<binding_type> make_binding(const Problem &p) {
            return std::make_unique<binding_type>(p, kernel_, state_, std::ref(workspace_),
                                                  std::false_type{});
        }
        auto lock() const {
            std::unique_lock guard(mutex_, std::try_to_lock);
            if (!guard.owns_lock())
                throw std::runtime_error(
                    "chain is already in use (concurrent or reentrant access)");
            return guard;
        }
        template <typename F> decltype(auto) run(RandomNumberGenerator &rng, F &&f) {
            auto guard = lock();
            ExclusiveLease<RandomNumberGenerator> rng_lease(rng);
            return std::forward<F>(f)();
        }
        ExclusiveLease<Kernel> lease_;
        const Problem *problem_;
        std::shared_ptr<const Problem> problem_owner_;
        Kernel &kernel_;
        state_type state_;
        const Index sampling_dimension_, target_dimension_;
        workspace_type workspace_;
        std::unique_ptr<binding_type> binding_;
        mutable std::mutex mutex_;
    };

    template <typename P, typename K>
    auto bind_chain(const P &problem, K &kernel, typename K::state_type &&state) {
        return std::make_unique<OwnedStateChain<P, K>>(problem, kernel, std::move(state));
    }
    template <typename P, typename K>
    auto bind_chain(std::shared_ptr<const P> problem, K &kernel, typename K::state_type &&state) {
        return std::make_unique<OwnedStateChain<P, K>>(std::move(problem), kernel,
                                                       std::move(state));
    }
    template <typename P, typename K>
        requires requires { typename P::geometry_type; }
    auto bind_chain(const P &&, K &, typename K::state_type &&) = delete;
} // namespace hopsy
#endif
