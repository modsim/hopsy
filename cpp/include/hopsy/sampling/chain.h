#ifndef HOPSY_BOUND_CHAIN_H
#define HOPSY_BOUND_CHAIN_H

#include <functional>
#include <stdexcept>
#include <utility>
#include <hopsy/problem/problem.h>
#include <hopsy/sampling/step.h>

namespace hopsy {
    namespace detail {
        template <typename P, typename K, typename S>
        void validate_binding(const P &problem, const K &kernel, const S &state) {
            if (state.size() == 0 || !state.allFinite() ||
                state.size() != kernel.proposal().state.size())
                throw std::invalid_argument(
                    "BoundChain requires a finite state matching the kernel dimension");
            if constexpr (requires { problem.geometry().num_dimension(); }) {
                if (state.size() != problem.geometry().num_dimension())
                    throw std::invalid_argument(
                        "BoundChain state and sampling geometry dimensions differ");
            }
            if constexpr (requires { problem.coordinates().sampling_dimension(); }) {
                if (state.size() != problem.coordinates().sampling_dimension())
                    throw std::invalid_argument(
                        "BoundChain state and coordinate map dimensions differ");
            }
            if (!problem.geometry().is_feasible(state))
                throw std::invalid_argument(
                    "BoundChain initial state is outside the sampling geometry");
        }
    } // namespace detail
    // Exclusive, typed binding of an existing state/kernel pair to one immutable
    // problem. Caller-owned objects must outlive this view and must not be used
    // through another binding concurrently. Changing geometry needs a fresh
    // kernel/workspace. Rebinding a target starts a new cache lifetime once.
    template <typename Problem, typename Kernel, typename Workspace> class BoundChain {
      public:
        using state_type = typename Kernel::state_type;
        using scalar_type = typename Kernel::scalar_type;
        template <bool Validate = true>
        BoundChain(const Problem &problem, Kernel &kernel, state_type &state, Workspace workspace,
                   std::bool_constant<Validate> = {})
            : problem_(problem), kernel_(kernel), state_(state), workspace_(std::move(workspace)),
              evaluation_(problem, state) {
            if constexpr (Validate)
                detail::validate_binding(problem, kernel, state);
            if constexpr (requires { kernel.invalidate_target_cache(); })
                kernel.invalidate_target_cache();
        }
        BoundChain(const Problem &&, Kernel &, state_type &, Workspace) = delete;
        BoundChain(const BoundChain &) = delete;
        BoundChain &operator=(const BoundChain &) = delete;
        BoundChain(BoundChain &&) = default;
        BoundChain &operator=(BoundChain &&) = delete;

        bool step(RandomNumberGenerator &rng) {
            // Always-accepted line kernels never prepare a target-dependent
            // reverse distribution. Keep exception cleanup out of that path.
            using Target = std::remove_cvref_t<decltype(sampling_target())>;
            if constexpr (Kernel::template always_accept<Target>) {
                return hopsy::step(kernel_, state_, rng, sampling_target(), problem_.geometry(),
                                   mutable_workspace());
            } else {
                try {
                    return hopsy::step(kernel_, state_, rng, sampling_target(), problem_.geometry(),
                                       mutable_workspace());
                } catch (...) {
                    kernel_.reject(state_);
                    throw;
                }
            }
        }
        const state_type &sampling_state() const noexcept { return state_; }
        Vector<scalar_type> target_state() const {
            Vector<scalar_type> result(target_dimension());
            problem_.coordinates().to_target(state_, result);
            return result;
        }
        Index sampling_dimension() const noexcept { return state_.size(); }
        Index target_dimension() const noexcept {
            return problem_.coordinates().target_dimension(state_.size());
        }
        decltype(auto) sampling_target() noexcept { return evaluation_.target(problem_); }
        const auto &geometry_workspace() const noexcept {
            return std::as_const(detail::unwrap(workspace_));
        }
        const Kernel &kernel() const noexcept { return kernel_; }
        auto step_size() const
            requires requires(const Kernel &k) { k.step_size(); }
        {
            return kernel_.step_size();
        }
        void set_step_size(scalar_type sigma)
            requires requires(Kernel &k, scalar_type value) { k.set_step_size(value); }
        {
            kernel_.set_step_size(sigma);
        }

        template <typename Derived>
        void draw_into(RandomNumberGenerator &rng, Eigen::MatrixBase<Derived> &samples,
                       Index thinning = 1) {
            draw_into_impl<false>(rng, samples, thinning);
        }
        template <typename Derived>
        void draw_sampling_into(RandomNumberGenerator &rng, Eigen::MatrixBase<Derived> &samples,
                                Index thinning = 1) {
            draw_into_impl<true>(rng, samples, thinning);
        }
        Matrix<scalar_type> draw(RandomNumberGenerator &rng, Index count, Index thinning = 1) {
            if (count < 0 || thinning < 1)
                throw std::invalid_argument("Invalid sample count or thinning");
            Matrix<scalar_type> result(target_dimension(), count);
            draw_into(rng, result, thinning);
            return result;
        }
        Matrix<scalar_type> draw_sampling(RandomNumberGenerator &rng, Index count,
                                          Index thinning = 1) {
            if (count < 0 || thinning < 1)
                throw std::invalid_argument("Invalid sample count or thinning");
            Matrix<scalar_type> result(sampling_dimension(), count);
            draw_sampling_into(rng, result, thinning);
            return result;
        }

      private:
        auto &mutable_workspace() noexcept { return detail::unwrap(workspace_); }
        template <bool Sampling, typename Derived>
        void draw_into_impl(RandomNumberGenerator &rng, Eigen::MatrixBase<Derived> &samples,
                            Index thinning) {
            const Index rows = Sampling ? sampling_dimension() : target_dimension();
            if (thinning < 1 || samples.rows() != rows)
                throw std::invalid_argument("Invalid output dimension or thinning");
            for (Index sample = 0; sample < samples.cols(); ++sample) {
                for (Index iteration = 0; iteration < thinning; ++iteration)
                    step(rng);
                if constexpr (Sampling || Problem::identity_coordinates)
                    samples.col(sample) = state_;
                else {
                    auto column = samples.col(sample);
                    problem_.coordinates().to_target(state_, column);
                }
            }
        }
        const Problem &problem_;
        Kernel &kernel_;
        state_type &state_;
        Workspace workspace_;
        [[no_unique_address]] detail::TargetEvaluation<Problem, state_type> evaluation_;
    };

    template <typename P, typename K, typename W>
    auto bind_chain(const P &problem, K &kernel, typename K::state_type &state, W &workspace) {
        return BoundChain<P, K, std::reference_wrapper<W>>(problem, kernel, state,
                                                           std::ref(workspace));
    }
    template <typename P, typename K>
    auto bind_chain(const P &problem, K &kernel, typename K::state_type &state) {
        detail::validate_binding(problem, kernel, state);
        auto workspace = K::make_workspace(problem.geometry(), state);
        return BoundChain<P, K, decltype(workspace)>(problem, kernel, state, std::move(workspace),
                                                     std::false_type{});
    }
    namespace detail {
        // Owning frontends have already validated and encapsulated state and
        // geometry. Rebind only the target; do not repeat O(m*n) feasibility
        // checks on every Python single-step call.
        template <typename P, typename K, typename W>
        auto bind_existing_chain(const P &problem, K &kernel, typename K::state_type &state,
                                 W &workspace) {
            return BoundChain<P, K, std::reference_wrapper<W>>(
                problem, kernel, state, std::ref(workspace), std::false_type{});
        }
    } // namespace detail
    // Borrowing a temporary problem would leave dangling references.
    template <typename P, typename K, typename... Args>
    auto bind_chain(const P &&, K &, typename K::state_type &, Args &&...) = delete;
} // namespace hopsy
#endif
