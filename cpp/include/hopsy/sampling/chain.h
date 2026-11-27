#ifndef HOPSY_BOUND_CHAIN_H
#define HOPSY_BOUND_CHAIN_H

#include <functional>
#include <stdexcept>
#include <utility>
#include <hopsy/problem/problem.h>
#include <hopsy/sampling/draw.h>
#include <hopsy/kernel/geometry_support.h>

namespace hopsy {
    namespace detail {
        template <typename P, typename K, typename S>
        void validate_binding(const P &problem, const K &kernel, const S &state) {
            if constexpr (!kernel_supports_geometry_v<K, typename P::geometry_type>)
                throw std::invalid_argument(
                    "kernel does not support this geometry: BilinearPolytope requires uniform or "
                    "Gaussian-displacement Hit-and-Run; Gaussian-family and overrelaxed kernels "
                    "are unsupported");
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
            return with_step(rng, [](const auto &step) { return step(); });
        }
        const state_type &sampling_state() const noexcept { return state_; }
        std::optional<double> state_log_density() const noexcept { return density_cache_.state(); }
        std::optional<double> proposal_log_density() const noexcept {
            return density_cache_.proposal();
        }
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
            draw_into_impl<false>(rng, samples, thinning, detail::ignore_sample);
        }
        template <typename Derived>
        void draw_sampling_into(RandomNumberGenerator &rng, Eigen::MatrixBase<Derived> &samples,
                                Index thinning = 1) {
            draw_into_impl<true>(rng, samples, thinning, detail::ignore_sample);
        }
        Matrix<scalar_type> draw(RandomNumberGenerator &rng, Index count, Index thinning = 1) {
            return draw_impl<false>(rng, count, thinning, detail::ignore_sample);
        }
        Matrix<scalar_type> draw_sampling(RandomNumberGenerator &rng, Index count,
                                          Index thinning = 1) {
            return draw_impl<true>(rng, count, thinning, detail::ignore_sample);
        }

        // Observe each retained draw after thinning. The callback can read the
        // state and its caches through the const binding.
        template <typename Observe>
        Matrix<scalar_type> draw_observed(RandomNumberGenerator &rng, Index count, Index thinning,
                                           const Observe &observe) {
            return draw_impl<false>(rng, count, thinning, observe);
        }

      private:
        auto &mutable_workspace() noexcept { return detail::unwrap(workspace_); }
        decltype(auto) transition_target() noexcept {
            if constexpr (detail::CachesLogDensity<Kernel> &&
                          !target_is_uniform_v<decltype(sampling_target())>)
                return detail::CachedTarget(sampling_target(), density_cache_);
            else
                return sampling_target();
        }
        bool transition(RandomNumberGenerator &rng, const auto &target, const auto &geometry,
                        auto &workspace) {
            using Target = std::remove_cvref_t<decltype(target)>;
            if constexpr (detail::CachesLogDensity<Kernel> && !target_is_uniform_v<Target> &&
                          !Kernel::template always_accept<Target>) {
                return detail::step_with_density_cache(kernel_, state_, rng, target, geometry,
                                                       workspace, density_cache_);
            } else {
                return hopsy::step(kernel_, state_, rng, target, geometry, workspace);
            }
        }
        template <typename Run> decltype(auto) with_step(RandomNumberGenerator &rng, const Run &run) {
            // Resolve the bound target and workspace once per batch.
            decltype(auto) target = transition_target();
            const auto &geometry = problem_.geometry();
            auto &workspace = mutable_workspace();
            const auto step = [&] { return transition(rng, target, geometry, workspace); };
            using Target = std::remove_cvref_t<decltype(target)>;
            if constexpr (Kernel::template always_accept<Target>) {
                return run(step);
            } else {
                try {
                    return run(step);
                } catch (...) {
                    detail::cancel_transition(kernel_, state_, density_cache_);
                    throw;
                }
            }
        }

        template <bool Sampling, typename Observe>
        Matrix<scalar_type> draw_impl(RandomNumberGenerator &rng, Index count, Index thinning,
                                      const Observe &observe) {
            if (count < 0 || thinning < 1)
                throw std::invalid_argument("Invalid sample count or thinning");
            Matrix<scalar_type> samples(Sampling ? sampling_dimension() : target_dimension(), count);
            draw_into_impl<Sampling>(rng, samples, thinning, observe);
            return samples;
        }

        template <bool Sampling, typename Derived, typename Observe>
        void draw_into_impl(RandomNumberGenerator &rng, Eigen::MatrixBase<Derived> &samples,
                            Index thinning, const Observe &observe) {
            const Index rows = Sampling ? sampling_dimension() : target_dimension();
            if (thinning < 1 || samples.rows() != rows)
                throw std::invalid_argument("Invalid output dimension or thinning");
            with_step(rng, [&](const auto &step) {
                detail::collect_draws(samples.cols(), thinning, step, [&](Index sample) {
                    if constexpr (Sampling || Problem::identity_coordinates)
                        samples.col(sample) = state_;
                    else {
                        auto column = samples.col(sample);
                        problem_.coordinates().to_target(state_, column);
                    }
                    observe(sample, std::as_const(*this));
                });
            });
        }
        const Problem &problem_;
        Kernel &kernel_;
        state_type &state_;
        Workspace workspace_;
        [[no_unique_address]] detail::TargetEvaluation<Problem, state_type> evaluation_;
        detail::LogDensityCache density_cache_;
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
    // Borrowing a temporary problem would leave dangling references.
    template <typename P, typename K, typename... Args>
    auto bind_chain(const P &&, K &, typename K::state_type &, Args &&...) = delete;
} // namespace hopsy
#endif
