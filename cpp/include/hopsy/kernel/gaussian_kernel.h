#ifndef GAUSSIAN_KERNEL_H
#define GAUSSIAN_KERNEL_H

#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <utility>

#include <hopsy/geometry/workspace.h>
#include <hopsy/kernel/drift.h>
#include <hopsy/kernel/gaussian_traits.h>
#include <hopsy/kernel/geometry_support.h>
#include <hopsy/kernel/metric.h>
#include <hopsy/target/log_density_cache.h>
#include <hopsy/target/traits.h>
#include <hopsy/random_number_generator.h>

namespace hopsy {
    // The caller supplies a valid state/kernel pair and immutable geometry.
    // BoundChain fixes the target and its cache lifetime. Advanced low-level
    // calls must keep the target fixed or explicitly invalidate between runs.
    template <typename State, typename Drift, typename Metric> class GaussianKernel {
      public:
        using state_type = State;
        using scalar_type = typename State::Scalar;
        static_assert(std::floating_point<scalar_type>);
        static constexpr bool cache_log_density = true;
        static constexpr bool symmetric_proposal = gaussian_proposal_is_symmetric_v<Drift, Metric>;
        template <typename Target> static constexpr bool always_accept = false;

        struct Proposal {
            State state;
            scalar_type step_size{};
        };

      private:
        enum class Phase { idle, proposed, evaluated };
        static constexpr bool zero_drift = std::same_as<Drift, ZeroDrift>;
        static constexpr bool quadratic_drift = [] {
            if constexpr (requires { Drift::scales_quadratically; })
                return Drift::scales_quadratically;
            else
                return false;
        }();
        struct NoReverseData {
            NoReverseData(const Metric &, const State &) noexcept {}
        };
        struct ReverseData {
            Metric metric;
            // For quadratic drifts these are coefficients of sigma^2, including
            // the 1/2 factor, not scaled means. Other policies use pending-only
            // displacements and do not reuse drift across transitions.
            State current_drift, candidate_drift, delta;
            bool current_metric_valid = false;
            bool candidate_metric_valid = false;
            bool current_drift_valid = false;
            bool candidate_drift_valid = false;
            ReverseData(const Metric &m, const State &x)
                : metric(m), delta(x) {
                if constexpr (!zero_drift) {
                    current_drift = x;
                    candidate_drift = x;
                }
            }
        };

      public:
        GaussianKernel(const State &initial_state, const Drift &drift, const Metric &metric,
                       scalar_type step_size)
            : drift_(drift), metric_(metric), step_size_(step_size),
              proposal_{.state = initial_state}, noise_(initial_state),
              reverse_(metric, initial_state) {
            check_step_size(step_size);
            // Zero-drift kernels never read drift buffers; leave dynamic vectors
            // empty instead of allocating state-sized storage for them.
            if constexpr (!zero_drift)
                drift_buffer_ = initial_state;
        }

        template <typename Geometry>
        static EmptyWorkspace make_workspace(const Geometry &, const State &) noexcept {
            return {};
        }

        // After accept(), proposal.state holds the previous accepted state.
        [[nodiscard]] const Proposal &proposal() const noexcept { return proposal_; }
        [[nodiscard]] bool has_pending_proposal() const noexcept { return phase_ != Phase::idle; }
        // Covariance is sigma^2 G(x)^-1. All target/drift caches are unscaled.
        [[nodiscard]] scalar_type step_size() const noexcept { return step_size_; }
        void set_step_size(scalar_type sigma) {
            require_idle();
            check_step_size(sigma);
            step_size_ = sigma;
        }

        // Call between transitions after changing the mathematical target,
        // including constraints/weights represented by the target. Geometry-only
        // metric factors remain valid. Changing state/geometry needs a new kernel.
        void invalidate_target_cache() {
            require_idle();
            clear_target_data();
        }

        // Cold reinitialization of a binding may replace state/geometry too.
        void invalidate_state_cache() {
            require_idle();
            clear_target_data();
            if constexpr (!symmetric_proposal) {
                reverse_.current_metric_valid = false;
                reverse_.candidate_metric_valid = false;
            }
        }

        template <typename Target, typename Geometry, typename Workspace>
        void propose(const State &state, RandomNumberGenerator &rng, const Target &target,
                     const Geometry &geometry, Workspace &workspace) {
            static_assert(kernel_supports_geometry_v<GaussianKernel, Geometry>,
                          "Gaussian-family kernels do not support BilinearPolytope");
            require_idle();
            if (state.size() != proposal_.state.size())
                throw std::invalid_argument("GaussianKernel state dimension changed");
            density_cache_.clear_proposal();
            proposal_.step_size = step_size_;
            if constexpr (symmetric_proposal) {
                metric_.prepare(state, geometry, std::as_const(workspace));
            } else {
                reverse_.candidate_metric_valid = false;
                reverse_.candidate_drift_valid = false;
                if (!reverse_.current_metric_valid) {
                    if (!metric_.prepare(state, geometry, std::as_const(workspace)))
                        throw std::domain_error("GaussianKernel requires a finite "
                                                "positive-definite metric at the current state");
                    reverse_.current_metric_valid = true;
                }
            }
            if constexpr (!zero_drift) {
                if constexpr (!symmetric_proposal && quadratic_drift) {
                    if (!reverse_.current_drift_valid) {
                        drift_.compute(state, scalar_type{1}, target, std::as_const(metric_),
                                       geometry, std::as_const(workspace), reverse_.current_drift);
                        reverse_.current_drift_valid = true;
                    }
                    drift_buffer_ = proposal_.step_size * reverse_.current_drift;
                    drift_buffer_ *= proposal_.step_size;
                } else {
                    drift_.compute(state, proposal_.step_size, target, std::as_const(metric_),
                                   geometry, std::as_const(workspace), drift_buffer_);
                }
            }
            metric_.sample(rng, noise_);
            if constexpr (zero_drift)
                proposal_.state = state + proposal_.step_size * noise_;
            else
                proposal_.state = state + drift_buffer_ + proposal_.step_size * noise_;
            phase_ = Phase::proposed;
        }

        template <typename Target, typename Geometry, typename Workspace>
        [[nodiscard]] double log_acceptance_ratio(const State &state, const Target &target,
                                                  const Geometry &geometry,
                                                  const Workspace &workspace) {
            if (!has_pending_proposal())
                throw std::logic_error("GaussianKernel has no pending proposal");
            phase_ = Phase::proposed;
            if (!proposal_.state.allFinite() || !geometry.is_feasible(proposal_.state))
                return -std::numeric_limits<double>::infinity();

            double target_ratio = 0.;
            if constexpr (!target_is_uniform_v<Target>) {
                if constexpr (requires { target.log_density_increment(state, proposal_.state); })
                    target_ratio = target.log_density_increment(state, proposal_.state);
                else
                    target_ratio = detail::CachedTarget(target, density_cache_)
                                       .log_density_increment(state, proposal_.state);
            }
            // Target support can be narrower than geometric support. Reject
            // zero/undefined target mass before reverse gradients/factorization.
            if (std::isnan(target_ratio) ||
                target_ratio == -std::numeric_limits<double>::infinity())
                return target_ratio;

            double correction = 0.;
            if constexpr (!symmetric_proposal) {
                if (!reverse_.candidate_metric_valid) {
                    if (!reverse_.metric.prepare(proposal_.state, geometry, workspace))
                        return -std::numeric_limits<double>::infinity();
                    reverse_.candidate_metric_valid = true;
                }
                if constexpr (!zero_drift) {
                    if (!reverse_.candidate_drift_valid) {
                        const auto scale = quadratic_drift ? scalar_type{1} : proposal_.step_size;
                        drift_.compute(proposal_.state, scale, target,
                                       std::as_const(reverse_.metric), geometry, workspace,
                                       reverse_.candidate_drift);
                        reverse_.candidate_drift_valid = true;
                    }
                    reverse_.delta =
                        (proposal_.state - state - drift_buffer_) / proposal_.step_size;
                } else {
                    reverse_.delta = (proposal_.state - state) / proposal_.step_size;
                }
                const double forward = metric_.squared_norm(reverse_.delta);
                reverse_.delta = (state - proposal_.state) / proposal_.step_size;
                if constexpr (!zero_drift) {
                    if constexpr (quadratic_drift)
                        reverse_.delta -= proposal_.step_size * reverse_.candidate_drift;
                    else
                        reverse_.delta -= reverse_.candidate_drift / proposal_.step_size;
                }
                correction = reverse_.metric.log_sqrt_determinant() -
                             metric_.log_sqrt_determinant() +
                             0.5 * (forward - reverse_.metric.squared_norm(reverse_.delta));
            }
            const double ratio = target_ratio + correction;
            // Accept requires a usable evaluation; reject can cancel any proposal.
            if (!std::isnan(ratio) && ratio != -std::numeric_limits<double>::infinity())
                phase_ = Phase::evaluated;
            return ratio;
        }

        template <typename Geometry, typename Workspace>
        void accept(State &state, const Geometry &geometry, Workspace &workspace) {
            if (phase_ != Phase::evaluated)
                throw std::logic_error("GaussianKernel accept requires a usable MH evaluation");
            geometry.reset_workspace(proposal_.state, workspace);
            state.swap(proposal_.state);
            density_cache_.accept_proposal();
            if constexpr (!symmetric_proposal) {
                std::swap(metric_, reverse_.metric);
                reverse_.current_metric_valid = true;
                if constexpr (!zero_drift && quadratic_drift) {
                    reverse_.current_drift.swap(reverse_.candidate_drift);
                    reverse_.current_drift_valid = reverse_.candidate_drift_valid;
                }
            }
            finish_proposal();
        }

        // Idempotent cancellation is also safe after a target evaluation throws.
        void reject(const State &) noexcept { finish_proposal(); }

      private:
        void require_idle() const {
            if (has_pending_proposal())
                throw std::logic_error("GaussianKernel has a pending proposal");
        }
        void clear_target_data() noexcept {
            density_cache_ = {};
            if constexpr (!symmetric_proposal) {
                reverse_.current_drift_valid = false;
                reverse_.candidate_drift_valid = false;
            }
        }
        void finish_proposal() noexcept {
            phase_ = Phase::idle;
            density_cache_.clear_proposal();
            if constexpr (!symmetric_proposal) {
                reverse_.candidate_metric_valid = false;
                reverse_.candidate_drift_valid = false;
            }
        }
        static void check_step_size(scalar_type sigma) {
            if (!std::isfinite(sigma) || sigma <= scalar_type{0})
                throw std::invalid_argument("GaussianKernel step size must be finite and positive");
        }
        Drift drift_;
        Metric metric_;
        scalar_type step_size_;
        Proposal proposal_;
        State noise_, drift_buffer_;
        [[no_unique_address]] std::conditional_t<symmetric_proposal, NoReverseData, ReverseData>
            reverse_;
        Phase phase_ = Phase::idle;
        // Used by direct kernel callers. Bound sampling supplies its own cache.
        detail::LogDensityCache density_cache_;
    };
    template <typename State>
    using DikinWalkKernel = GaussianKernel<State, ZeroDrift, DikinMetric<typename State::Scalar>>;
    // Dikin-metric CSmMALA; Fisher mixing remains a separate feature.
    template <typename State>
    using CSmMALAKernel =
        GaussianKernel<State, ClippedGradientDrift, DikinMetric<typename State::Scalar>>;
} // namespace hopsy
#endif // GAUSSIAN_KERNEL_H
