#ifndef HOPSY_LOG_DENSITY_CACHE_H
#define HOPSY_LOG_DENSITY_CACHE_H

#include <functional>
#include <optional>
#include <hopsy/target/traits.h>

namespace hopsy::detail {
    // The owner starts a new proposal before stepping and commits its density
    // only if that proposal became the accepted configuration.
    class LogDensityCache {
      public:
        std::optional<double> state() const noexcept { return state_; }
        std::optional<double> proposal() const noexcept { return proposal_; }
        void set_state(double value) noexcept { state_ = value; }
        void clear_proposal() noexcept { proposal_.reset(); }
        void accept_proposal() noexcept { state_ = proposal_; }

        template <typename Evaluate> double state(Evaluate evaluate) {
            if (!state_)
                state_ = evaluate();
            return *state_;
        }

        template <typename Evaluate> double proposal(Evaluate evaluate) {
            if (!proposal_)
                proposal_ = evaluate();
            return *proposal_;
        }

      private:
        std::optional<double> state_, proposal_;
    };

    template <typename Kernel>
    concept CachesLogDensity = requires { requires Kernel::cache_log_density; };

    template <typename Target> class CachedTarget {
      public:
        CachedTarget(const Target &target, LogDensityCache &cache)
            : target_(target), cache_(cache) {}

        template <typename State> double log_density(const State &state) const {
            return static_cast<double>(target_.log_density(state));
        }

        // The transform is applied only to the ratio's operands. The shared
        // cache always holds values from the untransformed target.
        template <typename State, typename Transform = std::identity>
        double log_density_increment(const State &state, const State &proposal,
                                     Transform transform = {}) const {
            const double current = transform(cache_.state([&] { return log_density(state); }));
            const double candidate =
                transform(cache_.proposal([&] { return log_density(proposal); }));
            return candidate - current;
        }

        template <typename State>
            requires requires(const Target &target, const State &state) {
                target.log_gradient(state);
            }
        decltype(auto) log_gradient(const State &state) const {
            return target_.log_gradient(state);
        }

        template <typename State, typename Output>
            requires requires(const Target &target, const State &state, Output &out) {
                target.log_gradient(state, out);
            }
        void log_gradient(const State &state, Output &out) const {
            target_.log_gradient(state, out);
        }

      private:
        const Target &target_;
        LogDensityCache &cache_;
    };
} // namespace hopsy::detail

namespace hopsy {
    template <typename Target>
    struct target_is_uniform<detail::CachedTarget<Target>> : target_is_uniform<Target> {};
} // namespace hopsy

#endif
