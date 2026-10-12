#ifndef HIT_AND_RUN_H
#define HIT_AND_RUN_H

#include <cassert>
#include <utility>

#include <hopsy/kernel/direction.h>
#include <hopsy/kernel/traits.h>
#include <hopsy/random_number_generator.h>
#include <hopsy/target/traits.h>

namespace hopsy {
    // State is an owning vector. DirectionGenerator owns its direction buffer:
    // generate(rng) and direction() return the same retained direction.
    // Its direction selection must be state-independent (as in the supplied
    // generators); state-dependent direction laws need their own MH correction.
    template<typename State, typename DirectionGenerator, typename LineKernel>
    class HitAndRunKernel {
    public:
        using state_type = State;
        using scalar_type = typename State::Scalar;

        struct Proposal {
            State state;
            scalar_type step{};
            double log_proposal_ratio{};
        };

        template<typename Target>
        static constexpr bool always_accept =
                target_is_uniform_v<Target> && !line_kernel_requires_mh_v<LineKernel>;

        HitAndRunKernel(const State &initial_state,
                        DirectionGenerator direction_generator,
                        LineKernel line_kernel)
            : direction_generator_(std::move(direction_generator)),
              line_kernel_(std::move(line_kernel)),
              proposal_{.state = initial_state} {
        }

        // The proposal belongs to the kernel and is overwritten by propose().
        // After accept(), its state buffer contains the previous accepted state.
        [[nodiscard]] const Proposal &proposal() const noexcept { return proposal_; }

        // Keep the state and its workspace unchanged between propose() and the
        // corresponding accept()/reject(). In particular, another line_domain()
        // call must not overwrite workspace.direction_product while pending.
        template<typename Target, typename Geometry, typename Workspace>
        void propose(const State &state, RandomNumberGenerator &rng,
                     const Target &, const Geometry &geometry,
                     Workspace &workspace) {
            const auto &direction = direction_generator_.generate(rng);
            const auto &domain = geometry.line_domain(state, direction, workspace);
            proposal_.step = line_kernel_.propose(rng, domain);
            proposal_.log_proposal_ratio = line_kernel_.log_proposal_ratio(domain, proposal_.step);
            step_direction(direction, state, proposal_.step, proposal_.state);
        }

        template<typename Target, typename Geometry, typename Workspace>
        [[nodiscard]] double log_acceptance_ratio(
            const State &state, const Target &target,
            const Geometry &, const Workspace &) const {
            return static_cast<double>(target.log_density(proposal_.state)) -
                   static_cast<double>(target.log_density(state)) +
                   proposal_.log_proposal_ratio;
        }

        template<typename Target, typename Geometry, typename Workspace>
        void accept(State &state, const Target &, const Geometry &geometry,
                    Workspace &workspace) {
            geometry.advance_workspace(proposal_.step, direction_generator_.direction(), workspace);
            accept_direction(direction_generator_.direction(), state, proposal_.state);
        }

        template<typename Target, typename Geometry, typename Workspace>
        void reject(const State &state, const Target &, const Geometry &, Workspace &) noexcept {
            reject_direction(direction_generator_.direction(), state, proposal_.state);
        }

    private:
        [[no_unique_address]] DirectionGenerator direction_generator_;
        [[no_unique_address]] LineKernel line_kernel_;
        Proposal proposal_;
    };
}

#endif // HIT_AND_RUN_H
