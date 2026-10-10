#ifndef HIT_AND_RUN_H
#define HIT_AND_RUN_H

#include <hopsy/kernel/traits.h>
#include <hopsy/random_number_generator.h>

namespace hopsy {
    template<typename State, typename DirectionGenerator, typename LineKernel>
    class HitAndRunKernel {
    public:
        struct Proposal {
            DirectionGenerator::direction_type direction;
            LineKernel::step_type step;
        };

        template<typename DomainGeometry>
        [[nodiscard]] auto propose(
            const State &state,
            RandomNumberGenerator &rng,
            const DomainGeometry &domain_geometry
        ) {
            Proposal proposal;
            proposal.direction.noalias() = direction_generator_.generate(rng);
            auto &line_domain = domain_geometry.line_domain(state, direction);
            auto step = line_kernel_.propose(rng, line_domain);

            return proposal;
        }

        double log_acceptance_probability() {
            if constexpr (requires_metropolis_hastings_v<LineKernel>) {
                return line_kernel_.log_proposal_ratio(line_domain, proposal.step);
            } else {
                return 0.;
            }
        }

        template<typename Geometry, typename Workspace>
        auto accept(Proposal &proposal, Geometry &geometry, Workspace &workspace) {
            geometry.advance_workspace(proposal.step, proposal.direction, workspace);
            // hit and run (we should only compute this
            state + proposal.direction * proposal.step;
            // coordinate hit and run
            state(coordinate) += proposal.step;
        }

        auto reject() {
        }

    private:
        [[no_unique_address]] DirectionGenerator direction_generator_;
        [[no_unique_address]] LineKernel line_kernel_;
    };
}

#endif //HIT_AND_RUN_H
