#ifndef HIT_AND_RUN_H
#define HIT_AND_RUN_H

#include <hopsy/random_number_generator.h>

namespace hopsy {

    template <typename DirectionGenerator, typename LineKernel, typename ProposalGeometry>
    class HitAndRunKernel {
    public:
        template <typename DirectionContextType, typename Direction, typename LineDomain>
        struct LineKernelContext {
            const DirectionContextType& direction_context;
            const Direction& direction;
            const LineDomain& domain;
        };

        template <typename State, typename Direction, typename LineTransition>
        struct Proposal {
            State state;
            Direction direction;
            LineTransition line_transition;
        };

        template <typename State, typename Target, typename DomainGeometry>
        [[nodiscard]] auto propose(
            const State& state,
            RandomNumberGenerator& rng,
            const Target& target,
            const DomainGeometry& domain_geometry
            // const ProposalGeometry& proposal_geometry
        ) {
            auto& direction = direction_generator_.generate(rng);

            auto& DomainGeometry::line_domain_type domain_geometry.line_domain(state, direction);

            line_kernel_.propose(rng);

            // return Proposal<State, decltype(move.direction), decltype(move.line_transition)>{
            //     .state = std::move(proposal_state),
            //     .direction = std::move(move.direction),
            //     .line_transition = std::move(move.line_transition)
            // };
        }

        template <typename State, typename Target, typename Geometry, typename Workspace>
        auto advance(
            State& state,
            RandomNumberGenerator& rng,
            const Target& target,
            const Geometry& geometry,
            Workspace& workspace
        ) {

        }

    private:
        [[no_unique_address]] DirectionGenerator direction_generator_;
        [[no_unique_address]] LineKernel line_kernel_;
    };

}

#endif //HIT_AND_RUN_H
