#ifndef HIT_AND_RUN_H
#define HIT_AND_RUN_H

namespace hopsy {

    template <typename DirectionGenerator, typename LineKernel>
    class HitAndRunKernel {
    public:
        template <typename State, typename Target, typename Geometry, typename Workspace>
        struct DirectionContext {
            const State& state;
            RandomNumberGenerator& rng;
            const Target& target;
            const Geometry& geometry;
            const Workspace& workspace;
        };

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

        template <typename State, typename Target, typename Geometry, typename Workspace>
        [[nodiscard]] auto propose(
            const State& state,
            RandomNumberGenerator& rng,
            const Target& target,
            const Geometry& geometry,
            const Workspace& workspace
        ) {
            auto move = draw_move(
                state,
                rng,
                target,
                geometry,
                workspace
            );

            State proposal_state = state;

            if (move.line_transition.accepted) {
                geometry.apply_step(
                    proposal_state,
                    move.direction,
                    move.line_transition.step
                );
            }

            return Proposal<State, decltype(move.direction), decltype(move.line_transition)>{
                .state = std::move(proposal_state),
                .direction = std::move(move.direction),
                .line_transition = std::move(move.line_transition)
            };
        }

        template <typename State, typename Target, typename Geometry, typename Workspace>
        auto advance(
            State& state,
            RandomNumberGenerator& rng,
            const Target& target,
            const Geometry& geometry,
            Workspace& workspace
        ) {
            auto move = draw_move(
                state,
                rng,
                target,
                geometry,
                workspace
            );

            if (move.line_transition.accepted) {
                geometry.apply_step(
                    state,
                    move.direction,
                    move.line_transition.step,
                    workspace
                );
            }

            return move.line_transition;
        }

    private:
        template <typename Direction, typename LineTransition>
        struct Move {
            Direction direction;
            LineTransition line_transition;
        };

        template <typename State, typename Target, typename Geometry, typename Workspace>
        [[nodiscard]] auto draw_move(
            const State& state,
            RandomNumberGenerator& rng,
            const Target& target,
            const Geometry& geometry,
            const Workspace& workspace
        ) {
            DirectionContext<State, Target, Geometry, Workspace> direction_context{
                .state = state,
                .rng = rng,
                .target = target,
                .geometry = geometry,
                .workspace = workspace
            };

            auto direction = direction_generator_.generate(direction_context);

            auto domain = geometry.line_domain(
                    state,
                    direction,
                    workspace
                );

            LineKernelContext<decltype(direction_context), decltype(direction), decltype(domain)> line_context{
                .direction_context = direction_context,
                .direction = direction,
                .domain = domain
            };

            auto line_transition =
                line_kernel_.advance(line_context);

            return Move<decltype(direction), decltype(line_transition)>{
                .direction = std::move(direction),
                .line_transition = std::move(line_transition)
            };
        }


        [[no_unique_address]] DirectionGenerator direction_generator_;
        [[no_unique_address]] LineKernel line_kernel_;
    };

}

#endif //HIT_AND_RUN_H
