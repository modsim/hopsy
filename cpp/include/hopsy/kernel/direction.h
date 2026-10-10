#ifndef DIRECTION_H
#define DIRECTION_H

#include <hopsy/linalg.h>
#include <cassert>

namespace hopsy {
    struct CoordinateDirection {
        Index coordinate;
    };

    template<typename State, typename Scalar, typename Output>
    void step_direction(const CoordinateDirection &direction, const State &state,
                        Scalar distance, Output &output) {
        assert(direction.coordinate >= 0 && direction.coordinate < state.size());
        output(direction.coordinate) = state(direction.coordinate) + distance;
    }

    template<typename Derived, typename State, typename Scalar, typename Output>
    void step_direction(const Eigen::MatrixBase<Derived> &direction, const State &state,
                        Scalar distance, Output &output) {
        assert(direction.size() == state.size());
        output.noalias() = state + distance * direction;
    }

    template<typename State>
    void accept_direction(
        const CoordinateDirection& direction,
        State& state,
        State& candidate
    ) {
        const auto i = direction.coordinate;
        state(i) = candidate(i);
    }

    template<typename State>
    void reject_direction(
        const CoordinateDirection& direction,
        const State& state,
        State& candidate
    ) {
        const auto i = direction.coordinate;
        candidate(i) = state(i);
    }

    template<typename Derived, typename State>
    void accept_direction(
        const Eigen::MatrixBase<Derived>&,
        State& state,
        State& candidate
    ) {
        state.swap(candidate);
    }

    template<typename Derived, typename State>
    void reject_direction(
        const Eigen::MatrixBase<Derived>&,
        const State&,
        State&
    ) noexcept {}
}

#endif // DIRECTION_H
