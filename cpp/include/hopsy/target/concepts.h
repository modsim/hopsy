#ifndef TARGET_H
#define TARGET_H

#include <concepts>

namespace hopsy {
    template <typename Target, typename State>
    concept TargetDensity =
        requires(
        const Target& target,
        const State& state
    )
    {
            target.log_density(state);
    };

    template <typename Target, typename State>
    concept DifferentiableTargetDensity =
        TargetDensity<Target, State> &&
        requires(
        const Target& target,
        const State& state
    )
    {
        target.log_gradient(state);
    };

    template <typename Target, typename State>
    concept TwiceDifferentiableTargetDensity =
        DifferentiableTargetDensity<Target, State> &&
        requires(
        const Target& target,
        const State& state
    )
    {
        target.log_hessian(state);
    };
}

#endif //TARGET_H
