#ifndef PROBLEM_H
#define PROBLEM_H

#include <concepts>

namespace hopsy {
    template <typename Geometry, typename TargetDensity>
    concept Problem =
        requires(
        const typename Geometry::state_type& state,
        const TargetDensity& target_density
    )
    {
        target.log_density(state);
    };
}

namespace hopsy {

#endif //PROBLEM_H
