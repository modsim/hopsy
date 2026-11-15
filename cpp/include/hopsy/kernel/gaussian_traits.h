#ifndef HOPSY_GAUSSIAN_TRAITS_H
#define HOPSY_GAUSSIAN_TRAITS_H

#include <type_traits>

namespace hopsy {
    struct ZeroDrift;
    struct IdentityMetric;

    // Symmetry means q(y|x) == q(x|y), not that a transition always accepts.
    // Unknown policy pairs conservatively require forward/reverse evaluation.
    template <typename Drift, typename Metric>
    struct gaussian_proposal_is_symmetric : std::false_type {};

    template <>
    struct gaussian_proposal_is_symmetric<ZeroDrift, IdentityMetric> : std::true_type {};

    template <typename Drift, typename Metric>
    inline constexpr bool gaussian_proposal_is_symmetric_v =
        gaussian_proposal_is_symmetric<Drift, Metric>::value;
} // namespace hopsy
#endif
