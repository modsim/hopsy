#ifndef TRAITS_H
#define TRAITS_H

#include <type_traits>

namespace hopsy {
    struct line_kernel_requires_mh : std::true_type {
    };

    template<typename LineKernel>
    inline constexpr bool line_kernel_requires_mh_v =
            line_kernel_requires_mh<std::remove_cvref_t<LineKernel> >::value;
}

#endif //TRAITS_H
