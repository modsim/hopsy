#ifndef HOPSY_TARGET_TRAITS_H
#define HOPSY_TARGET_TRAITS_H

#include <type_traits>

namespace hopsy {
    template<typename>
    struct target_is_uniform : std::false_type {
    };

    template<typename Target>
    inline constexpr bool target_is_uniform_v =
            target_is_uniform<std::remove_cvref_t<Target> >::value;
}

#endif
