#ifndef HOPSY_KERNEL_GEOMETRY_SUPPORT_H
#define HOPSY_KERNEL_GEOMETRY_SUPPORT_H
#include <type_traits>
namespace hopsy {
    template <typename State, typename Direction, typename Line> class HitAndRunKernel;
    class UniformLineKernel;
    class GaussianLineKernel;
    template <typename Geometry>
    inline constexpr bool geometry_requires_hit_and_run = [] {
        if constexpr (requires { Geometry::hit_and_run_only; })
            return Geometry::hit_and_run_only;
        else
            return false;
    }();
    // Existing geometries retain their contracts. Marked geometries (including
    // subclasses) opt in only through the audited Hit-and-Run specialization.
    template <typename Kernel, typename Geometry>
    struct kernel_supports_geometry : std::bool_constant<!geometry_requires_hit_and_run<Geometry>> {
    };
    template <typename Line> struct line_kernel_supports_interval_unions : std::false_type {};
    template <> struct line_kernel_supports_interval_unions<UniformLineKernel> : std::true_type {};
    template <> struct line_kernel_supports_interval_unions<GaussianLineKernel> : std::true_type {};
    template <typename State, typename Direction, typename Line, typename Geometry>
    struct kernel_supports_geometry<HitAndRunKernel<State, Direction, Line>, Geometry>
        : std::bool_constant<!geometry_requires_hit_and_run<Geometry> ||
                             line_kernel_supports_interval_unions<Line>::value> {};
    template <typename Kernel, typename Geometry>
    inline constexpr bool kernel_supports_geometry_v =
        kernel_supports_geometry<std::remove_cvref_t<Kernel>, std::remove_cvref_t<Geometry>>::value;
} // namespace hopsy
#endif
