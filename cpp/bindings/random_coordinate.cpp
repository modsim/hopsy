#include "common.h"
namespace hopsy::bindings {
    void bind_random_coordinate_types(nb::module_ &m) {
        bind_hit_and_run<RandomCoordinateDirectionGenerator, UniformLineKernel>(
            m, "UniformRandomCoordinateHitAndRunKernel");
        bind_hit_and_run<RandomCoordinateDirectionGenerator, GaussianLineKernel>(
            m, "GaussianRandomCoordinateHitAndRunKernel");
        bind_hit_and_run<RandomCoordinateDirectionGenerator, OverRelaxedLineKernel>(
            m, "OverRelaxedRandomCoordinateHitAndRunKernel");
    }
} // namespace hopsy::bindings
