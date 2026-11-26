#include "tempering.h"
namespace hopsy::bindings {
    void bind_random_coordinate_types(nb::module_ &m) {
        bind_hit_and_run_with_tempering<RandomCoordinateDirectionGenerator, UniformLineKernel>(
            m, "UniformRandomCoordinateHitAndRunKernel");
        bind_hit_and_run_with_tempering<RandomCoordinateDirectionGenerator, GaussianLineKernel>(
            m, "GaussianRandomCoordinateHitAndRunKernel");
        bind_hit_and_run_with_tempering<RandomCoordinateDirectionGenerator, OverRelaxedLineKernel>(
            m, "OverRelaxedRandomCoordinateHitAndRunKernel");
    }
} // namespace hopsy::bindings
