#include "common.h"
namespace hopsy::bindings {
    void bind_cycling_types(nb::module_ &m) {
        bind_hit_and_run<CyclingCoordinateDirectionGenerator, UniformLineKernel>(
            m, "UniformCoordinateHitAndRunKernel");
        bind_hit_and_run<CyclingCoordinateDirectionGenerator, GaussianLineKernel>(
            m, "GaussianCoordinateHitAndRunKernel");
        bind_hit_and_run<CyclingCoordinateDirectionGenerator, OverRelaxedLineKernel>(
            m, "OverRelaxedCoordinateHitAndRunKernel");
    }
} // namespace hopsy::bindings
