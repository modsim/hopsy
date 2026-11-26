#include "tempering.h"
namespace hopsy::bindings {
    void bind_cycling_types(nb::module_ &m) {
        bind_hit_and_run_with_tempering<CyclingCoordinateDirectionGenerator, UniformLineKernel>(
            m, "UniformCoordinateHitAndRunKernel");
        bind_hit_and_run_with_tempering<CyclingCoordinateDirectionGenerator, GaussianLineKernel>(
            m, "GaussianCoordinateHitAndRunKernel");
        bind_hit_and_run_with_tempering<CyclingCoordinateDirectionGenerator, OverRelaxedLineKernel>(
            m, "OverRelaxedCoordinateHitAndRunKernel");
    }
} // namespace hopsy::bindings
