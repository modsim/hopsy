#include "common.h"
namespace hopsy::bindings {
    void bind_dense_types(nb::module_ &m) {
        bind_hit_and_run<RandomDirectionGenerator<double>, UniformLineKernel>(
            m, "UniformHitAndRunKernel");
        bind_hit_and_run<RandomDirectionGenerator<double>, GaussianLineKernel>(
            m, "GaussianHitAndRunKernel");
        bind_hit_and_run<RandomDirectionGenerator<double>, OverRelaxedLineKernel>(
            m, "OverRelaxedHitAndRunKernel");
    }
} // namespace hopsy::bindings
