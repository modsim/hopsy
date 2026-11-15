#include "common.h"
namespace hopsy::bindings {
    void bind_gaussian_types(nb::module_ &m) {
        bind_gaussian<GaussianKernel<State, ZeroDrift, IdentityMetric>>(m,
                                                                        "GaussianRandomWalkKernel");
        bind_gaussian<DikinWalkKernel<State>>(m, "DikinWalkKernel");
        bind_gaussian<CSmMALAKernel<State>>(m, "CSmMALAKernel");
    }
} // namespace hopsy::bindings
