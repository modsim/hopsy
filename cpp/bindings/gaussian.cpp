#include "tempering.h"
namespace hopsy::bindings {
    void bind_gaussian_types(nb::module_ &m) {
        bind_gaussian_with_tempering<GaussianKernel<State, ZeroDrift, IdentityMetric>>(
            m, "GaussianRandomWalkKernel");
        bind_gaussian_with_tempering<DikinWalkKernel<State>>(m, "DikinWalkKernel");
        bind_gaussian_with_tempering<CSmMALAKernel<State>>(m, "CSmMALAKernel");
    }
} // namespace hopsy::bindings
