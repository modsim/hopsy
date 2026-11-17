#ifndef HOPSY_TARGET_ABI_V1_H
#define HOPSY_TARGET_ABI_V1_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define HOPSY_TARGET_CAPSULE_NAME "hopsy.target.v1"
/* The producer owns the immutable descriptor and context until its capsule
 * destructor runs. Consumers retain the capsule throughout evaluation.
 * Set abi_version=1 and structure_size=sizeof(hopsy_target_v1).
 * dimension is fixed and positive. Density is required; Hessian requires gradient.
 *
 * x borrows dimension contiguous doubles for this call. out borrows one double
 * for log density, dimension doubles for its gradient, or dimension*dimension
 * doubles for its dense row-major Hessian. Derivatives are of log density.
 * Return 0 after writing all output; nonzero status reports failure.
 * Callbacks must not throw or use Python. Shared contexts must permit concurrent
 * calls; otherwise the caller must serialize access or use separate contexts.
 * No pointer or buffer may be retained beyond its documented lifetime. */
typedef int (*hopsy_target_callback_v1)(void *context, const double *x, double *out);
typedef struct hopsy_target_v1 {
    uint32_t abi_version;
    size_t structure_size;
    size_t dimension;
    void *context;
    hopsy_target_callback_v1 log_density;
    hopsy_target_callback_v1 log_gradient;
    hopsy_target_callback_v1 log_hessian;
} hopsy_target_v1;
#ifdef __cplusplus
}
#endif
#endif
