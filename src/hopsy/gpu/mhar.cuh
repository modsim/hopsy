#pragma once
#include "dmatrix.h"
#include "dvector.h"
#include "curandwrapper.h"
#include "cublaswrapper.h"
#include "cudawrappers.h"
#include "helper.h"
#include <cub/cub.cuh>
#include <cuda/std/limits>

namespace hopsy {
    namespace GPU {

        template <typename PRNGenerator>
        __global__ void initPrngStates(PRNGenerator *states, int seed);

        template <typename Real, typename PRNGenerator, int ThreadsPerBlock>
        __global__ void genRndDir(Real* A, int rows, int cols, PRNGenerator* global_states);

        template <typename Real, typename PRNGenerator>
        void launchGenRndDir(Real* A, int rows, int cols, PRNGenerator* global_states, int ThreadsPerBlock, int BlocksPerGrid, cudaStream_t stream=0);

        template <typename Real, typename PRNGenerator, int ThreadsPerBlock>
        __global__ void sampleStepSize(Real* AD, Real* slack, Real* alpha, Real* b, int rows, int cols, PRNGenerator* global_states);

        template <typename Real, typename PRNGenerator>
        void launchSampleStepSize(Real* AD, Real* slack, Real* alpha, Real* b, int rows, int cols, PRNGenerator* global_states, int ThreadsPerBlock, int BlocksPerGrid, cudaStream_t stream=0);

        template <typename Real>
        __global__ void XupdateNoSave(Real* X, Real* D, Real* alpha,  int rows, int cols);

        template <typename Real>
        __global__ void XupdateWithSaveIterantArray(Real *X, Real *D, Real *alpha, Real *Xsave, int rows, int cols, int *t, int thinning);
    }
}