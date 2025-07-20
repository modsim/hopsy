#pragma once
#include "helper.h"

namespace hopsy {
    namespace GPU {
        /**
         * @brief RAII Wrapper class for managing GPU memory allocation of PRNG state arrays.
         *
         * @tparam PRNGenerator The type representing the PRNG state (e.g., curandState).
         */
        template<typename PRNGenerator>
        struct PRNGState{
            PRNGenerator* states;
            public:
                PRNGState(int nThreads){ CUDA_CHECK(cudaMalloc((void **)&states, nThreads * sizeof(PRNGenerator))); }
                ~PRNGState(){ CUDA_CHECK(cudaFree(states)); }
        };

    }
}