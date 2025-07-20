#pragma once
#include "dmatrix.h"
#include <Eigen/Dense>


namespace hopsy {
    namespace GPU {
        
        struct CudaEvent{
            cudaEvent_t e;
            CudaEvent(){cudaEventCreate(&e);}
            CudaEvent(unsigned int flags){cudaEventCreateWithFlags(&e, flags);}
            ~CudaEvent(){cudaEventDestroy(e);}
            operator cudaEvent_t() const { return e; }
        };

        struct CudaStream{
            cudaStream_t s;
            CudaStream(){cudaStreamCreate(&s);}
            CudaStream(unsigned int flags){cudaStreamCreateWithFlags(&s, flags);}
            ~CudaStream(){cudaStreamDestroy(s);}
            operator cudaStream_t() const { return s; }
        };

        template <typename T>
        struct DevPtr{
            T* ptr;
            DevPtr(){   cudaMalloc(&ptr, sizeof(T));    }
            DevPtr(T val){  cudaMalloc(&ptr, sizeof(T)); cudaMemcpy(ptr, &val, sizeof(T), cudaMemcpyHostToDevice); }
            ~DevPtr(){  cudaFree(ptr);   }
            T toHost() const { T val; cudaMemcpy(&val, ptr, sizeof(T), cudaMemcpyDeviceToHost); return val; }
            void set(T val) { cudaMemcpy(ptr, &val, sizeof(T), cudaMemcpyHostToDevice); }
            operator T*() const { return ptr; }
        };

        template <typename Func>
        inline std::pair<int, int> getOptimalLaunchConfig(Func kernelFunc) {
            int maxBlocksPerSM = 0;
            int optimalThreadsPerBlock = 0;
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&maxBlocksPerSM, &optimalThreadsPerBlock, kernelFunc, 0, 0));
            return {optimalThreadsPerBlock, maxBlocksPerSM};
        }
    }
}