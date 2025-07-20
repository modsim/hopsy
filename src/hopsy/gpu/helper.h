#pragma once    
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

// Column major matrix indexing
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

// CUDA API error checking
#define CUDA_CHECK(cuda_error) \
    do { \
        cudaError_t err = cuda_error; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while (0)

// CUBLAS API error checking
#define CUBLAS_CHECK(cublas_error) \
    do { \
        cublasStatus_t err = cublas_error; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error: " << cublasGetStatusString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while (0)


// CURAND API error checking
#define CURAND_CHECK(err)                                                      \
  do {                                                                         \
    curandStatus_t err_ = (err);                                               \
    if (err_ != CURAND_STATUS_SUCCESS) {                                       \
      std::printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("curand error");                                \
    }                                                                          \
  } while (0)
