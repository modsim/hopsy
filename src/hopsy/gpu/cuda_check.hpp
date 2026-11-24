#ifndef HOPSY_GPU_CUDA_CHECK_HPP
#define HOPSY_GPU_CUDA_CHECK_HPP

// Error checking for CUDA runtime, cuBLAS and cuRAND calls.
//
// The *_CHECK macros throw std::runtime_error, which pybind11 turns into a
// Python RuntimeError. The *_CHECK_NOEXCEPT variants only report to stderr and
// are meant for destructors and other cleanup paths that must not throw.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace hopsy {
    namespace gpu {
        namespace detail {
            inline std::string errorLocation(const char *file, int line) {
                return std::string(" at ") + file + ":" + std::to_string(line);
            }

            inline std::string cudaErrorMessage(cudaError_t err, const char *call,
                                                const char *file, int line) {
                return std::string(call) + " failed (" + cudaGetErrorName(err) +
                       ": " + cudaGetErrorString(err) + ")" + errorLocation(file, line);
            }

            inline std::string cublasErrorMessage(cublasStatus_t status, const char *call,
                                                  const char *file, int line) {
                return std::string(call) + " failed (" + cublasGetStatusName(status) +
                       ": " + cublasGetStatusString(status) + ")" +
                       errorLocation(file, line);
            }

            // cuRAND has no status-to-string function.
            inline std::string curandErrorMessage(curandStatus_t status, const char *call,
                                                  const char *file, int line) {
                return std::string(call) + " failed (curandStatus_t " +
                       std::to_string(static_cast<int>(status)) + ")" +
                       errorLocation(file, line);
            }

            inline void cudaCheck(cudaError_t err, const char *call, const char *file,
                                  int line) {
                if (err != cudaSuccess) {
                    throw std::runtime_error(cudaErrorMessage(err, call, file, line));
                }
            }

            inline void cudaCheckNoexcept(cudaError_t err, const char *call,
                                          const char *file, int line) noexcept {
                if (err != cudaSuccess) {
                    std::cerr << "hopsy.gpu: " << cudaErrorMessage(err, call, file, line)
                              << " (ignored in cleanup)" << std::endl;
                }
            }

            inline void cublasCheck(cublasStatus_t status, const char *call,
                                    const char *file, int line) {
                if (status != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error(cublasErrorMessage(status, call, file, line));
                }
            }

            inline void cublasCheckNoexcept(cublasStatus_t status, const char *call,
                                            const char *file, int line) noexcept {
                if (status != CUBLAS_STATUS_SUCCESS) {
                    std::cerr << "hopsy.gpu: "
                              << cublasErrorMessage(status, call, file, line)
                              << " (ignored in cleanup)" << std::endl;
                }
            }

            inline void curandCheck(curandStatus_t status, const char *call,
                                    const char *file, int line) {
                if (status != CURAND_STATUS_SUCCESS) {
                    throw std::runtime_error(curandErrorMessage(status, call, file, line));
                }
            }
        }// namespace detail
    }// namespace gpu
}// namespace hopsy

#define CUDA_CHECK(call) ::hopsy::gpu::detail::cudaCheck((call), #call, __FILE__, __LINE__)
#define CUDA_CHECK_NOEXCEPT(call) \
    ::hopsy::gpu::detail::cudaCheckNoexcept((call), #call, __FILE__, __LINE__)

#define CUBLAS_CHECK(call) \
    ::hopsy::gpu::detail::cublasCheck((call), #call, __FILE__, __LINE__)
#define CUBLAS_CHECK_NOEXCEPT(call) \
    ::hopsy::gpu::detail::cublasCheckNoexcept((call), #call, __FILE__, __LINE__)

#define CURAND_CHECK(call) \
    ::hopsy::gpu::detail::curandCheck((call), #call, __FILE__, __LINE__)

// Does not synchronize, so errors raised while the kernel runs
// surface at the next synchronizing call.
#define CUDA_CHECK_LAST_ERROR() CUDA_CHECK(cudaGetLastError())

#endif// HOPSY_GPU_CUDA_CHECK_HPP
