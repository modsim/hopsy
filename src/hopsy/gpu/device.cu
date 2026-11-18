#include "device.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse.h>
#include <library_types.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace hopsy {
    namespace gpu {
        namespace {
            void check(cudaError_t err, const char *call) {
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string(call) + " failed (" +
                                             cudaGetErrorName(err) +
                                             "): " + cudaGetErrorString(err));
                }
            }

            // "major.minor.patch" from a math library's <lib>GetProperty function.
            template<typename Status>
            std::string libraryVersion(Status (*getProperty)(libraryPropertyType, int *),
                                       Status success, const char *call) {
                const libraryPropertyType properties[] = {MAJOR_VERSION, MINOR_VERSION,
                                                          PATCH_LEVEL};
                std::string version;
                for (libraryPropertyType property : properties) {
                    int value = 0;
                    Status status = getProperty(property, &value);
                    if (status != success) {
                        throw std::runtime_error(std::string(call) + " failed (status " +
                                                 std::to_string(static_cast<int>(status)) +
                                                 ")");
                    }
                    version += (version.empty() ? "" : ".") + std::to_string(value);
                }
                return version;
            }
        }// namespace

        int deviceCount() {
            int count = 0;
            cudaError_t err = cudaGetDeviceCount(&count);
            if (err != cudaSuccess) {
                // CUDA unusable (no device, no/too old driver, init failure)
                std::cerr << "hopsy.gpu: no usable CUDA device ("
                          << cudaGetErrorName(err) << ": " << cudaGetErrorString(err)
                          << ")" << std::endl;
                // Reset the runtime's last-error state so it does not leak into later
                // calls.
                cudaGetLastError();
                return 0;
            }
            return count;
        }

        bool isAvailable() { return deviceCount() > 0; }

        int runtimeVersion() {
            int version = 0;
            check(cudaRuntimeGetVersion(&version), "cudaRuntimeGetVersion");
            return version;
        }

        int driverVersion() {
            int version = 0;
            check(cudaDriverGetVersion(&version), "cudaDriverGetVersion");
            return version;
        }

        DeviceProperties deviceProperties(int index) {
            int count = deviceCount();
            if (index < 0 || index >= count) {
                throw std::out_of_range("invalid CUDA device index " +
                                        std::to_string(index) + " (" +
                                        std::to_string(count) + " device(s) visible)");
            }
            cudaDeviceProp prop;
            check(cudaGetDeviceProperties(&prop, index), "cudaGetDeviceProperties");
            return DeviceProperties{index, prop.name, prop.major, prop.minor,
                                    prop.totalGlobalMem};
        }

        std::map<std::string, std::string> mathLibraryVersions() {
            return {
                    {"cublas", libraryVersion(&cublasGetProperty, CUBLAS_STATUS_SUCCESS,
                                              "cublasGetProperty")},
                    {"cusparse", libraryVersion(&cusparseGetProperty,
                                                CUSPARSE_STATUS_SUCCESS,
                                                "cusparseGetProperty")},
                    {"curand", libraryVersion(&curandGetProperty, CURAND_STATUS_SUCCESS,
                                              "curandGetProperty")},
            };
        }
    }// namespace gpu
}// namespace hopsy
