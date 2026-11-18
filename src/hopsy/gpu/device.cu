#include "device.hpp"

#include <cuda_runtime.h>

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
    }// namespace gpu
}// namespace hopsy
