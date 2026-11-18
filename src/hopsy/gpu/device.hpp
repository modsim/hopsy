#ifndef HOPSY_GPU_DEVICE_HPP
#define HOPSY_GPU_DEVICE_HPP

#include <cstddef>
#include <string>

// Kept free of CUDA headers; can be included from host-only translation units.

namespace hopsy {
    namespace gpu {
        struct DeviceProperties {
            int index;
            std::string name;
            int major;
            int minor;
            std::size_t totalMemory;
        };

        // Number of visible CUDA devices.
        int deviceCount();

        // True if at least one CUDA device can be used.
        bool isAvailable();

        // CUDA runtime (`libcudart`) version hopsy.gpu was built against, e.g. 12020
        // for 12.2.
        int runtimeVersion();

        // Latest CUDA version supported by the installed driver, 0 if no driver.
        int driverVersion();

        // Properties of device `index`.
        DeviceProperties deviceProperties(int index);
    }// namespace gpu
}// namespace hopsy

#endif// HOPSY_GPU_DEVICE_HPP
