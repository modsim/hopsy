#include "device_utils.h"
#include <cuda_runtime.h>
#include "helper.h"



namespace hopsy {
    namespace GPU {

        void set_device(int device) {
            CUDA_CHECK(cudaSetDevice(device));
        }

        std::vector<std::string> list_devices() {
            int count;
            CUDA_CHECK(cudaGetDeviceCount(&count));
            std::vector<std::string> devices;
            for (int i = 0; i < count; ++i) {
                cudaDeviceProp prop;
                CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
                std::ostringstream oss;
                oss << "[" << i << "] " << prop.name;
                devices.push_back(oss.str());
            }
            return devices;
        }
    }
}
