    #pragma once


    namespace hopsy {
        namespace GPU {
            
            template<typename Real>
            __global__ void initializeKernel(Real* data, int len, Real value) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < len) {
                    data[idx] = value;
                }
            }
        }
    }        