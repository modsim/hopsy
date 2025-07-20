#include "dvector.h"
#include "helper.h"
#include "dvector.cuh"

namespace hopsy {
    namespace GPU {
        template<typename Real>
        DVector<Real>::DVector(const HVector<Real>& v) : len(v.size()) {
            size_t size = len * sizeof(Real);
            CUDA_CHECK(cudaMalloc(&dvec, size));
            CUDA_CHECK(cudaMemcpy(dvec, v.data(), size, cudaMemcpyHostToDevice));
        }

        template<typename Real>
        DVector<Real>::DVector(int len_) : len(len_) {
            size_t size = len * sizeof(Real);
            CUDA_CHECK(cudaMalloc(&dvec, size));
        }

        template<typename Real>
        DVector<Real>::DVector(int len_, Real value) : len(len_) {
            size_t size = len * sizeof(Real);
            CUDA_CHECK(cudaMalloc(&dvec, size));
            int blockSize = 256;
            int numBlocks = (len + blockSize - 1) / blockSize;
            initializeKernel<Real><<<numBlocks, blockSize>>>(dvec, len, value);
        }

        template<typename Real>
        void DVector<Real>::fill(Real value) {
            int blockSize = 256;
            int numBlocks = (len + blockSize - 1) / blockSize;
            initializeKernel<Real><<<numBlocks, blockSize>>>(dvec, len, value);
        }

        template<typename Real>
        HVector<Real> DVector<Real>::toHost() const {
            HVector<Real> v(len);
            CUDA_CHECK(cudaMemcpy(v.data(), dvec, len * sizeof(Real), cudaMemcpyDeviceToHost));
            return v;
        }

        template<typename Real>
        DVector<Real>::~DVector() {
            cudaFree(dvec);
            dvec = nullptr;
        }

        // Copy constructor (deep copy)
        template<typename Real>
        DVector<Real>::DVector(const DVector& other) : len(other.len) {
            cudaMalloc(&dvec, len * sizeof(Real));  // Allocate new memory on the device
            cudaMemcpy(dvec, other.dvec, len * sizeof(Real), cudaMemcpyDeviceToDevice);  // Copy the data from the other device vector
        }

        // Move constructor
        template<typename Real>
        DVector<Real>::DVector(DVector&& other) noexcept : dvec(other.dvec), len(other.len) {
            other.dvec = nullptr;  // Nullify the other object's pointer
            other.len = 0;
        }
        
        // Copy assignment operator
        template<typename Real>
        DVector<Real>& DVector<Real>::operator=(const DVector& other) {
            if (this != &other) {  // Check for self-assignment
                // Free existing memory
                if (dvec) {
                    cudaFree(dvec);
                }

                // Allocate new memory and copy the data
                len = other.len;
                cudaMalloc(&dvec, len * sizeof(Real));
                cudaMemcpy(dvec, other.dvec, len * sizeof(Real), cudaMemcpyDeviceToDevice);
            }
            return *this;
        }

        // Move assignment operator
        template<typename Real>
        DVector<Real>& DVector<Real>::operator=(DVector&& other) noexcept {
            if (this != &other) {  // Check for self-assignment
                // Free existing memory
                if (dvec) {
                    cudaFree(dvec);
                }

                // Move data from the other object
                dvec = other.dvec;
                len = other.len;

                // Nullify the other object's pointer to prevent double free
                other.dvec = nullptr;
                other.len = 0;
            }
            return *this;
        }

        template class DVector<double>;
        template class DVector<int>;
    }
}