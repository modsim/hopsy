#pragma once
#include <Eigen/Dense>


namespace hopsy {
    namespace GPU {

        template<typename Real>
        using HVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

        /**
            * @brief Device vector class for managing 1D arrays on the GPU.
            *
            * Key features:
            * - Stores vector data in device memory (GPU) for use in CUDA kernels.
            * - Supports construction from host vectors (Eigen) or by specifying length and value.
            * - Provides a fill method to set all elements to a given value using a CUDA kernel.
            * - Offers a method to copy data back to the host as an Eigen vector.
            *
            * @tparam Real The floating-point type of the vector elements
            */
        template<typename Real>
        struct DVector{

            // Data allocated on device
            Real    *dvec;

            // Meta data available on host
            int len;

            public:
            
            // Construct from Eigen vector
            DVector(const HVector<Real>& v);

            // Construct from input length
            DVector(int len_);

            // Construct from input length set to a value
            DVector(int len_, Real value);

            // Fill the vector with a value
            void fill(Real value);

            ~DVector();

            // Returns eigen vector on host
            HVector<Real> toHost() const;

            // Copy constructor (deep copy)
            DVector(const DVector& other);

            // Move constructor
            DVector(DVector&& other) noexcept;

            // Copy assignment operator
            DVector& operator=(const DVector& other);

            // Move assignment operator
            DVector& operator=(DVector&& other) noexcept;

        };
    }
}        