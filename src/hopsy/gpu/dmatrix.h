#pragma once
#include "Eigen/Dense"
#include "dvector.h"

namespace hopsy {
    namespace GPU {

        template<typename Real>
        using HMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

        /**
         * @brief Device matrix class for managing 2D arrays on the GPU.
         *
         * - Stores matrix data in device memory (GPU) for use in CUDA kernels.
         * - Supports construction from host matrices/vectors (Eigen), device vectors, or by shape.
         * - Provides copy and move constructors, but disables assignment operators to avoid
         *   accidental shallow copies.
         * - Offers a method to copy data back to the host as an Eigen matrix.
         * - Allows creation of submatrix views without additional memory allocation.
         *
         * @tparam Real The floating-point type of the matrix elements
         */
        template<typename Real>
        struct DMatrix {
            
            // Data allocated on device
            Real *dmat;
            
            // Meta data available on host
            int rows;
            int cols;
            bool ownsData;

            public:

            // Construct from Eigen
            DMatrix(const HMatrix<Real>& m);

            // Construct from input dimensions
            DMatrix(int rows, int cols);

            // Construct from vector and broadcast
            DMatrix(const HVector<Real>& v, int cols);

            // Construct from device vector and broadcast
            DMatrix(const DVector<Real>& v, int cols);

            // Construct from vector with a given number of cols, and broadcast to 'z' cols
            DMatrix(const HVector<Real>& v, int cols, int z);

            // Construct subMatrix view
            DMatrix(Real *dmat, int rows, int cols);

            // Copy constructor
            DMatrix(const DMatrix &other);

            // Move constructor
            DMatrix(DMatrix &&other) noexcept;

            // Delete copy assignment
            DMatrix& operator=(const DMatrix &other) = delete;

            // Delete move assignment
            DMatrix& operator=(DMatrix &&other) = delete;

            // Destructor
            ~DMatrix();

            // Copy to host
            HMatrix<Real> toHost() const;

            // Submatrix view
            DMatrix<Real> subMatrix(int rowStart, int colStart, int rowEnd, int colEnd) const;
        };
    }
}        