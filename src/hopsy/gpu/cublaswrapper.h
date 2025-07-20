#pragma once
#include "helper.h"

namespace hopsy {
    namespace GPU {

        /**
         * @brief RAII wrapper for cuBLAS handle and common linear algebra operations on the GPU.
         *
         * Key features:
         * - RAII for cuBLAS handle
         * - Wrappers for matrix-matrix (GeMM), matrix-vector (GeMV), and
         *   symmetric rank-k/1 updates (Syrk/Syr1).
         * - Also scaling, diagonal multiplication, and Frobenius norm.
         * - Designed to work with DMatrix and DVector device containers.
         *
         * Usage:
         * @code
         *   CUBLASHandle handle;
         *   handle.GeMM(A, B, C, 1.0, 0.0); // C = A * B
         * @endcode
         */
        struct CUBLASHandle {
            cublasHandle_t h;

            CUBLASHandle() { CUBLAS_CHECK(cublasCreate(&h)); }
            ~CUBLASHandle() { CUBLAS_CHECK(cublasDestroy(h)); }
            operator cublasHandle_t() const { return h; }

            /// @brief Perform C = alpha * A * B + beta * C
            template <typename Real>
            __host__ void GeMM(const DMatrix<Real> &A, const DMatrix<Real> &B, DMatrix<Real> &C, Real alpha, Real beta, cublasOperation_t transa = CUBLAS_OP_N, cublasOperation_t transb = CUBLAS_OP_N) {
            CUBLAS_CHECK(cublasDgemm(h, transa, transb, A.rows, B.cols, A.cols, &alpha, A.dmat, A.rows, B.dmat, B.rows, &beta, C.dmat, C.rows));
            }
            
            /// @brief Perform C = A*A^T, rank k update
            template <typename Real>
            __host__ void Syrk(const DMatrix<Real>& A, DMatrix<Real>& C, cublasFillMode_t  uplo, Real alpha, Real beta) {
                CUBLAS_CHECK(cublasDsyrk(h, uplo, CUBLAS_OP_N, A.rows, A.cols, &alpha, A.dmat, A.rows, &beta, C.dmat, C.rows));
            }

            /// @brief Perform C = C + alpha * x@x^T, rank 1 update
            template <typename Real>
            __host__ void Syr1(const DVector<Real>& x, DMatrix<Real>& C, cublasFillMode_t  uplo, Real alpha) {
                CUBLAS_CHECK(cublasDsyr(h, uplo, C.rows, &alpha, x.dvec, 1, C.dmat, C.rows));
            }

            /// @brief Perform y = alpha * A * x + beta * y
            template <typename Real>
            __host__ void GeMV(const DMatrix<Real>& A, const DVector<Real>& x, DVector<Real>& y, Real alpha, Real beta) {
                CUBLAS_CHECK(cublasDgemv(h, CUBLAS_OP_N, A.rows, A.cols, &alpha, A.dmat, A.rows, x.dvec, 1, &beta, y.dvec, 1));
            }

            /// @brief Perform  C = A * diag(X)
            template <typename Real>
            __host__ void DgMM(const DMatrix<Real>& A, const DVector<Real>& X, DMatrix<Real>& C) {
                CUBLAS_CHECK(cublasDdgmm(h, CUBLAS_SIDE_RIGHT, A.rows, A.cols, A.dmat, A.rows, X.dvec, 1, C.dmat, C.rows));
            }

            /// @brief Perform result = norm_F(X)
            template <typename Real>
            __host__ void NrmF(const DMatrix<Real>& X, Real &result) {
                CUBLAS_CHECK(cublasDnrm2(h, X.cols*X.rows, X.dmat, 1, &result));
            }

            /// @bried Perform X = alpha*X
            template <typename Real>
            __host__ void Scal(const DMatrix<Real>& X, Real alpha) {
                CUBLAS_CHECK(cublasDscal(h, X.cols*X.rows, &alpha, X.dmat, 1));
            }

        };

    }
}