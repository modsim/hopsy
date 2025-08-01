#pragma once
#include <vector>
#include <numeric>
#include "dmatrix.h"
#include "dvector.h"
#include "cublaswrapper.h"
#include "helper.h"
#include <cub/cub.cuh>
#include <cuda/std/limits>
#include "cudawrappers.h"

namespace hopsy {
    namespace GPU {

        /**
         * @brief CUDA kernel to compute mean and variance using Welford's algorithm.
         *
         * Computes the mean and variance for each chain and dimension in parallel.
         *
         * @param samples Input samples (device).
         * @param nspc Number of samples per chain.
         * @param nchains Number of chains.
         * @param dim Number of dimensions.
         * @param mean_g Output means (device).
         * @param var_g Output variances (device).
         */
        template <typename Real>
        __global__ void welfords_mean_var_kernel(const Real *samples, int nspc, int nchains, int dim, Real *mean_g, Real *var_g);

        template <typename Real>
        __global__ void rhat_kernel(const Real* mean, const Real* var, int nchains, int dim, int nspc, Real* rhat_g);

        /**
         * @brief Computes the nested R-hat metric for improved convergence diagnostics.
         *
         * Implements a nested R-hat computation by grouping chains into superchains.
         *
         * @param samples_d Device matrix of samples (dim x nspc*nchains).
         * @param nspc Number of samples per chain.
         * @param nchains Number of chains.
         * @param dim Number of dimensions.
         * @param K Number of superchains.
         * @return Device vector of nested R-hat values for each dimension.
         */
        template <typename Real>
        DVector<Real> NestedRhat(const DMatrix<Real>& samples_d, int nspc, int nchains, int dim, int K);

        template <typename Real>
        __global__ void subchains_reduction(const Real *means, const Real *vars, int K, int M, int dim, int nchain, Real *subc_mean_of_means, Real *subc_sum_of_var_and_mean);

        template <typename Real>
        __global__ void superchains_reduction(const Real *subc_mean_of_means, const Real *subc_sum_of_var_and_mean, int K, int dim, Real *nestR);
    }
}