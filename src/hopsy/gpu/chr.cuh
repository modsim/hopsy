#pragma once
#include "dmatrix.h"
#include "dvector.h"
#include "curandwrapper.h"
#include "cublaswrapper.h"
#include "cudawrappers.h"
#include "helper.h"
#include <cub/cub.cuh>
#include <cuda/std/limits>


namespace hopsy {
    namespace GPU {

        /**
         * @brief CUDA kernel for performing the Coordinated Hit-and-Run (CHR) sampling.
         *
         * @tparam Real The data type for real numbers
         * @tparam PRNGenerator The type of the pseudo-random number generator.
         * @tparam ThreadsPerBlock The number of threads per block for the kernel.
         * @param A Pointer to the matrix A in device memory.
         * @param slack Pointer to the slack vector in device memory.
         * @param x0 Pointer to the initial point vector in device memory.
         * @param samples Pointer to the output samples in device memory.
         * @param rows The number of rows in the matrix A.
         * @param cols The number of columns in the matrix A, aka the flux dimension.
         * @param global_states Pointer to the global states for the PRN generator.
         * @param samples_per_chain The number of samples to generate per Markov chain.
         * @param thinning The thinning factor for the Markov chain.
         * @param nchains The number of Markov chains to run in parallel.
         */
        template <typename Real, typename PRNGenerator, int ThreadsPerBlock>
        __global__ void chrKernel(Real* A, Real* slack, Real* x0, Real* samples, int rows, int cols, PRNGenerator* global_states, int samples_per_chain, int thinning, int nchains);


        /**
         * @brief Launches the CUDA kernel for the CHR (Coordinate Hit-and-Run) sampling algorithm.
         *
         * This function initializes and launches a CUDA kernel to perform CHR sampling on a convex polytope.
         * It generates samples starting from initial points, using the provided random number generator states.
         *
         * @tparam Real Numeric type for computations
         * @tparam PRNGenerator Type of the pseudo-random number generator.
         * @param A Constraint matrix defining the polytope (each row represents a constraint).
         * @param slack Matrix of slack variables for each constraint and sample.
         * @param x0 Matrix of initial points for each Markov chain.
         * @param samples Output matrix to store the generated samples.
         * @param global_states Array of random number generator states for each thread.
         * @param samples_per_chain Number of samples to generate per Markov chain.
         * @param thinning Number of steps between recorded samples (thinning factor).
         * @param threads_per_block Number of CUDA threads per block.
         * @param blocks_per_grid Number of CUDA blocks to launch.
         */
        template <typename Real, typename PRNGenerator>
        void launchChrKernel(DMatrix<Real> &A, DMatrix<Real> &slack, DMatrix<Real> &x0, DMatrix<Real> &samples, PRNGenerator *global_states, int samples_per_chain, int thinning, int threads_per_block, int blocks_per_grid);

        /**
         * @brief Initializes the states of a pseudo-random number generator (PRNG) for use in CUDA kernels.
         *
         * This kernel function sets up an array of PRNG states, one for each thread, using the provided seed.
         * It is typically called before launching kernels that require random number generation on the GPU.
         *
         * @tparam PRNGenerator The type of the PRNG state (e.g., curandState).
         * @param states Pointer to an array of PRNG state objects in device memory.
         * @param seed The seed value to initialize the PRNG states.
         * @param nstates The total number of PRNG states to initialize (typically matches the number of threads).
         */
        template <typename PRNGenerator>
        __global__ void initPrngStatesCHR(PRNGenerator *states, int seed, int nstates);
    }
}