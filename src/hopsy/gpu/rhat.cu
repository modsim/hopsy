#include "rhat.cuh"
#include "gpusamplers.h"

namespace hopsy {
    namespace GPU {

        DVector<double> rhat(const DMatrix<double>& samples_d, int nspc, int nchains, int dim){

            std::pair<int, int> launchConfig = getOptimalLaunchConfig(welfords_mean_var_kernel<double>);
            int threads = launchConfig.first;
            int blocks = (dim + threads - 1) / threads;

            DMatrix<double> mean_d(dim, nchains);
            DMatrix<double> var_d(dim, nchains);
            DVector<double> rhat_d(dim);

            // Calculate mean and variance using Welford's method
            welfords_mean_var_kernel<<<nchains, threads>>>(samples_d.dmat, nspc, nchains, dim, mean_d.dmat, var_d.dmat);
            rhat_kernel<<<blocks, threads>>>(mean_d.dmat, var_d.dmat, nchains, dim, nspc, rhat_d.dvec);

            return rhat_d;
        }


        template <typename Real>
        __global__ void welfords_mean_var_kernel(const Real* samples, int nspc, int nchains, int dim, Real* mean_g, Real* var_g) {
            int tid = threadIdx.x;
            int bid = blockIdx.x;
            int blockstride = blockDim.x;

            // Grid-Stride
            for (int i = tid; i < dim; i+=blockstride){

                // Welford's method
                Real mean = 0, var = 0, delta = 0;
                for (int j = 0; j < nspc; ++j){
                    int coloffset = j * nchains;
                    delta = samples[IDX2C(i, bid + coloffset, dim)] - mean;
                    mean += delta / (j + 1);
                    var += delta * (samples[IDX2C(i, bid + coloffset, dim)] - mean);
                }
                var /= (nspc - 1);

                // Copy to global memory
                mean_g[IDX2C(i, bid, dim)] = mean;
                var_g[IDX2C(i, bid, dim)] = var;
            }
        }

        template<typename Real>
        __global__ void rhat_kernel(const Real* mean, const Real* var, int nchains, int dim, int nspc, Real* rhat_g) {
            int gtid = threadIdx.x + blockIdx.x * blockDim.x;
            int blockstride = blockDim.x * gridDim.x;

            // Grid-Stride
            for (int i = gtid; i < dim; i+=blockstride){
                Real W = 0;         // Mean of variances
                Real mean_mean = 0;     // Mean of means
                Real B = 0;             // nchains * Variance of means
                Real delta = 0;

                // Welford's method
                for (int j = 0; j < nchains; ++j) {
                    Real local_mean = mean[IDX2C(i, j, dim)];
                    W += var[IDX2C(i, j, dim)];
                    delta = local_mean - mean_mean;
                    mean_mean += delta / (j + 1);
                    B += delta * (local_mean - mean_mean);
                }
                B = nspc * B / (nchains - 1.0);
                W /= nchains;
                rhat_g[i] = sqrt((((nspc - 1.0) / nspc) * W + B / nspc) / W);
            }
        }


        /********************
         **** Nested Rhat****
        *******************/

        // template <typename Real>
        // DVector<Real> NestedRhat(const DMatrix<Real>& samples_d, int nspc, int nchains, int dim, int K){
        //     // Load samples
        //     if (nchains % K != 0) {
        //         std::cerr << "Error: nchains (" << nchains << ") is not divisible by number of superchains K (" << K << ")." << std::endl;
        //         exit(EXIT_FAILURE);
        //     }
        //     int M = nchains / K;    // Number of subchains, i.e. each superchain has M subchains
        //     std::cout << "Number of Superchains: " << K << ", Number of Subchains: " << M << std::endl;

        //     DMatrix<Real> mean_d(dim, nchains), var_d(dim, nchains);
        //     DMatrix<Real> subc_mean_of_means(dim, K); // Mean of means of subchains
        //     DMatrix<Real> subc_sum_of_var_and_mean(dim, K); // Sum of variance and mean of subchains
        //     DVector<Real> nestR(dim); // Nested Rhat

        //     //Superchain reduction launch config
        //     std::pair<int, int> launchConfig = getOptimalLaunchConfig(superchains_reduction<Real>);
        //     std::cout<< "Superchain reduction Launch Config: Threads: " << launchConfig.first << " Blocks: " << launchConfig.second << std::endl;
        //     int supc_threads = launchConfig.first;
        //     int supc_blocks = (dim + supc_threads - 1) / supc_threads;

            
        //     CudaEvent start, stop;
            
        //     cudaEventRecord(start);
        //     welfords_mean_var_kernel<<<nchains, dim>>>(samples_d.dmat, nspc, nchains, dim, mean_d.dmat, var_d.dmat);
        //     CUDA_CHECK(cudaGetLastError());
        //     subchains_reduction<<<K, dim>>>(mean_d.dmat, var_d.dmat, K, M, dim, nchains, subc_mean_of_means.dmat, subc_sum_of_var_and_mean.dmat);
        //     CUDA_CHECK(cudaGetLastError());
        //     superchains_reduction<<<supc_blocks, supc_threads>>>(subc_mean_of_means.dmat, subc_sum_of_var_and_mean.dmat, K, dim, nestR.dvec);
        //     CUDA_CHECK(cudaGetLastError());
        //     cudaEventRecord(stop);
        //     cudaEventSynchronize(stop);
        //     printElapsedTime(start, stop, "Rhat Computation");

        //     return nestR;
        // }
        // template DVector<double> NestedRhat(const DMatrix<double>& , int , int , int, int );

        template <typename Real>
        __global__ void subchains_reduction(const Real* means, const Real* vars, int K, int M, int dim, int nchain, Real* subc_mean_of_means, Real* subc_sum_of_var_and_mean){
            int tid = threadIdx.x;
            int bid = blockIdx.x;
            int threadstride = blockDim.x;
        
            // Block stride
            for (int i = tid; i < dim; i+= threadstride){

                Real partial_subc_mean_of_means = 0, partial_subc_var_of_means = 0, partial_subc_mean_of_vars = 0, delta = 0;
                for (int j = 0; j < M; ++j){
                    Real local_mean = means[IDX2C(i, j + bid * M, dim)];
                    partial_subc_mean_of_vars += vars[IDX2C(i, j + bid * M, dim)];
                    delta = local_mean - partial_subc_mean_of_means;
                    partial_subc_mean_of_means += delta / (j + 1);
                    partial_subc_var_of_means += delta * (local_mean - partial_subc_mean_of_means);
                }
                partial_subc_var_of_means /= (M - 1);
                partial_subc_mean_of_vars /= M;

                subc_mean_of_means[IDX2C(i,bid, dim)] = partial_subc_mean_of_means;
                subc_sum_of_var_and_mean[IDX2C(i,bid, dim)] = partial_subc_var_of_means + partial_subc_mean_of_vars;
            }

        }


        template <typename Real>
        __global__ void superchains_reduction(const Real* subc_mean_of_means, const Real* subc_sum_of_var_and_mean, int K, int dim, Real* nestR){
            int gtid = threadIdx.x + blockIdx.x * blockDim.x;
            int blockstride = blockDim.x * gridDim.x;

            // Grid-Stride
            for (int i = gtid; i < dim; i+=blockstride){
                Real B = 0;
                Real W = 0;
                Real temp_mean = 0;
                Real delta = 0;

                // Welford's method
                for (int j = 0; j < K; ++j) {
                    Real local_mean = subc_mean_of_means[IDX2C(i, j, dim)];
                    W += subc_sum_of_var_and_mean[IDX2C(i, j, dim)];
                    delta = local_mean - temp_mean;
                    temp_mean += delta / (j + 1);
                    B += delta * (local_mean - temp_mean);
                }
                B = B / (K - 1.0);
                W /= K;
                nestR[i] = sqrt(1.0 + B / W);
            }
        }
    }
}