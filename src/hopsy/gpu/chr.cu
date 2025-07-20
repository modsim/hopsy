#include "chr.cuh"
#include "gpusamplers.h"

namespace hopsy {
    namespace GPU {

        DMatrix<double> CoordinateHitAndRun(    DMatrix<double>& A_d,
                                        DVector<double>& b_d,
                                        DMatrix<double>& X_d,
                                        int nspc,
                                        int thinningfactor,
                                        int nchains,
                                        int tpb_ss)
        {
            // TODO: Implement logic for launch config when tpb_ss are not provided
            int N = A_d.cols;
            int thinning = (thinningfactor > 0) ? thinningfactor * N : 1;

            DMatrix<double> samples_d(N, nchains * nspc), slack_d(b_d, nchains);

            // Init cuBLAS handle
            CUBLASHandle handle;

            // Init slack: slack = b - A*x0
            handle.GeMM(A_d, X_d, slack_d, -1.0, 1.0);

            // Init PRNG
            PRNGState<curandStateMRG32k3a> gen(nchains);
            std::pair<int, int> launchConfig = getOptimalLaunchConfig(initPrngStatesCHR<curandStateMRG32k3a>);
            int blocksPerGrid = (nchains + launchConfig.first - 1) / launchConfig.first;
            initPrngStatesCHR<<<launchConfig.first, blocksPerGrid>>>(gen.states, 0, nchains);
            CUDA_CHECK(cudaGetLastError());

            // Launch CHR kernel
            launchChrKernel(A_d, slack_d, X_d, samples_d, gen.states, nspc, thinning, tpb_ss, nchains);

            return samples_d;
        }

        template <typename Real, typename PRNGenerator, int ThreadsPerBlock>
        __global__ void chrKernel(Real* A, Real* slack, Real* X0, Real* samples, int rows, int cols, PRNGenerator* global_states, int samples_per_chain, int thinning, int nchains){

            int tid = threadIdx.x;
            int bid = blockIdx.x;
            int threadstride = blockDim.x;

            // Shared memory for x
            extern __shared__ Real x_s[];
            // Copy x0 to shared memory
            for (int i = tid; i < cols; i += threadstride){
                x_s[i] = X0[IDX2C(i, bid, cols)];
            }

            using BlockReduce = cub::BlockReduce<Real, ThreadsPerBlock>;
            __shared__ typename BlockReduce::TempStorage temp_storage_max;
            __shared__ typename BlockReduce::TempStorage temp_storage_min;
            __shared__ Real alpha;

            for (int t = 0; t < samples_per_chain * thinning; ++t){
                
                bool save = t % thinning == 0;
                int colOffset = static_cast<int>(t / thinning) * nchains;
                int e = t % cols;

                Real partial_max = cuda::std::numeric_limits<Real>::lowest();
                Real partial_min = cuda::std::numeric_limits<Real>::max();
                for (int i = tid; i < rows; i += threadstride){

                    // Get slack and projected direction
                    Real s = slack[IDX2C(i, bid, rows)];
                    Real ae = A[IDX2C(i, e, rows)];
                    
                    // Get step size bounds
                    Real inv_dist = ae / s;
                    inv_dist = isnan(inv_dist) || isinf(inv_dist) ? Real(0.0) : inv_dist;
                    partial_max = fmax(partial_max, inv_dist);
                    partial_min = fmin(partial_min, inv_dist);
                }
                Real aggregate_max = BlockReduce(temp_storage_max).Reduce(partial_max, cub::Max());
                Real aggregate_min = BlockReduce(temp_storage_min).Reduce(partial_min, cub::Min());
                if(tid==0){
                    PRNGenerator localState = global_states[bid];
                    Real usample = curand_uniform_double(&localState);
                    alpha = (1/aggregate_min) + usample * ((1/aggregate_max) - (1/aggregate_min));
                    x_s[e] += alpha;
                    global_states[bid] = localState;
                }
                __syncthreads();


                // save x
                if(save){
                    for (int i = tid; i < cols; i += threadstride)
                        samples[IDX2C(i, bid + colOffset, cols)] = x_s[i];
                }
                
                // Update slack
                for (int i = tid; i < rows; i += threadstride){
                    slack[IDX2C(i, bid, rows)] -= alpha * A[IDX2C(i, e, rows)];
                }
            }
        }
        template __global__ void chrKernel<double, curandState, 32>(double*, double*, double*, double*, int, int, curandState*, int, int, int);
        template __global__ void chrKernel<double, curandState, 64>(double*, double*, double*, double*, int, int, curandState*, int, int, int);
        template __global__ void chrKernel<double, curandState, 128>(double*, double*, double*, double*, int, int, curandState*, int, int, int);
        template __global__ void chrKernel<double, curandState, 256>(double*, double*, double*, double*, int, int, curandState*, int, int, int);
        template __global__ void chrKernel<double, curandState, 512>(double*, double*, double*, double*, int, int, curandState*, int, int, int);
        template __global__ void chrKernel<double, curandState, 1024>(double*, double*, double*, double*, int, int, curandState*, int, int, int);

        template <typename Real, typename PRNGenerator>
        void launchChrKernel(DMatrix<Real>& A, DMatrix<Real>& slack, DMatrix<Real>& x0, DMatrix<Real>& samples, PRNGenerator* global_states, int samples_per_chain, int thinning, int threads_per_block, int blocks_per_grid){
            int rows = A.rows;
            int cols = A.cols;
            size_t shared_memory_size = (cols) * sizeof(Real);
            switch(threads_per_block){
                case 32:
                    chrKernel<Real, PRNGenerator, 32><<<blocks_per_grid, threads_per_block, shared_memory_size, 0>>>(A.dmat, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid);
                    break;
                case 64:    
                    chrKernel<Real, PRNGenerator, 64><<<blocks_per_grid, threads_per_block, shared_memory_size, 0>>>(A.dmat, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid);
                    break;
                case 128:
                    chrKernel<Real, PRNGenerator, 128><<<blocks_per_grid, threads_per_block, shared_memory_size, 0>>>(A.dmat, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid);
                    break;
                case 256:
                    chrKernel<Real, PRNGenerator, 256><<<blocks_per_grid, threads_per_block, shared_memory_size, 0>>>(A.dmat, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid);
                    break;
                case 512:
                    chrKernel<Real, PRNGenerator, 512><<<blocks_per_grid, threads_per_block, shared_memory_size, 0>>>(A.dmat, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid);
                    break;
                case 1024:
                    chrKernel<Real, PRNGenerator, 1024><<<blocks_per_grid, threads_per_block, shared_memory_size, 0>>>(A.dmat, slack.dmat, x0.dmat, samples.dmat, rows, cols, global_states, samples_per_chain, thinning, blocks_per_grid);
                    break;
                default:
                    throw std::runtime_error("Invalid threads per block");
                    break;
                }
            CUDA_CHECK(cudaGetLastError());
        }

        template <typename PRNGenerator>
        __global__ void initPrngStatesCHR(PRNGenerator *states, int seed, int nstates){
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if(tid < nstates)
                curand_init(seed,tid,0,&states[tid]);
        }

    }
}