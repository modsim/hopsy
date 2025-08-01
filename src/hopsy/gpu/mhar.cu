#include "mhar.cuh"
#include "gpusamplers.h"


namespace hopsy {
    namespace GPU {
        DMatrix<double> WarumUp(DMatrix<double>& A_d,
                                DVector<double>& b_d,
                                DVector<double>& x0_d,
                                int nwarmup,
                                int nchains,
                                int tpb_rd,
                                int tpb_ss)
        {

            // TODO: Implement logic for launch config when tpb_rd and tpb_ss are not provided

            int M = A_d.rows;
            int N = A_d.cols;
            DMatrix<double> X_d(x0_d, nchains), D_d(N, nchains), AD_d(M, nchains), slack_d(b_d, nchains);
            DVector<double> alpha_d(nchains);
            PRNGState<curandStateMRG32k3a> gen(tpb_rd * nchains);
            initPrngStates<<<nchains, tpb_rd>>>(gen.states, 1234);

            dim3 tpb_xup_nosave(256);
            dim3 bpg_xup_nosave((N + tpb_xup_nosave.x - 1) / tpb_xup_nosave.x, nchains);

            CUBLASHandle handle;
            CudaStream captureStream;
            cudaGraph_t WarmupGraph;
            cudaGraphExec_t warumup_graph;
            bool warmup_g_created = false;
            CUBLAS_CHECK(cublasSetStream(handle, captureStream));
            if(!warmup_g_created){
                CUDA_CHECK(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal));
                launchGenRndDir(D_d.dmat, D_d.rows, D_d.cols, gen.states, tpb_rd, nchains, captureStream);
                handle.GeMM(A_d, D_d, AD_d, 1.0, 0.0);
                handle.GeMM(A_d, X_d, slack_d, -1.0, 1.0);
                launchSampleStepSize(AD_d.dmat, slack_d.dmat, alpha_d.dvec, b_d.dvec, AD_d.rows, AD_d.cols, gen.states, tpb_ss, nchains, captureStream);
                XupdateNoSave<<<bpg_xup_nosave, tpb_xup_nosave, 0, captureStream>>>(X_d.dmat, D_d.dmat, alpha_d.dvec, X_d.rows, X_d.cols);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamEndCapture(captureStream, &WarmupGraph));
                CUDA_CHECK(cudaGraphInstantiate(&warumup_graph, WarmupGraph, NULL, NULL, 0));
                warmup_g_created = true;
            }

            // Warm Up
            for (int i = 0; i < nwarmup; i++){
                CUDA_CHECK(cudaGraphLaunch(warumup_graph, 0));
            }

            // Cleanup
            CUDA_CHECK(cudaGraphExecDestroy(warumup_graph));
            CUDA_CHECK(cudaGraphDestroy(WarmupGraph));

            return X_d;
        }

        DMatrix<double> MatrixHitAndRun     ( DMatrix<double>& A_d,
                                                DVector<double>& b_d,
                                                DMatrix<double>& X_d,
                                                int nspc,
                                                int thinning,
                                                int nchains,
                                                int tpb_rd,
                                                int tpb_ss)
        {
            // TODO: Implement logic for launch config when tpb_rd and tpb_ss are not provided

            // Dimensions
            int N = A_d.cols;       // Flux Dimension
            int M = A_d.rows;       // Constraint Dimension

            DMatrix<double> samples_d(N, nspc * nchains), D_d(N, nchains), AD_d(M, nchains), slack_d(b_d, nchains);
            DVector<double> alpha_d(nchains);

            // Initialize random states
            PRNGState<curandStateMRG32k3a> gen(tpb_rd*nchains);
            initPrngStates<<<nchains, tpb_rd>>>(gen.states, 1234);
            CUDA_CHECK(cudaGetLastError());

            // Init Library handles
            CUBLASHandle handle;

            // Setup graphs
            CudaStream captureStream;
            cudaGraph_t FinalSamplingGraph;             //class
            cudaGraphExec_t final_sampling_graph;    //instance
            bool final_sampling_g_created = false;
            CUBLAS_CHECK(cublasSetStream(handle, captureStream));
            // Create Final Sampling graph
            DVector<int> t_d(nchains, 0);
            if (!final_sampling_g_created)
            {
                CUDA_CHECK(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal));
                launchGenRndDir(D_d.dmat, D_d.rows, D_d.cols, gen.states, tpb_rd, nchains, captureStream);
                handle.GeMM(A_d, D_d, AD_d, 1.0, 0.0);
                handle.GeMM(A_d, X_d, slack_d, -1.0, 1.0);
                launchSampleStepSize(AD_d.dmat, slack_d.dmat, alpha_d.dvec, b_d.dvec, AD_d.rows, AD_d.cols, gen.states, tpb_ss, nchains, captureStream);
                XupdateWithSaveIterantArray<<<nchains, tpb_rd, 0, captureStream>>>(X_d.dmat, D_d.dmat, alpha_d.dvec, samples_d.dmat, X_d.rows, X_d.cols, t_d.dvec, thinning);
                CUDA_CHECK(cudaStreamEndCapture(captureStream, &FinalSamplingGraph));
                CUDA_CHECK(cudaGraphInstantiate(&final_sampling_graph, FinalSamplingGraph, NULL, NULL, 0));
                final_sampling_g_created = true;
            }

            // Launch the sampling graph
            for (int i = 0; i < nspc * thinning; i++)
            {
                CUDA_CHECK(cudaGraphLaunch(final_sampling_graph, 0));
            }

            CUDA_CHECK(cudaGraphExecDestroy(final_sampling_graph));
            CUDA_CHECK(cudaGraphDestroy(FinalSamplingGraph));
            
            return samples_d;
        }  
        
        
        template <typename PRNGenerator>
        __global__ void initPrngStates(PRNGenerator *states, int seed){
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            curand_init(seed,tid,0,&states[tid]);
        }

        template <typename Real, typename PRNGenerator, int ThreadsPerBlock>
        __global__ void genRndDir(Real* A, int rows, int cols, PRNGenerator* global_states){
            // Thread indexes
            int tid = threadIdx.x;                  // Thread ID in a block
            int bid = blockIdx.x;                   // Block ID
            int gtid = tid + bid * blockDim.x;      // Global thread ID in the grid
            int blockstride = gridDim.x;
            int threadstride = blockDim.x;

            using BlockReduce = cub::BlockReduce<Real, ThreadsPerBlock>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            __shared__ Real broadcast_norm;

            PRNGenerator localState = global_states[gtid];

            // Generate and accumulate norm
            for (int j = bid; j < cols; j += blockstride){

                // Partial sum for a thread in a block
                Real partial_sum = 0.0;

                for (int i = tid; i < rows; i += 2 * threadstride){

                    // Sample from Normal and accumulate squared value
                    double2 X = curand_normal2_double(&localState);     // sampled with Box-Muller, so generating 2 better than 1
                    A[IDX2C(i, j, rows)] = X.x;
                    partial_sum += X.x * X.x;          
                    if (i + threadstride < rows) {
                        A[IDX2C(i + threadstride, j, rows)] = X.y;
                        partial_sum += X.y * X.y;
                    }
                }

                // Block reduction
                Real aggregate = BlockReduce(temp_storage).Sum(partial_sum);

                // Broadcast the block sum to all threads in the block
                if(tid == 0){
                    broadcast_norm = sqrt(aggregate);
                }
                __syncthreads();

                // Normalize the column
                for (int i = tid; i < rows; i += threadstride){
                    A[IDX2C(i, j, rows)] /= broadcast_norm;
                }

                __syncthreads();    // Sync threads before next column
            }

            // Store the state back
            global_states[gtid] = localState;
        }
        // Explicit instantiation
        template __global__ void genRndDir<double, curandState, 32>(double*, int, int, curandState*);
        template __global__ void genRndDir<double, curandState, 64>(double*, int, int, curandState*);
        template __global__ void genRndDir<double, curandState, 128>(double*, int, int, curandState*);
        template __global__ void genRndDir<double, curandState, 256>(double*, int, int, curandState*);
        template __global__ void genRndDir<double, curandState, 512>(double*, int, int, curandState*);
        template __global__ void genRndDir<double, curandState, 1024>(double*, int, int, curandState*);

        template <typename Real, typename PRNGenerator>
        void launchGenRndDir(Real* A, int rows, int cols, PRNGenerator* global_states, int ThreadsPerBlock, int BlocksPerGrid, cudaStream_t stream){
            switch (ThreadsPerBlock)
            {
            case 32:
                genRndDir<Real, PRNGenerator, 32><<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(A, rows, cols, global_states);
                break;
            case 64:
                genRndDir<Real, PRNGenerator, 64><<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(A, rows, cols, global_states);
                break;
            case 128:
                genRndDir<Real, PRNGenerator, 128><<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(A, rows, cols, global_states);
                break;
            case 256:
                genRndDir<Real, PRNGenerator, 256><<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(A, rows, cols, global_states);
                break;
            case 512:
                genRndDir<Real, PRNGenerator, 512><<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(A, rows, cols, global_states);
                break;
            case 1024:
                genRndDir<Real, PRNGenerator, 1024><<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(A, rows, cols, global_states);
                break;
            default:
                break;
            }
            CUDA_CHECK(cudaGetLastError());
        }


        template <typename Real, typename PRNGenerator, int ThreadsPerBlock>
        __global__ void sampleStepSize(Real* AD, Real* slack, Real* alpha, Real* b, int rows, int cols, PRNGenerator* global_states){
            int tid = threadIdx.x;                  // Thread ID in a block
            int bid = blockIdx.x;                   // Block ID
            int blockstride = gridDim.x;
            int threadstride = blockDim.x;
            PRNGenerator localState;
            if (tid == 0)
            {
                localState = global_states[bid];
            }

            using BlockReduce = cub::BlockReduce<Real, ThreadsPerBlock>;
            __shared__ typename BlockReduce::TempStorage temp_storage_max;
            __shared__ typename BlockReduce::TempStorage temp_storage_min;

            for (int j = bid; j < cols; j += blockstride) {
                Real partial_max = cuda::std::numeric_limits<Real>::lowest();
                Real partial_min = cuda::std::numeric_limits<Real>::max();
                for (int i = tid; i < rows; i += threadstride){
                    Real inv_dist = AD[IDX2C(i, j, rows)] / slack[IDX2C(i, j, rows)];
                    inv_dist = isnan(inv_dist) || isinf(inv_dist) ? Real(0.0) : inv_dist;
                    partial_max = fmax(partial_max, inv_dist);
                    partial_min = fmin(partial_min, inv_dist);
                    slack[IDX2C(i, j, rows)] = b[i];                    // Reset slack to b
                }
                Real aggregate_max = BlockReduce(temp_storage_max).Reduce(partial_max, cub::Max());
                Real aggregate_min = BlockReduce(temp_storage_min).Reduce(partial_min, cub::Min());
                if(tid==0) {
                    Real usample = curand_uniform_double(&localState);
                    alpha[j] = (1/aggregate_min) + usample * ((1/aggregate_max) - (1/aggregate_min));
                }
            __syncthreads();
            }

            if(tid==0) {
            global_states[bid] = localState;
            }
        }
        // Explicit instantiation
        template __global__ void sampleStepSize<double, curandState, 32>(double*, double*, double*, double*, int, int, curandState*);
        template __global__ void sampleStepSize<double, curandState, 64>(double*, double*, double*, double*, int, int, curandState*);
        template __global__ void sampleStepSize<double, curandState, 128>(double*, double*, double*, double*, int, int, curandState*);
        template __global__ void sampleStepSize<double, curandState, 256>(double*, double*, double*, double*, int, int, curandState*);
        template __global__ void sampleStepSize<double, curandState, 512>(double*, double*, double*, double*, int, int, curandState*);
        template __global__ void sampleStepSize<double, curandState, 1024>(double*, double*, double*, double*, int, int, curandState*);

        template <typename Real, typename PRNGenerator>
        void launchSampleStepSize(Real* AD, Real* slack, Real* alpha, Real* b, int rows, int cols, PRNGenerator* global_states, int ThreadsPerBlock, int BlocksPerGrid, cudaStream_t stream){
            switch(ThreadsPerBlock){
                case 32:
                    sampleStepSize<Real, PRNGenerator, 32><<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(AD, slack, alpha, b, rows, cols, global_states);
                    break;
                case 64:
                    sampleStepSize<Real, PRNGenerator, 64><<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(AD, slack, alpha, b, rows, cols, global_states);
                    break;
                case 128:
                    sampleStepSize<Real, PRNGenerator, 128><<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(AD, slack, alpha, b, rows, cols, global_states);
                    break;
                case 256:
                    sampleStepSize<Real, PRNGenerator, 256><<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(AD, slack, alpha, b, rows, cols, global_states);
                    break;
                case 512:
                    sampleStepSize<Real, PRNGenerator, 512><<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(AD, slack, alpha, b, rows, cols, global_states);
                    break;
                case 1024:
                    sampleStepSize<Real, PRNGenerator, 1024><<<BlocksPerGrid, ThreadsPerBlock, 0, stream>>>(AD, slack, alpha, b, rows, cols, global_states);
                    break;
                default:
                    break;
            }
            CUDA_CHECK(cudaGetLastError());
        }


        template <typename Real>
        __global__ void XupdateNoSave(Real* X, Real* D, Real* alpha,  int rows, int cols){
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int col = blockIdx.y;
            for (int j = col; j < cols; j += gridDim.y){
                for (int i = idx; i < rows; i += blockDim.x * gridDim.x){
                    X[IDX2C(i, j, rows)] += alpha[j] * D[IDX2C(i, j, rows)];
                }
            }
        }


        template<typename Real>
        __global__ void XupdateWithSaveIterantArray(Real* X, Real* D, Real* alpha, Real* Xsave, int rows, int cols, int* t, int thinning){
            int tid = threadIdx.x;                  // Thread ID in a block
            int bid = blockIdx.x;                   // Block ID
            int blockstride = gridDim.x;
            int threadstride = blockDim.x;
            __shared__ bool save;
            __shared__ int colOffset;

            if(tid==0){
                if(t[bid] % thinning == 0){
                    save = true;
                    colOffset = static_cast<int>(t[bid] / thinning) * cols;
                } else{ save = false; }
                t[bid]++;
            }
            __syncthreads();
            for (int j = bid; j < cols; j += blockstride){
                for (int i = tid; i < rows; i+=threadstride){
                    X[IDX2C(i, j, rows)] += alpha[j] * D[IDX2C(i, j, rows)];
                    if(save){
                        Xsave[IDX2C(i, j + colOffset, rows)] = X[IDX2C(i, j, rows)];
                    }
                }
            }

        }
    }




}