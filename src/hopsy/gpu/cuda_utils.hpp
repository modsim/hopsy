#ifndef HOPSY_GPU_CUDA_UTILS_HPP
#define HOPSY_GPU_CUDA_UTILS_HPP

// Small CUDA helpers: column-major indexing, RAII wrappers for streams and
// graphs, launch configuration.

#include <cuda_runtime.h>

#include <exception>
#include <utility>

#include "cuda_check.hpp"

// Column-major index of element (i, j) with leading dimension ld.
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

namespace hopsy {
    namespace gpu {
        class CudaStream {
        public:
            explicit CudaStream(unsigned int flags = cudaStreamDefault) {
                CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
            }

            ~CudaStream() {
                if (stream_ != nullptr) {
                    CUDA_CHECK_NOEXCEPT(cudaStreamDestroy(stream_));
                }
            }

            // Deleting copy also prevents implicit creation of move
            CudaStream(const CudaStream &) = delete;
            CudaStream &operator=(const CudaStream &) = delete;

            // Implicit conversion
            operator cudaStream_t() const { return stream_; }

        private:
            cudaStream_t stream_ = nullptr;
        };

        // Executable CUDA graph, recorded by stream capture.
        // Non-Copyable and Non-Movable
        // Example use:
        //     CudaStream stream;
        //     CudaGraph step(stream, [&](cudaStream_t s) {
        //         kernel<<<grid, block, 0, s>>>(...);
        //         handle.GeMM(...);   // cuBLAS handle set to `stream` beforehand
        //     });
        //     for (int i = 0; i < n; ++i) step.launch(stream);
        //
        // Only the instantiated cudaGraphExec_t is kept; the captured cudaGraph_t is
        // destroyed right after instantiation (the executable graph does not depend
        // on it). If `record` throws, the capture is ended and the partial graph is
        // destroyed before the exception propagates, so `stream` stays usable.
        class CudaGraph {
        public:
            /**
             * @brief Captures the work issued by `record` on `stream` into an
             *        executable CUDA graph.
             *
             * @tparam Record Callable that captures graph; Passed in as universal ref
             *         to accept mutable lambdas. Refer
             *         https://johannesugb.github.io/cpu-programming/how-to-pass-lambda-functions-in-C++/
             * @param stream Stream to capture on. All work in the graph must be
             *        issued on this stream.
             * @param record Called once as `record(stream)` during capture.
             * @param mode   Stream capture mode, forwarded to
             *        `cudaStreamBeginCapture`. Defaults to `cudaStreamCaptureModeGlobal`.
             *
             * @throws std::runtime_error if capture or instantiation fails, or
             *         rethrows whatever `record` throws (the partial graph is
             *         destroyed first, so `stream` stays usable).
             */
            template<typename Record>
            CudaGraph(cudaStream_t stream, Record &&record,
                      cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal) {
                CUDA_CHECK(cudaStreamBeginCapture(stream, mode));
                try {
                    record(stream);
                } catch (...) {
                    cudaGraph_t partial = nullptr;
                    CUDA_CHECK_NOEXCEPT(cudaStreamEndCapture(stream, &partial));
                    if (partial != nullptr) {
                        CUDA_CHECK_NOEXCEPT(cudaGraphDestroy(partial));
                    }
                    throw;
                }
                cudaGraph_t graph = nullptr;
                CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
                cudaError_t err = cudaGraphInstantiate(&exec_, graph, 0);
                CUDA_CHECK_NOEXCEPT(cudaGraphDestroy(graph));
                CUDA_CHECK(err);
            }

            ~CudaGraph() {
                if (exec_ != nullptr) {
                    CUDA_CHECK_NOEXCEPT(cudaGraphExecDestroy(exec_));
                }
            }

            CudaGraph(const CudaGraph &) = delete;
            CudaGraph &operator=(const CudaGraph &) = delete;

            // Enqueue one run of the graph on `stream` (asynchronous).
            void launch(cudaStream_t stream = 0) const {
                CUDA_CHECK(cudaGraphLaunch(exec_, stream));
            }

        private:
            cudaGraphExec_t exec_ = nullptr;
        };

        /**
         * @brief Block size that maximizes occupancy for `kernel` on the current
         *        device (cudaOccupancyMaxPotentialBlockSize).
         *
         * The result is a multiple of the warp size but not necessarily a power
         * of two. The grid size is left to the caller, since it depends on the
         * problem size, e.g. `(n + blockSize - 1) / blockSize`.
         *
         * @tparam Kernel Kernel function type (pointer to a `__global__` function).
         *
         * @param kernel              Kernel to compute the block size for.
         * @param dynamicSharedMemory Dynamic shared memory per block in bytes, as
         *                            passed in the launch.
         * @param blockSizeLimit      Upper bound on the block size; 0 means no limit.
         *
         * @return Threads per block.
         *
         * @throws std::runtime_error if the occupancy query fails.
         */
        template<typename Kernel>
        int optimalBlockSize(Kernel kernel, size_t dynamicSharedMemory = 0,
                             int blockSizeLimit = 0) {
            int minGridSize = 0;
            int blockSize = 0;
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
                    &minGridSize, &blockSize, kernel, dynamicSharedMemory,
                    blockSizeLimit));
            return blockSize;
        }
    }// namespace gpu
}// namespace hopsy

#endif// HOPSY_GPU_CUDA_UTILS_HPP
