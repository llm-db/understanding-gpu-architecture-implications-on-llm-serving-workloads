#include <cuda_runtime.h>
#include <torch/extension.h>
#include <rocwmma/rocwmma.hpp>


namespace gemm_unified {

// Helper macro for HIP errors
#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(expression)                                     \
    if (auto status = (expression); status != hipSuccess) {             \
        fprintf(stderr, "hip error: '%s'(%d) at %s:%d\n",               \
                hipGetErrorString(status), status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                             \
    }
#endif


const int ROCWMMA_M = 32;
const int ROCWMMA_N = 32;

const int ROCWMMA_K = 16;

__global__ void finegemm_8_kernel(int M, int K, int N, const at::Half *A,
                                  const at::Half *B, at::Half *result) {
    using namespace rocwmma;
    const half *a = reinterpret_cast<const half *>(A);
    const half *b = reinterpret_cast<const half *>(B);
    half *c = reinterpret_cast<half *>(result);
    half *d = reinterpret_cast<half *>(result);

    int lda = K;
    int ldb = N;
    int ldd = N;
    // find out this block

    auto fragA = rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                                   half, row_major>();
    auto fragB = rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K,
                                   half, row_major>();
    auto fragAcc = rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N,
                                     ROCWMMA_K, half>();
    // Tile using a 2D grid
    auto majorWarp = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
    auto minorWarp = (blockIdx.y * blockDim.y + threadIdx.y);

    // Target C block
    auto cRow = majorWarp * ROCWMMA_M;
    auto cCol = minorWarp * ROCWMMA_N;

    // Bounds check
    if (cRow < M && cCol < N) {
        // fragAcc = A x B
        for (int i = 0; i < K; i += ROCWMMA_K) {
            // Load the inputs
            rocwmma::load_matrix_sync(fragA, a + (cRow * lda + i), lda);
            rocwmma::load_matrix_sync(fragB, b + (i) * ldb + cCol, ldb);

            // Matrix multiply - accumulate using MFMA units
            rocwmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
        }

        // Store to D
        rocwmma::store_matrix_sync(d + (cRow * ldd + cCol), fragAcc, ldd,
                                   rocwmma::mem_row_major);
    }

}

at::Tensor finegemm_8(const at::Tensor &a, const at::Tensor &b) {
    long M = a.size(0);
    long K = a.size(1);
    long N = b.size(1);

    int lda = K;
    int ldb = K;
    int ldc = N;
    int ldd = ldc;

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::zeros({M, N}, a_contig.options());
    at::Half *a_ptr = a_contig.data_ptr<at::Half>();
    at::Half *b_ptr = b_contig.data_ptr<at::Half>();
    at::Half *result_ptr = result.data_ptr<at::Half>();
    
    const int BM = 4 * 32;
    const int BN = 4 * 32;
    dim3 block(256, 4);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    finegemm_8_kernel<<<grid, block>>>(M, K, N, a_ptr, b_ptr, result_ptr);

    return result;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(gemm_unified, CUDA, m) { m.impl("finegemm_8", &finegemm_8); }

}  // namespace gemm_unified