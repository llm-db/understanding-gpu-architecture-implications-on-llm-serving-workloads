#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdio>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#include "rocwmma/internal/types.hpp"

namespace gemm_unified {

using namespace rocwmma;

__global__ void finegemm_11_kernel(int M, int K, int N, const at::Half *A,
                                   const at::Half *B, at::Half *result) {
    __shared__ half a_smem[2][128][16];
    __shared__ half b_smem[2][16][128];

    const half* a = reinterpret_cast<const half*>(A);
    const half* b = reinterpret_cast<const half*>(B);
    half* d = reinterpret_cast<half*>(result);

    fragment<matrix_a, 32, 32, 16, half, col_major> a_frag[2];
    fragment<matrix_b, 32, 32, 16, half, row_major> b_frag[2];
    fragment<accumulator, 32, 32, 16, half, row_major> c_frag[2][2];

    int4 a_tmp;
    int4 b_tmp;


    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            fill_fragment(c_frag[i][j], __float2half(0.0));
        }
    }

    const int tid = threadIdx.x;
    const int warp_id = threadIdx.x / 64;
    const int warp_x = warp_id / 2;
    const int warp_y = warp_id % 2;

    const int c_dram_x = blockIdx.x * 128;
    const int c_dram_y = blockIdx.y * 128;

    {
        // preload the first part
        int k = 0;
        {
            // load A into smem
            const int a_smem_x = tid / 2;
            const int a_smem_y = 8 * (tid % 2);
            const int a_dram_x = a_smem_x + c_dram_x;
            const int a_dram_y = a_smem_y + k * 16;
            *reinterpret_cast<int4 *>(&a_smem[0][a_smem_x][a_smem_y]) = *reinterpret_cast<const int4 *>(a + (a_dram_x) * K + (a_dram_y));
        }
        {
            // load B into smem
            const int b_smem_x = tid / 16;
            const int b_smem_y = 8 * (tid % 16);
            const int b_dram_x = b_smem_x + k * 16;
            const int b_dram_y = b_smem_y + c_dram_y;
            *reinterpret_cast<int4 *>(&b_smem[0][b_smem_x][b_smem_y]) = *reinterpret_cast<const int4 *>(b + (b_dram_x) * N + (b_dram_y));
        }
    }
    __syncthreads();

    for (int k = 0; k < (K / 16); k++) {
        int k_stage = k % 2;
        // load next stage
        int k_next = k + 1;
        int k_next_stage = k_next % 2;
        k_next = min(k_next, (K/16) - 1);
        {
            {
                // load A into smem
                const int a_smem_x = tid / 2;
                const int a_smem_y = 8 * (tid % 2);
                const int a_dram_x = a_smem_x + c_dram_x;
                const int a_dram_y = a_smem_y + k_next * 16;
                a_tmp = *reinterpret_cast<const int4 *>(a + (a_dram_x) * K + (a_dram_y));
            }
            {
                // load B into smem
                const int b_smem_x = tid / 16;
                const int b_smem_y = 8 * (tid % 16);
                const int b_dram_x = b_smem_x + k_next * 16;
                const int b_dram_y = b_smem_y + c_dram_y;
                b_tmp = *reinterpret_cast<const int4 *>(b + (b_dram_x) * N + (b_dram_y));
                
            }

        }
        // __syncthreads();
        const int a_smem_x_offset = warp_x * 64;
        const int b_smem_y_offset = warp_y * 64;
        for (int m = 0; m < 2; m++) {
            load_matrix_sync(a_frag[m], &(a_smem[k_stage][a_smem_x_offset + m * 32][0]), 16);
        }

        for (int n = 0; n < 2; n++) {
            load_matrix_sync(b_frag[n], &(b_smem[k_stage][0][b_smem_y_offset + 32 * n]), 128);
        }

        for (int m = 0; m < 2; m++) {
            for (int n = 0; n < 2; n++) {
                mma_sync(c_frag[m][n], a_frag[m], b_frag[n], c_frag[m][n]);
            }
        }

        {
            const int a_smem_x = tid / 2;
            const int a_smem_y = 8 * (tid % 2);
            *reinterpret_cast<int4 *>(&a_smem[k_next_stage][a_smem_x][a_smem_y]) = a_tmp;
            const int b_smem_x = tid / 16;
            const int b_smem_y = 8 * (tid % 16);
            *reinterpret_cast<int4 *>(&b_smem[k_next_stage][b_smem_x][b_smem_y]) = b_tmp;
        }
        __syncthreads();
    }


    for (int m = 0; m < 2; m++) {
        for (int n = 0; n < 2; n++) {
            store_matrix_sync((d + (c_dram_x + warp_x * 64 + m * 32) * N + (c_dram_y + warp_y * 64 + n * 32)), c_frag[m][n], N);
        }
    }
}

at::Tensor finegemm_11(const at::Tensor &a, const at::Tensor &b) {
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
    dim3 block(64 * 4);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    finegemm_11_kernel<<<grid, block>>>(M, K, N, a_ptr, b_ptr, result_ptr);

    return result;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(gemm_unified, CUDA, m) {
    m.impl("finegemm_11", &finegemm_11);
}

}  // namespace gemm_unified