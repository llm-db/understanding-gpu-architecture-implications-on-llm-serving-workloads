// This kernel we will do warpsplit -1 (adjusting which thread is loading which
// part of SMEM to REG)

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cassert>
#include <cstdio>

#include "c10/util/Exception.h"

namespace gemm_unified {

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void finegemm_6_kernel(int M, int K, int N, const at::Half *a,
                                  const at::Half *b, at::Half *result) {
    const half *a_ptr = reinterpret_cast<const half *>(a);
    const half *b_ptr = reinterpret_cast<const half *>(b);
    half *result_ptr = reinterpret_cast<half *>(result);
    // const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int id_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;

    // load into register, we are doing warp splitting here
    // we assume a [8][4] warp structure
    const int warp_id = id_in_block / 32;     // in [0, 8)
    const int id_in_warp = id_in_block % 32;  // in [0, 32)
    const int thread_result_x =
        (blockIdx.x * blockDim.x + (warp_id % 4) * 4 + id_in_warp % 4) * TN;
    const int thread_result_y =
        (blockIdx.y * blockDim.y + (warp_id / 4) * 8 + id_in_warp / 4) * TM;
    // const int thread_result_x = idx * TN;
    // const int thread_result_y = idy * TM;

    __shared__ half a_shared[BK][BM];
    __shared__ half b_shared[BK][BN];
    // registers to hold TM * TN mma
    half TM_reg[TM] = {__int2half_rd(0)};
    half TN_reg[TN] = {__int2half_rd(0)};
    half TM_TN_reg[TM][TN] = {__int2half_rd(0)};
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            TM_TN_reg[i][j] = __int2half_rd(0);
        }
    }

    // tile by tile compute -> potential room for split-k
    for (int k_dim_base = 0; k_dim_base < K; k_dim_base += BK) {
        // load into shared A
        const int a_block_y = blockIdx.y * BM;
        const int a_block_x =
            k_dim_base;  // as we assumed this is always dividable
        const half *a_block = a_ptr + a_block_y * K + a_block_x;
        const int slice_per_row_a = BK / 8;
        const int a_thread_y_offset = id_in_block / slice_per_row_a;
        const int a_thread_x_offset = id_in_block % slice_per_row_a;
        const int a_block_y_stepsize = total_threads / slice_per_row_a;
        for (int a_block_y_step = 0; a_block_y_step < BM;
             a_block_y_step += a_block_y_stepsize) {
            const int y_to_load = a_block_y_step + a_thread_y_offset;
            const int x_to_load = a_thread_x_offset * 8;
            if (y_to_load < BM) {
                if (y_to_load + a_block_y < M) {
                    const float4 tmp = *reinterpret_cast<const float4 *>(
                        a_block + y_to_load * K + x_to_load);
                    const half2 tmp_x =
                        *reinterpret_cast<const half2 *>(&tmp.x);
                    const half2 tmp_y =
                        *reinterpret_cast<const half2 *>(&tmp.y);
                    const half2 tmp_z =
                        *reinterpret_cast<const half2 *>(&tmp.z);
                    const half2 tmp_w =
                        *reinterpret_cast<const half2 *>(&tmp.w);
                    a_shared[x_to_load][y_to_load] =
                        *reinterpret_cast<const half *>(&tmp_x.x);
                    a_shared[x_to_load + 1][y_to_load] =
                        *reinterpret_cast<const half *>(&tmp_x.y);
                    a_shared[x_to_load + 2][y_to_load] =
                        *reinterpret_cast<const half *>(&tmp_y.x);
                    a_shared[x_to_load + 3][y_to_load] =
                        *reinterpret_cast<const half *>(&tmp_y.y);
                    a_shared[x_to_load + 4][y_to_load] =
                        *reinterpret_cast<const half *>(&tmp_z.x);
                    a_shared[x_to_load + 5][y_to_load] =
                        *reinterpret_cast<const half *>(&tmp_z.y);
                    a_shared[x_to_load + 6][y_to_load] =
                        *reinterpret_cast<const half *>(&tmp_w.x);
                    a_shared[x_to_load + 7][y_to_load] =
                        *reinterpret_cast<const half *>(&tmp_w.y);
                } else {
                    a_shared[x_to_load][y_to_load] = __int2half_rd(0);
                    a_shared[x_to_load + 1][y_to_load] = __int2half_rd(0);
                    a_shared[x_to_load + 2][y_to_load] = __int2half_rd(0);
                    a_shared[x_to_load + 3][y_to_load] = __int2half_rd(0);
                    a_shared[x_to_load + 4][y_to_load] = __int2half_rd(0);
                    a_shared[x_to_load + 5][y_to_load] = __int2half_rd(0);
                    a_shared[x_to_load + 6][y_to_load] = __int2half_rd(0);
                    a_shared[x_to_load + 7][y_to_load] = __int2half_rd(0);
                }
            }
        }
        // load into shared B
        const int b_block_x = blockIdx.x * BN;
        const int b_block_y = k_dim_base;
        const half *b_block = b_ptr + b_block_y * N + b_block_x;
        const int slice_per_row_b = BN / 8;
        const int b_thread_x_offset = id_in_block % slice_per_row_b;
        const int b_thread_y_offset = id_in_block / slice_per_row_b;
        const int b_block_y_stepsize = total_threads / slice_per_row_b;
        for (int b_block_y_step = 0; b_block_y_step < BK;
             b_block_y_step += b_block_y_stepsize) {
            const int y_to_load = b_block_y_step + b_thread_y_offset;
            const int x_to_load = b_thread_x_offset * 8;
            if (x_to_load < BN && y_to_load < BK) {
                if (x_to_load + b_block_x < N) {
                    const float4 tmp = *reinterpret_cast<const float4 *>(
                        b_block + y_to_load * N + x_to_load);
                    const half2 tmp_x =
                        *reinterpret_cast<const half2 *>(&tmp.x);
                    const half2 tmp_y =
                        *reinterpret_cast<const half2 *>(&tmp.y);
                    const half2 tmp_z =
                        *reinterpret_cast<const half2 *>(&tmp.z);
                    const half2 tmp_w =
                        *reinterpret_cast<const half2 *>(&tmp.w);
                    b_shared[y_to_load][x_to_load] =
                        *reinterpret_cast<const half *>(&tmp_x.x);
                    b_shared[y_to_load][x_to_load + 1] =
                        *reinterpret_cast<const half *>(&tmp_x.y);
                    b_shared[y_to_load][x_to_load + 2] =
                        *reinterpret_cast<const half *>(&tmp_y.x);
                    b_shared[y_to_load][x_to_load + 3] =
                        *reinterpret_cast<const half *>(&tmp_y.y);
                    b_shared[y_to_load][x_to_load + 4] =
                        *reinterpret_cast<const half *>(&tmp_z.x);
                    b_shared[y_to_load][x_to_load + 5] =
                        *reinterpret_cast<const half *>(&tmp_z.y);
                    b_shared[y_to_load][x_to_load + 6] =
                        *reinterpret_cast<const half *>(&tmp_w.x);
                    b_shared[y_to_load][x_to_load + 7] =
                        *reinterpret_cast<const half *>(&tmp_w.y);
                    // b_shared[y_to_load][x_to_load] = b_block[y_to_load * N +
                    // x_to_load];
                } else {
                    b_shared[y_to_load][x_to_load] = __int2half_rd(0);
                    b_shared[y_to_load][x_to_load + 1] = __int2half_rd(0);
                    b_shared[y_to_load][x_to_load + 2] = __int2half_rd(0);
                    b_shared[y_to_load][x_to_load + 3] = __int2half_rd(0);
                    b_shared[y_to_load][x_to_load + 4] = __int2half_rd(0);
                    b_shared[y_to_load][x_to_load + 5] = __int2half_rd(0);
                    b_shared[y_to_load][x_to_load + 6] = __int2half_rd(0);
                    b_shared[y_to_load][x_to_load + 7] = __int2half_rd(0);
                    // b_shared[y_to_load][x_to_load] = __int2half_rd(0);
                }
            }
        }

        __syncthreads();
        const int a_shared_y_offset =
            (8 * (warp_id / 4) + (id_in_warp / 4)) * TM;
        const int b_shared_x_offset =
            (4 * (warp_id % 4) + (id_in_warp % 4)) * TN;
        // const int a_shared_y_offset = threadIdx.y * TN;
        // const int b_shared_x_offset = threadIdx.x * TM;
        for (int K_loop = 0; K_loop < BK; K_loop++) {
#pragma unroll
            for (int i = 0; i < TM; i++) {
                TM_reg[i] = a_shared[K_loop][a_shared_y_offset + i];
            }
#pragma unroll
            for (int i = 0; i < TN; i++) {
                TN_reg[i] = b_shared[K_loop][b_shared_x_offset + i];
            }
#pragma unroll
            for (int i = 0; i < TM; i++) {
                half2 m_val2 = __half2half2(TM_reg[i]);
                for (int j = 0; j < TN; j += 2) {
                    half2 n_val2 = __halves2half2(TN_reg[j], TN_reg[j + 1]);
                    half2 mult2 = __hmul2(m_val2, n_val2);
                    TM_TN_reg[i][j] =
                        __hadd(TM_TN_reg[i][j], __low2half(mult2));
                    TM_TN_reg[i][j + 1] =
                        __hadd(TM_TN_reg[i][j + 1], __high2half(mult2));
                }
            }
        }
        __syncthreads();
    }

    // result writeback
    if (thread_result_x < N && thread_result_y < M) {
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < TN; j += 8) {
                const int write_x = thread_result_x + j;
                const int write_y = thread_result_y + i;
                float4 tmp;
                half2 tmp_x =
                    __halves2half2(TM_TN_reg[i][j], TM_TN_reg[i][j + 1]);
                half2 tmp_y =
                    __halves2half2(TM_TN_reg[i][j + 2], TM_TN_reg[i][j + 3]);
                half2 tmp_z =
                    __halves2half2(TM_TN_reg[i][j + 4], TM_TN_reg[i][j + 5]);
                half2 tmp_w =
                    __halves2half2(TM_TN_reg[i][j + 6], TM_TN_reg[i][j + 7]);
                tmp.x = *reinterpret_cast<float *>(&tmp_x);
                tmp.y = *reinterpret_cast<float *>(&tmp_y);
                tmp.z = *reinterpret_cast<float *>(&tmp_z);
                tmp.w = *reinterpret_cast<float *>(&tmp_w);
                reinterpret_cast<float4 *>(
                    &result_ptr[write_y * N + write_x])[0] = tmp;
            }
        }
    }
}

at::Tensor finegemm_6(const at::Tensor &a, const at::Tensor &b) {
    long M = a.size(0);
    long K = a.size(1);
    long N = b.size(1);
    TORCH_CHECK(K == b.size(0));
    TORCH_CHECK(a.dtype() == at::kHalf);
    TORCH_CHECK(b.dtype() == at::kHalf);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);

    // for half2 data type, we assume K is even
    TORCH_CHECK(K % 2 == 0);

    // assume M, N, K are divisible by TM, TN
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    TORCH_CHECK(M % TM == 0);
    TORCH_CHECK(N % TN == 0);
    TORCH_CHECK(K % BK == 0);
    TORCH_CHECK(BK % 8 == 0);
    TORCH_CHECK(BM % 8 == 0);
    TORCH_CHECK(BN % 8 == 0);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::zeros({M, N}, a_contig.options());
    const at::Half *a_ptr = a_contig.data_ptr<at::Half>();
    const at::Half *b_ptr = b_contig.data_ptr<at::Half>();
    at::Half *result_ptr = result.data_ptr<at::Half>();
    int numel = a_contig.numel();

    // Launch the kernel
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM) / BM);
    finegemm_6_kernel<BM, BN, BK, 8, 8>
        <<<grid, block>>>(M, K, N, a_ptr, b_ptr, result_ptr);
    cudaDeviceSynchronize();
    return result;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(gemm_unified, CUDA, m) { m.impl("finegemm_6", &finegemm_6); }

}  // namespace gemm_unified