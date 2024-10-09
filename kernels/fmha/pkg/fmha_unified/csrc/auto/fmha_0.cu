// naive FA-2
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdio>
#include "cuda_fp16.h"


namespace fmha_unified {

template <const int N_block, const int N_warp>
__global__ void finefmha_0_kernel(const at::Half* Q, const at::Half* K,
                                  const at::Half* V, const int H_q,
                                  const int H_kv, const int N_q, const int N_kv, const int d, const at::Half softmax_scale,
                                  at::Half* O, bool causal, bool is_decode, int64_t decode_length) {
    // Q O [B(batch_size), H(head), N_q(seq_length), d(head_embed)]
    // K V [B(batch_size), H(head), N_kv(seq_length), d(head_embed)]
    const half* q = reinterpret_cast<const half*>(Q);
    const half* k = reinterpret_cast<const half*>(K);
    const half* v = reinterpret_cast<const half*>(V);
    half* o = reinterpret_cast<half*>(O);
    const half scale = *reinterpret_cast<const half*>(&softmax_scale);

    const int token_block_idx = blockIdx.x;
    const int head_block_idx = blockIdx.y;
    const int batch_block_idx = blockIdx.z;

    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;
    const int idx_in_warp = tid % 32;

    const int q_kv_ratio = H_q / H_kv;

    // deal with GQA here
    const half* q_block = q + batch_block_idx * (H_q * N_q * d) +
                          head_block_idx * (N_q * d) +
                          token_block_idx * (N_block * d);
    const half* k_block = k + batch_block_idx * (H_kv * N_kv * d) +
                          (head_block_idx / q_kv_ratio) * (N_kv * d);
    const half* v_block = v + batch_block_idx * (H_kv * N_kv * d) +
                          (head_block_idx / q_kv_ratio) * (N_kv * d);
    half* o_block = o + batch_block_idx * (H_q * N_q * d) +
                    head_block_idx * (N_q * d) + token_block_idx * (N_block * d);

    // not for k and v, they work on the whole N of K and V
    const half* q_warp = q_block + warp_idx * (N_warp * d);
    half* o_warp = o_block + warp_idx * (N_warp * d);

    __shared__ half q_smem[4][N_warp][128];
    // no transpose here
    __shared__ half k_smem[N_warp][16];
    __shared__ half v_smem[N_warp][128];
    __shared__ half s_p_smem[4][N_warp][N_warp];
    __shared__ half o_smem[4][N_warp][128];
    __shared__ half m_smem[4][N_warp];
    __shared__ half l_smem[4][N_warp];
    __shared__ half m_diff_smem[4][N_warp];

    half left_reg[4] = {__float2half(0.0)};
    half up_reg[2] = {__float2half(0.0)};
    half mma_result[4][2]{__float2half(0.0)};

    half pv_left_reg[4] = {__float2half(0.0)};
    half pv_up_reg[16] = {__float2half(0.0)};
    half pv_mma_result[4][16] = {__float2half(0.0)};

    // Set register arrays to zero
    for (int i = 0; i < 4; i++) {
        left_reg[i] = __float2half(0.0);
        for (int j = 0; j < 2; j++) {
            mma_result[i][j] = __float2half(0.0);
        }
    }
    for (int i = 0; i < 2; i++) {
        up_reg[i] = __float2half(0.0);
    }

    for (int i = 0; i < 4; i++) {
        pv_left_reg[i] = __float2half(0.0);
        for (int j = 0; j < 16; j++) {
            pv_mma_result[i][j] = __float2half(0.0);
        }
    }

    for (int i = 0; i < 16; i++) {
        pv_up_reg[i] = __float2half(0.0);
    }

    auto& warp_q_smem = q_smem[warp_idx];

    auto& warp_s_p_smem = s_p_smem[warp_idx];
    auto& warp_o_smem = o_smem[warp_idx];
    auto& warp_m_smem = m_smem[warp_idx];
    auto& warp_l_smem = l_smem[warp_idx];
    auto& warp_m_diff_smem = m_diff_smem[warp_idx];

    for (int i = 0; i < (N_block / (N_warp * 4)); i++) {
        // init
        // o = 0
        if (idx_in_warp < N_warp) {
            for (int k = 0; k < d; k++) {
                warp_o_smem[idx_in_warp][k] = __float2half(0.0);
            }
            // l = 0
            warp_l_smem[idx_in_warp] = __float2half(0.0);
            // m = -inf
            warp_m_smem[idx_in_warp] = __hneg(__float2half(INFINITY));
        }
        int j_max = 0;
        if (causal) {
            j_max = min(token_block_idx * (N_block / N_warp) + i * 4 + warp_idx + 4, (N_kv / N_warp));
        } else {
            j_max = (N_kv / N_warp);
        }
        for (int j = 0; j < j_max; j++) {
            // 1. get S
            // for 1.1 load q and k
            // for 1.2 accumulate s
            // 1.3 store s back to smem
            for (int k = 0; k < (d / 16); k++) {
                if (j == 0) {
                    for (int m = 0; m < (N_warp / (32 / 16)); m++) {
                        const int smem_x = m * (32 / 16) + idx_in_warp / 16;
                        const int smem_y = idx_in_warp % 16;
                        warp_q_smem[smem_x][smem_y + k * 16] =
                            *(q_warp + (smem_x + i * (N_warp * 4)) * d +
                              (smem_y + k * 16));
                    }
                }
                for (int m = 0; m < (N_warp / (128 / 16)); m++) {
                    const int smem_x = m * (128 / 16) + tid / 16;
                    const int smem_y = tid % 16;
                    k_smem[smem_x][smem_y] =
                        *(k_block + (smem_x + j * N_warp) * d +
                          (smem_y + k * 16));
                }
                __syncthreads();
                for (int m = 0; m < 16; m++) {
                    // within a warp, each tiny block is 4 * 2, and each thread
                    // is responsible for 4 of these blocks
                    const int tiny_x = idx_in_warp / 8;
                    const int tiny_y = idx_in_warp % 8;
                    for (int p = 0; p < 2; p++) {
                        left_reg[p] = warp_q_smem[tiny_x * 2 + p][m + k * 16];
                        left_reg[p + 2] =
                            warp_q_smem[tiny_x * 2 + p + 8][m + k * 16];
                    }
                    for (int q = 0; q < 1; q++) {
                        const int logical_x = m;
                        const int logical_y = tiny_y * 1 + q;
                        up_reg[q] = k_smem[logical_y][logical_x];
                        up_reg[q + 1] = k_smem[logical_y + 8][logical_x];
                    }
                    for (int p = 0; p < 2 * 2; p++) {
                        for (int q = 0; q < 1 * 2; q++) {
                            mma_result[p][q] = __hadd(mma_result[p][q], __hmul(left_reg[p], up_reg[q]));
                        }
                    }
                }
                __syncthreads();
            }
            for (int m = 0; m < 2; m++) {
                for (int n = 0; n < 2; n++) {
                    for (int p = 0; p < 2; p++) {
                        for (int q = 0; q < 1; q++) {
                            const int reg_x = m * 2 + p;
                            const int reg_y = n * 1 + q;
                            const int smem_x =
                                m * 8 + (idx_in_warp / 8) * 2 + p;
                            const int smem_y =
                                n * 8 + (idx_in_warp % 8) * 1 + q;
                            if (causal && token_block_idx * N_block + warp_idx * N_warp + smem_x < j * 16 + smem_y) {
                                warp_s_p_smem[smem_x][smem_y] = __hneg(__float2half(INFINITY));
                            } else if (is_decode && j * 16 + smem_y > decode_length) {
                                warp_s_p_smem[smem_x][smem_y] = __hneg(__float2half(INFINITY));
                            } else {
                                warp_s_p_smem[smem_x][smem_y] = __hmul(scale, mma_result[reg_x][reg_y]);
                            }
                        }
                    }
                }
            }

            __syncthreads();

            // Set register arrays to zero
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 2; j++) {
                    mma_result[i][j] = __float2half(0.0);
                }
            }

            // 2. get m
            if (idx_in_warp < 16) {
                half previous_max = warp_m_smem[idx_in_warp];
                for (int k = 0; k < 16; k++) {
                    warp_m_smem[idx_in_warp] =
                        __hmax(warp_m_smem[idx_in_warp],
                               warp_s_p_smem[idx_in_warp][k]);
                }
                warp_m_diff_smem[idx_in_warp] =
                    __hadd(previous_max, __hneg(warp_m_smem[idx_in_warp]));
            }

            // 3. get p in place
            for (int m = 0; m < (16 / (32 / 16)); m++) {
                const int x_offset = m * 2 + idx_in_warp / 16;
                const int y_offset = idx_in_warp % 16;
                warp_s_p_smem[x_offset][y_offset] = hexp(__hadd(warp_s_p_smem[x_offset][y_offset], __hneg(warp_m_smem[x_offset])));
            }
            // 4. get l
            if (idx_in_warp < 16) {
                half row_sum = __float2half(0.0);
                for (int m = 0; m < 16; m++) {
                    row_sum = __hadd(warp_s_p_smem[idx_in_warp][m], row_sum);
                }
                warp_l_smem[idx_in_warp] = __hadd(__hmul(hexp(warp_m_diff_smem[idx_in_warp]), warp_l_smem[idx_in_warp]), row_sum);
            }

            // 5. get o
            // for 5.1 load p and v
            // for 5.2 accumulate o
            // 5.3 store o back to smem
            for (int m = 0; m < 16; m++) {
                for (int n = 0; n < (d / 128); n++) {
                    const int smem_x = m;
                    const int smem_y = n * 32 + tid;
                    const int dram_x = j * 16 + m;
                    const int dram_y = n * 32 + tid;
                    v_smem[smem_x][smem_y] = *(v_block + (dram_x)*d + (dram_y));
                }
            }
            __syncthreads();
            for (int k = 0; k < 16; k++) {
                for (int m = 0; m < 2; m++) {
                    pv_left_reg[m] =
                        warp_s_p_smem[m + 2 * (idx_in_warp / 8)][k];
                    pv_left_reg[m + 2] =
                        warp_s_p_smem[m + 2 * (idx_in_warp / 8) + 8][k];
                }
                for (int m = 0; m < 8; m++) {
                    pv_up_reg[m] = v_smem[k][m + 8 * (idx_in_warp % 8)];
                    pv_up_reg[m + 8] =
                        v_smem[k][m + 8 * (idx_in_warp % 8) + 64];
                }

                for (int m = 0; m < 2 * 2; m++) {
                    for (int n = 0; n < 8 * 2; n++) {
                        pv_mma_result[m][n] = __hadd(pv_mma_result[m][n], __hmul(pv_left_reg[m], pv_up_reg[n]));
                    }
                }
            }

            for (int m = 0; m < 2; m++) {
                for (int n = 0; n < 2; n++) {
                    for (int p = 0; p < 2; p++) {
                        for (int q = 0; q < 8; q++) {
                            const int reg_x = m * 2 + p;
                            const int reg_y = n * 8 + q;
                            const int smem_x =
                                m * 8 + (idx_in_warp / 8) * 2 + p;
                            const int smem_y =
                                n * 64 + (idx_in_warp % 8) * 8 + q;
                            warp_o_smem[smem_x][smem_y] =
                                __hadd(pv_mma_result[reg_x][reg_y], __hmul(warp_o_smem[smem_x][smem_y], hexp(warp_m_diff_smem[smem_x])));
                        }
                    }
                }
            }
            __syncthreads();
            for (int i = 0; i < 4; i++) {
                pv_left_reg[i] = __float2half(0.0);
                for (int j = 0; j < 16; j++) {
                    pv_mma_result[i][j] = __float2half(0.0);
                }
            }
        }
        // adjust o and write to final output
        if (idx_in_warp < 16) {
            for (int k = 0; k < d; k++) {
                warp_o_smem[idx_in_warp][k] =
                    __hmul(warp_o_smem[idx_in_warp][k], __hdiv(__int2half_rd(1), warp_l_smem[idx_in_warp]));
            }
        }
        // write to o_warp
        for (int m = 0; m < 16; m++) {
            for (int n = 0; n < (d / 32); n++) {
                const int smem_x = m;
                const int smem_y = n * 32 + idx_in_warp;
                *(o_warp + (i * (N_warp * 4)) * d + (smem_x)*d + smem_y) =
                    warp_o_smem[smem_x][smem_y];
            }
        }
        __syncthreads();
    }

    return;
}

torch::Tensor finefmha_0(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                         bool causal, bool is_decode, int64_t decode_length) {
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N_q = Q.size(2);
    const int N_kv = K.size(2);
    const int d = Q.size(3);

    const int H_kv = K.size(1);
    const float softmax_scale = 1.0 / sqrt(d);

    TORCH_CHECK(Q.dtype() == at::kHalf);
    TORCH_CHECK(K.dtype() == at::kHalf);
    TORCH_CHECK(V.dtype() == at::kHalf);
    TORCH_INTERNAL_ASSERT(Q.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(K.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(V.device().type() == at::DeviceType::CUDA);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q, Q.options());

    dim3 grid_dim(ceil(N_q / 64), H, B);  // batch_size x num_heads
    dim3 block_dim(128);                 // Bc threads per block

    finefmha_0_kernel<64, 16><<<grid_dim, block_dim>>>(
        Q.data_ptr<at::Half>(), K.data_ptr<at::Half>(), V.data_ptr<at::Half>(),
        H, H_kv, N_q, N_kv, d, softmax_scale, O.data_ptr<at::Half>(),
        causal, is_decode, decode_length);
    return O;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(fmha_unified, CUDA, m) { m.impl("finefmha_0", &finefmha_0); }

}  // namespace fmha_unified