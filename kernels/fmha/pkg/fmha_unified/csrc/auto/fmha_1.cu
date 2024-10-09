// naive FA-2
// I got the block size wrong in previous versions, adjust to 64, 64
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "c10/core/ScalarType.h"
#include "cuda_fp16.h"


namespace fmha_unified {

template <const int N_block, const int N_warp, const bool causal,
          const bool is_decode>
__global__ void finefmha_1_kernel(const at::Half* Q, const at::Half* K,
                                  const at::Half* V, const int H_q,
                                  const int H_kv, const int N_q, const int N_kv,
                                  const int d, const at::Half softmax_scale,
                                  at::Half* O, int64_t decode_length) {
    // Q O [B(batch_size), H(head), N_q(seq_length), d(head_embed)]
    // K V [B(batch_size), H(head), N_kv(seq_length), d(head_embed)]
    const half* q = reinterpret_cast<const half*>(Q);
    const half* k = reinterpret_cast<const half*>(K);
    const half* v = reinterpret_cast<const half*>(V);
    half* o = reinterpret_cast<half*>(O);
    const half scale =
        *reinterpret_cast<const half*>(&softmax_scale);

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
                           head_block_idx * (N_q * d) +
                           token_block_idx * (N_block * d);

    // 16K each
    __shared__ half o_smem[4 * N_warp][128];
    __shared__ half q_smem[4 * N_warp][16];
    __shared__ half k_smem[4 * N_warp][16];
    __shared__ half v_smem[16][128];
    // no transpose here

    // 2K in total
    __shared__ half s_p_smem[64][64];
    __shared__ half m_smem[4 * N_warp];
    __shared__ half l_smem[4 * N_warp];
    __shared__ half m_diff_smem[4 * N_warp];

    half left_reg[8] = {__float2half(0.0)};
    half up_reg[8] = {__float2half(0.0)};
    half mma_result[8][8]{__float2half(0.0)};

// Set register arrays to zero

    for (int i = 0; i < 8; i++) {
        left_reg[i] = __float2half(0.0);
        for (int j = 0; j < 8; j++) {
            mma_result[i][j] = __float2half(0.0);
        }
    }

    for (int i = 0; i < 8; i++) {
        up_reg[i] = __float2half(0.0);
    }

    for (int i = 0; i < (N_block / (N_warp * 4)); i++) {
        // init
        // o = 0
        // for (int m = 0; m < 64; m++) {
        //     o_smem[m][tid] = 0;
        // }
        __syncthreads();
        for (int m = 0; m < 8; m++) {
            const int smem_x = 8 * m + tid / 16;
            const int smem_y = 8 * (tid % 16);
            *reinterpret_cast<int4*>(&(o_smem[smem_x][smem_y])) =
                make_int4(0, 0, 0, 0);
        }

        if (tid < 64) {
            l_smem[tid] = __float2half(0.0);
            m_smem[tid] = __hneg(__float2half(INFINITY));
        }

        int j_max = 0;
        if (causal) {
            const int max_i_block = token_block_idx * ceil(N_block / (4 * N_warp)) + 1;
            j_max = min(max_i_block, (N_kv / (4 * N_warp)));
        } else {
            j_max = ceil(N_kv / (4 * N_warp));
        }


        for (int j = 0; j < j_max; j++) {
            // 1. get S
            // for 1.1 load q and k
            // for 1.2 accumulate s
            // 1.3 store s back to smem
            for (int k = 0; k < (d / 16); k++) {
                {
                    // load q
                    const int smem_x = tid / 2;
                    const int smem_y = 8 * (tid % 2);
                    const int dram_x = i * 64 + smem_x;
                    const int dram_y = k * 16 + smem_y;
                    *reinterpret_cast<int4 *>(&q_smem[smem_x][smem_y]) = *reinterpret_cast<const int4 *>(q_block + (dram_x) * d + dram_y);
                }
                {
                    // load k
                    const int smem_x = tid / 2;
                    const int smem_y = 8 * (tid % 2);
                    const int dram_x = j * 64 + smem_x;
                    const int dram_y = k * 16 + smem_y;
                    *reinterpret_cast<int4 *>(&k_smem[smem_x][smem_y]) = *reinterpret_cast<const int4 *>(k_block + (dram_x) * d + dram_y);
                }
                __syncthreads();


                for (int m = 0; m < 16; m++) {
                    const int warp_x = warp_idx / 2;
                    const int warp_y = warp_idx % 2;

                    for (int p = 0; p < 4; p++) {
                        const int reg_x1 = p;
                        const int reg_x2 = reg_x1 + 4;
                        const int q_smem_x1 = warp_x * 32 + 4 * (idx_in_warp / 8) + p;
                        const int q_smem_x2 = q_smem_x1 + 16;
                        const int q_smem_y = m;
                        left_reg[reg_x1] = q_smem[q_smem_x1][q_smem_y];
                        left_reg[reg_x2] = q_smem[q_smem_x2][q_smem_y];
                    }

                    for (int q = 0; q < 2; q++) {
                        const int reg_x1 = q;
                        const int reg_x2 = reg_x1 + 2;
                        const int k_smem_x = m;
                        const int k_smem_y1 = warp_y * 32 + 2 * (idx_in_warp % 8) + q;
                        const int k_smem_y2 = k_smem_y1 + 16;
                        up_reg[reg_x1] = k_smem[k_smem_y1][k_smem_x];
                        up_reg[reg_x2] = k_smem[k_smem_y2][k_smem_x];
                    }

                    for (int p = 0; p < 2 * 4; p++) {
                        for (int q = 0; q < 2 * 2; q++) {
                            mma_result[p][q] = __hadd(mma_result[p][q], __hmul(left_reg[p], up_reg[q]));
                        }
                    }
                }
                __syncthreads();
            }

            for (int m = 0; m < 2; m++) {
                for (int n = 0; n < 2; n++) {
                    for (int p = 0; p < 4; p++) {
                        for (int q = 0; q < 2; q++) {
                            const int reg_x = m * 4 + p;
                            const int reg_y = n * 2 + q;
                            const int warp_x = warp_idx / 2;
                            const int warp_y = warp_idx % 2;
                            const int t_x = idx_in_warp / 8;
                            const int t_y = idx_in_warp % 8;
                            const int smem_x = warp_x * 32 + m * 16 + t_x * 4 + p;
                            const int smem_y = warp_y * 32 + n * 16 + t_y * 2 + q;
                            if (causal && token_block_idx * N_block + smem_x < j * 64 + smem_y) {
                                s_p_smem[smem_x][smem_y] = __hneg(__float2half(INFINITY));
                            } else if (is_decode && j * 64 + smem_y > decode_length) {
                                s_p_smem[smem_x][smem_y] = __hneg(__float2half(INFINITY));
                            } else {
                                s_p_smem[smem_x][smem_y] = __hmul(scale, mma_result[reg_x][reg_y]); 
                            }
                        }
                    }
                }
            }

            // Set register arrays to zero
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    mma_result[i][j] = __float2half(0.0);
                }
            }
            __syncthreads();

            // 2. get m
            if (tid < 64) {
                half previous_max = m_smem[tid];
                for (int k = 0; k < 64; k++) {
                    m_smem[tid] = __hmax(m_smem[tid], s_p_smem[tid][k]);
                }   
                m_diff_smem[tid] = hexp(__hadd(previous_max, __hneg(m_smem[tid])));
            }

            __syncthreads();
            // 3. get p in place
            for (int m = 0; m < 32; m++) {
                const int sp_x = m * 2 + tid / 64;
                const int sp_y = tid % 64;
                s_p_smem[sp_x][sp_y] = hexp(__hadd(s_p_smem[sp_x][sp_y], __hneg(m_smem[sp_x])));
            }

            __syncthreads();
            if (tid < 64) {
                // 4. get l
                half row_sum = __float2half(0.0);
                for (int m = 0; m < 64; m++) {
                    row_sum = __hadd(row_sum, s_p_smem[tid][m]);
                }
                l_smem[tid] = __hadd(__hmul(m_diff_smem[tid], l_smem[tid]), row_sum);
            }

            for(int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    mma_result[i][j] = __float2half(0.0);
                }
            }

            __syncthreads();
            // 5. get o
            // for 5.1 load p and v
            // for 5.2 accumulate o
            // 5.3 store o back to smem
            for (int k = 0; k < (64/16); k++) {
                {
                    // load v
                    for (int m = 0; m < 2; m++) {
                        const int smem_x = m * 8 + tid / 16;
                        const int smem_y = 8 * (tid % 16);
                        const int dram_x = j * 64 + k * 16 + smem_x;
                        const int dram_y = smem_y;
                        *reinterpret_cast<int4 *>(&v_smem[smem_x][smem_y]) = *reinterpret_cast<const int4 *>(v_block + (dram_x) * d + dram_y);
                    }
                }
                __syncthreads();
                for (int m = 0; m < 16; m++) {
                    const int warp_x = warp_idx / 2;
                    const int warp_y = warp_idx % 2;
                    for (int p = 0; p < 4; p++) {
                        const int reg_x1 = p;
                        const int reg_x2 = p + 4;
                        const int p_smem_x1 = 32 * warp_x + 4 * (idx_in_warp / 8) + p;
                        const int p_smem_x2 = p_smem_x1 + 16;
                        const int p_smem_y = k * 16 + m;
                        left_reg[reg_x1] = s_p_smem[p_smem_x1][p_smem_y];
                        left_reg[reg_x2] = s_p_smem[p_smem_x2][p_smem_y];
                    }
                    
                    for (int q = 0; q < 4; q++) {
                        const int reg_x1 = q;
                        const int reg_x2 = q + 4;
                        const int v_smem_x = m;
                        const int v_smem_y1 = 64 * warp_y + 4 * (idx_in_warp % 8) + q;
                        const int v_smem_y2 = v_smem_y1 + 32;
                        up_reg[reg_x1] = v_smem[v_smem_x][v_smem_y1];
                        up_reg[reg_x2] = v_smem[v_smem_x][v_smem_y2];
                    }

                    for (int p = 0; p < 2 * 4; p++) {
                        for (int q = 0; q < 2 * 4; q++) {
                            mma_result[p][q] = __hadd(mma_result[p][q], __hmul(left_reg[p], up_reg[q]));;
                        }
                    }
                }
                __syncthreads();
            }


            // write back using accumulate
            for (int m = 0; m < 2; m++) {
                for (int n = 0; n < 2; n++) {
                    for (int p = 0; p < 4; p++) {
                        for (int q = 0; q < 4; q++) {
                            const int reg_x = m * 4 + p;
                            const int reg_y = n * 4 + q;
                            const int warp_x = warp_idx / 2;
                            const int warp_y = warp_idx % 2;
                            const int t_x = idx_in_warp / 8;
                            const int t_y = idx_in_warp % 8;
                            const int smem_x = 32 * warp_x + 16 * m + 4 * t_x + p;
                            const int smem_y = 64 * warp_y + 32 * n + 4 * t_y + q;
                            o_smem[smem_x][smem_y] =__hadd( __hmul(o_smem[smem_x][smem_y], m_diff_smem[smem_x]), mma_result[reg_x][reg_y]);
                        }
                    }
                }
            }
            __syncthreads();
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    mma_result[i][j] = __float2half(0.0);
                }
            }
            
        }  // finish reading all the k and v, now we have to adjust o
        __syncthreads();
        // adjust o and write to final output
        if (tid < 64) {
            for (int k = 0; k < d; k++) {
                o_smem[tid][k] = __hmul(o_smem[tid][k], __hdiv(__float2half(1.0), l_smem[tid]));
            }
        }
        __syncthreads();
        
        // write to o_warp

        for (int m = 0; m < 8; m++) {
            const int smem_x = 8 * m + tid / 16;
            const int smem_y = 8 * (tid % 16);
            const int dram_x = i * 64 + smem_x;
            const int dram_y = smem_y;
            *reinterpret_cast<int4*>(o_block + (dram_x)*d + dram_y) = *reinterpret_cast<int4*>(&(o_smem[smem_x][smem_y]));
        }
    }

    return;
}

torch::Tensor finefmha_1(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
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

    if (is_decode) {
        TORCH_CHECK(!causal);
    }

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q, Q.options());

    dim3 grid_dim(ceil(N_q / 64), H, B);  // batch_size x num_heads
    dim3 block_dim(128);                  // Bc threads per block

    if (is_decode) {
        finefmha_1_kernel<64, 16, false, true><<<grid_dim, block_dim>>>(
            Q.data_ptr<at::Half>(), K.data_ptr<at::Half>(),
            V.data_ptr<at::Half>(), H, H_kv, N_q, N_kv, d, softmax_scale,
            O.data_ptr<at::Half>(), decode_length);
    } else if (causal) {
        finefmha_1_kernel<64, 16, true, false><<<grid_dim, block_dim>>>(
            Q.data_ptr<at::Half>(), K.data_ptr<at::Half>(),
            V.data_ptr<at::Half>(), H, H_kv, N_q, N_kv, d, softmax_scale,
            O.data_ptr<at::Half>(), decode_length);
    } else {
        finefmha_1_kernel<64, 16, false, false><<<grid_dim, block_dim>>>(
            Q.data_ptr<at::Half>(), K.data_ptr<at::Half>(),
            V.data_ptr<at::Half>(), H, H_kv, N_q, N_kv, d, softmax_scale,
            O.data_ptr<at::Half>(), decode_length);
    }

    return O;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(fmha_unified, CUDA, m) { m.impl("finefmha_1", &finefmha_1); }

}  // namespace fmha_unified