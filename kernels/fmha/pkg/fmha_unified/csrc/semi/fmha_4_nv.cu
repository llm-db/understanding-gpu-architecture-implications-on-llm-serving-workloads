// async memory load
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <mma.h>

#include <cuda_pipeline.h>


namespace fmha_unified {

template <const int N_block, const int N_warp, const bool causal,
          const bool is_decode, const int d>
__global__ void finefmha_4_kernel(const at::Half* Q, const at::Half* K,
                                  const at::Half* V, const int H_q,
                                  const int H_kv, const int N_q, const int N_kv,
                                  const at::Half softmax_scale,
                                  at::Half* O, int64_t decode_length) {
    // Q O [B(batch_size), H(head), N_q(seq_length), d(head_embed)]
    // K V [B(batch_size), H(head), N_kv(seq_length), d(head_embed)]
    const half* q = reinterpret_cast<const half*>(Q);
    const half* k = reinterpret_cast<const half*>(K);
    const half* v = reinterpret_cast<const half*>(V);
    half* o = reinterpret_cast<half*>(O);

    const int token_block_idx = blockIdx.x;
    const int head_block_idx = blockIdx.y;
    const int batch_block_idx = blockIdx.z;

    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;

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
    __shared__ half q_smem[2][4 * N_warp][16];
    __shared__ half k_smem[2][4 * N_warp][16];
    __shared__ half v_smem[2][16][128];
    // no transpose here

    // 2K in total
    __shared__ half s_p_smem[64][64];
    __shared__ half m_smem[4 * N_warp];
    __shared__ half l_smem[4 * N_warp];
    __shared__ half m_diff_smem[4 * N_warp];

    const int warp_x = warp_idx / 2;
    const int warp_y = warp_idx % 2;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_frag[2];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_frag[2];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> V_frag[4];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_frag[2][4];

    

    for (int i = 0; i < (N_block / (N_warp * 4)); i++) {
        // init
        // o = 0
        // for (int m = 0; m < 64; m++) {
        //     o_smem[m][tid] = 0;
        // }
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            const int smem_x = 8 * m + tid / 16;
            const int smem_y = 8 * (tid % 16);
            *reinterpret_cast<int4*>(&(o_smem[smem_x][smem_y])) =
                make_int4(0, 0, 0, 0);
        }
        if (tid < 64) {
            l_smem[tid] = 0;
            m_smem[tid] = -INFINITY;
        }

        int j_max = 0;
        if (causal) {
            const int max_i_block = token_block_idx * ceil(N_block / (4 * N_warp)) + 1;
            j_max = min(max_i_block, (N_kv / (4 * N_warp)));
        } else {
            j_max = ceil(N_kv / (4 * N_warp));
        }

        {
            // start to fill the pipeline
            int k = 0;
            int j = 0;
            {
                // load q
                const int smem_x = tid / 2;
                const int smem_y = 8 * (tid % 2);
                const int dram_x = i * 64 + smem_x;
                const int dram_y = k * 16 + smem_y;
                // *reinterpret_cast<int4 *>(&q_smem[0][smem_x][smem_y]) = *reinterpret_cast<const int4 *>(q_block + (dram_x) * d + dram_y);
                __pipeline_memcpy_async(&q_smem[0][smem_x][smem_y], q_block + (dram_x) * d + dram_y, sizeof(half) * 8);
            }
            {
                // load k
                const int smem_x = tid / 2;
                const int smem_y = 8 * (tid % 2);
                const int dram_x = j * 64 + smem_x;
                const int dram_y = k * 16 + smem_y;
                // *reinterpret_cast<int4 *>(&k_smem[0][smem_x][smem_y]) = *reinterpret_cast<const int4 *>(k_block + (dram_x) * d + dram_y);
                __pipeline_memcpy_async(&k_smem[0][smem_x][smem_y], k_block + (dram_x) * d + dram_y, sizeof(half) * 8);
            }
            __pipeline_commit();

        }

        for (int j = 0; j < j_max; j++) {
            // 1. get S
            // for 1.1 load q and k
            // for 1.2 accumulate s
            // 1.3 store s back to smem
            {
                // clean up accumulator
                #pragma unroll
                for (int m = 0; m < 2; m++) {
                    #pragma unroll
                    for (int n = 0; n < 2; n++) {
                        nvcuda::wmma::fill_fragment(C_frag[m][n], 0);
                    }
                }
            }

            for (int k = 0; k < (d / 16); k++) {
                const int k_stage = k % 2;
                {
                    // preload next stage
                    const int k_next = k + 1;
                    if (k_next < (d / 16)) {
                        const int k_next_stage = k_next % 2;
                        {
                            // load q
                            const int smem_x = tid / 2;
                            const int smem_y = 8 * (tid % 2);
                            const int dram_x = i * 64 + smem_x;
                            const int dram_y = k_next * 16 + smem_y;
                            // *reinterpret_cast<int4 *>(&q_smem[0][smem_x][smem_y]) = *reinterpret_cast<const int4 *>(q_block + (dram_x) * d + dram_y);
                            __pipeline_memcpy_async(&q_smem[k_next_stage][smem_x][smem_y], q_block + (dram_x) * d + dram_y, sizeof(half) * 8);
                        }
                        {
                            // load k
                            const int smem_x = tid / 2;
                            const int smem_y = 8 * (tid % 2);
                            const int dram_x = j * 64 + smem_x;
                            const int dram_y = k_next * 16 + smem_y;
                            // *reinterpret_cast<int4 *>(&k_smem[0][smem_x][smem_y]) = *reinterpret_cast<const int4 *>(k_block + (dram_x) * d + dram_y);
                            __pipeline_memcpy_async(&k_smem[k_next_stage][smem_x][smem_y], k_block + (dram_x) * d + dram_y, sizeof(half) * 8);
                        }
                    } else {
                        {
                            // preload the first element of V
                            #pragma unroll
                            for (int m = 0; m < 2; m++) {
                                const int smem_x = m * 8 + tid / 16;
                                const int smem_y = 8 * (tid % 16);
                                // k = 0 for k
                                const int dram_x = j * 64 + 0 * 16 + smem_x;
                                const int dram_y = smem_y;
                                // *reinterpret_cast<int4 *>(&v_smem[0][smem_x][smem_y]) = *reinterpret_cast<const int4 *>(v_block + (dram_x) * d + dram_y);
                                __pipeline_memcpy_async(&v_smem[0][smem_x][smem_y], v_block + (dram_x) * d + dram_y, sizeof(half) * 8);
                            }
                        }
                    }
                    __pipeline_commit();
                }
                __pipeline_wait_prior(1);
                __syncthreads();
                const int q_smem_x = warp_x * 32;
                const int k_smem_x = warp_y * 32;
                #pragma unroll
                for (int m = 0; m < 2; m++) {
                    const int frag_q_smem_x = q_smem_x + 16 * m;
                    nvcuda::wmma::load_matrix_sync(A_frag[m], q_smem[k_stage][frag_q_smem_x], 16);
                    
                }
                #pragma unroll
                for (int n = 0; n < 2; n++) {        
                    const int frag_k_smem_x = k_smem_x + 16 * n;
                    nvcuda::wmma::load_matrix_sync(B_frag[n], k_smem[k_stage][frag_k_smem_x], 16);
                }
                #pragma unroll
                for (int m = 0; m < 2; m++) {
                    #pragma unroll
                    for (int n = 0; n < 2; n++) {
                        nvcuda::wmma::mma_sync(C_frag[m][n], A_frag[m], B_frag[n], C_frag[m][n]);
                    }
                }
            }

            {
                // write back to s_p_smem
                #pragma unroll
                for (int m = 0; m < 2; m++) {
                    #pragma unroll
                    for (int n = 0; n < 2; n++) {
                        const int s_p_smem_x = warp_x * 32 + m * 16;
                        const int s_p_smem_y = warp_y * 32 + n * 16;
                        for (int p = 0; p < C_frag[0][0].num_elements; p++) {
                            C_frag[m][n].x[p] *= softmax_scale;
                        }
                        nvcuda::wmma::store_matrix_sync(&s_p_smem[s_p_smem_x][s_p_smem_y], C_frag[m][n], 64, nvcuda::wmma::mem_row_major);
                    }
                }
            }

            // 2. get m
            __syncthreads();

            if (tid < 64) {
                if (j < j_max - 1) {
                    half previous_max = m_smem[tid];
                    #pragma unroll
                    for (int k = 0; k < 64; k++) {
                        m_smem[tid] = __hmax(m_smem[tid], s_p_smem[tid][k]);
                    }   
                    m_diff_smem[tid] = hexp(previous_max - m_smem[tid]);
                } else {
                    half previous_max = m_smem[tid];
                    #pragma unroll
                    for (int k = 0; k < 64; k++) {
                        const int q_in_seq = token_block_idx * N_block + tid;
                        const int k_in_seq = j * 64 + k;
                        if (causal && q_in_seq < k_in_seq) {
                            s_p_smem[tid][k] = -INFINITY;
                        } else if (k_in_seq > decode_length) {
                            s_p_smem[tid][k] = -INFINITY;
                        }
                        m_smem[tid] = __hmax(m_smem[tid], s_p_smem[tid][k]);
                    }   
                    m_diff_smem[tid] = hexp(previous_max - m_smem[tid]);
                }
            }

            __syncthreads();
            // 3. get p in place
            #pragma unroll
            for (int m = 0; m < 32; m++) {
                const int sp_x = m * 2 + tid / 64;
                const int sp_y = tid % 64;
                s_p_smem[sp_x][sp_y] = hexp(s_p_smem[sp_x][sp_y] -  m_smem[sp_x]);
            }

            __syncthreads();
            if (tid < 64) {
                // 4. get l
                half row_sum = 0;
                #pragma unroll
                for (int m = 0; m < 64; m++) {
                    row_sum += s_p_smem[tid][m];
                }
                l_smem[tid] = m_diff_smem[tid] * l_smem[tid] + row_sum;
            }
            // make adjustment early, so that later we can use pure mma
            #pragma unroll
            for (int m = 0; m < 64; m++) {
                o_smem[m][tid] *= m_diff_smem[m];
            }

            // 5. get o
            // for 5.1 load p and v
            // for 5.2 accumulate o
            // 5.3 store o back to smem
            {
                // clean up accumulator
                #pragma unroll
                for (int m = 0; m < 2; m++) {
                    #pragma unroll
                    for (int n = 0; n < 4; n++) {
                        const int o_frag_x = 32 * warp_x + 16 * m;
                        const int o_frag_y = 64 * warp_y + 16 * n;
                        nvcuda::wmma::load_matrix_sync(C_frag[m][n], &o_smem[o_frag_x][o_frag_y], 128, nvcuda::wmma::mem_row_major);
                    }
                }
            }
            #pragma unroll
            for (int k = 0; k < (64/16); k++) {
                const int k_stage = k % 2;
                {
                    // preload next stage
                    const int k_next = k + 1;
                    if (k_next < (64/16)) {
                        {
                            const int k_next_stage = k_next % 2;
                            // load v
                            #pragma unroll
                            for (int m = 0; m < 2; m++) {
                                const int smem_x = m * 8 + tid / 16;
                                const int smem_y = 8 * (tid % 16);
                                const int dram_x = j * 64 + k_next * 16 + smem_x;
                                const int dram_y = smem_y;
                                __pipeline_memcpy_async(&v_smem[k_next_stage][smem_x][smem_y], v_block + (dram_x) * d + dram_y, sizeof(half) * 8);
                                // *reinterpret_cast<int4 *>(&v_smem[smem_x][smem_y]) = *reinterpret_cast<const int4 *>(v_block + (dram_x) * d + dram_y);
                            }
                        }
                    } else {
                        {
                            // preload the first element of k and v
                            const int k = 0;
                            const int j_next = min(j + 1, (j_max - 1));
                            {
                                // load q
                                const int smem_x = tid / 2;
                                const int smem_y = 8 * (tid % 2);
                                const int dram_x = i * 64 + smem_x;
                                const int dram_y = k * 16 + smem_y;
                                // *reinterpret_cast<int4 *>(&q_smem[0][smem_x][smem_y]) = *reinterpret_cast<const int4 *>(q_block + (dram_x) * d + dram_y);
                                __pipeline_memcpy_async(&q_smem[0][smem_x][smem_y], q_block + (dram_x) * d + dram_y, sizeof(half) * 8);
                            }
                            {
                                // load k
                                const int smem_x = tid / 2;
                                const int smem_y = 8 * (tid % 2);
                                const int dram_x = j_next * 64 + smem_x;
                                const int dram_y = k * 16 + smem_y;
                                // *reinterpret_cast<int4 *>(&k_smem[0][smem_x][smem_y]) = *reinterpret_cast<const int4 *>(k_block + (dram_x) * d + dram_y);
                                __pipeline_memcpy_async(&k_smem[0][smem_x][smem_y], k_block + (dram_x) * d + dram_y, sizeof(half) * 8);
                            }
                        }
                    }
                    __pipeline_commit();
                }
                __pipeline_wait_prior(1);
                __syncthreads();
                #pragma unroll
                for (int m = 0; m < 2; m++) {
                    const int sp_x = 32 * warp_x + 16 * m;
                    const int sp_y = k * 16;
                    nvcuda::wmma::load_matrix_sync(A_frag[m], &s_p_smem[sp_x][sp_y], 64);
                }
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    const int v_x = 0;
                    const int v_y = 64 * warp_y + 16 * n;
                    nvcuda::wmma::load_matrix_sync(V_frag[n], &v_smem[k_stage][v_x][v_y], 128);
                }
                #pragma unroll
                for (int m = 0; m < 2; m++) {
                    #pragma unroll
                    for (int n = 0; n < 4; n++) {
                        nvcuda::wmma::mma_sync(C_frag[m][n], A_frag[m], V_frag[n], C_frag[m][n]);
                    }
                }
            }

            #pragma unroll
            for (int m = 0; m < 2; m++) {
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    const int o_frag_x = 32 * warp_x + 16 * m;
                    const int o_frag_y = 64 * warp_y + 16 * n;
                    nvcuda::wmma::store_matrix_sync(&o_smem[o_frag_x][o_frag_y], C_frag[m][n], 128, nvcuda::wmma::mem_row_major);
                }
            }
        }  // finish reading all the k and v, now we have to adjust o
        __syncthreads();
        // adjust o and write to final output
        
        #pragma unroll
        for (int m = 0; m < 64; m++) {
            o_smem[m][tid] = o_smem[m][tid] / l_smem[m];
        }

        __syncthreads();
        // write to o_warp
        #pragma unroll
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

torch::Tensor finefmha_4(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
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

    TORCH_CHECK(d == 128);

    if (is_decode) {
        TORCH_CHECK(!causal);
    }

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q, Q.options());

    dim3 grid_dim(ceil(N_q / 64), H, B);  // batch_size x num_heads
    dim3 block_dim(128);                  // Bc threads per block

    if (is_decode) {
        finefmha_4_kernel<64, 16, false, true, 128><<<grid_dim, block_dim>>>(
            Q.data_ptr<at::Half>(), K.data_ptr<at::Half>(),
            V.data_ptr<at::Half>(), H, H_kv, N_q, N_kv, softmax_scale,
            O.data_ptr<at::Half>(), decode_length);
    } else if (causal) {
        finefmha_4_kernel<64, 16, true, false, 128><<<grid_dim, block_dim>>>(
            Q.data_ptr<at::Half>(), K.data_ptr<at::Half>(),
            V.data_ptr<at::Half>(), H, H_kv, N_q, N_kv, softmax_scale,
            O.data_ptr<at::Half>(), decode_length);
    } else {
        finefmha_4_kernel<64, 16, false, false, 128><<<grid_dim, block_dim>>>(
            Q.data_ptr<at::Half>(), K.data_ptr<at::Half>(),
            V.data_ptr<at::Half>(), H, H_kv, N_q, N_kv, softmax_scale,
            O.data_ptr<at::Half>(), decode_length);
    }

    return O;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(fmha_unified, CUDA, m) { m.impl("finefmha_4", &finefmha_4); }

}  // namespace fmha_unified