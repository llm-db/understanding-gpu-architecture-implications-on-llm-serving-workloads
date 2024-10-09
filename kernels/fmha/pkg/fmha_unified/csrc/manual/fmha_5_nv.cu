// nothing touches shared memory
// note: this kernel is not fully implemented, it is put here just to study its performance implications
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <mma.h>

#include <cuda_pipeline.h>

#include <kittens.cuh>
#include "c10/core/ScalarType.h"
#include "c10/util/BFloat16.h"
#include "c10/util/Exception.h"
#include "common/base_types.cuh"
#include "ops/warp/register/tile/conversions.cuh"

namespace fmha_unified {


#define NUM_WORKERS 8 // This kernel uses 16 workers in parallel per block, to help issue instructions more quickly.

using namespace kittens; // this kernel only handles headdim=64 for simplicity. Also n should be a multiple of 256 here.
template <const bool is_causal>
__global__ void attend_ker64(int n, const at::BFloat16* __restrict__ __q__, const at::BFloat16* __restrict__ __k__, const at::BFloat16* __restrict__ __v__, at::BFloat16* __o__) {

    auto warpid        = kittens::warpid();
    const int token_block_idx = blockIdx.x;
    const int head_block_idx = blockIdx.y;
    const int batch_block_idx = blockIdx.z;
    const int N_q = n;
    const int N_kv = n;
    constexpr int d = 128;
    constexpr int H_q = 32;
    constexpr int H_kv = 8;
    const int q_kv_ratio = H_q / H_kv;
    const bf16* _q = reinterpret_cast<const bf16 *>(__q__ + batch_block_idx * (H_q * N_q * d) +
                                 head_block_idx * (N_q * d) + token_block_idx * (128 * d));  
    const bf16* _k = reinterpret_cast<const bf16 *>(__k__ + batch_block_idx * (H_kv * N_kv * d) +
                                 (head_block_idx / q_kv_ratio) * (N_kv * d));
    const bf16* _v = reinterpret_cast<const bf16 *>(__v__ + batch_block_idx * (H_kv * N_kv * d) +
                                 (head_block_idx / q_kv_ratio) * (N_kv * d));
    bf16* _o = reinterpret_cast<bf16 *>(__o__ + batch_block_idx * (H_q * N_q * d) + head_block_idx * (N_q * d) + token_block_idx * (128 * d));


    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    // K and V live in shared memory -- this is about all that will fit.
    st_bf_1x8 (&k_smem)[NUM_WORKERS] = al.allocate<st_bf_1x8, NUM_WORKERS>();
    st_bf_1x8 (&v_smem)[NUM_WORKERS] = al.allocate<st_bf_1x8, NUM_WORKERS>();

    // Initialize all of the register tiles.
    rt_bf_1x8<> q_reg, k_reg, v_reg; // v_reg need to be swapped into col_l
    rt_fl_1x1<> att_block;
    rt_bf_1x1<> att_block_mma;
    rt_fl_1x8<> o_reg;
    rt_fl_1x1<>::col_vec max_vec_last, max_vec; // these are column vectors for the attention block
    rt_fl_1x1<>::col_vec norm_vec_last, norm_vec; // these are column vectors for the attention block

    // each warp loads its own Q tile of 16x64, and then multiplies by 1/sqrt(d)
    load(q_reg, _q + (warpid)*q_reg.num_elements, q_reg.cols);
    mul(q_reg, q_reg, __float2bfloat16(0.08838834764831843f)); // temperature adjustment

    // zero flash attention L, M, and O registers.
    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_reg);

    // iterate over k, v for these q's that have been loaded
    int kv_blocks = N_kv / (q_reg.rows*NUM_WORKERS);
    if constexpr (is_causal) {
        kv_blocks = min(kv_blocks, (token_block_idx + 1));
    }
    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {

        // each warp loads its own chunk of k, v into shared memory
        load(v_smem[warpid], _v + (kv_idx*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
        load(k_smem[warpid], _k + (kv_idx*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
        __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

        int max_subtile = NUM_WORKERS;
        if constexpr (is_causal) {
            if (kv_idx == kv_blocks - 1) {
                max_subtile = warpid + 1;
            }
        }
        // now each warp goes through all of the subtiles, loads them, and then does the flash attention internal alg.
        for(int subtile = 0; subtile < max_subtile; subtile++) {

            load(k_reg, k_smem[subtile]); // load k from shared into registers

            zero(att_block); // zero 16x16 attention tile
            mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T

            if constexpr (is_causal) {
                if(kv_idx == kv_blocks - 1 && subtile == max_subtile - 1) {
                    make_causal(att_block, att_block, -INFINITY);
                }
            }

            copy(norm_vec_last, norm_vec);
            copy(max_vec_last,  max_vec);

            row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
            sub_row(att_block, att_block, max_vec); // subtract max from attention -- now all <=0
            exp(att_block, att_block); // exponentiate the block in-place.

            sub(max_vec_last, max_vec_last, max_vec); // subtract new max from old max to find the new normalization.
            exp(max_vec_last, max_vec_last); // exponentiate this vector -- this is what we need to normalize by.
            mul(norm_vec, norm_vec, max_vec_last); // and the norm vec is now normalized.

            row_sum(norm_vec, att_block, norm_vec); // accumulate the new attention block onto the now-rescaled norm_vec
            div_row(att_block, att_block, norm_vec); // now the attention block is correctly normalized

            mul(norm_vec_last, norm_vec_last, max_vec_last); // normalize the previous norm vec according to the new max
            div(norm_vec_last, norm_vec_last, norm_vec); // normalize the previous norm vec according to the new norm

            copy(att_block_mma, att_block); // convert to bf16 for mma_AB

            load(v_reg, v_smem[subtile]); // load v from shared into registers.
            rt_bf_1x8<ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

            mul_row(o_reg, o_reg, norm_vec_last); // normalize o_reg in advance of mma_AB'ing onto it
            mma_AB(o_reg, att_block_mma, v_reg_col, o_reg); // mfma onto o_reg with the local attention@V matmul.
        }
        __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
    }

    store(_o + (warpid)*q_reg.num_elements, o_reg, q_reg.cols); // write out o. compiler has an issue with register usage if d is made constexpr q_reg.rows :/
    
}

torch::Tensor finefmha_5(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                         bool causal, bool is_decode, int64_t decode_length) {
    // TORCH_CHECK(!causal);
    TORCH_CHECK(!is_decode);
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N_q = Q.size(2);
    const int N_kv = K.size(2);
    const int d = Q.size(3);

    const int H_kv = K.size(1);
    const float softmax_scale = 1.0 / sqrt(d);

    TORCH_CHECK(Q.dtype() == at::kBFloat16);
    TORCH_CHECK(K.dtype() == at::kBFloat16);
    TORCH_CHECK(V.dtype() == at::kBFloat16);
    TORCH_INTERNAL_ASSERT(Q.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(K.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(V.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(d == 128);
    TORCH_CHECK(N_q % 128 == 0);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q, Q.options());

    dim3 grid_dim(ceil(N_q / 128), H, B);  // batch_size x num_heads
    dim3 block_dim(32 * NUM_WORKERS);       
    unsigned long mem_size = 100000; // need to launch two blocks if possible.
    
    if (causal) {
        cudaFuncSetAttribute(
        attend_ker64<true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
        );           // Bc threads per block
        attend_ker64<true><<<grid_dim, block_dim, (70000)>>> (
            N_q, Q.data_ptr<at::BFloat16>(), K.data_ptr<at::BFloat16>(), V.data_ptr<at::BFloat16>(), O.data_ptr<at::BFloat16>());
    } else {
        cudaFuncSetAttribute(
        attend_ker64<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
        );           // Bc threads per block
        attend_ker64<false><<<grid_dim, block_dim, (70000)>>> (
            N_q, Q.data_ptr<at::BFloat16>(), K.data_ptr<at::BFloat16>(), V.data_ptr<at::BFloat16>(), O.data_ptr<at::BFloat16>());
    }
    return O;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(fmha_unified, CUDA, m) { m.impl("finefmha_5", &finefmha_5); }

}  // namespace fmha_unified