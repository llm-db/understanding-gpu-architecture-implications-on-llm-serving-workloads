// from here on we start to use tensor core through CUDA WMMA API
// mostly learnt from https://github.com/Bruce-Lee-LY/cuda_hgemm
// wmma_async with a multi stage pipeline


#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include <cuda_pipeline.h>

namespace gemm_unified {

template <const int BM, const int BN, const int BK, const int WM, const int WN>
__global__ void finegemm_10_kernel(int M, int K, int N, const at::Half *a,
                                  const at::Half *b, at::Half *result) {
using namespace nvcuda;
    const half *a_ptr = reinterpret_cast<const half *>(a);
    const half *b_ptr = reinterpret_cast<const half *>(b);
    half *c_ptr = reinterpret_cast<half *>(result);
    // find out this block
    const int c_block_y = blockIdx.y * BM;
    const int c_block_x = blockIdx.x * BN;
    // find out this thread & warp
    const int id_in_block = threadIdx.x;
    const int warp_id = id_in_block / 32;
    // const int id_in_warp = id_in_block % 32;
    // step along K and accumulate results
    __shared__ half a_smem[2 * (BK + 8) * BM];
    __shared__ half b_smem[2 * (BN + 8) * BK];
    constexpr int a_shared_mem_buff_size = (BK + 8) * BM;
    constexpr int b_shared_mem_buff_size = (BN + 8) * BK;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> A_frag[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> C_frag[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    // prefetch the zero-th k_idx to start this pipeline
    // pipeline.producer_acquire();
    {
            // load data from dram into smem with async load from CC >= 8.0
            int k_idx = 0;
            const int a_k_y = c_block_y;
            const int a_k_x = k_idx * BK;
            const half *a_k = a_ptr + a_k_y * K + a_k_x;
            const int b_k_y = k_idx * BK;
            const int b_k_x = c_block_x;
            const half *b_k = b_ptr + b_k_y * N + b_k_x;
            constexpr int a_k_y_step = 128 / (BK / 8);
            #pragma unroll
            for (int i = 0; (i + a_k_y_step) <= BM; i += a_k_y_step) {
                const int y_idx = id_in_block / (BK / 8);
                const int x_idx = id_in_block % (BK / 8);
                __pipeline_memcpy_async(a_smem + (i + y_idx) * (BK + 8) + (x_idx * 8), a_k + (i + y_idx) * K + (x_idx * 8), sizeof(half) * 8);
            }
            constexpr int b_k_y_step = 128 / (BN / 8);
            #pragma unroll
            for (int i = 0; (i + b_k_y_step) <= BK; i += b_k_y_step) {
                const int y_idx = id_in_block / (BN / 8);
                const int x_idx = id_in_block % (BN / 8);
                __pipeline_memcpy_async( b_smem + (i + y_idx) * (BN + 8) + (x_idx * 8), b_k + (i + y_idx) * N + (x_idx * 8), sizeof(half) * 8);
            }
            __pipeline_commit();
    }

    // now we start the 'real' pipeline
    for (int k_idx = 0; k_idx < (K / BK); k_idx++) {
        const int k_idx_stage = k_idx % 2;
        {
            // preload the next one
            int k_idx_next = k_idx + 1;
            
            // pipeline.producer_acquire();
            const int k_idx_next_stage = k_idx_next % 2;
            if (k_idx_next == (K / BK)) {
                k_idx_next -= 1;
            }
            const int a_k_y = c_block_y;
            const int a_k_x = k_idx_next * BK;
            const half *a_k = a_ptr + a_k_y * K + a_k_x;
            const int b_k_y = k_idx_next * BK;
            const int b_k_x = c_block_x;
            const half *b_k = b_ptr + b_k_y * N + b_k_x;
            // load data from dram into smem with async load from CC >= 8.0
            constexpr int a_k_y_step = 128 / (BK / 8);
            #pragma unroll
            for (int i = 0; (i + a_k_y_step) <= BM; i += a_k_y_step) {
                const int y_idx = id_in_block / (BK / 8);
                const int x_idx = id_in_block % (BK / 8);
                __pipeline_memcpy_async(a_smem + k_idx_next_stage * a_shared_mem_buff_size + (i + y_idx) * (BK + 8) + (x_idx * 8), a_k + (i + y_idx) * K + (x_idx * 8), sizeof(half) * 8);
            }
            constexpr int b_k_y_step = 128 / (BN / 8);
            #pragma unroll
            for (int i = 0; (i + b_k_y_step) <= BK; i += b_k_y_step) {
                const int y_idx = id_in_block / (BN / 8);
                const int x_idx = id_in_block % (BN / 8);
                __pipeline_memcpy_async(b_smem + k_idx_next_stage * b_shared_mem_buff_size + (i + y_idx) * (BN + 8) + (x_idx * 8), b_k + (i + y_idx) * N + (x_idx * 8), sizeof(half) * 8);
            }
            __pipeline_commit();
        }
        __pipeline_wait_prior(1);
        __syncthreads();
        {
            // now we actually consume the data
            constexpr int warp_row = BN / WN;
            const int warp_x_base = WN * int(warp_id % warp_row);
            const int warp_y_base = WM * int(warp_id / warp_row);
            const int warp_a_x_base = 0;
            const int warp_a_y_base = warp_y_base;
            const int warp_b_x_base = warp_x_base;
            const int warp_b_y_base = 0;
            // directly load all of B
            #pragma unroll
            for (int b_idx = 0; b_idx < (WN / 16); b_idx++) {
                const half* b_load_base = b_smem + k_idx_stage * b_shared_mem_buff_size + (warp_b_y_base) * (BN + 8) + (warp_b_x_base + b_idx * 16);
                wmma::load_matrix_sync(B_frag[b_idx], b_load_base,(BN + 8));
            }
            // load a tile of A and compute with all of B
            #pragma unroll
            for (int a_idx = 0; a_idx < (WM / 16); a_idx++) {
                const half* a_load_base = a_smem + k_idx_stage * a_shared_mem_buff_size + (warp_a_y_base + a_idx * 16) * (BK + 8) + (warp_a_x_base);
                wmma::load_matrix_sync(A_frag[a_idx], a_load_base, (BK + 8));
            }
            #pragma unroll
            for (int a_idx = 0; a_idx < (WM / 16); a_idx++) {
                for (int b_idx = 0; b_idx < (WN / 16); b_idx++) {
                    wmma::mma_sync(C_frag[a_idx][b_idx], A_frag[a_idx], B_frag[b_idx], C_frag[a_idx][b_idx]);
                }
            }
        }
    } // end of kidx loop

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j ++) {
            const int c_ptr_x = c_block_x + WN * int(warp_id % (BN / WN)) + j * 16;
            const int c_ptr_y = c_block_y + WM * int(warp_id / (BN / WN)) + i * 16;
            half* c_out = c_ptr + (c_ptr_y) * N + (c_ptr_x);
            wmma::store_matrix_sync(c_out, C_frag[i][j], N, wmma::mem_row_major);
        }
    }
}

at::Tensor finegemm_10(const at::Tensor &a, const at::Tensor &b) {
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
    const int BK = 16;
    const int TM = 8;
    const int TN = 4;
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

    // Launch the kernel
    dim3 block(128);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    finegemm_10_kernel<BM, BN, BK, 64, 64>
        <<<grid, block>>>(M, K, N, a_ptr, b_ptr, result_ptr);
    return result;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(gemm_unified, CUDA, m) { m.impl("finegemm_10", &finegemm_10); }

}  // namespace gemm_unified