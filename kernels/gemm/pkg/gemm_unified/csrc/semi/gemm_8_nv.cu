// from here on we start to use tensor core through CUDA WMMA API
// mostly learnt from https://github.com/Bruce-Lee-LY/cuda_hgemm
// we'll be doing wmma_base => so we do not care about thread level computations anymore
// wmma_async, wmma_async_stage2 and wmma_async_stage3


// This kernel we will do warpsplit -1 (adjusting which thread is loading which
// part of SMEM to REG)
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

namespace gemm_unified {

template <const int BM, const int BN, const int BK, const int WM, const int WN>
__global__ void finegemm_8_kernel(int M, int K, int N, const at::Half *a,
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
    // step along K and accumulate results
    __shared__ half a_smem[BK * BM];
    __shared__ half b_smem[BN * BK];
    // __shared__ half c_smem[BN * BM];
    

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> A_frag[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> C_frag[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    for (int k_idx = 0; k_idx < (K / BK); k_idx++) {
        const int a_k_y = c_block_y;
        const int a_k_x = k_idx * BK;
        const half *a_k = a_ptr + a_k_y * K + a_k_x;
        const int b_k_y = k_idx * BK;
        const int b_k_x = c_block_x;
        const half *b_k = b_ptr + b_k_y * N + b_k_x;
        {
            // load data from dram into smem
            // TODO: change this to a vector load
            const int a_k_y_step = blockDim.x / (BK / 8);
            for (int i = 0; (i + a_k_y_step) <= BM; i += a_k_y_step) {
                const int y_idx = id_in_block / (BK / 8);
                const int x_idx = id_in_block % (BK / 8);
                *reinterpret_cast<int4 *>(a_smem + (i + y_idx) * BK + (x_idx * 8)) = *reinterpret_cast<const int4 *>(a_k + (i + y_idx) * K + (x_idx * 8));
            }
            const int b_k_y_step = blockDim.x / (BN / 8);
            for (int i = 0; (i + b_k_y_step) <= BK; i += b_k_y_step) {
                const int y_idx = id_in_block / (BN / 8);
                const int x_idx = id_in_block % (BN / 8);
                *reinterpret_cast<int4 *>(b_smem + (i + y_idx) * BN + (x_idx * 8)) = *reinterpret_cast<const int4 *>(b_k + (i + y_idx) * N + (x_idx * 8));
            }
        }
        __syncthreads();
        {
            constexpr int warp_row = BN / WN;
            const int warp_x_base = WN * int(warp_id % warp_row);
            const int warp_y_base = WM * int(warp_id / warp_row);
            const int warp_a_x_base = 0;
            const int warp_a_y_base = warp_y_base;
            const int warp_b_x_base = warp_x_base;
            const int warp_b_y_base = 0;
            // directly load all of B
            for (int b_idx = 0; b_idx < (WN / 16); b_idx++) {
                const half* b_load_base = b_smem + (warp_b_y_base) * BN + (warp_b_x_base + b_idx * 16);
                wmma::load_matrix_sync(B_frag[b_idx], b_load_base,BN);
            }
            // load a tile of A and compute with all of B
            for (int a_idx = 0; a_idx < (WM / 16); a_idx++) {
                const half* a_load_base = a_smem + (warp_a_y_base + a_idx * 16) * BK + (warp_a_x_base);
                wmma::load_matrix_sync(A_frag[a_idx], a_load_base, BK);
            }

            for (int a_idx = 0; a_idx < (WM / 16); a_idx++) {
                for (int b_idx = 0; b_idx < (WN / 16); b_idx++) {
                    wmma::mma_sync(C_frag[a_idx][b_idx], A_frag[a_idx], B_frag[b_idx], C_frag[a_idx][b_idx]);
                }
            }
        }
        __syncthreads();
    } // end of kidx loop


    // write back results, this is not worth it! just writeback to global
    // first write to shared memory
    // int shared_writeback_x_base = WN * int(warp_id % (BN / WN));
    // int shared_writeback_y_base = WM * int(warp_id / (BN / WN));
    // for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 4; j ++) {
    //         half *shared_writeback = c_smem + (shared_writeback_y_base + i * 16) * BN + (shared_writeback_x_base + j * 16);
    //         wmma::store_matrix_sync(shared_writeback, C_frag[i][j], BN, wmma::mem_row_major);
    //     }
    // }

    // then to global memory
    // __syncthreads();
    // const int gmem_y_step = blockDim.x * 8 / (BN);
    // const int gmem_x_offset = 8 * (id_in_block % (BN / 8));
    // const int gmem_y_offset = id_in_block / (BN / 8);
    // for (int i = 0; i < BM; i += gmem_y_step) {
    //     half *gmem_writeback = c_ptr + (c_block_y + i + gmem_y_offset) * N + (c_block_x + gmem_x_offset);
    //     half *smem_writeback = c_smem + (i + gmem_y_offset) * BN + (gmem_x_offset);
    //     *reinterpret_cast<int4 *>(gmem_writeback) = *reinterpret_cast<int4 *>(smem_writeback);
    // }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j ++) {
            const int c_ptr_x = c_block_x + WN * int(warp_id % (BN / WN)) + j * 16;
            const int c_ptr_y = c_block_y + WM * int(warp_id / (BN / WN)) + i * 16;
            half* c_out = c_ptr + (c_ptr_y) * N + (c_ptr_x);
            wmma::store_matrix_sync(c_out, C_frag[i][j], N, wmma::mem_row_major);
        }
    }
}

at::Tensor finegemm_8(const at::Tensor &a, const at::Tensor &b) {
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

    finegemm_8_kernel<BM, BN, BK, 64, 64>
        <<<grid, block>>>(M, K, N, a_ptr, b_ptr, result_ptr);
    return result;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(gemm_unified, CUDA, m) { m.impl("finegemm_8", &finegemm_8); }

}  // namespace gemm_unified