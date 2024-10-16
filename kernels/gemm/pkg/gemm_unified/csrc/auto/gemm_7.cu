// This kernel we will do warpsplit -1 (adjusting which thread is loading which
// part of SMEM to REG)
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace gemm_unified {

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WSUBM, const int WSUBN, const int TM, const int TN>
__global__ void finegemm_7_kernel(int M, int K, int N, const at::Half *a,
                                  const at::Half *b, at::Half *result) {
    const half *a_ptr = reinterpret_cast<const half *>(a);
    const half *b_ptr = reinterpret_cast<const half *>(b);
    half *c_ptr = reinterpret_cast<half *>(result);
    // find out this block
    const int c_block_y = blockIdx.y * BM;
    const int c_block_x = blockIdx.x * BN;
    // find out this thread & warp
    const int id_in_block = threadIdx.x;
    const int warp_id = id_in_block / 32;
    const int id_in_warp = id_in_block % 32;
    // step along K and accumulate results
    __shared__ half a_smem[BM * BK];
    __shared__ half b_smem[BN * BK];
    half a_reg[TM * (WM / WSUBM)];
    half b_reg[TN * (WN / WSUBN)];
    half c_reg[(TM * (WM / WSUBM)) * (TN * (WN / WSUBN))] = {__int2half_rd(0)};
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
            const int a_k_y_step = blockDim.x / BK;
            for (int i = 0; (i + a_k_y_step) <= BM; i += a_k_y_step) {
                const int y_idx = id_in_block / BK;
                const int x_idx = id_in_block % BK;
                a_smem[(x_idx)*BM + i + y_idx] = a_k[(i + y_idx) * K + x_idx];
            }
            const int b_k_y_step = blockDim.x / BN;
            for (int i = 0; (i + b_k_y_step) <= BK; i += b_k_y_step) {
                const int y_idx = id_in_block / BN;
                const int x_idx = id_in_block % BN;
                b_smem[(i + y_idx) * BN + x_idx] = b_k[(i + y_idx) * N + x_idx];
            }
        }
        __syncthreads();
        // Print content of a_smem if thread coordinates are x=0 and y=0
        {
            constexpr int warp_row = BN / WN;
            const int warp_x_base = WN * int(warp_id % warp_row);
            const int warp_y_base = WM * int(warp_id / warp_row);
            const int thread_per_warp_row = WSUBN / TN;
            // load data from smem to reg & calculation
            for (int Bkidx = 0; Bkidx < BK; Bkidx += 1) {
                const int warp_a_x_base =
                    warp_y_base + int(id_in_warp / thread_per_warp_row) * TM;
                const int warp_a_y_base = Bkidx;
                const int warp_b_x_base =
                    warp_x_base + int(id_in_warp % thread_per_warp_row) * TN;
                const int warp_b_y_base = Bkidx;
                // load into reg
                for (int i = 0; i < (WM / WSUBM); i++) {
                    for (int j = 0; j < TM; j++) {
                        a_reg[i * TM + j] =
                            a_smem[(warp_a_y_base)*BM + warp_a_x_base +
                                   i * WSUBM + j];
                    }
                }

                for (int i = 0; i < (WN / WSUBN); i++) {
                    for (int j = 0; j < TN; j++) {
                        b_reg[i * TN + j] =
                            b_smem[(warp_b_y_base)*BN + warp_b_x_base +
                                   i * WSUBN + j];
                    }
                }

                for (int i = 0; i < TM * (WM/WSUBM); i++) {
                    for (int j = 0; j < TN * (WN/WSUBN); j++) {
                        c_reg[(i)* TN * (WN/WSUBN) + j] = __hadd(__hmul(a_reg[i], b_reg[j]), c_reg[(i)* TN * (WN/WSUBN) + j]);
                    }
                }
            } // end of Bkidx 0-BK loop
        }
        __syncthreads();
    } // end of kidx loop



    // write back results
    for (int i = 0; i < (WM/WSUBM); i++) {
        for (int j = 0; j < (WN/WSUBN); j++) {
            for (int k = 0; k < TM; k++) {
                for (int l = 0; l < TN; l++) {
                    const int c_ptr_x = c_block_x + WN * int(warp_id % (BN / WN)) + int(id_in_warp % (WSUBN/TN)) * TN + j * WSUBN + l;
                    const int c_ptr_y = c_block_y + WM * int(warp_id / (BN / WN)) + int(id_in_warp / (WSUBN/TN)) * TM + i * WSUBM + k;
                    const int c_reg_x = j * TN + l;
                    const int c_reg_y = i * TM + k;
                    c_ptr[(c_ptr_y) * N + c_ptr_x] = c_reg[(c_reg_y) * TN * (WN/WSUBN) + c_reg_x];
                }
            }
        }
    }
}

at::Tensor finegemm_7(const at::Tensor &a, const at::Tensor &b) {
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

    finegemm_7_kernel<BM, BN, BK, 64, 64, 64, 16, 8, 4>
        <<<grid, block>>>(M, K, N, a_ptr, b_ptr, result_ptr);
    return result;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(gemm_unified, CUDA, m) { m.impl("finegemm_7", &finegemm_7); }

}  // namespace gemm_unified