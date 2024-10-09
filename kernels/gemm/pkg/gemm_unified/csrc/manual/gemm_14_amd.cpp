// wider & better but still not so good TN
#include <torch/extension.h>
#include <cstdio>

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#include "rocwmma/internal/types.hpp"

#define ROCWMMA_M 32
#define ROCWMMA_N 32
#define ROCWMMA_K 16
#define BLOCKS_X 2
#define BLOCKS_Y 2

namespace gemm_unified {
using namespace rocwmma;
__global__ void __launch_bounds__(256)
    finegemm_14_kernel(uint32_t m, uint32_t n, uint32_t k, at::Half const *A,
                       at::Half const *B, at::Half *D) {
    ///
    /// Perform initial global pre-fetch
    ///

    const half* a = reinterpret_cast<const half*>(A);
    const half* b = reinterpret_cast<const half*>(B);
    half* d = reinterpret_cast<half*>(D);

    HIP_DYNAMIC_SHARED(void*, localMemPtr);
    auto*    ldsPtrLo = reinterpret_cast<half*>(localMemPtr);
    auto*    ldsPtrHi = ldsPtrLo + (256 * 32);
    half* zj_smem = reinterpret_cast<half *>(localMemPtr);

    constexpr int smem_buf_stage_step = 256 * 32;
    const int tid = threadIdx.y * 128 + threadIdx.x;
    const int warp_id = tid / 64;
    const int id_in_warp = tid % 64;
    const int warp_x = warp_id / 2;
    const int warp_y = warp_id % 2;
    const int c_dram_x = blockIdx.x * 128;
    const int c_dram_y = blockIdx.y * 128;
    const int M = m;
    const int N = n;
    const int K = k;
    half a_tmp[2][8];
    half b_tmp[2][8];
    {
        // preload the first part
        int k_next = 0;
        int k_stage = k_next % 2;
        {
            // load A into reg
            for (int i = 0; i < 2; i++) {
                const int a_dram_x = c_dram_x + i * 64 + (tid / 4);
                const int a_dram_y = k_next * 32 + 8 * (tid % 4);
                *reinterpret_cast<int4 *>(&a_tmp[i][0]) = *reinterpret_cast<const int4 *>(a + (a_dram_x) * K + (a_dram_y));
            }
        }
        {
            // load B into reg
            for (int i = 0; i < 2; i++) {
                const int b_dram_x = c_dram_y + i * 64 + (tid / 4);
                const int b_dram_y = k_next * 32 + 8 * (tid % 4);
                *reinterpret_cast<int4 *>(&b_tmp[i][0]) = *reinterpret_cast<const int4 *>(b + (b_dram_x) * K + b_dram_y);
            }
        }
        {
            // write A into smem
            for (int i = 0; i < 2; i++) {
                const int a_smem_x = i * 64 + tid / 4;
                const int a_smem_y = 8 * (tid % 4);
                *reinterpret_cast<int4 *>(zj_smem + k_stage * smem_buf_stage_step + (a_smem_x) * (32) + (a_smem_y)) = *reinterpret_cast<int4 *>(&a_tmp[i][0]);
            }
            
        }
        {
            // write B into smem
            for (int i = 0; i < 2; i++) {
                const int b_smem_x = 128 + 64 * i + tid / 4;
                const int b_smem_y = 8 * (tid % 4);
                *reinterpret_cast<int4 *>(zj_smem + k_stage * smem_buf_stage_step + (b_smem_x) * (32) + (b_smem_y)) = *reinterpret_cast<int4 *>(&b_tmp[i][0]);
            }            
        }
    }

    ///
    /// Synchronize warps and memory
    ///
    synchronize_workgroup();

    using halfx4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) __fp16;
    using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
    floatx16 zj_fragAcc[BLOCKS_X][BLOCKS_Y];
    
    #pragma unroll
    for (int i = 0; i < BLOCKS_X; i++) {
        #pragma unroll
        for (int j = 0; j < BLOCKS_Y; j++) {
            #pragma unroll
            for (int m = 0; m < 16; m++) {
                zj_fragAcc[i][j][m] = {0};
            }
        }
    }
    using zj_mFrag = halfx4[4];
    ///
    /// Accumulate A * B for all mfma frags in warp tile
    ///
    for(uint32_t currentK = 32; currentK < k; currentK += 32)
    {
        zj_mFrag zj_fragA[BLOCKS_X];
        zj_mFrag zj_fragB[BLOCKS_Y];
        {
            // load the required stuff for this time
            const half* a_ptr = reinterpret_cast<const half*>(ldsPtrLo);
            const int a_warp_offset = 64 * warp_x;
            zj_fragA[0][0] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32)));
            zj_fragA[0][1] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32) + 8));
            zj_fragA[0][2] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32) + 16));
            zj_fragA[0][3] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32) + 24));

            zj_fragA[1][0] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32)));
            zj_fragA[1][1] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32) + 8));
            zj_fragA[1][2] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32) + 16));
            zj_fragA[1][3] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32) + 24));
            const half* b_ptr = a_ptr + (128) * 32;
            const int b_warp_offset = 64 * warp_y;
            zj_fragB[0][0] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32)));
            zj_fragB[0][1] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32) + 8));
            zj_fragB[0][2] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32) + 16));
            zj_fragB[0][3] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32) + 24));

            zj_fragB[1][0] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32)));
            zj_fragB[1][1] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32) + 8));
            zj_fragB[1][2] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32) + 16));
            zj_fragB[1][3] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32) + 24));
        }
        
        {
            int k_next = (currentK / 32);
            int k_stage = k_next % 2;
            k_next = min(k_next, (k / 32) - 1);
            {
                // load A into reg
                for (int i = 0; i < 2; i++) {
                    const int a_dram_x = c_dram_x + i * 64 + (tid / 4);
                    const int a_dram_y = k_next * 32 + 8 * (tid % 4);
                    *reinterpret_cast<int4 *>(&a_tmp[i][0]) = *reinterpret_cast<const int4 *>(a + (a_dram_x) * K + (a_dram_y));
                }
            }
            {
                // load B into reg
                for (int i = 0; i < 2; i++) {
                    const int b_dram_x = c_dram_y + i * 64 + (tid / 4);
                    const int b_dram_y = k_next * 32 + 8 * (tid % 4);
                    *reinterpret_cast<int4 *>(&b_tmp[i][0]) = *reinterpret_cast<const int4 *>(b + (b_dram_x) * K + b_dram_y);
                }
            }
        }

        {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    #pragma unroll
                    for (int m = 0; m < 4; m++) {
                        zj_fragAcc[i][j] = __builtin_amdgcn_mfma_f32_32x32x8f16(zj_fragA[i][m], zj_fragB[j][m], zj_fragAcc[i][j], 0, 0, 0);
                    }
                }
            }
        }

        {
            int k_next = (currentK / 32);
            int k_stage = k_next % 2;
            {
                // write A into smem
                for (int i = 0; i < 2; i++) {
                    const int a_smem_x = i * 64 + tid / 4;
                    const int a_smem_y = 8 * (tid % 4);
                    *reinterpret_cast<int4 *>(zj_smem + k_stage * smem_buf_stage_step + (a_smem_x) * (32) + (a_smem_y)) = *reinterpret_cast<int4 *>(&a_tmp[i][0]);
                }
            }
            {
                // write B into smem
                for (int i = 0; i < 2; i++) {
                    const int b_smem_x = 128 + 64 * i + tid / 4;
                    const int b_smem_y = 8 * (tid % 4);
                    *reinterpret_cast<int4 *>(zj_smem + k_stage * smem_buf_stage_step + (b_smem_x) * (32) + (b_smem_y)) = *reinterpret_cast<int4 *>(&b_tmp[i][0]);
                }
            }
        }

        // Make sure that all waves have finished reading / writing to lds for currentK.
        synchronize_workgroup();

        // Swap Lds buffers
        auto* tmp = ldsPtrLo;
        ldsPtrLo  = ldsPtrHi;
        ldsPtrHi  = tmp;
    }
    
    zj_mFrag zj_fragA[BLOCKS_X];
    zj_mFrag zj_fragB[BLOCKS_Y];
    {
        // load the required stuff for this time
        const half* a_ptr = reinterpret_cast<const half*>(ldsPtrLo);
        const int a_warp_offset = 64 * warp_x;
        zj_fragA[0][0] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32)));
        zj_fragA[0][1] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32) + 8));
        zj_fragA[0][2] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32) + 16));
        zj_fragA[0][3] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32) + 24));

        zj_fragA[1][0] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32)));
        zj_fragA[1][1] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32) + 8));
        zj_fragA[1][2] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32) + 16));
        zj_fragA[1][3] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32) + 24));
        const half* b_ptr = a_ptr + (128) * 32;
        const int b_warp_offset = 64 * warp_y;
        zj_fragB[0][0] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32)));
        zj_fragB[0][1] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32) + 8));
        zj_fragB[0][2] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32) + 16));
        zj_fragB[0][3] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32)) * 32 + (4 * (id_in_warp / 32) + 24));

        zj_fragB[1][0] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32)));
        zj_fragB[1][1] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32) + 8));
        zj_fragB[1][2] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32) + 16));
        zj_fragB[1][3] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32) + 32) * 32 + (4 * (id_in_warp / 32) + 24));
    }
    {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    zj_fragAcc[i][j] = __builtin_amdgcn_mfma_f32_32x32x8f16(zj_fragA[i][m], zj_fragB[j][m], zj_fragAcc[i][j], 0, 0, 0);
                }
            }
        }
    }

    {
        // write back results
        const int warp_x_offset = c_dram_x + 64 * warp_x;
        const int warp_y_offset = c_dram_y + 64 * warp_y;
        for (int m = 0; m < 2; m++) {
            for (int n = 0; n < 2; n++) {
                // floatx16 = zj_fragAcc[m][n];
                const int frag16_x = 32 * m + 4 * (id_in_warp / 32);
                const int frag16_y = 32 * n + (id_in_warp % 32);
                for (int p = 0; p < 4; p++) {
                    const int frag4_start_index = p * 4;
                    const int frag4_x = frag16_x + p * 8;
                    for (int q = 0; q < 4; q++) {
                        const int d_dram_x = warp_x_offset + frag4_x + q;
                        const int d_dram_y = warp_y_offset + frag16_y;
                        *(d + (d_dram_x) * N + (d_dram_y)) = __float2half(zj_fragAcc[m][n][frag4_start_index + q]);
                    }
                }
            }
        }

    }
}

at::Tensor finegemm_14(const at::Tensor &a, const at::Tensor &b) {
    long M = a.size(0);
    long K = a.size(1);
    long N = b.size(1);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::zeros({M, N}, a_contig.options());
    at::Half *a_ptr = a_contig.data_ptr<at::Half>();
    at::Half *b_ptr = b_contig.data_ptr<at::Half>();
    at::Half *result_ptr = result.data_ptr<at::Half>();

    const int BM = 4 * 32;
    const int BN = 4 * 32;
    dim3 block(128, 2);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    int ldsusage
        = 2u * sizeof(half) * (128 + 128) * (32);

    finegemm_14_kernel<<<grid, block, ldsusage>>>(M, N, K, a_ptr, b_ptr, result_ptr);

    return result;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(gemm_unified, CUDA, m) {
    m.impl("finegemm_14", &finegemm_14);
}

}  // namespace gemm_unified