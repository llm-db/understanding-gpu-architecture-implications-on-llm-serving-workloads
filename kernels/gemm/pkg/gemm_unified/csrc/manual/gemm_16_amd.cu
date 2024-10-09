// replicate the fast impl of ROCWMMA NT
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
    finegemm_16_kernel(uint32_t m, uint32_t n, uint32_t k, at::Half const *A,
                       at::Half const *B, at::Half *D) {
    ///
    /// Perform initial global pre-fetch
    ///

    const half *a = reinterpret_cast<const half *>(A);
    const half *b = reinterpret_cast<const half *>(B);
    half *d = reinterpret_cast<half *>(D);

    HIP_DYNAMIC_SHARED(void*, localMemPtr);
    auto*    ldsPtrLo = reinterpret_cast<half *>(localMemPtr);
    auto*    ldsPtrHi = ldsPtrLo + (256 * 16);
    half* zj_smem = reinterpret_cast<half *>(localMemPtr);

    half* out = reinterpret_cast<half *>(d);

    constexpr int smem_buf_stage_step = 256 * 16;
    const int tid = threadIdx.y * 128 + threadIdx.x;
    const int warp_id = tid / 64;
    const int id_in_warp = tid % 64;
    const int warp_x = warp_id / 2;
    const int warp_y = warp_id % 2;
    const int c_dram_x = blockIdx.x * 128;
    const int c_dram_y = blockIdx.y * 128;
    const int M = m;
    const int N = n;
    half a_tmp[8];
    half b_tmp[8];

    {
        // preload the first part
        int k_next = 0;
        int k_stage = k_next % 2;
        {
            // load A into reg
            const int a_dram_x = k_next * 16 + (tid / 16);
            const int a_dram_y = c_dram_x + 8 * (tid % 16);
            *reinterpret_cast<int4 *>(&a_tmp) = *reinterpret_cast<const int4 *>(a + (a_dram_x) * M + (a_dram_y));
        }
        {
            // load B into reg
            const int b_dram_x = k_next * 16 + (tid / 16);
            const int b_dram_y = c_dram_y + 8 * (tid % 16);
            *reinterpret_cast<int4 *>(b_tmp) = *reinterpret_cast<const int4 *>(b + (b_dram_x) * N + b_dram_y);
            
        }
        {
            // write A into smem
            const int a_smem_x = tid / 16;
            const int a_smem_y = 8 * (tid % 16);
            *reinterpret_cast<int4 *>(zj_smem + k_stage * smem_buf_stage_step + (a_smem_x) * 128 + (a_smem_y)) = *reinterpret_cast<int4 *>(a_tmp);

        }
        {
            // write B into smem
            const int b_smem_x = 16 + tid / 16;
            const int b_smem_y = 8 * (tid % 16);
            *reinterpret_cast<int4 *>(zj_smem + k_stage * smem_buf_stage_step + (b_smem_x) * 128 + (b_smem_y)) = *reinterpret_cast<int4 *>(b_tmp);
        }
    }

    ///
    /// Synchronize warps and memory
    ///
    synchronize_workgroup();

    using halfx4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) __fp16;
    using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
    using zj_mFrag = halfx4[2];
    floatx16 zj_fragAcc[BLOCKS_X][BLOCKS_Y];
    zj_mFrag zj_fragA[BLOCKS_X];
    zj_mFrag zj_fragB[BLOCKS_Y];
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
    


    ///
    /// Accumulate A * B for all mfma frags in warp tile
    ///
    __syncthreads();
    for(uint32_t currentK = ROCWMMA_K; currentK < k; currentK += ROCWMMA_K)
    {
        {
            // load the required stuff for this time
            const __fp16* a_ptr = reinterpret_cast<const __fp16*>(ldsPtrLo);
            const int a_warp_offset_0 = 64 * warp_x;
            const int a_warp_offset_1 = 64 * warp_x + 32;
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int smem_x = 4 * (id_in_warp / 32) + i;
                const int smem_y = a_warp_offset_0 + id_in_warp % 32;
                zj_fragA[0][0][i] = *(a_ptr + (smem_x) * 128 + (smem_y));
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int smem_x = 8 + 4 * (id_in_warp / 32) + i;
                const int smem_y = a_warp_offset_0 + id_in_warp % 32;
                zj_fragA[0][1][i] = *(a_ptr + (smem_x) * 128 + (smem_y));
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int smem_x = 4 * (id_in_warp / 32) + i;
                const int smem_y = a_warp_offset_1 + id_in_warp % 32;
                zj_fragA[1][0][i] = *(a_ptr + (smem_x) * 128 + (smem_y));
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int smem_x = 8 + 4 * (id_in_warp / 32) + i;
                const int smem_y = a_warp_offset_1 + id_in_warp % 32;
                zj_fragA[1][1][i] = *(a_ptr + (smem_x) * 128 + (smem_y));
            }

            const __fp16* b_ptr = a_ptr + (128) * 16;
            const int b_warp_offset_0 = 64 * warp_y;
            const int b_warp_offset_1 = 64 * warp_y + 32;
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int smem_x = 4 * (id_in_warp / 32) + i;
                const int smem_y = b_warp_offset_0 + id_in_warp % 32;
                zj_fragB[0][0][i] = *(b_ptr + (smem_x) * 128 + (smem_y));
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int smem_x = 8 + 4 * (id_in_warp / 32) + i;
                const int smem_y = b_warp_offset_0 + id_in_warp % 32;
                zj_fragB[0][1][i] = *(b_ptr + (smem_x) * 128 + (smem_y));
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int smem_x = 4 * (id_in_warp / 32) + i;
                const int smem_y = b_warp_offset_1 + id_in_warp % 32;
                zj_fragB[1][0][i] = *(b_ptr + (smem_x) * 128 + (smem_y));
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int smem_x = 8 + 4 * (id_in_warp / 32) + i;
                const int smem_y = b_warp_offset_1 + id_in_warp % 32;
                zj_fragB[1][1][i] = *(b_ptr + (smem_x) * 128 + (smem_y));
            }
            
        }
        
        
        {
            int k_next = (currentK / 16);
            int k_stage = k_next % 2;
            k_next = min(k_next, (k / 16) - 1);
            {
                // load A into reg
                const int a_dram_x = k_next * 16 + (tid / 16);
                const int a_dram_y = c_dram_x + 8 * (tid % 16);
                *reinterpret_cast<int4 *>(&a_tmp) = *reinterpret_cast<const int4 *>(a + (a_dram_x) * M + (a_dram_y));
            }
            {
                // load B into reg
                const int b_dram_x = k_next * 16 + (tid / 16);
                const int b_dram_y = c_dram_y + 8 * (tid % 16);
                *reinterpret_cast<int4 *>(b_tmp) = *reinterpret_cast<const int4 *>(b + (b_dram_x) * N + b_dram_y);
            }
        }


        {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    #pragma unroll
                    for (int m = 0; m < 2; m++) {
                        zj_fragAcc[i][j] = __builtin_amdgcn_mfma_f32_32x32x8f16(zj_fragA[i][m], zj_fragB[j][m], zj_fragAcc[i][j], 0, 0, 0);
                    }
                }
            }
        }

        {
            int k_next = (currentK / 16);
            int k_stage = k_next % 2;
            {
                // write A into smem
                const int a_smem_x = tid / 16;
                const int a_smem_y = 8 * (tid % 16);
                *reinterpret_cast<int4 *>(zj_smem + k_stage * smem_buf_stage_step + (a_smem_x) * 128 + (a_smem_y)) = *reinterpret_cast<int4 *>(a_tmp);

            }
            {
                // write B into smem
                const int b_smem_x = 16 + tid / 16;
                const int b_smem_y = 8 * (tid % 16);
                *reinterpret_cast<int4 *>(zj_smem + k_stage * smem_buf_stage_step + (b_smem_x) * 128 + (b_smem_y)) = *reinterpret_cast<int4 *>(b_tmp);
            }
        }

        // Make sure that all waves have finished reading / writing to lds for currentK.
        __syncthreads();

        // Swap Lds buffers
        auto* tmp = ldsPtrLo;
        ldsPtrLo  = ldsPtrHi;
        ldsPtrHi  = tmp;
    }
    
    
    {
        // load the required stuff for this time
        const __fp16* a_ptr = reinterpret_cast<const __fp16*>(ldsPtrLo);
        const int a_warp_offset_0 = 64 * warp_x;
        const int a_warp_offset_1 = 64 * warp_x + 32;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int smem_x = 4 * (id_in_warp / 32) + i;
            const int smem_y = a_warp_offset_0 + id_in_warp % 32;
            zj_fragA[0][0][i] = *(a_ptr + (smem_x) * 128 + (smem_y));
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int smem_x = 8 + 4 * (id_in_warp / 32) + i;
            const int smem_y = a_warp_offset_0 + id_in_warp % 32;
            zj_fragA[0][1][i] = *(a_ptr + (smem_x) * 128 + (smem_y));
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int smem_x = 4 * (id_in_warp / 32) + i;
            const int smem_y = a_warp_offset_1 + id_in_warp % 32;
            zj_fragA[1][0][i] = *(a_ptr + (smem_x) * 128 + (smem_y));
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int smem_x = 8 + 4 * (id_in_warp / 32) + i;
            const int smem_y = a_warp_offset_1 + id_in_warp % 32;
            zj_fragA[1][1][i] = *(a_ptr + (smem_x) * 128 + (smem_y));
        }

        const __fp16* b_ptr = a_ptr + (128) * 16;
        const int b_warp_offset_0 = 64 * warp_y;
        const int b_warp_offset_1 = 64 * warp_y + 32;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int smem_x = 4 * (id_in_warp / 32) + i;
            const int smem_y = b_warp_offset_0 + id_in_warp % 32;
            zj_fragB[0][0][i] = *(b_ptr + (smem_x) * 128 + (smem_y));
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int smem_x = 8 + 4 * (id_in_warp / 32) + i;
            const int smem_y = b_warp_offset_0 + id_in_warp % 32;
            zj_fragB[0][1][i] = *(b_ptr + (smem_x) * 128 + (smem_y));
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int smem_x = 4 * (id_in_warp / 32) + i;
            const int smem_y = b_warp_offset_1 + id_in_warp % 32;
            zj_fragB[1][0][i] = *(b_ptr + (smem_x) * 128 + (smem_y));
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int smem_x = 8 + 4 * (id_in_warp / 32) + i;
            const int smem_y = b_warp_offset_1 + id_in_warp % 32;
            zj_fragB[1][1][i] = *(b_ptr + (smem_x) * 128 + (smem_y));
        }
            
    }

    {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                #pragma unroll
                for (int m = 0; m < 2; m++) {
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

at::Tensor finegemm_16(const at::Tensor &a, const at::Tensor &b) {
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
        = 2u * sizeof(half) * (128 + 128) * (ROCWMMA_K);

    finegemm_16_kernel<<<grid, block, ldsusage>>>(M, N, K, a_ptr, b_ptr, result_ptr);

    return result;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(gemm_unified, CUDA, m) {
    m.impl("finegemm_16", &finegemm_16);
}

}  // namespace gemm_unified