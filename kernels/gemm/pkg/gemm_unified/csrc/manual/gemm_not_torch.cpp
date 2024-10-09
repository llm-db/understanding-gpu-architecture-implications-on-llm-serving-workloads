#include <cstdio>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/amd_hip_vector_types.h>
#include <iostream>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>


ROCWMMA_KERNEL void __launch_bounds__(256) gemm_rocwmma_d(uint32_t       m,
                                                          uint32_t       n,
                                                          uint32_t       k,
                                                          InputT const*  a,
                                                          InputT const*  b,
                                                          OutputT const* c,
                                                          OutputT*       d,
                                                          uint32_t       lda,
                                                          uint32_t       ldb,
                                                          uint32_t       ldc,
                                                          uint32_t       ldd,
                                                          ComputeT       alpha,
                                                          ComputeT       beta)
{
    ///
    /// Perform initial global pre-fetch
    ///

    HIP_DYNAMIC_SHARED(void*, localMemPtr);
    auto*    ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
    auto*    ldsPtrHi = ldsPtrLo + (256 * 16);
    half* zj_smem = reinterpret_cast<half *>(localMemPtr);

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
        int k = 0;
        int k_stage = k % 2;
        {
            // load A into reg
            const int a_dram_x = k * 16 + 8 * (tid / 128);
            const int a_dram_y = c_dram_x + (tid % 128);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                a_tmp[i] = *(a + (a_dram_x + i) * M + (a_dram_y));
            }
            // const int a_dram_x = k * 16 + (tid / 16);
            // const int a_dram_y = c_dram_x + 8 * (tid % 16);
            // *reinterpret_cast<int4 *>(&a_tmp) = *reinterpret_cast<const int4 *>(a + (a_dram_x) * M + (a_dram_y));
        }
        {
            // load B into reg
            const int b_dram_x = k * 16 + 8 * (tid / 128);
            const int b_dram_y = c_dram_y + (tid % 128);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                b_tmp[i] = *(b + (b_dram_x + i) * N + (b_dram_y));
            }
        }
        {
            // write A into smem
            const int a_smem_x = tid % 128;
            const int a_smem_y = 8 * (tid / 128);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                *(zj_smem + k_stage * smem_buf_stage_step + (a_smem_x) * (16) + (a_smem_y + i)) = a_tmp[i];
            }
        }
        {
            // write B into smem
            const int b_smem_x = 128 + tid % 128;
            const int b_smem_y = 8 * (tid / 128);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                *(zj_smem + k_stage * smem_buf_stage_step + (b_smem_x) * (16) + (b_smem_y + i)) = b_tmp[i];
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
    using zj_mFrag = halfx4[2];
    ///
    /// Accumulate A * B for all mfma frags in warp tile
    ///
    for(uint32_t currentK = ROCWMMA_K; currentK < k; currentK += ROCWMMA_K)
    {
        zj_mFrag zj_fragA[BLOCKS_X];
        zj_mFrag zj_fragB[BLOCKS_Y];
        {
            // load the required stuff for this time
            const half* a_ptr = reinterpret_cast<const half*>(ldsPtrLo);
            const int a_warp_offset = 64 * warp_x;
            zj_fragA[0][0] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32)) * 16 + (4 * (id_in_warp / 32)));
            zj_fragA[0][1] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32)) * 16 + (4 * (id_in_warp / 32) + 8));
            zj_fragA[1][0] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32) + 32) * 16 + (4 * (id_in_warp / 32)));
            zj_fragA[1][1] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32) + 32) * 16 + (4 * (id_in_warp / 32) + 8));
            const half* b_ptr = a_ptr + (128) * 16;
            const int b_warp_offset = 64 * warp_y;
            zj_fragB[0][0] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32)) * 16 + (4 * (id_in_warp / 32)));
            zj_fragB[0][1] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32)) * 16 + (4 * (id_in_warp / 32) + 8));
            zj_fragB[1][0] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32) + 32) * 16 + (4 * (id_in_warp / 32)));
            zj_fragB[1][1] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32) + 32) * 16 + (4 * (id_in_warp / 32) + 8));
        }
        
        {
            int k_next = (currentK / 16);
            int k_stage = k_next % 2;
            k_next = min(k_next, (k / 16) - 1);
            {
                // load A into reg
                const int a_dram_x = k_next * 16 + 8 * (tid / 128);
                const int a_dram_y = c_dram_x + (tid % 128);
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    a_tmp[i] = *(a + (a_dram_x + i) * M + (a_dram_y));
                }
                // const int a_dram_x = k_next * 16 + (tid / 16);
                // const int a_dram_y = c_dram_x + 8 * (tid % 16);
                // *reinterpret_cast<int4 *>(&a_tmp) = *reinterpret_cast<const int4 *>(a + (a_dram_x) * M + (a_dram_y));
            }
            {
                // load B into reg
                const int b_dram_x = k_next * 16 + 8 * (tid / 128);
                const int b_dram_y = c_dram_y + (tid % 128);
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    b_tmp[i] = *(b + (b_dram_x + i) * N + (b_dram_y));
                }
                // const int b_dram_x = k_next * 16 + (tid / 16);
                // const int b_dram_y = c_dram_y + 8 * (tid % 16);
                // *reinterpret_cast<int4 *>(&b_tmp) = *reinterpret_cast<const int4 *>(b + (b_dram_x) * N + (b_dram_y));
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
                const int a_smem_x = tid % 128;
                const int a_smem_y = 8 * (tid / 128);
                *reinterpret_cast<int4 *>(zj_smem + k_stage * smem_buf_stage_step + (a_smem_x) * (16) + (a_smem_y)) = *reinterpret_cast<int4 *>(&a_tmp);
            }
            {
                // write B into smem
                const int b_smem_x = 128 + tid % 128;
                const int b_smem_y = 8 * (tid / 128);
                *reinterpret_cast<int4 *>(zj_smem + k_stage * smem_buf_stage_step + (b_smem_x) * (16) + (b_smem_y)) = *reinterpret_cast<int4 *>(&b_tmp);
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
        zj_fragA[0][0] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32)) * 16 + (4 * (id_in_warp / 32)));
        zj_fragA[0][1] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32)) * 16 + (4 * (id_in_warp / 32) + 8));
        zj_fragA[1][0] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32) + 32) * 16 + (4 * (id_in_warp / 32)));
        zj_fragA[1][1] = *reinterpret_cast<const halfx4 *>(a_ptr + (a_warp_offset + (id_in_warp % 32) + 32) * 16 + (4 * (id_in_warp / 32) + 8));
        const half* b_ptr = a_ptr + (128) * 16;
        const int b_warp_offset = 64 * warp_y;
        zj_fragB[0][0] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32)) * 16 + (4 * (id_in_warp / 32)));
        zj_fragB[0][1] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32)) * 16 + (4 * (id_in_warp / 32) + 8));
        zj_fragB[1][0] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32) + 32) * 16 + (4 * (id_in_warp / 32)));
        zj_fragB[1][1] = *reinterpret_cast<const halfx4 *>(b_ptr + (b_warp_offset + (id_in_warp % 32) + 32) * 16 + (4 * (id_in_warp / 32) + 8));
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
                        *(d + (d_dram_x) * N + (d_dram_y)) = zj_fragAcc[m][n][frag4_start_index + q];
                    }
                }
            }
        }

    }
}




ROCWMMA_HOST void gemm_test(uint32_t m, uint32_t n, uint32_t k, ComputeT alpha, ComputeT beta)
{
    // Runtime checks for host parameters
    uint32_t hTBLOCK_X    = isGfx9() ? gfx9Params::TBLOCK_X : gfx11Params::TBLOCK_X;
    uint32_t hTBLOCK_Y    = isGfx9() ? gfx9Params::TBLOCK_Y : gfx11Params::TBLOCK_Y;
    uint32_t hBLOCKS_X    = isGfx9() ? gfx9Params::BLOCKS_X : gfx11Params::BLOCKS_X;
    uint32_t hBLOCKS_Y    = isGfx9() ? gfx9Params::BLOCKS_Y : gfx11Params::BLOCKS_Y;
    uint32_t hROCWMMA_M   = isGfx9() ? gfx9Params::ROCWMMA_M : gfx11Params::ROCWMMA_M;
    uint32_t hROCWMMA_N   = isGfx9() ? gfx9Params::ROCWMMA_N : gfx11Params::ROCWMMA_N;
    uint32_t hROCWMMA_K   = isGfx9() ? gfx9Params::ROCWMMA_K : gfx11Params::ROCWMMA_K;
    uint32_t hWARP_TILE_X = hBLOCKS_X * hROCWMMA_M;
    uint32_t hWARP_TILE_Y = hBLOCKS_Y * hROCWMMA_N;

    // Runtime warp calculation (host code needs to query warpsize dynamically)
    auto warpSize = getWarpSize();
    auto macroTileSize
        = rocwmma::make_coord2d(hTBLOCK_X / warpSize * hWARP_TILE_X, hTBLOCK_Y * hWARP_TILE_Y);

    // Device check for supported block and wave sizes
    if(isGfx11() && (hROCWMMA_M != 16 || hROCWMMA_N != 16))
    {
        std::cout << "Unsupported block size!\n";
        return;
    }

    if(isGfx9() && (hROCWMMA_M != hROCWMMA_N) || (hROCWMMA_M != 16 && hROCWMMA_M != 32))
    {
        std::cout << "Unsupported block size!\n";
        return;
    }

    if(isGfx11() && getWarpSize() != Constants::AMDGCN_WAVE_SIZE_32)
    {
        std::cout << "Unsupported wave size!\n";
        return;
    }

    if(isGfx9() && getWarpSize() != Constants::AMDGCN_WAVE_SIZE_64)
    {
        std::cout << "Unsupported wave size!\n";
        return;
    }

    // Bounds check
    if((m < get<0>(macroTileSize) || n < get<1>(macroTileSize) || k < hROCWMMA_K)
        || (m % hROCWMMA_M || n % hROCWMMA_N || k % hROCWMMA_K))
    {
        std::cout << "Unsupported matrix size!\n";
        return;
    }

    // Layouts = N_T_N_N
    int lda = m;
    int ldb = n;
    int ldc = m;
    int ldd = ldc;

    std::cout << "Initializing host data..." << std::endl;

    // Initialize input matrices
    std::vector<InputT>  matrixA(m * k);
    std::vector<InputT>  matrixB(k * n);
    std::vector<OutputT> matrixC(m * n);

    // Fill outputs with NaN to catch contamination
    std::vector<OutputT> matrixD(m * n, std::numeric_limits<OutputT>::signaling_NaN());

    fillRand(matrixA.data(), m, k);
    // float16_t index = 0;
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < k; j++) {
    //         matrixA[j + i * k] = index;
    //         index += 1;
    //     }
    // }
    fillRand(matrixB.data(), k, n);
    // index = 0;
    // for (int i = 0; i < k; i++) {
    //     for (int j = 0; j < n; j++) {
    //         matrixB[j + i * n] = index;
    //         index -= 1;
    //     }
    // }
    fillRand(matrixC.data(), m, n);

    std::cout << "Initializing device data..." << std::endl;

    // Allocate and copy device memory
    InputT*  d_a;
    InputT*  d_b;
    OutputT* d_c;
    OutputT* d_d;

    const size_t bytesA = matrixA.size() * sizeof(InputT);
    const size_t bytesB = matrixB.size() * sizeof(InputT);
    const size_t bytesC = matrixC.size() * sizeof(OutputT);
    const size_t bytesD = matrixD.size() * sizeof(OutputT);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
    CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

auto blockDim = dim3(hTBLOCK_X, hTBLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceilDiv(m, get<0>(macroTileSize)),
                        rocwmma::ceilDiv(n, get<1>(macroTileSize)));

    std::cout << "Launching GEMM kernel..." << std::endl;
    std::cout << "gridDim (" << gridDim.x << " " << gridDim.y << ")" << " blockdim ("
                << blockDim.x << " " << blockDim.y << ")" << std::endl;

    // Uses 2 lds blocks for prefetch loop (A and B)
    int ldsusage
        = 2u * sizeof(InputT) * (get<0>(macroTileSize) + get<1>(macroTileSize)) * (ROCWMMA_K);

    ////
    auto rocwmmaKernel = [&]() {
        hipExtLaunchKernelGGL(gemm_rocwmma_d,
                                gridDim,
                                blockDim,
                                ldsusage,
                                0,
                                nullptr,
                                nullptr,
                                0,
                                m,
                                n,
                                k,
                                d_a,
                                d_b,
                                d_c,
                                d_d,
                                lda,
                                ldb,
                                ldc,
                                ldd,
                                alpha,
                                beta);
    };

    constexpr uint32_t warmups    = 5u;
    constexpr uint32_t recordRuns = 20u;

    // Warm-up runs, not recorded
    for(uint32_t i = 0; i < warmups; ++i)
    {
        rocwmmaKernel();
    }

    // Actual recorded runs
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    CHECK_HIP_ERROR(hipEventRecord(startEvent));
    for(uint32_t i = 0; i < recordRuns; ++i)
    {
        rocwmmaKernel();
    }
    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));

    auto gFlops = calculateGFlops(m, n, k);
    auto tFlopsPerSec
        = calculateTFlopsPerSec(m, n, k, static_cast<double>(elapsedTimeMs), recordRuns);

    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // Echo performance
    std::cout << "TBlockX, TBlockY, " << "BlocksX, BlocksY, " << "BlkM, BlkN, BlkK, "
                << "MatM, MatN, MatK, " << "alpha, lda, ldb, " << "beta, ldc, ldd, "
                << "elapsedMs, Problem Size(GFlops), TFlops/s" << std::endl;

    std::cout << hTBLOCK_X << ", " << hTBLOCK_Y << ", " << hBLOCKS_X << ", " << hBLOCKS_Y
                << ", " << hROCWMMA_M << ", " << hROCWMMA_N << ", " << hROCWMMA_K << ", " << m
                << ", " << n << ", " << k << ", " << alpha << ", " << lda << ", " << ldb << ", "
                << beta << ", " << ldc << ", " << ldd << ", " << elapsedTimeMs << ", " << gFlops
                << ", " << tFlopsPerSec << std::endl;

    std::cout << "Validating result with reference..." << std::endl;

    if((uint64_t)m * (uint64_t)n * (uint64_t)k > (2048ull * 2048ull * 2048ull))
    {
        std::cout << "Please wait. Large sizes can take a while!" << std::endl;
    }

    // Bring kernel result back to host
    CHECK_HIP_ERROR(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost));

    // Setup and run reference computation
    std::vector<OutputT> matrixD_ref(m * n, std::numeric_limits<OutputT>::signaling_NaN());
    gemm_cpu_h<InputT, OutputT, ComputeT, DataLayoutA, DataLayoutB, DataLayoutC>(
        m,
        n,
        k,
        matrixA.data(),
        matrixB.data(),
        matrixC.data(),
        matrixD_ref.data(),
        lda,
        ldb,
        ldc,
        ldd,
        alpha,
        beta);

    auto res = compareEqual(matrixD.data(), matrixD_ref.data(), m * n);

    if(std::get<0>(res) == false)
    {
        // float32_t max_diff = 0;
        // for (int i = 0; i < 128; i++) {
        //     for (int j = 0; j < 128; j++) {
        //         if (std::isnan(abs(__half2float(matrixD.data()[i * 128 + j] - matrixD_ref.data()[i * 128 + j])))) {
        //             printf("NAN: %d %d", i, j);
        //         } else if (abs(__half2float(matrixD.data()[i * 128 + j] - matrixD_ref.data()[i * 128 + j])) > 0.1) {
        //             printf("DIFF: %d %d", i, j);
        //         }
        //     }
        // }
        int x_base = 8;
        int y_base = 0;
        std::cout << "FAILED\n";
        for (int i = x_base; i < x_base + 8; i++) {
            for (int j = y_base; j < y_base + 8; j++) {
                printf("%f ", __half2float(matrixD.data()[i * 128 + j]));
            }
            printf("\n");
        }
        printf("\n");
        for (int i = x_base; i < x_base + 8; i++) {
            for (int j = y_base; j < y_base + 8; j++) {
                printf("%f ", __half2float(matrixD_ref.data()[i * 128 + j]));
            }
            printf("\n");
        }
    }
    else
    {
        std::cout << "PASSED\n";
    }

    std::cout << "Max relative error: " << std::get<1>(res) << std::endl;

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));

    std::cout << "Finished!" << std::endl;
}

int main()
{
    gemm_test(4096, 4096, 4096, 1, 0);
    return 0;
}