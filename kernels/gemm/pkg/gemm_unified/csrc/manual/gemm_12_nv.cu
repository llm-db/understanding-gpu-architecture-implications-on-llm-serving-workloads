// from here we start to use CUTE do deal with all spatial partitions

// from here on we start to use tensor core through CUDA WMMA API
// mostly learnt from https://github.com/Bruce-Lee-LY/cuda_hgemm
// wmma_async with a multi stage pipeline


#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include <cuda_pipeline.h>

#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cute/layout.hpp"
#include "cute/tensor_impl.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"

#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>


namespace gemm_unified {

at::Tensor finegemm_12(const at::Tensor &a, const at::Tensor &b) {
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);
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
    const cutlass::half_t *a_ptr = reinterpret_cast<const cutlass::half_t *>(a_contig.data_ptr<at::Half>());
    const cutlass::half_t *b_ptr = reinterpret_cast<const cutlass::half_t *>(b_contig.data_ptr<at::Half>());
    cutlass::half_t *result_ptr = reinterpret_cast<cutlass::half_t *>(result.data_ptr<at::Half>());

    // Launch the kernel
    // dim3 block(128);
    // dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    // Define the GEMM operation
    using namespace cute;
    auto shape  = Shape <_3>{};
    auto stride = Stride<_3>{};
    crd2idx(   16, shape, stride);
    crd2idx(   make_coord(16), shape, stride);
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,                           // ElementA
        cutlass::layout::RowMajor,              // LayoutA
        cutlass::half_t,                           // ElementB
        cutlass::layout::RowMajor,              // LayoutB
        cutlass::half_t,                           // ElementOutput
        cutlass::layout::RowMajor,              // LayoutOutput
        cutlass::half_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm75
    >;
    Gemm gemm_op;
    Gemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                              {a_ptr, K},    // Tensor-ref for source matrix A
                              {b_ptr, N},    // Tensor-ref for source matrix B
                              {result_ptr, N},    // Tensor-ref for source matrix C
                              {result_ptr, N}    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              );

    gemm_op(args);

    // gemm('T', 'N', M, N, K, 1.0, a_ptr, K, b_ptr, N, 0.0, result_ptr, N);

    // finegemm_12_kernel<BM, BN, BK, 64, 64>
    //     <<<grid, block>>>(M, K, N, a_ptr, b_ptr, result_ptr);
    return result;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(gemm_unified, CUDA, m) { m.impl("finegemm_12", &finegemm_12); }

}  // namespace gemm_unified