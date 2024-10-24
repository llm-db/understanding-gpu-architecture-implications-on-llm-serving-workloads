#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace gemm_unified {

__global__ void finegemm_0_kernel(int M, int K, int N, const at::Half *a,
                                  const at::Half *b, at::Half *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < M && idy < N) {
        at::Half sum = 0.0;
        for (int i = 0; i < K; i++) {
            at::Half a_val = a[idx * K + i];
            at::Half b_val = b[i * N + idy];
            sum += a_val * b_val;
        }
        result[idx * N + idy] = sum;
    }
}

at::Tensor finegemm_0(const at::Tensor &a, const at::Tensor &b) {
    long M = a.size(0);
    long K = a.size(1);
    long N = b.size(1);
    TORCH_CHECK(K == b.size(0));
    TORCH_CHECK(a.dtype() == at::kHalf);
    TORCH_CHECK(b.dtype() == at::kHalf);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::empty({M, N}, a_contig.options());
    const at::Half *a_ptr = a_contig.data_ptr<at::Half>();
    const at::Half *b_ptr = b_contig.data_ptr<at::Half>();
    at::Half *result_ptr = result.data_ptr<at::Half>();
    int numel = a_contig.numel();

    // Launch the kernel
    dim3 block(32, 32);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    finegemm_0_kernel<<<grid, block>>>(M, K, N, a_ptr, b_ptr, result_ptr);
    return result;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(gemm_unified, CUDA, m) { m.impl("finegemm_0", &finegemm_0); }

}  // namespace gemm_unified