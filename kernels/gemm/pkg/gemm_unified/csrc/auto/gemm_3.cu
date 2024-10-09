// This kernel we try to use shared memory to improve performance
// TODO: fix this kernel
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace gemm_unified {

__global__ void finegemm_3_kernel(int M, int K, int N, const at::Half *a,
                                  const at::Half *b, at::Half *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    __shared__ at::Half a_shared[32][32];
    __shared__ at::Half b_shared[32][32];

    if (idx < N && idy < M) {
        for (int sram_block = 0; sram_block < K; sram_block += 32) {

            // load data into shared memory
            int a_x = sram_block * 32 + threadIdx.x;
            int a_y = idy;
            int b_x = idx;
            int b_y = sram_block * 32 + threadIdx.y;

            if (a_x < K) {
                a_shared[threadIdx.y][threadIdx.x] = a[a_y * K + a_x];
            } else {
                a_shared[threadIdx.y][threadIdx.x] = 0.0;
            }
            
            if (b_y < K) {
                b_shared[threadIdx.y][threadIdx.x] = b[b_y * N + b_x];
            } else {
                b_shared[threadIdx.y][threadIdx.x] = 0.0;
            }

            __syncthreads();

            // compute
            at::Half sum = 0.0;
            for (int i = 0; i < 32; i += 2) {
                half2 a_val2 = __halves2half2(a_shared[threadIdx.y][i], a_shared[threadIdx.y][i + 1]);
                half2 b_val2 = __halves2half2(b_shared[i][threadIdx.x], b_shared[i + 1][threadIdx.x]);
                half2 sum2 = __hmul2(a_val2, b_val2);
                // add operator is not reloaded in ROCm, use intrinsic instead
                sum += __hadd(__low2half(sum2), __high2half(sum2));
            }
            __syncthreads();
            result[idy * N + idx] += sum;

        }
    }
}

at::Tensor finegemm_3(const at::Tensor &a, const at::Tensor &b) {
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

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::zeros({M, N}, a_contig.options());
    const at::Half *a_ptr = a_contig.data_ptr<at::Half>();
    const at::Half *b_ptr = b_contig.data_ptr<at::Half>();
    at::Half *result_ptr = result.data_ptr<at::Half>();
    int numel = a_contig.numel();

    // Launch the kernel
    dim3 block(32, 32);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    finegemm_3_kernel<<<grid, block>>>(M, K, N, a_ptr, b_ptr, result_ptr);
    return result;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(gemm_unified, CUDA, m) { m.impl("finegemm_3", &finegemm_3); }

}  // namespace gemm_unified