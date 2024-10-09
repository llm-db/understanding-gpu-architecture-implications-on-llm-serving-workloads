// Adapted from https://github.com/tspeterkim/flash-attention-minimal/tree/main
// this is a typical Flash Attention 1 style kernel, I add causal into it
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <cstdio>
#include <math_constants.h>

namespace fmha_unified {

__global__ void finefmha_0_kernel(const float *Q, const float *K, const float *V,
                              const int N, const int d, const int Tc,
                              const int Tr, const int Bc, const int Br,
                              const float softmax_scale, float *l, float *m,
                              float *O, bool causal) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float *Qi = sram;
    float *Kj = &sram[tile_size];
    float *Vj = &sram[tile_size * 2];
    float *S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {
        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++) {
            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] =
                    Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                const int q_in_N = i * Br + tx;
                const int k_in_N = j * Bc + y;
                if (causal && k_in_N > q_in_N) {
                    sum += -CUDART_INF_F;
                }
                S[(Bc * tx) + y] = sum;
                if (sum > row_m) row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = expf(S[(Bc * tx) + y] - row_m);
                if (isnan(S[(Bc * tx) + y])) {
                    // very very strange, not sure why this can happen
                    S[(Bc * tx) + y] = 0.0;
                }
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) +
                              (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] =
                    (1 / row_l_new) *
                    ((row_l_prev * __expf(row_m_prev - row_m_new) *
                      O[qkv_offset + (tile_size * i) + (tx * d) + x]) +
                     (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner
                          // loop
    }
}

torch::Tensor finefmha_0(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 32;
    const int Br = 32;

    const int B = Q.size(0);
    const int nh = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tc = ceil((float)N / Bc);
    const int Tr = ceil((float)N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);

    // Calculate SRAM size needed per block
    const int sram_size =
        (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock,
                           0);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);    // Bc threads per block

    finefmha_0_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Tc,
        Tr, Bc, Br, softmax_scale, l.data_ptr<float>(), m.data_ptr<float>(),
        O.data_ptr<float>(), causal);
    return O;
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(fmha_unified, CUDA, m) { m.impl("finefmha_0", &finefmha_0); }

}  // namespace fmha_unified