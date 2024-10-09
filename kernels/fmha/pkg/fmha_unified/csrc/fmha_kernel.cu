#include <torch/extension.h>

namespace fmha_unified {

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(fmha_unified, m) {
  m.def("finefmha_0(Tensor q, Tensor k, Tensor v, bool causal, bool is_decode, int decode_length) -> Tensor");
  m.def("finefmha_1(Tensor q, Tensor k, Tensor v, bool causal, bool is_decode, int decode_length) -> Tensor");
  m.def("finefmha_2(Tensor q, Tensor k, Tensor v, bool causal, bool is_decode, int decode_length) -> Tensor");
  m.def("finefmha_3(Tensor q, Tensor k, Tensor v, bool causal, bool is_decode, int decode_length) -> Tensor");
  m.def("finefmha_4(Tensor q, Tensor k, Tensor v, bool causal, bool is_decode, int decode_length) -> Tensor");
  m.def("finefmha_5(Tensor q, Tensor k, Tensor v, bool causal, bool is_decode, int decode_length) -> Tensor");
  m.def("finefmha_6(Tensor q, Tensor k, Tensor v, bool causal, bool is_decode, int decode_length) -> Tensor");
  m.def("finefmha_7(Tensor q, Tensor k, Tensor v, bool causal, bool is_decode, int decode_length) -> Tensor");
}

}  // namespace gemm_unified