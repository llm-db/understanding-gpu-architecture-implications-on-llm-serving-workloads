#include <torch/extension.h>

namespace gemm_unified {

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(gemm_unified, m) {
    m.def("finegemm_0(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_1(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_2(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_3(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_4(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_5(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_6(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_7(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_8(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_9(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_10(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_11(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_12(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_13(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_14(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_15(Tensor a, Tensor b) -> Tensor");
    m.def("finegemm_16(Tensor a, Tensor b) -> Tensor");
}

}  // namespace gemm_unified