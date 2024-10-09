import torch
from torch import Tensor

__all__ = ["fine_gemm"]


def fine_gemm(a: Tensor, b: Tensor, kernel_index: int) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    # find out all the kernels
    local_vars = {}
    exec(f"""kernel = torch.ops.gemm_unified.finegemm_{kernel_index}""", globals(), local_vars)
    kernel = local_vars['kernel']
    return kernel.default(a, b) # type: ignore
