import torch
from torch import Tensor
import torch.nn.functional as F

__all__ = ["fine_fmha"]


def fine_fmha(q: Tensor, k: Tensor, v: Tensor, causal: bool, is_decode, kernel_index: int) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    # find out all the kernels
    local_vars = {}
    exec(f"""kernel = torch.ops.fmha_unified.finefmha_{kernel_index}""", globals(), local_vars)
    kernel = local_vars['kernel']

    # pad the q, k and v to be divisible by 64
    # size=(batch_size, kv_head, seq_len, head_embd)
    seq_length = q.size(2)
    decode_length = k.size(2)
    def pad_to_64(tensor: Tensor) -> Tensor:
        pad_size = (64 - tensor.size(2) % 64) % 64
        return F.pad(tensor, (0, 0, 0, pad_size))
    
    q = pad_to_64(q).contiguous()
    k = pad_to_64(k).contiguous()
    v = pad_to_64(v).contiguous()
    
    o = kernel.default(q, k, v, causal, is_decode, decode_length) # type: ignore
    
    return o[:, :, :seq_length, :]
