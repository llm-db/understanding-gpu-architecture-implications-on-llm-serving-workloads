# Adapted from https://github.com/tspeterkim/flash-attention-minimal/tree/main

import torch
from torch.nn import functional as F

# Load the CUDA kernel as a python module
import fmha_unified

def torch_attn(q, k, v, is_causal):
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    if q.size(1) != k.size(1):
        k = repeat_kv(k, 4)
        v = repeat_kv(v, 4)
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def correct(kernel_index, GQA=False, is_causal=False, is_decode=False, dtype=torch.bfloat16):
    # Use small model params, otherwise slower than manual attention. See caveats in README.
    batch_size = 1
    
    if GQA:
        n_head = 4
        kv_head = int(n_head / 4)
    else:
        n_head = 1
        kv_head = n_head
    
    seq_len = 128
    head_embd = 128
    # fix the random seed to 42
    torch.manual_seed(42)
    if is_decode:
        q_len = 1
    else:
        q_len = seq_len
    
    q = torch.randn(size=(batch_size, n_head, q_len, head_embd), dtype=dtype).cuda() * 0.1
    k = torch.randn(size=(batch_size, kv_head, seq_len, head_embd), dtype=dtype).cuda() * 0.1
    v = torch.randn(size=(batch_size, kv_head, seq_len, head_embd), dtype=dtype).cuda() * 0.1

    # o_correct = torch_attn(q, k, v, is_causal)
    # matmul of q and k in the last two dimensions
    o_correct = torch.einsum('bhqd,bhkd->bhqk', q, k)

    o = fmha_unified.ops.fine_fmha(q, k, v, is_causal, is_decode, kernel_index)
    if not torch.allclose(o, o_correct, rtol=0, atol=5e-02):
        # q_slice = q[0, 0, :64, :]
        # k_slice = k[0, 0, :64, :]
        # p (q_slice @ k_slice.T)[:8, :8]
        not_close_mask = ~torch.isclose(o, o_correct, rtol=0, atol=5e-02)
        not_close_indices = torch.nonzero(not_close_mask)
        print(f"Indices of elements that are not close: {not_close_indices}")
        breakpoint()
    

# correct(0, GQA=True, is_causal=True, is_decode=False, dtype=torch.half)
# correct(1, GQA=True, is_causal=True, is_decode=False, dtype=torch.half)
# correct(2, GQA=True, is_causal=True, is_decode=False, dtype=torch.half)
# correct(3, GQA=False, is_causal=True, is_decode=False, dtype=torch.half)
correct(5, GQA=False, is_causal=True, is_decode=False, dtype=torch.half)
# correct(4, GQA=True, is_causal=True, is_decode=False, dtype=torch.half)
# correct(5, GQA=True, is_causal=False, is_decode=False, dtype=torch.bfloat16)
# correct(5, GQA=False, is_causal=True, is_decode=False, dtype=torch.bfloat16)

# q_slice = q[0,0,:,:]
# k_slice = k[0,0,:,:]
# import math
# s_p = 1 / math.sqrt(128) * q_slice @ (k_slice.T)
# s_max = torch.max(s_p, dim=1).values
# # s_p minus s_max by broadcasting
# s_p = s_p - s_max[:, None]
# s_p = torch.exp(s_p)
# s_p_sum = torch.sum(s_p, dim=1)
# l_final = 1 / s_p_sum


# v_slice = v[0,0,:,:]
# o = (s_p @ v_slice) 

# # o = diag(l_final) @ o
# o = o * l_final[:, None]