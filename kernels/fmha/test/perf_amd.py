import torch
from torch.nn import functional as F

# Load the CUDA kernel as a python module
import fmha_unified
from torch.profiler import profile, record_function, ProfilerActivity
from tabulate import tabulate
from flash_attn import flash_attn_func

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


def call_flash_attn(q, k, v, is_causal):
    # exchange the second and third dimensions
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    return flash_attn_func(q, k, v, causal=is_causal)

def profile_finefmha_kernel(B, H, N, d, GQA, is_causal, is_decode, kernel_index, dtype=torch.bfloat16):
   # Use small model params, otherwise slower than manual attention. See caveats in README.
    batch_size = B
    n_head = H
    if GQA:
        kv_head = int(n_head / 4)
    else:
        kv_head = n_head
    seq_len = N
    head_embd = d

    q = torch.randn(size=(batch_size, n_head, seq_len, head_embd), dtype=dtype).cuda()
    k = torch.randn(size=(batch_size, kv_head, seq_len, head_embd), dtype=dtype).cuda()
    v = torch.randn(size=(batch_size, kv_head, seq_len, head_embd), dtype=dtype).cuda()

    for i in range(20):
        if kernel_index == -1:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            ) as prof:
                with record_function("fmha_torch"):
                    o_correct = torch_attn(q, k, v, is_causal)
        elif kernel_index == -2:
             with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            ) as prof:
                with record_function("fmha_flash"):
                    o_correct = call_flash_attn(q, k, v, is_causal)
        else:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            )as prof:
                with record_function(f"fmha_finefmha_{kernel_index}"):
                   o = fmha_unified.ops.fine_fmha(q, k, v, is_causal, is_decode, kernel_index)

        # Extract profiling data
        key_averages = prof.key_averages()
        table_data = []
        for item in key_averages:
            table_data.append(
                [
                    item.key,
                    item.device_time,
                    # item.cpu_time,
                ]
            )

    import pandas as pd
    table_data = pd.DataFrame(table_data, columns=["Key", "CUDA Time"])
    # get the average of CUDA Time and CPU Time grouped by Key
    table_data = table_data.groupby("Key").mean().reset_index()
    # order the table by CUDA Time in descending order
    table_data = table_data.sort_values(by="CUDA Time", ascending=False)
    # truncate the text in the Key column to 20 characters
    table_data["Key"] = table_data["Key"].str[:25]
    
    table_data["TFLOPS"] = (4 * seq_len * seq_len * head_embd * n_head / (table_data["CUDA Time"] * 1e-6)) * 1e-12
    
    # Define table headers
    headers = ["Key", "CUDA Time", "TFLOPS"]

    # Pretty print the table
    print(tabulate(table_data[:3], headers=headers, tablefmt="pretty"))
    # return the tflops of the first row
    return table_data.iloc[0]["TFLOPS"]


if __name__ == "__main__":
    # create a table for the finefmha kernel, each row is a different kernel: 0, 2, 4, 5, -2, -1, each column is different input size: 2**8 to 2**13
    import pandas as pd
    df = pd.DataFrame(columns=["2^8", "2^9", "2^10", "2^11", "2^12", "2^13", "2^14"], index=["Kernel 0", "Kernel 2", "Kernel -2", "Kernel -1"])
    for i in [0, 2, -2, -1]:
        for j in range(8, 15):
            tflops = profile_finefmha_kernel(1, 32, 2**j, 128, True, True, False, i, dtype=torch.half)
            df.at[f"Kernel {i}", f"2^{j}"] = tflops
    print(df)
    # save the table to a csv file
    df.to_csv("finefmha_perf.csv")
    print("Done")
    
    
    profile_finefmha_kernel(1, 32, 2048, 128, True, True, False, 0, dtype=torch.half)
    # profile_finefmha_kernel(1, 32, 2048, 128, True, True, False, 1, dtype=torch.half)
    profile_finefmha_kernel(1, 32, 2048, 128, True, True, False, 2, dtype=torch.half)
    # profile_finefmha_kernel(1, 32, 2048, 128, True, True, False, 3, dtype=torch.half)
    # profile_finefmha_kernel(1, 32, 2048, 128, True, True, False, 4, dtype=torch.half)
    # profile_finefmha_kernel(1, 32, 2048, 128, True, True, False, 6, dtype=torch.half)
    # profile_finefmha_kernel(1, 32, 2048, 128, True, True, False, 5, dtype=torch.half)
    # profile_finefmha_kernel(1, 32, 2048, 128, True, True, False, 5, dtype=torch.half)
    profile_finefmha_kernel(1, 32, 2048, 128, True, True, False, -2, dtype=torch.half)
    profile_finefmha_kernel(1, 32, 2048, 128, True, True, False, -1, dtype=torch.half)
    