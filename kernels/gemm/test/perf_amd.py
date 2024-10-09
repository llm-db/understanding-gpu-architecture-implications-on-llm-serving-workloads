import torch
import gemm_unified
from torch.profiler import profile, record_function, ProfilerActivity
from tabulate import tabulate


def profile_finegemm_kernel(M, K, N, kernel_index, dtype=torch.half):
    a = torch.randn(M, K, dtype=dtype).cuda()
    b = torch.randn(K, N, dtype=dtype).cuda()

    for i in range(2):
        if kernel_index == -1:
            c = a @ b
        else:
            c = gemm_unified.ops.fine_gemm(a, b, kernel_index)
    
    for i in range(50):
        if kernel_index == -1:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            ) as prof:
                with record_function("gemm_cublas"):
                    c = a @ b
        else:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            )as prof:
                with record_function(f"gemm_finegemm_{kernel_index}"):
                    c = gemm_unified.ops.fine_gemm(a, b, kernel_index)

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
    # truncate the Key to 10 characters
    table_data["Key"] = table_data["Key"].str[:20]
    # add a new colume to calculate the TFLOPS
    table_data["TFLOPS"] = (2 * M * K * N / (table_data["CUDA Time"] * 1e-6)) * 1e-12
    # order the table by CUDA Time in descending order
    table_data = table_data.sort_values(by="CUDA Time", ascending=False)
    
    # Define table headers
    headers = ["Key", "CUDA Time", "TFLOPS"]
    # Pretty print the table
    print(tabulate(table_data[:5], headers=headers, tablefmt="pretty"))
    # return the tflops of the third row
    return table_data.iloc[0]["TFLOPS"]
    


if __name__ == "__main__":
    # build a table for all the kernels, each row is a kernel (0 to 12 excluding 6), each column is a matrix size (2^8 to 2^13)
    import pandas as pd
    df = pd.DataFrame(columns=[f"2^{i}" for i in range(8, 14)], index=[f"Kernel {i}" for i in range(-1, 17)])
    # delete the row for kernel 6
    df = df.drop("Kernel 6")
    df = df.drop("Kernel 9")
    df = df.drop("Kernel 10")
    for i in range(-1, 17):
        if i == 6 or i == 9 or i == 10:
            continue
        else:
            for j in range(8, 14):
                tflops = profile_finegemm_kernel(2**j, 2**j, 2**j, i)
                df.loc[f"Kernel {i}", f"2^{j}"] = tflops
    print(df)
    # save the table to a csv file
    df.to_csv("finegemm_perf.csv")
    
    print("DONE")
    
    # for i in range(8, 14):
    #     # print kernel size 2^i
    #     print(f"Kernel size 2^{i} = {2**i}")
    #     tflops = profile_finegemm_kernel(2**i, 2**i, 2**i, 11)
    #     print(f"TFLOPS = {tflops}")
    
    # for i in range(8, 14):
    #     # print kernel size 2^i
    #     print(f"Kernel size 2^{i} = {2**i}")
    #     tflops = profile_finegemm_kernel(2**i, 2**i, 2**i, 12)
    #     print(f"TFLOPS = {tflops}")
        
    # profile_finegemm_kernel(4096, 4096, 4096, 0)
    # profile_finegemm_kernel(4096, 4096, 4096, 1)
    # profile_finegemm_kernel(4096, 4096, 4096, 2)
    # profile_finegemm_kernel(4096, 4096, 4096, 3)
    # profile_finegemm_kernel(4096, 4096, 4096, 4)
    # profile_finegemm_kernel(4096, 4096, 4096, 5)
    # # profile_finegemm_kernel(4096, 4096, 4096, 6)
    # profile_finegemm_kernel(4096, 4096, 4096, 7)
    # profile_finegemm_kernel(4096, 4096, 4096, 8)
    # # # # profile_finegemm_kernel(4096, 4096, 4096, 9)
    # # profile_finegemm_kernel(4096, 4096, 4096, 10)
    # profile_finegemm_kernel(4096, 4096, 4096, 11)
    # profile_finegemm_kernel(4096, 4096, 4096, 12)
    # profile_finegemm_kernel(4096, 4096, 4096, 13)
    # profile_finegemm_kernel(4096, 4096, 4096, 14)
    # profile_finegemm_kernel(4096, 4096, 4096, 15)
    # profile_finegemm_kernel(4096, 4096, 4096, 16)
    # # unfair for sure, just to avoid the use of tensor cores
    # profile_finegemm_kernel(4096, 4096, 4096, -1)
