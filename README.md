This repository contains the code for Zijian Zhang's master's thesis titled <em>Understanding GPU Architecture Implications on LLM Serving Workloads</em>.
# Getting Started
```sh
git clone https://github.com/llm-db/understanding-gpu-architecture-implications-on-llm-serving-workloads.git
git submodule init
git submodule update
```

## Nvidia GPU
```sh
conda create -n finekernel_cu python=3.10
conda activate finekernel_cu
pip install -r requirements_cu.txt
```

To build the gemm kernels:
```sh
cd kernels/gemm/pkg
python setup.py install
cd ../test
python perf.py
```
To build the fmha kernels:
```sh
pip install flash-attn --no-build-isolation
cd kernels/fmha/pkg
python setup.py install
cd ../test
python perf.py
```

## AMD GPU
```sh
conda create -n finekernel_rocm python=3.10
conda activate finekernel_rocm
pip install -r requirements_rocm.txt
```
To build the gemm kernels:
```sh
cd kernels/gemm/pkg
python setup.py install
cd ../test
python perf_amd.py
```
To build the fmha kernels:

```sh
# flash attention for ROCm has to be built from source
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
# be prepared, this is a very very long long build
MAX_JOBS=32 python setup.py install
```

```sh
cd kernels/fmha/pkg
python setup.py install
cd ../test
python perf_amd.py
```

Do note that the previous script is does not build the 'register trick' version of Flash Attention on AMD ROCm. It's part of the composable kernel code base. And you can build that by the following strategy.

```sh
conda install anaconda::cmake
cp kernels/fmha/register_trick_amd/fmha_fwd.py kernels/deps/ck/example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py
cp kernels/fmha/register_trick_amd/cpp_symbol_map.py kernels/deps/ck/example/ck_tile/01_fmha/codegen/cpp_symbol_map.py
cd kernels/deps/ck
mkdir build && cd build
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx90a" ..
make tile_example_fmha_fwd -j32
# from here you can use all the sequence length you want
./bin/tile_example_fmha_fwd -b=1 -h=32 -h_k=8 -s=4096 -mask=t -kname=1
# and the output will be like
# [fp16|batch|bhsd] b:1, h:32/8, s:4096/4096, d:128/128, scale_s:0.0883883, bias:n, p_drop:0, lse:0, squant:0, mask:t(-1:0), v:r011000
# , fmha_fwd_d128_fp16_batch_shb_b128x128x32x128x32x128_r4x1x1_w32x32x16_qr_vr_psddv_mask, 3.955 ms, 69.50 TFlops, 33.94 GB/s, valid:y
# from here you can read out the flops statistics
```
