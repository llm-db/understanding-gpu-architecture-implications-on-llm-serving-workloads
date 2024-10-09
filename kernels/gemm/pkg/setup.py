# Copyright (c) 2023, Tri Dao.

import sys
import warnings
import os
import shutil
import glob
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess


import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
    ROCM_HOME,
    IS_HIP_EXTENSION,
)


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_ROCM = True
    else:
        IS_ROCM = False
else:
    if BUILD_TARGET == "cuda":
        IS_ROCM = False
    elif BUILD_TARGET == "rocm":
        IS_ROCM = True


# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f"linux_{platform.uname().machine}"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def check_if_rocm_home_none(global_option: str) -> None:
    if ROCM_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so hipcc won't be necessary
    # in that case.
    warnings.warn(f"{global_option} was requested, but hipcc was not found.")


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]


def rename_cpp_to_cu(cpp_files):
    new_cu_files = []
    for entry in cpp_files:
        # if the file is a .cpp file, rename it to .cu
        if os.path.splitext(entry)[1] == ".cpp":
            shutil.copy(entry, os.path.splitext(entry)[0] + ".cu")
            new_cu_files.append(os.path.splitext(entry)[0] + ".cu")
        else:
            new_cu_files.append(entry)
    return new_cu_files


def validate_and_update_archs(archs):
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a", "gfx940", "gfx941", "gfx942"]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported by Flash-Attention"


ext_modules = []

if not SKIP_CUDA_BUILD and not IS_ROCM:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
    # See https://github.com/pytorch/pytorch/pull/70650
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(
        os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")
    ):
        generator_flag = ["-DOLD_GENERATOR_PATH"]

    check_if_cuda_home_none("flash_attn")
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.6"):
            raise RuntimeError(
                "FlashAttention is only supported on CUDA 11.6 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )
    # cc_flag.append("-gencode")
    # cc_flag.append("arch=compute_75,code=sm_75")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if CUDA_HOME is not None:
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    ext_modules.append(
        CUDAExtension(
            name="gemm_unified._C",
            sources=[
                "gemm_unified/csrc/gemm_kernel.cpp",  # TODO: Add source files here
            ]
            + glob.glob("gemm_unified/csrc/auto/*.cu")
            + glob.glob("gemm_unified/csrc/semi/*_nv.cu")
            + glob.glob("gemm_unified/csrc/manual/*_nv.cu"),
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                        # "-G",
                        # "-g",
                        # "-lineinfo",
                        # "--ptxas-options=-v",
                        # "--ptxas-options=-O2",
                        # "-lineinfo",
                        # "-DFLASHATTENTION_DISABLE_BACKWARD",
                        # "-DFLASHATTENTION_DISABLE_DROPOUT",
                        # "-DFLASHATTENTION_DISABLE_ALIBI",
                        # "-DFLASHATTENTION_DISABLE_SOFTCAP",
                        # "-DFLASHATTENTION_DISABLE_UNEVEN_K",
                        # "-DFLASHATTENTION_DISABLE_LOCAL",
                    ]
                    + generator_flag
                    + cc_flag
                ),
            },
            include_dirs=[
                os.path.abspath("../../deps/cutlass/include"),
            ],
        )
    )
elif not SKIP_CUDA_BUILD and IS_ROCM:

    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
    # See https://github.com/pytorch/pytorch/pull/70650
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(
        os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")
    ):
        generator_flag = ["-DOLD_GENERATOR_PATH"]

    check_if_rocm_home_none("flash_attn")
    cc_flag = []

    archs = os.getenv("GPU_ARCHS", "gfx90a").split(";")
    validate_and_update_archs(archs)

    cc_flag = [f"--offload-arch={arch}" for arch in archs]

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    # generator_flag += ["--offload-arch=gfx90a"]
    ext_modules.append(
        CUDAExtension(
            name="gemm_unified._C",
            sources=rename_cpp_to_cu(
                [
                    "gemm_unified/csrc/gemm_kernel.cpp",  # TODO: Add source files here
                ]
                + glob.glob("gemm_unified/csrc/auto/*.cu")
                + glob.glob("gemm_unified/csrc/semi/*_amd.cpp")
                + glob.glob("gemm_unified/csrc/manual/*_amd.cpp"),
            ),
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    # "-mllvm",
                    # "-enable-post-misched=0",
                    # "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
                    # "-fgpu-flush-denormals-to-zero",
                    # "-DCK_ENABLE_BF16",
                    # "-DCK_ENABLE_BF8",
                    # "-DCK_ENABLE_FP16",
                    # "-DCK_ENABLE_FP32",
                    # "-DCK_ENABLE_FP64",
                    # "-DCK_ENABLE_FP8",
                    # "-DCK_ENABLE_INT8",
                    # "-DCK_USE_XDL","
                    "-DUSE_PROF_API=1",
                    "-D__HIP_PLATFORM_AMD__=1",
                    "-D__HIP_HCC_COMPAT_MODE__=1",
                    "-D__HIP_PLATFORM_HCC__=1",
                    "--driver-mode=g++",
                    "-Xclang",
                    "-fallow-half-arguments-and-returns",
                    "-Wno-format-nonliteral",
                    "-fclang-abi-compat=17",
                    "-DNDEBUG",
                    # "-DFLASHATTENTION_DISABLE_BACKWARD",
                ]
                + generator_flag
                + cc_flag,
            },
            include_dirs=[
                os.path.abspath("../../deps/rocWMMA/library/include"),    
            ],
        )
    )


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (
                1024**3
            )  # free memory in GB
            max_num_jobs_memory = int(
                free_memory_gb / 9
            )  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


class CustomInstallCommand(install):
    """Customized setuptools install command - deletes egg-info, build, and dist directories after install."""

    def run(self):
        install.run(self)
        self.cleanup()

    def cleanup(self):
        directories = ["build", "dist"]
        egg_info_dir = None

        # Find the egg-info directory
        for item in os.listdir("."):
            if item.endswith(".egg-info"):
                egg_info_dir = item
                break

        if egg_info_dir:
            directories.append(egg_info_dir)

        for directory in directories:
            if os.path.exists(directory):
                print(f"Removing directory: {directory}")
                shutil.rmtree(directory)


setup(
    name="gemm_unified",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension, "install": CustomInstallCommand},
    python_requires=">=3.8",
    install_requires=[
        "torch",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
)
