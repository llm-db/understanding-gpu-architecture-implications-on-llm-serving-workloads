"""
Run Llama2 with huggingface

Reference:
https://github.com/microsoft/DeepSpeedExamples/blob/master/inference/huggingface/zero_inference/run_model.py
"""

import argparse
import gc
import os
import time

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import sys
sys.path.append("..")
import utils


def get_hf_model(
    model_name,
    pin_memory,
    quant_bits,
    quant_group_size,
    cache_dir,
):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, attn_implementation="flash_attention_2",)
    
    pin_memory = bool(args.pin_memory)
    dtype = torch.float16

    if quant_bits == 4:
        raise NotImplementedError()

    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=dtype, config=config)
    model.to(torch.cuda.current_device())
    model = model.eval()

    return model


def inference(config, tokenizer, model, batch_size, trials, prompt_len, gen_len):
    prompts = ["Paris is the capital city of"] * batch_size
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt",
        padding="max_length", max_length=prompt_len)
    input_tokens.to(torch.cuda.current_device())

    # Run generation
    print(f"benchmark, prompt_len = {prompt_len}, gen_len = {gen_len}, input_ids.shape = {input_tokens.input_ids.shape}")

    total_timings = []
    for _ in range(trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        with torch.no_grad():
            output_ids = model.generate(**input_tokens, max_new_tokens=gen_len, do_sample=False)

        end_event.record()
        torch.cuda.synchronize()
        total_timings.append(start_event.elapsed_time(end_event) / 1000.0)

    # Check lengths
    input_lens = [len(x) for x in input_tokens.input_ids]
    output_lens = [len(x) for x in output_ids]
    assert all(x == prompt_len for x in input_lens)
    assert all(x == prompt_len + gen_len for x in output_lens)

    # Log output
    print(f"Summary:")
    print(f"total_timings = {total_timings}")
    total_latency = total_timings[-1]
    prefill_latency = 0.0
    prefill_throughput = 0.0
    decode_latency = 0.0
    decode_throughput = 0.0
    num_generated_tokens = batch_size * gen_len
    total_throughput = num_generated_tokens / total_latency
    gpu_peak_mem = torch.cuda.max_memory_allocated(torch.device("cuda"))

    model_size = utils.model_bytes(config)
    cache_size = utils.cache_bytes(config, batch_size, prompt_len + gen_len)
    log_str = utils.write_gen_benchmark_log(
        model_size,
        cache_size,
        gpu_peak_mem,
        prefill_latency,
        prefill_throughput,
        decode_latency,
        decode_throughput,
        total_latency,
        total_throughput,
    )
    print(log_str)

def run_generation(
    model_name,
    trials,
    batch_size,
    prompt_len,
    gen_len,
    local_rank,
    pin_memory,
    quant_bits,
    quant_group_size,
    cache_dir
):
    # Load tokenizer
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    print("load model")
    with torch.no_grad():
        model = get_hf_model(
            model_name,
            pin_memory,
            quant_bits,
            quant_group_size,
            cache_dir,
        )

    model.share_memory()

    processes = []
    for _ in range(2):
        p = mp.Process(target=inference, args=(config, tokenizer, model, batch_size, trials, prompt_len, gen_len))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    if local_rank != 0:
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Llama-2-7b-hf", help="model name or path")
    parser.add_argument("--trials", type=int, default=3,  help="Number of token generation iterations")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=512,  help="prompt length")
    parser.add_argument("--gen_len", type=int, default=32,  help="number of tokens to generate")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank for distributed inference")
    parser.add_argument("--pin_memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
    parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
    parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
    parser.add_argument("--cache_dir", type=str, default=os.getenv("HF_HOME", "/scratch/yonghe"), help="cache dir for model name")
    args = parser.parse_args()

    gc.collect()

    run_generation(
        args.model_name,
        args.trials,
        args.batch_size,
        args.prompt_len,
        args.gen_len,
        args.local_rank,
        args.pin_memory,
        args.quant_bits,
        args.quant_group_size,
        args.cache_dir,
    )

