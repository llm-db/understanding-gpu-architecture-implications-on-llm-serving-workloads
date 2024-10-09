"""
Run Llama3 with huggingface

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

import sys

sys.path.append("..")
import utils

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No AMD GPU device found or ROCm is not installed.")


def get_hf_model(
    model_name,
    pin_memory,
    quant_bits,
    quant_group_size,
    cache_dir,
):
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        attn_implementation="eager",
    )
    pin_memory = bool(args.pin_memory)
    dtype = torch.float16

    if quant_bits == 4:
        raise NotImplementedError()

    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir, torch_dtype=dtype, config=config
    )
    model.to(torch.cuda.current_device())
    model = model.eval()

    return model


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
    cache_dir,
):
    # Load tokenizer
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", cache_dir=cache_dir
    )
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

    utils.add_model_hooks(model)

    prompts = (
        [
            """Writing a 10,000-word story here would be impractical due to the length. However, I can certainly start a story and outline the rest for you. Here's the beginning of a tale along with a detailed outline:

### Title: The Chronicles of Eldoria

---

### Chapter 1: The Awakening

In the heart of the ancient forest of Eldoria, where the trees whispered secrets of old, a young girl named Aria stirred from her slumber. She was not an ordinary girl; she was the last of the Enchanters, a lineage believed to be extinct for centuries. Aria's silver hair glistened in the morning sun, and her emerald eyes sparkled with a mix of curiosity and determination.

Aria had always felt different, her connection to nature stronger than anyone else in her village. The day she turned sixteen, she discovered her true heritage. Guided by an old journal left by her mother, Aria learned of her destiny to protect Eldoria from an impending darkness.

Her journey began with a quest to find the Four Elemental Stones, each hidden in different corners of Eldoria. These stones were the key to awakening the ancient guardians who could help her combat the looming threat.

### Chapter 2: The First Stone - Terra

Aria's journey led her to the mountains of Terra, home to the Earth Stone. The path was treacherous, with rocky cliffs and narrow passages. Along the way, she encountered a group of travelers who warned her of the stone's guardian, a fierce dragon named Thalor.

With the help of her new companions, Aria devised a plan to outsmart Thalor. They discovered the dragon's weakness: a rare herb found only in the deepest caves of Terra. After a daring adventure into the caves, Aria obtained the herb and used it to lull Thalor into a deep sleep, allowing her to claim the Earth Stone.

### Chapter 3: The Second Stone - Aqua

The next leg of Aria's journey took her to the mystical Lake Serene, where the Water Stone lay hidden beneath its depths. The lake was guarded by the Nymph Queen, Seraphina, who challenged Aria to prove her worthiness.

Seraphina set three trials for Aria, each testing her courage, wisdom, and compassion. Through sheer determination and the guidance of her mother's journal, Aria passed the trials and earned the trust of Seraphina, who bestowed upon her the Water Stone.

### Chapter 4: The Third Stone - Ignis

Aria's quest then brought her to the fiery realm of Ignis, a land of volcanoes and molten lava. The Fire Stone was kept within the Temple of Flames, protected by the Phoenix, an immortal bird of fire.

Aria's journey through Ignis was fraught with danger, but she found an ally in a young blacksmith named Kael. Together, they navigated the perilous landscape and devised a way to communicate with the Phoenix. By showing respect and understanding, Aria convinced the Phoenix to grant her the Fire Stone.

### Chapter 5: The Fourth Stone - Aether

The final stone, Aether, was located in the ethereal Sky Realm, a place of floating islands and endless skies. To reach it, Aria had to harness the power of the other three stones and unlock her full potential as an Enchanter.

In the Sky Realm, Aria faced her greatest challenge yet: a sorcerer named Malakar, who sought to use the Elemental Stones for his own dark purposes. A fierce battle ensued, with Aria drawing upon all her strength and the bonds she had formed along her journey.

### Chapter 6: The Guardians Awaken

With the Four Elemental Stones in her possession, Aria returned to Eldoria and performed the ancient ritual to awaken the guardians. These majestic beings, representing Earth, Water, Fire, and Air, rose to her aid as the darkness began to spread.

Aria and the guardians confronted Malakar in an epic showdown. The battle raged across Eldoria, with each guardian showcasing their unique powers. In the end, it was Aria's unwavering spirit and her connection to the land that tipped the scales in their favor.

### Chapter 7: The Restoration of Eldoria

With Malakar defeated, peace returned to Eldoria. The land began to heal, and the guardians retreated to their respective realms, leaving Aria as the protector of Eldoria. She vowed to honor her lineage and ensure the safety of her homeland for generations to come.

Aria's journey had transformed her from a curious girl into a wise and powerful Enchanter. She knew that challenges would arise in the future, but with the support of her friends and the strength within her, she was ready to face them.

### Outline for Remaining Chapters:

- **Chapter 8: The Aftermath**: Aria rebuilds her village and helps her companions find their paths.
- **Chapter 9: A New Threat**: A new enemy emerges, threatening the fragile peace.
- **Chapter 10: Unity in Diversity**: Aria unites various races and beings of Eldoria to face the new threat.
- **Chapter 11: The Final Battle**: A climactic battle that tests Aria’s limits and the strength of her allies.
- **Chapter 12: The Legacy of the Enchanters**: Aria establishes a new order of Enchanters to protect Eldoria.

This detailed outline and the beginning of the story should give you a strong foundation to continue writing."""
            * 1000
        ]
        * batch_size
    )
    input_tokens = tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=prompt_len,
    )
    input_tokens.to(torch.cuda.current_device())

    # Run generation
    print(
        f"benchmark, prompt_len = {prompt_len}, gen_len = {gen_len}, input_ids.shape = {input_tokens.input_ids.shape}"
    )

    prefill_timings = []
    total_timings = []
    for _ in range(trials):
        start = time.time()
        with torch.no_grad():
            model.stage = "prefill"
            output_ids = model.generate(
                **input_tokens, max_new_tokens=gen_len, do_sample=False
            )
            prefill_timings.append(model.__duration__)
        end = time.time()
        total_timings.append(end - start)

    if local_rank != 0:
        return

    utils.remove_model_hooks(model)
    # Check lengths
    input_lens = [len(x) for x in input_tokens.input_ids]
    output_lens = [len(x) for x in output_ids]
    assert all(x == prompt_len for x in input_lens)
    assert all(x == prompt_len + gen_len for x in output_lens)

    # Log output
    print(f"Summary:")
    print(f"total_timings = {total_timings}")
    print(f"prefill_timings = {prefill_timings}")
    total_latency = total_timings[-1]
    prefill_latency = prefill_timings[-1]

    prefill_throughput = batch_size * prompt_len / prefill_latency
    decode_latency = total_latency - prefill_latency
    decode_throughput = batch_size * (gen_len - 1) / max(decode_latency, 1e-10)
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

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    show_str = "Outputs:\n" + 30 * "-" + "\n"
    for i in [0, (len(outputs) - 1) // 2, len(outputs) - 1]:
        show_str += f"{i}: {outputs[i]}\n"
        show_str += 30 * "-" + "\n"
    # print(show_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="model name or path",
    )
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of token generation iterations"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=8192, help="prompt length")
    parser.add_argument(
        "--gen_len", type=int, default=320, help="number of tokens to generate"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.getenv("LOCAL_RANK", "0")),
        help="local rank for distributed inference",
    )
    parser.add_argument(
        "--pin_memory",
        type=int,
        default=0,
        help="whether to pinned CPU memory for ZeRO offloading",
    )
    parser.add_argument(
        "--quant_bits",
        type=int,
        default=16,
        help="model weight quantization bits; either 4 or 8",
    )
    parser.add_argument(
        "--quant_group_size",
        type=int,
        default=64,
        help="model weight quantization group size",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.getenv("HF_HOME", "/scratch/yonghe"),
        help="cache dir for model name",
    )
    args = parser.parse_args()

    gc.collect()
    while True:
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
