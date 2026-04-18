#!/usr/bin/env python3
"""
Simple FP8 dynamic quantization for vLLM using llmcompressor.
Falls back to manual compressed-tensors FP8 if llmcompressor is incompatible.

Usage:
    python3 quantize-fp8.py google/gemma-4-26B-A4B-it ./gemma-4-26B-A4B-it-FP8
"""
import sys
import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from safetensors.torch import save_file

def quantize_fp8_dynamic(model_id: str, output_dir: str):
    print(f"Loading {model_id} in BF16...")

    # Try processor first (for multimodal), fall back to tokenizer
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception:
        processor = None
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Model loaded. Quantizing to FP8...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect quantized state dict
    state_dict = {}
    quantization_config = {
        "quant_method": "compressed-tensors",
        "quantization_config": {
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "tensor",
                        "dynamic": True,
                        "symmetric": True,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "tensor",
                        "dynamic": True,
                        "symmetric": True,
                    }
                }
            },
            "format": "float-quantized",
            "ignore": ["lm_head"],
            "quant_method": "compressed-tensors",
        }
    }

    total_params = 0
    quantized_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        # Quantize Linear weight tensors (skip lm_head, embeddings, norms)
        if "weight" in name and param.ndim == 2 and "lm_head" not in name and "embed" not in name and "norm" not in name:
            # Convert to FP8 E4M3
            scale = param.abs().max() / torch.finfo(torch.float8_e4m3fn).max
            fp8_param = (param / scale).to(torch.float8_e4m3fn)
            state_dict[name] = fp8_param
            state_dict[name.replace(".weight", ".weight_scale")] = scale.float()
            quantized_params += param.numel()
        else:
            state_dict[name] = param

    print(f"Quantized {quantized_params:,} / {total_params:,} params ({100*quantized_params/total_params:.1f}%)")

    # Save in shards
    print(f"Saving to {output_dir}...")

    # Split into shards of ~5GB
    shard_size = 5 * 1024 * 1024 * 1024  # 5 GiB
    current_shard = {}
    current_size = 0
    shard_idx = 1
    weight_map = {}
    shard_files = []

    for name, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > shard_size and current_shard:
            shard_name = f"model-{shard_idx:05d}-of-TOTAL.safetensors"
            save_file(current_shard, output_path / shard_name)
            shard_files.append(shard_name)
            print(f"  Saved shard {shard_idx} ({current_size / 1e9:.1f} GiB)")
            shard_idx += 1
            current_shard = {}
            current_size = 0

        current_shard[name] = tensor.cpu()
        weight_map[name] = f"model-{shard_idx:05d}-of-TOTAL.safetensors"
        current_size += tensor_size

    # Save last shard
    if current_shard:
        shard_name = f"model-{shard_idx:05d}-of-TOTAL.safetensors"
        save_file(current_shard, output_path / shard_name)
        shard_files.append(shard_name)
        print(f"  Saved shard {shard_idx} ({current_size / 1e9:.1f} GiB)")

    # Fix shard names with correct total
    total_shards = len(shard_files)
    for i, old_name in enumerate(shard_files):
        new_name = f"model-{i+1:05d}-of-{total_shards:05d}.safetensors"
        (output_path / old_name).rename(output_path / new_name)
        # Update weight map
        for k, v in weight_map.items():
            if v == old_name:
                weight_map[k] = new_name

    # Save index
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state_dict.values())},
        "weight_map": weight_map,
    }
    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Copy config and tokenizer files
    model.config.quantization_config = quantization_config
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if processor:
        processor.save_pretrained(output_dir)

    print(f"Done! FP8 model saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <model_id> <output_dir>")
        sys.exit(1)

    quantize_fp8_dynamic(sys.argv[1], sys.argv[2])
