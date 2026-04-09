import os
import json
import sys
import argparse
import torch
import gc
import shutil
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def stream_and_decompress(input_fp8_path, output_bf16_path):
    model_id = input_fp8_path
    output_dir = output_bf16_path
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Config Patching
    print("Patching configuration files...")
    for filename in os.listdir(model_id):
        filepath = os.path.join(model_id, filename)
        if not os.path.isfile(filepath):
            continue
            
        if filename == "config.json":
            with open(filepath, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            if "quantization_config" in config_data:
                del config_data["quantization_config"]
            
            config_data["torch_dtype"] = "bfloat16"
            if "dtype" in config_data:
                config_data["dtype"] = "bfloat16"
                
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)
            print(" > Successfully patched config.json")
            
        elif filename.endswith(".json") and filename != "model.safetensors.index.json":
            shutil.copy(filepath, output_dir)
            print(f" > Copied {filename}")

    # 2. Load Model
    SHARD_SIZE_BYTES = 5 * 1024 * 1024 * 1024 

    print("\nLoading INT4 model into RAM (~500GB). This will take over an hour...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # ---------------------------------------------------------
    # 3. PRE-FLIGHT DIAGNOSTIC CHECK
    # ---------------------------------------------------------
    print("\n--- RUNNING ARCHITECTURE DIAGNOSTIC ---")
    unique_compressed_types = set()
    
    for name, module in model.named_modules():
        if hasattr(module, "weight_packed"):
            unique_compressed_types.add(str(type(module)))
            
    print("Compressed module types found:")
    for t in unique_compressed_types:
        print(f" - {t}")
        
    # If any compressed module is NOT a standard Linear layer, we abort.
    if not all("Linear" in t for t in unique_compressed_types):
        print("\n[!] CRITICAL WARNING: Non-Linear compressed modules detected!")
        print("[!] The Identity Matrix trick may produce mathematically mangled weights.")
        print("[!] Aborting script to protect your conversion pipeline.")
        sys.exit(1)
        
    print("Diagnostic Passed: All compressed layers are standard Linear modules. Proceeding...\n")
    # ---------------------------------------------------------

    # 4. Start the heavy lifting
    print("Initializing streaming sharder...")
    index_dict = {"metadata": {"total_size": 0}, "weight_map": {}}
    
    current_shard = {}
    current_shard_size = 0
    shard_count = 1
    total_size = 0
    identities = {}
    
    print("Retrieving state dict references...")
    sd = model.state_dict()
    
    for key in tqdm(list(sd.keys()), desc="Processing & Sharding Tensors"):
        if key.endswith(".weight_scale") or key.endswith(".weight_shape"):
            continue

        if key.endswith(".weight_packed"):
            module_name = key.replace(".weight_packed", "")
            module = model.get_submodule(module_name)
            in_features = module.in_features
            
            if in_features not in identities:
                identities[in_features] = torch.eye(in_features, dtype=torch.bfloat16, device="cpu")
            
            with torch.no_grad():
                out = module(identities[in_features])
            
            if hasattr(module, "bias") and module.bias is not None:
                out = out - module.bias
                
            # .contiguous() added to prevent safetensors ValueError
            tensor = out.T.contiguous().clone()
            new_key = f"{module_name}.weight"
            del out
        else:
            # .contiguous() added here as well
            tensor = sd[key].contiguous().clone()
            new_key = key

        current_shard[new_key] = tensor
        tensor_size_bytes = tensor.numel() * 2
        
        current_shard_size += tensor_size_bytes
        total_size += tensor_size_bytes
        
        shard_filename = f"model-{shard_count:05d}.safetensors"
        index_dict["weight_map"][new_key] = shard_filename
        
        if current_shard_size >= SHARD_SIZE_BYTES:
            save_file(current_shard, os.path.join(output_dir, shard_filename))
            shard_count += 1
            
            del current_shard
            current_shard = {}
            current_shard_size = 0
            gc.collect()

    if len(current_shard) > 0:
        shard_filename = f"model-{shard_count:05d}.safetensors"
        save_file(current_shard, os.path.join(output_dir, shard_filename))
        
    index_dict["metadata"]["total_size"] = total_size
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index_dict, f, indent=2)

    print("\nSaving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print(f"\nSuccess! Streaming conversion complete. BF16 model ready at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FP8 model to BF16.")
    parser.add_argument("--input-fp8-hf-path", type=str, required=True, help="Path to the input FP8 HuggingFace model.")
    parser.add_argument("--output-bf16-hf-path", type=str, required=True, help="Path to save the output BF16 HuggingFace model.")
    args = parser.parse_args()

    stream_and_decompress(args.input_fp8_hf_path, args.output_bf16_hf_path)