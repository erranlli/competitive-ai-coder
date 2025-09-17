import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- Configuration ---
base_model_path = "Qwen/Qwen2-7B-Instruct"
adapter_path = "/root/rllm/trav_test/qwen2.5-7b-mot-lora-run7"
merged_model_path = "/root/rllm/trav_test/qwen2.5-7b-mot-merged"
# ---------------------

print(f"Loading base model from: {base_model_path}")
base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
                torch_dtype=torch.bfloat16,
                    device_map="cpu",  # Load on CPU to avoid using GPU memory
                        trust_remote_code=True,
                        )

print(f"Loading tokenizer from: {base_model_path}")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

print(f"Loading LoRA adapter from: {adapter_path}")
# Load the PEFT model
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging adapter weights into the base model...")
# Merge the weights and unload the adapter
model = model.merge_and_unload()
print("Merge complete.")

print(f"Saving merged model and tokenizer to: {merged_model_path}")
os.makedirs(merged_model_path, exist_ok=True)
model.save_pretrained(merged_model_path, safe_serialization=True)
tokenizer.save_pretrained(merged_model_path)

print("Successfully merged and saved the model.")
