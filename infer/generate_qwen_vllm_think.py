import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig


def ensure_dir(path: str) -> None:
    """Ensures that a directory exists."""
    os.makedirs(path, exist_ok=True)


def normalize_text(text: str) -> str:
    """Normalizes text by removing trailing whitespace from lines."""
    if text is None:
        return ""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def extract_code_from_text(generated: str, language_hint: str = "python") -> str:
    """Extracts code from a fenced code block in markdown-formatted text."""
    import re

    if generated is None:
        return ""
    fence_pattern = re.compile(r"```(?:" + re.escape(language_hint) + r"|\w+)?\n([\s\S]*?)```", re.IGNORECASE)
    match = fence_pattern.search(generated)
    if match:
        return match.group(1).strip()
    any_fence = re.compile(r"```\n([\s\S]*?)```")
    match = any_fence.search(generated)
    if match:
        return match.group(1).strip()
    return generated.strip()


def sanitize_for_filename(text: str) -> str:
    """Sanitizes a string to be used as a valid filename."""
    import re

    t = (text or "").strip().lower()
    t = t.replace("/", "-")
    t = t.replace(" ", "_")
    t = re.sub(r"[^a-z0-9._-]", "", t)
    t = re.sub(r"[-_]{2,}", "-", t)
    return t or "model"


def resolve_outfile(model_name: str, dataset_name: str, subset: str, split: str) -> str:
    """Derives a standardized output filename."""
    model_tag = sanitize_for_filename(model_name)
    ds_tag = sanitize_for_filename(dataset_name)
    subset_tag = sanitize_for_filename(subset)
    split_tag = sanitize_for_filename(split)
    return f"{model_tag}__{ds_tag}__{subset_tag}__{split_tag}__vllm.jsonl"


def get_model_config_fields(model_name: str) -> Tuple[int, int, int]:
    """Return (num_heads, num_kv_heads, max_position_embeddings) if available, else zeros."""
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        num_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", 0) or 0)
        max_len = int(getattr(cfg, "max_position_embeddings", 0) or 0)
        return num_heads, num_kv_heads, max_len
    except Exception:
        return 0, 0, 0


def detect_model_type(model_name: str) -> str:
    """Detect the model type for prompt optimization."""
    model_lower = model_name.lower()
    if "deepseek-r1" in model_lower:
        return "deepseek-r1"
    elif "qwen2.5" in model_lower:
        return "qwen2.5"
    elif "qwen3" in model_lower:
        return "qwen3"
    return "generic"


def build_prompt_from_row(
    row: Dict[str, Any], 
    enable_thinking: bool = True,
    model_name: str = "",
    explicit_thinking: bool = False,
    use_livecodebench_style: bool = False
) -> str:
    """
    Build a prompt for Codeforces problems with model-specific optimizations.
    
    Args:
        row: Dictionary containing problem data
        enable_thinking: Whether thinking is enabled globally
        model_name: Model name for type detection
        explicit_thinking: Whether to add explicit thinking instructions
        use_livecodebench_style: Whether to use LiveCodeBench-style formatting
    """
    # Extract problem components
    title = row.get("title") or row.get("id") or "Codeforces Problem"
    description = row.get("description", "")
    input_format = row.get("input_format", "")
    output_format = row.get("output_format", "")
    note = row.get("note", "")
    examples = row.get("examples", [])
    tags = row.get("tags", [])
    time_limit = row.get("time_limit", "")
    memory_limit = row.get("memory_limit", "")
    
    if use_livecodebench_style:
        # LiveCodeBench style prompt
        problem_text = description.strip()
        
        # Add input/output format inline
        if input_format:
            problem_text += f"\n\nInput Format:\n{input_format.strip()}"
        
        if output_format:
            problem_text += f"\n\nOutput Format:\n{output_format.strip()}"
        
        if note:
            problem_text += f"\n\n{note.strip()}"
        
        # Add tags and limits if available
        if tags:
            problem_text += f"\n\nTags:\n{tags}"
        
        if time_limit or memory_limit:
            limits = []
            if time_limit:
                limits.append(f"Time Limit: {time_limit}")
            if memory_limit:
                limits.append(f"Memory Limit: {memory_limit}")
            problem_text += f"\n{' | '.join(limits)}"
        
        # LiveCodeBench-style instruction
        instruction = (
            "### Format: Read the inputs from stdin solve the problem and write the answer to stdout "
            "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
            "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            "```python\n"
            "# YOUR CODE HERE\n"
            "```\n\n"
            "### Answer: (use the provided format with backticks)"
        )
        
        return f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n\n{problem_text}\n{instruction}"
    
    else:
        # Original enhanced style
        examples_txt_lines: List[str] = []
        for i, ex in enumerate(examples, 1):
            ex_in = (ex.get("input") or "").strip()
            ex_out = (ex.get("output") or "").strip()
            
            if ex_in or ex_out:
                examples_txt_lines.append(f"Example {i}:")
                examples_txt_lines.append(f"Input:\n{ex_in}")
                examples_txt_lines.append(f"Output:\n{ex_out}")
        
        # Build problem statement
        parts = [f"Title: {title}"]
        
        if description:
            parts.extend(["", "Problem Description:", description.strip()])
        
        if input_format:
            parts.extend(["", "Input Format:", input_format.strip()])
        
        if output_format:
            parts.extend(["", "Output Format:", output_format.strip()])
        
        if note:
            parts.extend(["", "Note:", note.strip()])
        
        if examples_txt_lines:
            parts.extend(["", "Examples:", "\n\n".join(examples_txt_lines)])
        
        # Add metadata if available
        metadata = []
        if tags:
            metadata.append(f"Tags: {tags}")
        if time_limit:
            metadata.append(f"Time Limit: {time_limit}")
        if memory_limit:
            metadata.append(f"Memory Limit: {memory_limit}")
        
        if metadata:
            parts.extend(["", "Constraints:", "\n".join(metadata)])
        
        problem_stmt = "\n\n".join(p for p in parts if normalize_text(p))
        
        # Determine if we should add explicit thinking based on model type
        model_type = detect_model_type(model_name)
        should_add_thinking = (
            explicit_thinking and 
            enable_thinking and 
            model_type != "deepseek-r1"
        )
        
        # Model-specific thinking encouragement
        thinking_section = ""
        if should_add_thinking:
            thinking_section = (
                "First, analyze the problem step by step:\n"
                "1. Understand what the problem is asking\n"
                "2. Identify the input/output format and constraints\n"
                "3. Consider the algorithm and approach needed\n"
                "4. Think about edge cases and time complexity\n\n"
            )
        
        # Enhanced instruction
        base_instruction = (
            "Write a correct and efficient Python 3 program that solves this problem.\n\n"
            "Requirements:\n"
            "- Read input from standard input (stdin) using input() or sys.stdin\n"
            "- Write output to standard output (stdout) using print()\n"
            "- Output format must match exactly (no extra spaces, newlines, or explanatory text)\n"
            "- Solution must handle all constraints and edge cases\n"
            "- Code should be efficient for typical competitive programming limits\n"
            "- Use only standard Python libraries\n"
        )
        
        if should_add_thinking:
            instruction = thinking_section + base_instruction
            instruction += "\nAfter your analysis, provide only the final code in a Python code block."
        else:
            instruction = base_instruction + "\nProvide only the code in a Python code block."
        
        return f"{problem_stmt}\n\n{instruction}"


@dataclass
class GenConfig:
    """Configuration for the generation script."""
    dataset_name: str
    subset: str
    split: str
    model_name: str
    checkpoint_path: str
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    tensor_parallel_size: int
    gpu_ids: str
    max_model_len: int
    dtype: str
    results_dir: str
    outfile: str
    batch_size: int
    skip_non_stdio: bool
    max_problems: int
    gpu_memory_utilization: float
    enable_thinking: bool
    explicit_thinking: bool
    use_livecodebench_style: bool  # New parameter


def apply_chat_template(tokenizer: Any, user_prompt: str, enable_thinking: bool) -> str:
    """Apply chat template with optional thinking capability."""
    if enable_thinking:
        system_prompt = (
            "You are a helpful assistant. You are capable of thinking, reasoning, and planning. "
            "You can express your thoughts within <think> and </think> tags."
        )
    else:
        system_prompt = "You are a helpful coding assistant."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_prompt


def chunked(iterable: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def load_rows(cfg: GenConfig) -> List[Dict[str, Any]]:
    """Load dataset rows with filtering."""
    ds = load_dataset(cfg.dataset_name, split=cfg.split, name=cfg.subset)
    rows: List[Dict[str, Any]] = []
    for row in ds:
        if cfg.skip_non_stdio and (row.get("input_mode") or "stdio") != "stdio":
            continue
        rows.append(row)
        if cfg.max_problems and len(rows) >= cfg.max_problems:
            break
    return rows


def generate_with_vllm(cfg: GenConfig) -> str:
    """Main function to configure vLLM and run generation."""
    ensure_dir(cfg.results_dir)
    out_path = os.path.join(cfg.results_dir, cfg.outfile or resolve_outfile(
        cfg.model_name, cfg.dataset_name, cfg.subset, cfg.split
    ))
    
    with open(out_path, "w", encoding="utf-8") as f:
        pass

    rows = load_rows(cfg)
    if not rows:
        print("No rows to process.")
        return out_path

    print(f"Loaded {len(rows)} problems from {cfg.dataset_name}:{cfg.subset}:{cfg.split}")
    
    # Show prompt configuration
    model_type = detect_model_type(cfg.model_name)
    print(f"Model type detected: {model_type}")
    print(f"Thinking enabled: {cfg.enable_thinking}")
    print(f"Explicit thinking instructions: {cfg.explicit_thinking}")
    print(f"LiveCodeBench style: {cfg.use_livecodebench_style}")
    if cfg.explicit_thinking and model_type == "deepseek-r1":
        print("Note: Explicit thinking disabled for DeepSeek-R1 (has built-in reasoning)")

    if cfg.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    
    num_visible_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(',')) if os.environ.get("CUDA_VISIBLE_DEVICES") else 1

    model_to_use = cfg.checkpoint_path if cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path) else cfg.model_name
    print(f"Using model: {model_to_use}")

    num_heads, num_kv_heads, model_max_len = get_model_config_fields(model_to_use)
    
    # *** AUTOMATIC CONTEXT LENGTH OVERRIDE ***
    if model_max_len > 0 and cfg.max_model_len > model_max_len:
        print(f"Warning: --max-model-len ({cfg.max_model_len}) exceeds the model's architectural limit ({model_max_len}).")
        print("Setting VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 to allow this override.")
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    max_tp = min(cfg.tensor_parallel_size, num_visible_gpus)
    chosen_tp = 1
    for tp in range(max_tp, 0, -1):
        if num_heads > 0 and num_heads % tp == 0 and (num_kv_heads is None or num_kv_heads == 0 or num_kv_heads % tp == 0):
            chosen_tp = tp
            break
    
    print(
        f"Model heads={num_heads or 'N/A'}, kv_heads={num_kv_heads or 'N/A'} | "
        f"Visible GPUs={num_visible_gpus} | "
        f"Requested TP={cfg.tensor_parallel_size} -> Using TP={chosen_tp}"
    )

    from vllm import LLM, SamplingParams

    llm_params = {
        "model": model_to_use,
        "trust_remote_code": True,
        "tensor_parallel_size": chosen_tp,
        "max_model_len": cfg.max_model_len,
        "dtype": cfg.dtype,
        "seed": cfg.seed,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
    }

    if cfg.enable_thinking:
        print("Enabling 'thinking' mode (prefix caching).")
        llm_params["enable_prefix_caching"] = True

    llm = LLM(**llm_params)

    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
        stop=None,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    
    pbar = tqdm(total=len(rows), desc="Processing prompts")
    for batch in chunked(rows, cfg.batch_size):
        prompts = [
            apply_chat_template(
                tokenizer, 
                build_prompt_from_row(
                    row, 
                    cfg.enable_thinking, 
                    cfg.model_name, 
                    cfg.explicit_thinking,
                    cfg.use_livecodebench_style
                ), 
                cfg.enable_thinking
            ) 
            for row in batch
        ]
        ids = [row.get("id") or (row.get("contest_id", "") + "/" + row.get("index", "")) for row in batch]
        
        outputs = llm.generate(prompts, sampling_params)

        with open(out_path, "a", encoding="utf-8") as w:
            for i, out in enumerate(outputs):
                raw_response = out.outputs[0].text if out.outputs else ""
                code_text = extract_code_from_text(raw_response, language_hint="python")
                
                rec = {
                    "id": ids[i],
                    "model": cfg.model_name,
                    "raw_response": raw_response,
                    "code": code_text,
                    "timestamp": time.time(),
                }
                w.write(json.dumps(rec) + "\n")
        
        pbar.update(len(batch))

    pbar.close()
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Codeforces solutions with vLLM")
    
    p.add_argument("--dataset-name", default="open-r1/codeforces")
    p.add_argument("--subset", default="default")
    p.add_argument("--split", default="test")
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--checkpoint-path", default="", help="Path to fine-tuned checkpoint")
    
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=38912, help="Max sequence length. Automatically handles overrides for models like Qwen3.")
    p.add_argument("--max-new-tokens", type=int, default=32768)
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    
    p.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode with <think> tags")
    p.add_argument("--explicit-thinking", action="store_true", help="Add explicit step-by-step thinking instructions to prompts")
    p.add_argument("--use-livecodebench-style", action="store_true", help="Use LiveCodeBench-style prompt formatting")
    
    p.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--results-dir", default="./solutions")
    p.add_argument("--outfile", default="")
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--skip-non-stdio", action=argparse.BooleanOptionalAction, default=True)
    
    p.set_defaults(enable_thinking=True, explicit_thinking=False, use_livecodebench_style=False)
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = GenConfig(**vars(args))
    
    generate_with_vllm(cfg)


if __name__ == "__main__":
    main()