import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
import tempfile
import shutil
import re
try:
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict as ds_convert
except Exception:
    ds_convert = None


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
        # Training-style prompt matching user's fine-tune instruction (sections and fenced examples)
        def fmt_limits(val, unit):
            try:
                v = float(val)
                if v.is_integer():
                    return f"{int(v)} {unit}"
                return f"{v} {unit}"
            except Exception:
                return f"{val} {unit}" if val else ""

        ex_lines: List[str] = []
        if isinstance(examples, list) and len(examples) > 0:
            for idx, ex in enumerate(examples):
                ex_in = (ex.get("input") or "").strip()
                ex_out = (ex.get("output") or "").strip()
                if ex_in or ex_out:
                    ex_lines.append("```input\n" + ex_in + "\n```")
                    ex_lines.append("```output\n" + ex_out + "\n```")
                    if idx != len(examples) - 1:
                        ex_lines.append("-----")
        examples_block = "\n".join(ex_lines)

        tl = fmt_limits(time_limit, "seconds") if time_limit else "2.0 seconds"
        ml = fmt_limits(memory_limit, "megabytes") if memory_limit else "256.0 megabytes"

        prompt_parts: List[str] = []
        prompt_parts.append(
            "You will be given a competitive programming problem.\n"
            "Analyze the maximum input constraints and identify the optimal algorithmic approach and data structures needed to process the largest possible test cases within the time and memory limits, then explain why your chosen implementation strategy is the most efficient solution. Please reason step by step about your solution approach, then provide a complete implementation in Python 3 that is thoroughly optimized for both speed and memory usage.\n\n"
            "Your solution must read input from standard input (input()), write output to standard output (print()).\n"
            "Do not include any debug prints or additional output.\n\n"
            "Put your final solution within a single code block:\n"
            "```python\n"
            "<your code here>\n"
            "```"
        )
        prompt_parts.append("\n# Problem\n\n" + (description or "").strip())

        constraints_lines: List[str] = ["## Constraints"]
        constraints_lines.append(f"Time limit per test: {tl}")
        constraints_lines.append(f"Memory limit per test: {ml}")
        prompt_parts.append("\n" + "\n".join(constraints_lines))

        if input_format:
            prompt_parts.append("\n## Input Format\n" + input_format.strip())
        if output_format:
            prompt_parts.append("\n## Output Format\n" + output_format.strip())
        if examples_block:
            prompt_parts.append("\n## Examples\n" + examples_block)
        if note:
            prompt_parts.append("\n## Note\n" + note.strip())

        return "\n\n".join([p for p in prompt_parts if normalize_text(p)])


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
    # Decoding controls
    no_repeat_ngram_size: int = 6
    repetition_penalty: float = 1.1
    stop: str = ""  # comma-separated list; defaults applied if empty


def apply_chat_template(tokenizer: Any, user_prompt: str, enable_thinking: bool) -> str:
    """Apply chat template with optional thinking capability.

    When thinking is disabled, avoid injecting a system message to better match
    instruction-tuned training prompts (single user message).
    """
    if enable_thinking:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. You are capable of thinking, reasoning, and planning. "
                    "You can express your thoughts within <think> and </think> tags."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [
            {"role": "user", "content": user_prompt}
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

    def _latest_checkpoint(run_dir: str) -> Optional[str]:
        try:
            cks = [d for d in os.listdir(run_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(run_dir, d))]
            if not cks:
                return None
            cks_sorted = sorted(cks, key=lambda n: int(re.sub(r"[^0-9]", "", n)) if re.search(r"\d+", n) else -1)
            return os.path.join(run_dir, cks_sorted[-1])
        except Exception:
            return None

    def _normalize_single_file(src_dir: str, dst_dir: str) -> bool:
        os.makedirs(dst_dir, exist_ok=True)
        copied = False
        for name in ["pytorch_model.bin", "model.safetensors"]:
            p = os.path.join(src_dir, name)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(dst_dir, name))
                copied = True
        # Nested bug case: dir named pytorch_model.bin/
        nested = os.path.join(src_dir, "pytorch_model.bin")
        if os.path.isdir(nested):
            inner_bin = os.path.join(nested, "pytorch_model.bin")
            inner_safe = os.path.join(nested, "model.safetensors")
            if os.path.isfile(inner_bin):
                shutil.copy2(inner_bin, os.path.join(dst_dir, "pytorch_model.bin"))
                copied = True
            elif os.path.isfile(inner_safe):
                shutil.copy2(inner_safe, os.path.join(dst_dir, "model.safetensors"))
                copied = True
        if copied:
            for fname in [
                "config.json","generation_config.json","tokenizer.json","tokenizer_config.json",
                "special_tokens_map.json","vocab.json","merges.txt","added_tokens.json","chat_template.jinja",
            ]:
                s = os.path.join(src_dir, fname)
                if os.path.isfile(s):
                    try: shutil.copy2(s, os.path.join(dst_dir, fname))
                    except Exception: pass
        return copied

    def prepare_model_dir() -> str:
        # 1) No checkpoint_path: return model_name
        if not cfg.checkpoint_path or not os.path.exists(cfg.checkpoint_path):
            return cfg.model_name

        path = cfg.checkpoint_path
        # If a run dir with checkpoint-*/ exists, pick latest
        if os.path.isdir(path) and any(n.startswith("checkpoint-") for n in os.listdir(path)):
            ck = _latest_checkpoint(path)
            if ck:
                path = ck

        # If this looks like a DeepSpeed checkpoint (has global_step*/ inside)
        if os.path.isdir(path) and any(n.startswith("global_step") for n in os.listdir(path)):
            if ds_convert is None:
                print("Warning: deepspeed not available; cannot auto-convert ZeRO checkpoint. Falling back to model_name.")
                return cfg.model_name
            tmpdir = tempfile.mkdtemp(prefix="vllm_ds2hf_")
            out_file = os.path.join(tmpdir, "pytorch_model.bin")
            try:
                print(f"Auto-converting DeepSpeed checkpoint: {path} -> {out_file}")
                ds_convert(path, out_file)
                # Save tokenizer and config alongside for completeness
                try:
                    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
                    tok.save_pretrained(tmpdir)
                except Exception: pass
                try:
                    conf = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
                    conf.save_pretrained(tmpdir)
                except Exception: pass
                return tmpdir
            except Exception as e:
                print(f"Auto-convert failed: {e}")
                return cfg.model_name

        # If directory already has a single-file weight or nested bug, normalize
        tmpdir = tempfile.mkdtemp(prefix="vllm_norm_")
        if _normalize_single_file(path, tmpdir):
            return tmpdir

        # If it's an HF sharded directory with index at root, vLLM can usually load directly
        index_root = os.path.join(path, "pytorch_model.bin.index.json")
        if os.path.isfile(index_root):
            return path

        # If shards are nested under pytorch_model.bin/, normalize them upward (copy index/shards)
        nested_index = os.path.join(path, "pytorch_model.bin", "pytorch_model.bin.index.json")
        if os.path.isfile(nested_index):
            try:
                # Copy all files from nested into tmpdir
                for n in os.listdir(os.path.dirname(nested_index)):
                    s = os.path.join(os.path.dirname(nested_index), n)
                    d = os.path.join(tmpdir, n)
                    if os.path.isfile(s): shutil.copy2(s, d)
                # Copy tokenizer/config from root
                for fname in [
                    "config.json","generation_config.json","tokenizer.json","tokenizer_config.json",
                    "special_tokens_map.json","vocab.json","merges.txt","added_tokens.json","chat_template.jinja",
                ]:
                    s = os.path.join(path, fname)
                    if os.path.isfile(s): shutil.copy2(s, os.path.join(tmpdir, fname))
                return tmpdir
            except Exception:
                pass

        # Otherwise fall back to model_name
        return cfg.model_name

    model_to_use = prepare_model_dir()
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

    # Initialize LLM engine
    llm = LLM(
        model=model_to_use,
        trust_remote_code=True,
        tensor_parallel_size=chosen_tp,
        max_model_len=cfg.max_model_len,
        dtype=cfg.dtype,
        seed=cfg.seed,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enable_prefix_caching=bool(cfg.enable_thinking),
    )

    # Build stop sequences (CLI or env). Default: no stops
    stop_list = None
    if getattr(cfg, "stop", ""):
        stop_list = [s for s in cfg.stop.split(",") if s]
    if stop_list is None:
        env_stop = os.environ.get("VLLM_STOP_OVERRIDES", "")
        if env_stop:
            parsed = [s for s in env_stop.split(",") if s]
            stop_list = parsed if parsed else None

    # Construct SamplingParams with broad compatibility across vLLM versions
    try:
        sampling_params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            stop=stop_list,
            repetition_penalty=cfg.repetition_penalty,
        )
    except TypeError:
        # Fallback without repetition_penalty for older vLLM versions
        sampling_params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            stop=stop_list,
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
        
        # Generate responses using synchronous LLM.generate
        outputs = llm.generate(prompts, sampling_params)
        
        with open(out_path, "a", encoding="utf-8") as w:
            for i, output in enumerate(outputs):
                # Get the generated text from the first (and typically only) completion
                raw_response = output.outputs[0].text if output.outputs else ""
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
    p.add_argument("--no-repeat-ngram-size", type=int, default=6)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--stop", type=str, default="", help="Comma-separated stop sequences; default to \n```,\n### Explanation,\n## Explanation")
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
    
    # Default: disable thinking to reduce verbosity for instruction-tuned code models
    p.set_defaults(enable_thinking=False, explicit_thinking=False, use_livecodebench_style=False)
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = GenConfig(**vars(args))
    
    generate_with_vllm(cfg)


if __name__ == "__main__":
    main()