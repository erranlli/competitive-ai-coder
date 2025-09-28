#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from tqdm import tqdm

# Ensure local package imports work
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CUR_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from model_util.file_util import ensure_dir, resolve_outfile, prepare_model_dir as resolve_model_dir
from model_util.text_util import normalize_text, extract_code_from_text
from model_util.model_util import get_model_config_fields, detect_model_type, load_tokenizer

try:
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict as ds_convert
except Exception:
    ds_convert = None


def build_prompt_from_row(row: Dict[str, Any], enable_thinking: bool, model_name: str, use_livecodebench_style: bool) -> str:
    """Build prompt for Codeforces problem."""
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
        problem_text = description.strip()
        if input_format:
            problem_text += f"\n\nInput Format:\n{input_format.strip()}"
        if output_format:
            problem_text += f"\n\nOutput Format:\n{output_format.strip()}"
        if note:
            problem_text += f"\n\n{note.strip()}"
        if tags:
            problem_text += f"\n\nTags:\n{tags}"
        if time_limit or memory_limit:
            limits = []
            if time_limit:
                limits.append(f"Time Limit: {time_limit}")
            if memory_limit:
                limits.append(f"Memory Limit: {memory_limit}")
            problem_text += f"\n{' | '.join(limits)}"

        instruction = (
            "### Format: Read the inputs from stdin solve the problem and write the answer to stdout "
            "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
            "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            "```python\n"
            "# YOUR CODE HERE\n"
            "```\n\n"
            "### Answer: (use the provided format with backticks)"
        )
        return f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program.\n\n{problem_text}\n{instruction}"

    # Default: instruction-tuned prompt
    prompt_parts: List[str] = []
    prompt_parts.append(
        "You will be given a competitive programming problem.\n"
        "Analyze maximum input constraints and identify optimal algorithm/data structures. "
        "Provide a complete Python3 implementation optimized for speed and memory.\n\n"
        "Your solution must read input from standard input and write output to stdout.\n\n"
        "Put your final solution within a single code block:\n"
        "```python\n<your code here>\n```"
    )
    prompt_parts.append("\n# Problem\n\n" + (description or "").strip())

    constraints_lines: List[str] = ["## Constraints"]
    constraints_lines.append(f"Time limit per test: {time_limit or '2.0 seconds'}")
    constraints_lines.append(f"Memory limit per test: {memory_limit or '256.0 megabytes'}")
    prompt_parts.append("\n" + "\n".join(constraints_lines))

    if input_format:
        prompt_parts.append("\n## Input Format\n" + input_format.strip())
    if output_format:
        prompt_parts.append("\n## Output Format\n" + output_format.strip())

    if examples:
        ex_lines: List[str] = []
        for idx, ex in enumerate(examples):
            ex_in = (ex.get("input") or "").strip()
            ex_out = (ex.get("output") or "").strip()
            if ex_in or ex_out:
                ex_lines.append("```input\n" + ex_in + "\n```")
                ex_lines.append("```output\n" + ex_out + "\n```")
                if idx != len(examples) - 1:
                    ex_lines.append("-----")
        prompt_parts.append("\n## Examples\n" + "\n".join(ex_lines))
    if note:
        prompt_parts.append("\n## Note\n" + note.strip())

    return "\n\n".join([p for p in prompt_parts if normalize_text(p)])


@dataclass
class GenConfig:
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
    use_livecodebench_style: bool
    repetition_penalty: float = 1.1
    stop: str = ""


def apply_chat_template(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    if enable_thinking:
        messages = [
            {"role": "system", "content": "You are a helpful assistant capable of reasoning with <think> tags."},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_prompt


def chunked(iterable: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def load_rows(cfg: GenConfig) -> List[Dict[str, Any]]:
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
    ensure_dir(cfg.results_dir)
    out_path = os.path.join(cfg.results_dir, cfg.outfile or resolve_outfile(cfg.model_name, cfg.dataset_name, cfg.subset, cfg.split))
    with open(out_path, "w", encoding="utf-8") as f:
        pass

    rows = load_rows(cfg)
    if not rows:
        print("No rows to process.")
        return out_path

    print(f"Loaded {len(rows)} problems from {cfg.dataset_name}:{cfg.subset}:{cfg.split}")

    model_type = detect_model_type(cfg.model_name)
    print(f"Model type detected: {model_type}")
    print(f"Thinking enabled: {cfg.enable_thinking}")
    print(f"LiveCodeBench style: {cfg.use_livecodebench_style}")

    if cfg.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    num_visible_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) if os.environ.get("CUDA_VISIBLE_DEVICES") else 1

    model_dir = Path(cfg.checkpoint_path)
    if (model_dir / "model.safetensors.index.json").exists():
        print("Detected sharded model.")
        model_to_use = str(model_dir)
    else:
        model_to_use = resolve_model_dir(cfg.checkpoint_path, cfg.model_name, ds_convert)

    chosen_tp = min(cfg.tensor_parallel_size, num_visible_gpus)

    # For qwen2.5, the max tensor parallel size that satisfies both attention head (28)
    # and vocab size (152064) divisibility is 4. We must cap it at 4.
    if model_type == "qwen2.5" and chosen_tp > 4:
        print(f"Warning: For Qwen2.5, tensor parallel size must be <= 4. "
              f"Reducing from {chosen_tp} to 4.")
        chosen_tp = 4

    llm = LLM(
        model=model_to_use,
        tensor_parallel_size=chosen_tp,
        max_model_len=cfg.max_model_len,
        dtype=cfg.dtype,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=cfg.seed,
    )

    stop_list = [s for s in cfg.stop.split(",") if s] if cfg.stop else None
    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
        repetition_penalty=cfg.repetition_penalty,
        stop=stop_list,
    )

    tokenizer = load_tokenizer(cfg.model_name)
    pbar = tqdm(total=len(rows), desc="Processing prompts")

    for batch in chunked(rows, cfg.batch_size):
        prompts = [apply_chat_template(tokenizer, build_prompt_from_row(row, cfg.enable_thinking, cfg.model_name, cfg.use_livecodebench_style), cfg.enable_thinking) for row in batch]
        ids = [row.get("id") or (row.get("contest_id", "") + "/" + row.get("index", "")) for row in batch]

        outputs = llm.generate(prompts, sampling_params)

        with open(out_path, "a", encoding="utf-8") as w:
            for i, output in enumerate(outputs):
                raw_response = output.outputs[0].text if output.outputs else ""
                code_text = extract_code_from_text(raw_response, language_hint="python")
                rec = {"id": ids[i], "model": cfg.model_name, "raw_response": raw_response, "code": code_text, "timestamp": time.time()}
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
    p.add_argument("--max-model-len", type=int, default=38912)
    p.add_argument("--max-new-tokens", type=int, default=32768)
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--stop", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--use-livecodebench-style", action="store_true")
    p.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--results-dir", default="./solutions")
    p.add_argument("--outfile", default="")
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--skip-non-stdio", action=argparse.BooleanOptionalAction, default=True)
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = GenConfig(**vars(args))
    generate_with_vllm(cfg)


if __name__ == "__main__":
    main()
