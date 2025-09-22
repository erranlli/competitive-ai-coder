## Inference (vLLM, Qwen variants)

`generate_qwen_vllm_think.py` runs batch inference over Codeforces datasets with vLLM, supporting long context, multi-GPU tensor-parallel, and optional “thinking” tokens for R1-style models.

### Requirements

- vLLM with CUDA, compatible transformers/tokenizers
- GPUs with sufficient memory for the chosen model and max length

### Common flags

- `--dataset-name`, `--subset`, `--split`: Hugging Face dataset spec (e.g., `open-r1/codeforces`, `default`, `test`)
- `--model-name`: HF model id (e.g., `Qwen/Qwen2.5-7B-Instruct`)
- `--checkpoint-path`: optional local fine-tuned weights dir
- `--batch-size`: per-request batch size sent to vLLM
- `--max-model-len`, `--max-new-tokens`: context and generation lengths
- `--tensor-parallel-size`, `--gpu-ids`: multi-GPU config
- `--dtype`: `bfloat16` recommended
- `--temperature`, `--top-p`: sampling controls
- `--enable-thinking`, `--explicit-thinking`: enable R1-style thinking tokens when supported
- `--results-dir`: output directory for JSONL solutions

### Examples

Baseline Qwen2.5-7B-Instruct:
```bash
python infer/generate_qwen_vllm_think.py \
  --dataset-name open-r1/codeforces --subset default --split test \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --batch-size 64 \
  --max-model-len 32768 --max-new-tokens 10240 \
  --tensor-parallel-size 8 --gpu-ids 0,1,2,3,4,5,6,7 \
  --dtype bfloat16 \
  --temperature 0.4 --top-p 0.95 \
  --results-dir /root/competitive-coding-ai/model_solutions
```

Using a fine-tuned checkpoint:
```bash
python infer/generate_qwen_vllm_think.py \
  --dataset-name open-r1/codeforces --subset default --split test \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --checkpoint-path /root/competitive-coding-ai/qwen2.5-7b-mot-full-run0 \
  --batch-size 64 \
  --max-model-len 32768 --max-new-tokens 10240 \
  --tensor-parallel-size 8 --gpu-ids 0,1,2,3,4,5,6,7 \
  --dtype bfloat16 \
  --temperature 0.2 --top-p 0.95 \
  --results-dir /root/competitive-coding-ai/model_solutions
```

Qwen3-8B with explicit thinking:
```bash
python infer/generate_qwen_vllm_think.py \
  --dataset-name open-r1/codeforces --subset default --split test \
  --model-name "Qwen/Qwen3-8B" \
  --batch-size 64 \
  --max-model-len 38912 --max-new-tokens 16384 \
  --tensor-parallel-size 8 --gpu-ids 0,1,2,3,4,5,6,7 \
  --dtype bfloat16 \
  --temperature 0.6 --top-p 0.95 \
  --enable-thinking --explicit-thinking \
  --results-dir /root/competitive-coding-ai/model_solutions
```

DeepCoder-14B preview (very long gen):
```bash
python infer/generate_qwen_vllm_think.py \
  --dataset-name open-r1/codeforces --subset default --split test \
  --model-name "agentica-org/DeepCoder-14B-Preview" \
  --batch-size 8 \
  --max-model-len 65536 --max-new-tokens 64000 \
  --tensor-parallel-size 8 --gpu-ids 0,1,2,3,4,5,6,7 \
  --dtype bfloat16 \
  --temperature 0.6 --top-p 0.95 \
  --results-dir /root/competitive-coding-ai/model_solutions
```

### Tips

- Keep `--max-model-len` + `--max-new-tokens` within GPU memory; prefer bfloat16.
- For large batches, ensure vLLM engine config matches tensor parallelism and memory.
- Store results in `model_solutions/` for downstream evaluation.

