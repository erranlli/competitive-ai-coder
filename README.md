## Competitive Coding AI

End-to-end pipeline for training, generating, and evaluating competitive programming solutions with Qwen-family models. The repo includes:

- Dataset preparation from DeepCoder trajectories (convert → filter by reward → decontaminate → visualize)
- SFT training with stability features (gradient clipping, loss re-weighting/masking for code-only learning)
- Checkpoint consolidation (DeepSpeed → HF → single-file) for deployment
- vLLM-based generation with code-centric decoding controls
- Robust, checker-aware evaluation via a Piston-compatible execution server

### Repository layout

- `datasets/`
  - `deepcoder_gen_data/`: convert DeepCoder trajectories into CoT/code datasets
  - `preprocess_data/`: decontamination, code filtering, Arrow exports (reward/test filtered)
  - `compare_datasets/`: dataset diffing utilities
  - `visualize/`: Gradio/web visualizer for dataset inspection
  - See `datasets/README.md` for details
- `train/`
  - `train_qwen_think.py`: main SFT trainer (with code-token loss re-weighting; imports from `model_util`)
  - `consolidate_cpu.py`: DeepSpeed → HF weights in-place
  - `merge_shards_cpu.py`: HF sharded → single-file directory
  - See `train/README.md` for usage
- `infer/`
  - `generate_qwen_vllm_think.py`: batched generation with vLLM (uses `model_util` for model/path resolution)
  - See `infer/README.md` for decoding tips
- `eval/`
  - `eval_with_piston_gentest_checker_stats.py`: checker-aware evaluation + stats
  - See `eval/README.md` for server notes and options
- `data_util/`: reusable pretty-printers, record lookup, and eval renderers used in notebooks
- `model_util/`: shared utilities for training and inference (see `model_util/README.md`)
- `benchmark/`: notebooks and small JSON builders for quick sanity checks (see `benchmark/README.md`)
- `test/`: Piston smoke tests (see `test/README.md`)

---

## Quickstart (TL;DR)

1) Prepare a code-only SFT dataset (from DeepCoder or other sources) using the filter scripts:

```bash
python datasets/preprocess_data/filter_and_save_cots.py \
  --subset solutions_py \
  --match-strategy robust \
  --endpoint http://localhost:2000 \
  --output-path ./codeforces_cots_high_quality.arrow \
  --num-workers 16
```

2) Train (Qwen2.5-7B Instruct) with stable settings:

```bash
torchrun --nproc_per_node=8 train/train_qwen_think.py \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --dataset-path ./codeforces_cots_high_quality.arrow \
  --output-dir ./runs/qwen25-7b \
  --deepspeed train/deepspeed_zero3.json \
  --num-train-epochs 1 --learning-rate 1e-5 --warmup-ratio 0.05 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 16 \
  --eval-steps 200 --save-steps 200 --logging-steps 50 \
  --enable-loss-reweight --loss-weight-code 2.0 --loss-weight-noncode 0.1
```

3) Consolidate weights for vLLM:

```bash
# DeepSpeed checkpoint → HF in-place (adds tokenizer/config)
python train/consolidate_cpu.py --output-dir ./runs/qwen25-7b --base-model-name Qwen/Qwen2.5-7B-Instruct

# HF sharded → single-file directory
python train/merge_shards_cpu.py --input-dir ./runs/qwen25-7b --output-dir ./deploy/qwen25-7b-single
```

4) Generate with vLLM (code-centric decoding):

```bash
python infer/generate_qwen_vllm_think.py \
  --dataset-name open-r1/codeforces --subset default --split test \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --checkpoint-path ./deploy/qwen25-7b-single \
  --batch-size 64 --max-model-len 32768 --max-new-tokens 1536 \
  --tensor-parallel-size 8 --gpu-ids 0,1,2,3,4,5,6,7 \
  --dtype bfloat16 --temperature 0.2 --top-p 0.9 \
  --no-repeat-ngram-size 6 --repetition-penalty 1.1 \
  --results-dir ./model_solutions
```

5) Evaluate (checker-aware, robust):

```bash
python eval/eval_with_piston_gentest_checker_stats.py \
  --solutions-path ./model_solutions/qwen-qwen2.5-7b-instruct__open-r1-codeforces__default__test__vllm.jsonl \
  --endpoint http://localhost:2000 \
  --generated-tests-dir /path/to/generated_tests \
  --max-generated-tests 0 \
  --results-dir results
```

---

## Dataset generation: from DeepCoder trajectories to SFT-ready code

The standard pipeline:

1) Convert solution trajectories → CoT/code format
- `datasets/deepcoder_gen_data/convert_trajectories_to_cots.py`: extracts successful trajectory solutions, constructs messages with instructions and Python code blocks

2) (Optional) Run generation from parquet
- `datasets/deepcoder_gen_data/run_rllm_from_parquet.py`: generate solutions using an existing model to enlarge the dataset

3) Filter by reward (execution/test correctness)
- `datasets/preprocess_data/filter_and_save_cots.py`: validates each sample by executing code against public/private tests via a Piston-like API; only fully passing samples are saved to Arrow.
- Robust matching modes: `--match-strategy {basic,robust}` controls how expected vs actual is compared (numeric tolerance, JSON-aware where applicable)

4) Decontamination / cleanup
- `datasets/preprocess_data/decontaminate_converted_cots.py`: removes near-duplicates or leaked items per your policy

5) Compare datasets and visualize
- `datasets/compare_datasets/compare_datasets.py` helps quantify differences across sets (e.g., DeepCoder vs MoT subsets)
- `datasets/visualize/visualize_cots_datasets_web.py` launches a small web app for manual inspection (problems, tests, and target code)

Outputs are typically saved with `Dataset.save_to_disk()` (Arrow) and fed directly into the trainer.

---

## Training (SFT) notes

- Script: `train/train_qwen_think.py`
- Stability features:
  - Gradient clipping (default `max_grad_norm=0.5`)
  - Cosine scheduler, warmup ratio configurable
  - Early stopping supported when evaluation is enabled
- Code-first supervision:
  - Enable per-token loss re-weighting: `--enable-loss-reweight`
  - Emphasize code tokens inside ```python blocks: `--loss-weight-code 2.0 --loss-weight-noncode 0.1`
  - Or mask non-code from loss completely: `--mask-noncode`
  - This curbs the model’s tendency to emit reasoning/explanations and focuses gradient signal on executable code
- Recommended starting hyperparameters (Qwen2.5-7B-Instruct):
  - `--learning-rate 1e-5` (try `5e-6` if unstable)
  - `--num-train-epochs 1`
  - `--per-device-train-batch-size 1 --gradient-accumulation-steps 16`
  - `--max-seq-length 8192–12288` for faster, more stable SFT; use longer contexts only if data demands

---

## Consolidation: from training outputs to deployable weights

- `train/consolidate_cpu.py --output-dir <run_dir> --base-model-name Qwen/Qwen2.5-7B-Instruct`
  - Finds latest `checkpoint-*` under `run_dir`, runs DeepSpeed → FP32 HF conversion, and saves tokenizer/config
- `train/merge_shards_cpu.py --input-dir <hf_dir> --output-dir <single_dir>`
  - Loads HF sharded weights on CPU and re-saves as a single-file model directory (handles nested shard layouts)

Use the final single-file directory with vLLM.

---

## Generation (vLLM)

- `infer/generate_qwen_vllm_think.py` configures vLLM and runs batched generation with a code-centric prompt.
- Decoding tips to reduce runtime_error/TLE and repetition:
  - `--max-new-tokens 1024–1536`
  - `--temperature 0.1–0.2 --top-p 0.9`
  - `--no-repeat-ngram-size 6 --repetition-penalty 1.1`
  - Keep `enable_thinking=False` for instruct SFT models unless your SFT explicitly trains reasoning style
- Results are written to `model_solutions/` as JSONL.

---

## Robust evaluation (checker-aware + large inputs)

- Script: `eval/eval_with_piston_gentest_checker_stats.py`
  - Executes solutions via a Piston-compatible API for official tests
  - Uses checker code (`generated_checker`) when provided to judge correctness
  - For large inputs, automatically replaces stdin with `input.txt` (works around JSON body limits)
  - Supports sampling/sorting of generated tests; skip budgets for huge testcases; concurrency for throughput
- Outputs:
  - Detailed per-case JSONL: `results/<solutions_prefix>/piston_eval_results.jsonl`
  - Metrics JSON: `results/<solutions_prefix>/piston_eval_metrics.json`

---

## Uploading datasets to Hugging Face Hub

Use the tools under `upload_hf/`:
- `inspect_schema.py` to inspect Arrow datasets
- `upload_public_datasets.py` to push to a HF dataset repo

---

## Troubleshooting checklist

- “Good training, bad evaluation”: ensure you consolidated the latest run; don’t evaluate an earlier single-file directory
- Too much verbosity / TLE / runtime_error
  - Prefer code-only or loss-reweighted SFT target
  - Use code-centric decoding params; shorten `max-new-tokens`
  - Filter/clean samples that lack a final python code fence
- Trainer error `num_items_in_batch`
  - Our custom trainer accepts `**kwargs`; make sure you’re on the updated script
- CUDA mismatch / arch warnings
  - Harmless if APIs are compatible; otherwise set `TORCH_CUDA_ARCH_LIST` or match CUDA across torch/driver
  