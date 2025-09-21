# Model Evaluation

This directory contains instructions to setup code evaluation docker and run model evaluation with respect to benchmarks. 

## Setup Evaluation Docker
Default docker configuration has several issuesm e.g. very small payload. Follow instructions below to setup a proper evaluation docker.  

### clone and enter repo
git clone https://github.com/engineer-man/piston

### Build your new, fixed image from the Dockerfile
docker build -t piston-fixed .

### Stop any old container
docker stop piston_api
docker rm piston_api

### Run the new image. The command is now simple and clean!
docker run \
    --privileged \
    -dit \
    -p 2000:2000 \
    --name piston_api \
    --env-file ./piston.env \
    piston-fixed

### Install latest python
cli/index.js ppman install python

# Run evaluation

python  eval_with_piston_gentest_checker_stats.py \
  --solutions-path model_solutions/deepseek-ai-deepseek-r1-0528-qwen3-8b__open-r1-codeforces__default__test__vllm.jsonl

python  eval/eval_with_piston_gentest_checker_stats.py \
--solutions-path /root/competitive-coding-ai/final_model_solutions/qwen-qwen2.5-7b-instruct__open-r1-codeforces__default__test__vllm.jsonl 2>&1 | tee final_repod_eval.txt

  python infer/generate_qwen_vllm_think.py   --dataset-name open-r1/codeforces --subset default --split test   --model-name "Qwen/Qwen2.5-7B-Instruct"   --checkpoint-path /root/rllm/trav_test/final_qwen2.5-7b-mot-full-run0   --batch-size 64   --max-model-len 32768 --max-new-tokens 10240   --tensor-parallel-size 8 --gpu-ids 0,1,2,3,4,5,6,7   --dtype bfloat16   --temperature 0.1 --top-p 0.95   --results-dir /root/competitive-coding-ai/final_model_solutions 2>&1 | tee final_reprod_ed.txt  

=== EVALUATION SUMMARY === 
{
  "num_attempted": 468,
  "num_correct": 20,
  "pass_at_1": 0.042735042735042736,
  "timestamp": 1758441492.0865378,
  "stats": {
    "num_problems_total": 468,
    "num_with_checker": 53,
    "with_checker_passed": 53,
    "with_checker_failed": 0,
    "num_without_checker": 415,
    "without_checker_passed": 20,
    "failure_reasons": {
      "runtime_error": 1632,
      "time_limit_exceeded": 495,
      "memory_limit_exceeded": 47,
      "failed_outputs_match": 8
    }
  }
}

Failure reasons summary:
  runtime_error: 1632   # 8 epochs: overfitted!!
  time_limit_exceeded: 495
  memory_limit_exceeded: 47
  failed_outputs_match: 8 # this is so low!!

Compared with baseline:
  failed_outputs_match: 890
  runtime_error: 560
  failed_checker: 503
  time_limit_exceeded: 12

(piston_env) root@ml-ai-ubuntu-gpu-h200x8-1128gb-nyc2:~/competitive-coding-ai# python  eval/eval_with_piston_gentest_checker_stats.py --solutions-path /root/competitive-coding-ai/model_solutions/qwen-qwen2.5-7b-instruct__open-r1-codeforces__default__test__vllm_origin_t01_top095.jsonl 2>&1 | tee final_qwen25_7b_instruct_t01_top095_baseline.txt

=== EVALUATION SUMMARY ===
{
  "num_attempted": 468,
  "num_correct": 29,
  "pass_at_1": 0.06196581196581197,
  "timestamp": 1758442843.1173468,
  "stats": {
    "num_problems_total": 468,
    "num_with_checker": 53,
    "with_checker_passed": 11,
    "with_checker_failed": 42,
    "num_without_checker": 415,
    "without_checker_passed": 26,
    "failure_reasons": {
      "runtime_error": 560,
      "failed_outputs_match": 890,
      "failed_checker": 503,
      "time_limit_exceeded": 12
    }
  }
}

Failure reasons summary:
  failed_outputs_match: 890
  runtime_error: 560
  failed_checker: 503
  time_limit_exceeded: 12

Saved detailed results to: /root/competitive-coding-ai/results/qwen-qwen2.5-7b-instruct__open-r1-codeforces__default__test__vllm_origin_t01_top095/piston_eval_results.jsonl
Saved metrics to: /root/competitive-coding-ai/results/qwen-qwen2.5-7b-instruct__open-r1-codeforces__default__test__vllm_origin_t01_top095/piston_eval_metrics.json
