
python infer/generate_qwen_vllm_think.py \
  --dataset-name open-r1/codeforces --subset default --split test \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --checkpoint-path /root/rllm/trav_test/final_qwen2.5-7b-mot-full-run0 \
  --batch-size 64 \
  --max-model-len 32768 --max-new-tokens 10240 \
  --tensor-parallel-size 8 --gpu-ids 0,1,2,3,4,5,6,7 \
  --dtype bfloat16 \
  --temperature 0.1 --top-p 0.95 \
  --results-dir /root/competitive-coding-ai/final_model_solutions
  

python infer/generate_qwen_vllm_think.py \
  --dataset-name open-r1/codeforces --subset default --split test \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --batch-size 64 \
  --max-model-len 32768 --max-new-tokens 10240 \
  --tensor-parallel-size 8 --gpu-ids 0,1,2,3,4,5,6,7 \
  --dtype bfloat16 \
  --temperature 0.4 --top-p 0.95 \
  --results-dir /root/competitive-coding-ai/model_solutions

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

  python infer/generate_qwen_vllm_think.py \
  --dataset-name open-r1/codeforces --subset default --split test \
  --model-name "Qwen/Qwen3-8B" \
  --batch-size 64 \
  --max-model-len 38912 --max-new-tokens 16384 \
  --tensor-parallel-size 8 --gpu-ids 0,1,2,3,4,5,6,7 \
  --dtype bfloat16 \
  --temperature 0.2 --top-p 0.95 \
  --results-dir /root/competitive-coding-ai/model_solutions

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


  python infer/generate_qwen_vllm_think.py \
  --dataset-name open-r1/codeforces --subset default --split test \
  --model-name "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --batch-size 24 \
  --max-model-len 32768 --max-new-tokens 8192 \
  --tensor-parallel-size 8 --gpu-ids 0,1,2,3,4,5,6,7 \
  --dtype bfloat16 \
  --temperature 0.2 --top-p 0.95 \
  --results-dir /root/competitive-coding-ai/model_solutions

  python infer/generate_qwen_vllm_think.py \
  --dataset-name open-r1/codeforces --subset default --split test \
  --model-name "agentica-org/DeepCoder-14B-Preview" \
  --batch-size 8 \
  --max-model-len 65536 --max-new-tokens 64000 \
  --tensor-parallel-size 8 --gpu-ids 0,1,2,3,4,5,6,7 \
  --dtype bfloat16 \
  --temperature 0.6 --top-p 0.95 \
  --results-dir /root/competitive-coding-ai/model_solutions

  python infer/generate_qwen_vllm_think.py \
  --dataset-name open-r1/codeforces --subset default --split test \
  --model-name "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" \
  --batch-size 64 \
  --max-model-len 38912 --max-new-tokens 32768 \
  --tensor-parallel-size 8 --gpu-ids 0,1,2,3,4,5,6,7 \
  --dtype bfloat16 \
  --temperature 0.6 --top-p 0.95 \
  --results-dir /root/competitive-coding-ai/model_solutions