1. Recover the old result: deepspeed problem? Training script problem? Data problem?
2. Train Deepcoder 16K context length

torchrun --nproc_per_node=8 /root/competitive-coding-ai/train/train_qwen_think.py \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --dataset-path "/mnt/data2/new_deepcoder_cots_arrow_appexp" \
  --output-dir "./deepcoder-run3-final-optimized" \
  --deepspeed "/root/competitive-coding-ai/train/deepspeed_zero3.json" \
  --report-to "wandb" --wandb-project "Final-Mixture-of-Thoughts-Finetuning" \
  --num-train-epochs 3 \
  --learning-rate 2e-5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --warmup-ratio 0.03 \
  --logging-steps 10 \
  --save-strategy "epoch" \
  --bf16_full_eval \
  2>&1 | tee deepcoder_final_optimized_run3.txt


torchrun --nproc_per_node=8 /root/competitive-coding-ai/train/train_qwen_think.py \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --dataset-path "/mnt/data2/new_deepcoder_cots_arrow_appexp" \
  --output-dir "./final_qwen2.5-7b-deepcoder-full-run_3ep2e_5" \
  --deepspeed "/root/competitive-coding-ai/train/deepspeed_zero2.json" \
  --report-to "wandb" --wandb-project "Final-Mixture-of-Thoughts-Finetuning" \
  --num-train-epochs 3 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.03 2>&1 | tee final_deepcoder_debug_run1.txt


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 /root/competitive-coding-ai/train/train_qwen_think.py \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --dataset-path "/mnt/data2/new_deepcoder_cots_arrow_appexp" \
  --output-dir "./final_qwen2.5-7b-deepcoder-full-run0" \
  --deepspeed "/root/competitive-coding-ai/train/deepspeed_zero3_h200.json" \
  --report-to "wandb" --wandb-project "Final-Mixture-of-Thoughts-Finetuning" \
  --num-train-epochs 8 \
  --learning-rate 4e-5 \
  --warmup-ratio 0.03 2>&1 | tee final_deepcoder_debug0.txt


torchrun --nproc_per_node=8 train_filtered_openai.py \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --dataset-path "/root/rllm/trav_test/filtered_datasets_flexible_match/successful_solutions" \
  --output-dir "./final_qwen2.5-7b-mot-full-run0" \
  --deepspeed "deepspeed_zero3.json" \
  --report-to "wandb" --wandb-project "Final-Mixture-of-Thoughts-Finetuning" \
  --num-train-epochs 8 \
  --learning-rate 4e-5 \
  --warmup-ratio 0.03 2>&1 | tee final_mot_debug0.txt
  
  
  \
  --resume-from-checkpoint last

## Gemini recommendation:
  torchrun --nproc_per_node=8 train_filtered_openai.py \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --dataset-path "/root/rllm/trav_test/filtered_datasets_flexible_match/successful_solutions" \
  --output-dir "./final_qwen2.5-7b-mot-run1-optimized" \
  --deepspeed "deepspeed_zero2_h200.json" \
  --report-to "wandb" --wandb-project "Final-Mixture-of-Thoughts-Finetuning" \
  --num-train-epochs 3 \
  --learning-rate 2e-5 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --warmup-ratio 0.03 \
  --logging-steps 10 2>&1 | tee final_mot_optimized_debug1.txt

--output-dir: Changed to a new directory to track your new experiment.
--deepspeed: Switched to the much faster deepspeed_zero2_h200.json.
--num-train-epochs: Reduced from 8 to 3 to prevent overfitting.
--learning-rate: Lowered from 4e-5 to a safer 2e-5.
--per-device-train-batch-size: Explicitly set to 4 for better GPU utilization.
--gradient-accumulation-steps: Explicitly set to 2 to maintain the same effective batch size.
--logging-steps: Added for more frequent feedback during training.


torchrun --nproc_per_node=8 train_filtered_openai.py \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --dataset-path "/PATH/TO/YOUR/12k_TRAIN_SPLIT" \
  --output-dir "./qwen2-7b-long-context-run1" \
  --deepspeed "deepspeed_zero3_h200_32k.json" \
  --report-to "wandb" --wandb-project "Long-Context-CoT-Finetuning" \
  \
  # -- Key Training Parameters --
  --num-train-epochs 2 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.03 \
  \
  # -- Memory and Batching Strategy --
  --max-seq-length 32768 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  \
  # -- Evaluation and Logging --
  --disable-training-eval false \
  --validation-split-percentage 5 \
  --eval-steps 200 \
  --logging-steps 10 \
  2>&1 | tee long_context_run1.txt




python infer/generate_qwen_vllm_think.py \
  --dataset-name open-r1/codeforces --subset default --split test \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --checkpoint-path  /root/competitive-coding-ai/final_qwen2.5-7b-deepcoder-SINGLE-FILE \
  --batch-size 64 \
  --max-model-len 32768 --max-new-tokens 10240 \
  --tensor-parallel-size 8 --gpu-ids 0,1,2,3,4,5,6,7 \
  --dtype bfloat16 \
  --temperature 0.2 --top-p 0.95 \
  --results-dir /root/competitive-coding-ai/final_sft_model_solutions
