#!/bin/bash
# Example consolidation commands for your training setup

set -e  # Exit on error

echo "=== vLLM Model Consolidation Examples ==="
echo "These examples show how to consolidate your trained models for vLLM inference"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_example() {
    echo -e "${GREEN}Example: $1${NC}"
    echo -e "${YELLOW}$2${NC}"
    echo ""
}

# Check if consolidation script exists
if [ ! -f "train/consolidate_for_vllm.py" ]; then
    echo -e "${RED}Error: consolidate_for_vllm.py not found in train/ directory${NC}"
    exit 1
fi

print_example "1. Consolidate latest checkpoint from a training run (safetensors format)" \
"python train/consolidate_for_vllm.py \\
  --input ./ft_runs_grad_norm10_16k_wu005_lr5e05_30epoch/qwen2.5-7b \\
  --output ./models/qwen2.5-7b-final \\
  --dtype bfloat16"

print_example "2. Consolidate specific checkpoint" \
"python train/consolidate_for_vllm.py \\
  --input ./ft_runs_grad_norm10_16k_wu005_lr5e05_30epoch/qwen2.5-7b/checkpoint-196 \\
  --output ./models/qwen2.5-7b-step196 \\
  --dtype bfloat16"

print_example "3. Batch process all your training experiments" \
"python train/batch_consolidate_for_vllm.py \\
  --input-pattern \"./ft_runs_*/qwen2.5-7b\" \\
  --output-dir ./consolidated_models \\
  --dtype bfloat16 \\
  --naming-template \"{run_name}_final\""

print_example "4. Process multiple checkpoints for comparison" \
"python train/batch_consolidate_for_vllm.py \\
  --input-dir ./ft_runs_grad_norm10_16k_wu005_lr5e05_30epoch/qwen2.5-7b \\
  --output-dir ./checkpoint_comparison \\
  --checkpoint-filter \"checkpoint-{50,100,150,196}\" \\
  --naming-template \"qwen25_step_{step}\" \\
  --dtype bfloat16"

print_example "5. Dry run to see what would be processed" \
"python train/batch_consolidate_for_vllm.py \\
  --input-pattern \"./ft_runs_*/qwen2.5-7b\" \\
  --output-dir ./consolidated_models \\
  --dry-run"

echo "=== Usage with vLLM ==="
echo ""
echo -e "${GREEN}After consolidation, use with vLLM (safetensors format supported):${NC}"
echo -e "${YELLOW}python infer/generate_qwen_vllm_think.py \\
  --model-name ./models/qwen2.5-7b-final \\
  --dataset-name open-r1/codeforces --subset default --split test \\
  --batch-size 64 \\
  --max-model-len 32768 --max-new-tokens 32000 \\
  --tensor-parallel-size 8 --gpu-ids 0,1,2,3,4,5,6,7 \\
  --dtype bfloat16 \\
  --temperature 0.1 --top-p 0.95 \\
  --results-dir ./model_solutions_consolidated \\
  --max-problems 64${NC}"

echo ""
echo -e "${GREEN}Note: Models are saved as model.safetensors by default for better performance${NC}"

echo ""
echo "=== Interactive Mode ==="
echo "Run this script with arguments to execute examples:"
echo ""
echo "Available commands:"
echo "  ./train/consolidate_examples.sh latest     - Consolidate latest checkpoint"
echo "  ./train/consolidate_examples.sh specific   - Consolidate specific checkpoint"
echo "  ./train/consolidate_examples.sh batch      - Batch process all experiments"
echo "  ./train/consolidate_examples.sh compare    - Process checkpoints for comparison"
echo "  ./train/consolidate_examples.sh dryrun     - Show what would be processed"

# Handle command line arguments
case "${1:-}" in
    "latest")
        echo -e "${GREEN}Executing: Consolidate latest checkpoint${NC}"
        python train/consolidate_for_vllm.py \
          --input ./ft_runs_grad_norm10_16k_wu005_lr5e05_30epoch/qwen2.5-7b \
          --output ./models/qwen2.5-7b-final \
          --dtype bfloat16
        ;;
    "specific")
        echo -e "${GREEN}Executing: Consolidate specific checkpoint${NC}"
        python train/consolidate_for_vllm.py \
          --input ./ft_runs_grad_norm10_16k_wu005_lr5e05_30epoch/qwen2.5-7b/checkpoint-196 \
          --output ./models/qwen2.5-7b-step196 \
          --dtype bfloat16
        ;;
    "batch")
        echo -e "${GREEN}Executing: Batch process all experiments${NC}"
        python train/batch_consolidate_for_vllm.py \
          --input-pattern "./ft_runs_*/qwen2.5-7b" \
          --output-dir ./consolidated_models \
          --dtype bfloat16 \
          --naming-template "{run_name}_final"
        ;;
    "compare")
        echo -e "${GREEN}Executing: Process checkpoints for comparison${NC}"
        python train/batch_consolidate_for_vllm.py \
          --input-dir ./ft_runs_grad_norm10_16k_wu005_lr5e05_30epoch/qwen2.5-7b \
          --output-dir ./checkpoint_comparison \
          --checkpoint-filter "checkpoint-{50,100,150,196}" \
          --naming-template "qwen25_step_{step}" \
          --dtype bfloat16
        ;;
    "dryrun")
        echo -e "${GREEN}Executing: Dry run${NC}"
        python train/batch_consolidate_for_vllm.py \
          --input-pattern "./ft_runs_*/qwen2.5-7b" \
          --output-dir ./consolidated_models \
          --dry-run
        ;;
    "")
        # No arguments, just show examples
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Use one of: latest, specific, batch, compare, dryrun"
        exit 1
        ;;
esac
