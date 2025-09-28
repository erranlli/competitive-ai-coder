#!/bin/bash
# Batch inference script for consolidated checkpoints

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CHECKPOINTS_DIR="/mnt/data3/all_checkpoints_grad_norm10_16k_wu005_lr5e05_30epoch"
OUTPUT_DIR="/mnt/data3/model_solution_grad_norm10_16k_wu005_lr5e05_30epoch"
MAX_PROBLEMS=64
GPUS_PER_CHECKPOINT=4
MAX_CONCURRENT=2
SORT_ORDER="step"

echo -e "${GREEN}=== Batch Inference for Consolidated Checkpoints ===${NC}"
echo "This script will run inference on all consolidated checkpoints using vLLM"
echo ""

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --checkpoints-dir DIR    Directory with consolidated checkpoints (default: $CHECKPOINTS_DIR)"
    echo "  --output-dir DIR         Output directory for results (default: $OUTPUT_DIR)"
    echo "  --max-problems N         Max problems per checkpoint (default: $MAX_PROBLEMS)"
    echo "  --gpus-per-checkpoint N  GPUs per checkpoint (default: $GPUS_PER_CHECKPOINT)"
    echo "  --max-concurrent N       Max concurrent checkpoints (default: $MAX_CONCURRENT)"
    echo "  --sort-order ORDER       Sort order: step, name, none (default: $SORT_ORDER)"
    echo "  --dry-run               Show what would be processed"
    echo "  --force                 Overwrite existing results"
    echo "  --help                  Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Process all checkpoints with defaults"
    echo "  $0 --dry-run                         # Show what would be processed"
    echo "  $0 --max-problems 32 --force         # Process 32 problems, overwrite existing"
    echo "  $0 --checkpoints-dir /path/to/models # Use different checkpoint directory"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoints-dir)
            CHECKPOINTS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-problems)
            MAX_PROBLEMS="$2"
            shift 2
            ;;
        --gpus-per-checkpoint)
            GPUS_PER_CHECKPOINT="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --sort-order)
            SORT_ORDER="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --force)
            FORCE="--force"
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Validate inputs
if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo -e "${RED}Error: Checkpoints directory does not exist: $CHECKPOINTS_DIR${NC}"
    echo "Please run the consolidation first or specify a different directory with --checkpoints-dir"
    exit 1
fi

# Count checkpoints
CHECKPOINT_COUNT=$(find "$CHECKPOINTS_DIR" -maxdepth 1 -type d -name "*step_*" | wc -l)
if [ "$CHECKPOINT_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No consolidated checkpoints found in $CHECKPOINTS_DIR${NC}"
    echo "Expected directories with names like 'qwen2.5-7b_step_XXX'"
    echo "Please run the consolidation script first."
    exit 1
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  Checkpoints directory: $CHECKPOINTS_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Checkpoints found: $CHECKPOINT_COUNT"
echo "  Max problems per checkpoint: $MAX_PROBLEMS"
echo "  GPUs per checkpoint: $GPUS_PER_CHECKPOINT"
echo "  Max concurrent checkpoints: $MAX_CONCURRENT"
echo "  Total GPU utilization: $((GPUS_PER_CHECKPOINT * MAX_CONCURRENT))/8 GPUs"
echo ""

if [ -n "$DRY_RUN" ]; then
    echo -e "${YELLOW}DRY RUN MODE - No actual inference will be performed${NC}"
    echo ""
fi

# Estimate time
ESTIMATED_TIME_PER_CHECKPOINT=300  # 5 minutes per checkpoint (conservative estimate)
TOTAL_TIME_SEQUENTIAL=$((CHECKPOINT_COUNT * ESTIMATED_TIME_PER_CHECKPOINT))
TOTAL_TIME_PARALLEL=$((TOTAL_TIME_SEQUENTIAL / MAX_CONCURRENT))

echo -e "${YELLOW}Time Estimates:${NC}"
echo "  Sequential processing: ~$((TOTAL_TIME_SEQUENTIAL / 60)) minutes"
echo "  Parallel processing (${MAX_CONCURRENT} concurrent): ~$((TOTAL_TIME_PARALLEL / 60)) minutes"
echo ""

# Ask for confirmation unless dry run
if [ -z "$DRY_RUN" ]; then
    echo -e "${YELLOW}This will process $CHECKPOINT_COUNT checkpoints in parallel.${NC}"
    echo -e "${YELLOW}Each checkpoint will generate results for $MAX_PROBLEMS problems.${NC}"
    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Run the batch inference
echo -e "${GREEN}Starting batch inference...${NC}"
echo ""

# Build command
CMD="python infer/batch_inference_vllm.py \
    --checkpoints-dir \"$CHECKPOINTS_DIR\" \
    --output-dir \"$OUTPUT_DIR\" \
    --max-problems $MAX_PROBLEMS \
    --gpus-per-checkpoint $GPUS_PER_CHECKPOINT \
    --max-concurrent $MAX_CONCURRENT \
    --sort-order $SORT_ORDER \
    --batch-size 64 \
    --continue-on-error"

if [ -n "$DRY_RUN" ]; then
    CMD="$CMD $DRY_RUN"
fi

if [ -n "$FORCE" ]; then
    CMD="$CMD $FORCE"
fi

echo -e "${YELLOW}Running command:${NC}"
echo "$CMD"
echo ""

# Execute the command
eval $CMD
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Batch inference completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Results are available in:${NC}"
    echo "  $OUTPUT_DIR"
    echo ""
    echo -e "${YELLOW}To analyze results, you can:${NC}"
    echo "  1. Check individual checkpoint results in subdirectories"
    echo "  2. Review the summary in $OUTPUT_DIR/batch_inference_results.json"
    echo "  3. Run evaluation scripts on the generated solutions"
else
    echo -e "${RED}❌ Batch inference failed with exit code $EXIT_CODE${NC}"
    echo ""
    echo -e "${YELLOW}Check the error messages above and:${NC}"
    echo "  1. Verify all checkpoints are properly consolidated"
    echo "  2. Check GPU availability and memory"
    echo "  3. Review the detailed results in $OUTPUT_DIR/batch_inference_results.json"
fi

exit $EXIT_CODE
