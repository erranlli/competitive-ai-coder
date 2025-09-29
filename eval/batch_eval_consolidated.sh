#!/bin/bash
# Batch evaluation script for consolidated checkpoint solutions

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
SOLUTIONS_DIR="/mnt/data3/model_solution_grad_norm10_16k_wu005_lr5e05_30epoch"
OUTPUT_DIR="/mnt/data3/eval_results_grad_norm10_16k_wu005_lr5e05_30epoch"
MAX_WORKERS=4
MAX_GENERATED_TESTS=0
GENERATED_TESTS_WORKERS=16
ENDPOINT="http://localhost:2000"
TIMEOUT=300

echo -e "${GREEN}=== Batch Evaluation for Checkpoint Solutions ===${NC}"
echo "This script will evaluate solutions from all checkpoints using the piston checker"
echo ""

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --solutions-dir DIR       Directory with checkpoint solutions (default: $SOLUTIONS_DIR)"
    echo "  --output-dir DIR          Output directory for evaluation results (default: $OUTPUT_DIR)"
    echo "  --max-workers N           Max parallel evaluation workers (default: $MAX_WORKERS)"
    echo "  --max-generated-tests N   Max generated tests per problem (default: $MAX_GENERATED_TESTS)"
    echo "  --generated-tests-workers N  Workers for generated tests (default: $GENERATED_TESTS_WORKERS)"
    echo "  --endpoint URL            Evaluation endpoint (default: $ENDPOINT)"
    echo "  --timeout N               Timeout per evaluation in seconds (default: $TIMEOUT)"
    echo "  --dry-run                 Show what would be processed"
    echo "  --force                   Overwrite existing results"
    echo "  --help                    Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Evaluate all solutions with defaults"
    echo "  $0 --dry-run                         # Show what would be processed"
    echo "  $0 --max-generated-tests 5 --force   # Generate 5 tests per problem, overwrite existing"
    echo "  $0 --solutions-dir /path/to/solutions # Use different solutions directory"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --solutions-dir)
            SOLUTIONS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --max-generated-tests)
            MAX_GENERATED_TESTS="$2"
            shift 2
            ;;
        --generated-tests-workers)
            GENERATED_TESTS_WORKERS="$2"
            shift 2
            ;;
        --endpoint)
            ENDPOINT="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
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
if [ ! -d "$SOLUTIONS_DIR" ]; then
    echo -e "${RED}Error: Solutions directory does not exist: $SOLUTIONS_DIR${NC}"
    echo "Please run the batch inference first or specify a different directory with --solutions-dir"
    exit 1
fi

# Count solution directories
SOLUTION_COUNT=$(find "$SOLUTIONS_DIR" -maxdepth 1 -type d -name "checkpoint-*" | wc -l)
if [ "$SOLUTION_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No checkpoint solution directories found in $SOLUTIONS_DIR${NC}"
    echo "Expected directories with names like 'checkpoint-49', 'checkpoint-98', etc."
    echo "Please run the batch inference script first."
    exit 1
fi

# Count total solution files
TOTAL_JSONL_FILES=$(find "$SOLUTIONS_DIR" -name "*.jsonl" | wc -l)
if [ "$TOTAL_JSONL_FILES" -eq 0 ]; then
    echo -e "${RED}Error: No JSONL solution files found in $SOLUTIONS_DIR${NC}"
    echo "Please ensure the inference script generated solution files."
    exit 1
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  Solutions directory: $SOLUTIONS_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Checkpoint directories: $SOLUTION_COUNT"
echo "  Total solution files: $TOTAL_JSONL_FILES"
echo "  Max workers: $MAX_WORKERS"
echo "  Max generated tests: $MAX_GENERATED_TESTS"
echo "  Generated tests workers: $GENERATED_TESTS_WORKERS"
echo "  Endpoint: $ENDPOINT"
echo "  Timeout per evaluation: ${TIMEOUT}s"
echo ""

if [ -n "$DRY_RUN" ]; then
    echo -e "${YELLOW}DRY RUN MODE - No actual evaluation will be performed${NC}"
    echo ""
fi

# Check if evaluation endpoint is available
if [ -z "$DRY_RUN" ]; then
    echo -e "${YELLOW}Checking evaluation endpoint availability...${NC}"
    if ! curl -s --connect-timeout 5 "$ENDPOINT" > /dev/null 2>&1; then
        echo -e "${RED}Warning: Cannot connect to evaluation endpoint $ENDPOINT${NC}"
        echo "Please ensure the piston checker service is running on port 2000"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
    else
        echo -e "${GREEN}✅ Evaluation endpoint is available${NC}"
    fi
    echo ""
fi

# Estimate time
ESTIMATED_TIME_PER_CHECKPOINT=60  # 1 minute per checkpoint (conservative estimate)
TOTAL_TIME_SEQUENTIAL=$((SOLUTION_COUNT * ESTIMATED_TIME_PER_CHECKPOINT))
TOTAL_TIME_PARALLEL=$((TOTAL_TIME_SEQUENTIAL / MAX_WORKERS))

echo -e "${YELLOW}Time Estimates:${NC}"
echo "  Sequential processing: ~$((TOTAL_TIME_SEQUENTIAL / 60)) minutes"
echo "  Parallel processing (${MAX_WORKERS} workers): ~$((TOTAL_TIME_PARALLEL / 60)) minutes"
echo ""

# Ask for confirmation unless dry run
if [ -z "$DRY_RUN" ]; then
    echo -e "${YELLOW}This will evaluate solutions from $SOLUTION_COUNT checkpoints in parallel.${NC}"
    echo -e "${YELLOW}Each checkpoint will be evaluated for $TOTAL_JSONL_FILES total solution files.${NC}"
    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Run the batch evaluation
echo -e "${GREEN}Starting batch evaluation...${NC}"
echo ""

# Build command
CMD="python eval/batch_eval_solutions.py \
    --solutions-dir \"$SOLUTIONS_DIR\" \
    --output-dir \"$OUTPUT_DIR\" \
    --max-workers $MAX_WORKERS \
    --max-generated-tests $MAX_GENERATED_TESTS \
    --generated-tests-workers $GENERATED_TESTS_WORKERS \
    --endpoint \"$ENDPOINT\" \
    --timeout $TIMEOUT \
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
    echo -e "${GREEN}✅ Batch evaluation completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Results are available in:${NC}"
    echo "  $OUTPUT_DIR"
    echo ""
    echo -e "${YELLOW}To analyze results, you can:${NC}"
    echo "  1. Check individual checkpoint evaluation results in subdirectories"
    echo "  2. Review the summary in $OUTPUT_DIR/batch_eval_results.json"
    echo "  3. Compare performance across different checkpoints"
    echo ""
    echo -e "${YELLOW}Example result files:${NC}"
    echo "  $OUTPUT_DIR/eval_checkpoint_49/eval_*.txt"
    echo "  $OUTPUT_DIR/eval_checkpoint_98/eval_*.txt"
    echo "  ..."
else
    echo -e "${RED}❌ Batch evaluation failed with exit code $EXIT_CODE${NC}"
    echo ""
    echo -e "${YELLOW}Check the error messages above and:${NC}"
    echo "  1. Verify all solution files are properly formatted"
    echo "  2. Check that the evaluation endpoint is running"
    echo "  3. Review the detailed results in $OUTPUT_DIR/batch_eval_results.json"
fi

exit $EXIT_CODE
