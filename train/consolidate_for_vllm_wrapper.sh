#!/bin/bash
# Wrapper script to run consolidation with the correct Python environment

# Find the correct Python executable
PYTHON_EXEC="python"

# Check for rllm environment
if [ -f "/home/g10/anaconda3/envs/rllm/bin/python" ]; then
    PYTHON_EXEC="/home/g10/anaconda3/envs/rllm/bin/python"
    echo "Using rllm environment: $PYTHON_EXEC"
elif command -v python3 &> /dev/null; then
    PYTHON_EXEC="python3"
    echo "Using system python3: $PYTHON_EXEC"
else
    echo "Using default python: $PYTHON_EXEC"
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the consolidation script with all passed arguments
exec "$PYTHON_EXEC" "$SCRIPT_DIR/consolidate_for_vllm.py" "$@"
