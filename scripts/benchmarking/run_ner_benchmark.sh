#!/bin/bash
# Named Entity Recognition Benchmarking Script for ModernAraBERT
# 
# This script runs NER benchmarks on ANERCorp dataset with the specified model.
# The script automatically downloads the dataset from HuggingFace.
#
# Usage:
#   ./scripts/benchmarking/run_ner_benchmark.sh [MODEL_NAME] [OUTPUT_DIR]
#
# Examples:
#   # Run NER benchmark with ModernAraBERT
#   ./scripts/benchmarking/run_ner_benchmark.sh modernbert ./results/ner
#
#   # Run with other models
#   ./scripts/benchmarking/run_ner_benchmark.sh arabert ./results/ner_arabert
#   ./scripts/benchmarking/run_ner_benchmark.sh mbert ./results/ner_mbert
#
#   # Compare multiple models
#   for model in modernbert arabert mbert arabert2 marbert camel; do
#       ./scripts/benchmarking/run_ner_benchmark.sh "$model" "./results/ner_$model"
#   done

set -e  # Exit on error

# Default values
MODEL_NAME="${1:-modernbert}"
OUTPUT_DIR="${2:-./results/ner}"
DATASET_NAME="${3:-asas-ai/ANERCorp}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname $(dirname "$SCRIPT_DIR"))"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ModernAraBERT NER Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME (will be downloaded from HuggingFace)"
echo "Output: $OUTPUT_DIR"
echo -e "${BLUE}========================================${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo -e "${GREEN}Running NER benchmark...${NC}"
echo -e "${BLUE}----------------------------------------${NC}"

# Run the Python benchmarking script with correct arguments
python -m src.benchmarking.ner.ner_benchmark \
    --model "$MODEL_NAME" \
    --dataset "$DATASET_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 16 \
    --epochs 3 \
    --patience 8 \
    --learning-rate 2e-5 \
    --fine-tune head-only \
    --inference-test

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}✅ NER benchmark completed successfully!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    
    # Find the results JSON file (it will have a timestamp in the name)
    RESULTS_FILE=$(find "$OUTPUT_DIR" -name "*_results.json" | head -1)
    
    if [ -n "$RESULTS_FILE" ] && [ -f "$RESULTS_FILE" ]; then
        echo "Results summary:"
        echo "  - JSON: $RESULTS_FILE"
        
        # Display key metrics if jq is available
        if command -v jq &> /dev/null; then
            echo ""
            echo "Key Metrics:"
            jq -r '.results | to_entries[] | "  - \(.key): \(.value)"' "$RESULTS_FILE" 2>/dev/null || true
        fi
    else
        echo "Results files not found in expected location."
        echo "Check the output directory for generated files."
    fi
else
    echo -e "${RED}❌ NER benchmark failed${NC}"
    exit 1
fi

