#!/bin/bash
# Sentiment Analysis Benchmarking Script for ModernAraBERT
# 
# This script runs sentiment analysis benchmarks on multiple datasets
# (HARD, AJGT, LABR, ASTD) with the specified model.
#
# Usage:
#   ./scripts/benchmarking/run_sa_benchmark.sh [MODEL_PATH] [DATASET] [OUTPUT_DIR]
#
# Examples:
#   # Run all SA benchmarks with ModernAraBERT
#   ./scripts/benchmarking/run_sa_benchmark.sh gizadatateam/ModernAraBERT all ./results/sa
#
#   # Run specific dataset
#   ./scripts/benchmarking/run_sa_benchmark.sh gizadatateam/ModernAraBERT HARD ./results/sa_hard
#
#   # Compare models on all datasets
#   for model in gizadatateam/ModernAraBERT aubmindlab/bert-base-arabertv2 bert-base-multilingual-cased; do
#       ./scripts/benchmarking/run_sa_benchmark.sh "$model" all "./results/sa_$(basename $model)"
#   done

set -e  # Exit on error

# Default values
MODEL_PATH="${1:-gizadatateam/ModernAraBERT}"
DATASET="${2:-all}"
OUTPUT_DIR="${3:-./results/sentiment_analysis}"
DATA_DIR="${DATA_DIR:-./data/benchmarking/sa}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname $(dirname "$SCRIPT_DIR"))"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ModernAraBERT Sentiment Analysis Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "Data directory: $DATA_DIR"
echo -e "${BLUE}========================================${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define datasets
if [ "$DATASET" == "all" ]; then
    DATASETS=("HARD" "AJGT" "LABR" "ASTD")
else
    DATASETS=("$DATASET")
fi

# Run benchmarks
for ds in "${DATASETS[@]}"; do
    echo ""
    echo -e "${GREEN}Running benchmark on $ds dataset...${NC}"
    echo -e "${BLUE}----------------------------------------${NC}"
    
    # Run the Python benchmarking script
    python -m src.benchmarking.sa.sa_benchmark \
        --model-name "$MODEL_PATH" \
        --dataset "$ds" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR/$ds" \
        --batch-size 16 \
        --num-epochs 200 \
        --patience 10 \
        --learning-rate 2e-5 \
        --freeze-encoder \
        --track-memory
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $ds benchmark completed successfully${NC}"
    else
        echo -e "${RED}❌ $ds benchmark failed${NC}"
        exit 1
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ All sentiment analysis benchmarks completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary of results:"
for ds in "${DATASETS[@]}"; do
    if [ -f "$OUTPUT_DIR/$ds/results.json" ]; then
        echo "  - $ds: $OUTPUT_DIR/$ds/results.json"
    fi
done

