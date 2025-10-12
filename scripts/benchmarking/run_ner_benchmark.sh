#!/bin/bash
# Named Entity Recognition Benchmarking Script for ModernAraBERT
# 
# This script runs NER benchmarks on ANERCorp dataset with the specified model.
#
# Usage:
#   ./scripts/benchmarking/run_ner_benchmark.sh [MODEL_PATH] [OUTPUT_DIR]
#
# Examples:
#   # Run NER benchmark with ModernAraBERT
#   ./scripts/benchmarking/run_ner_benchmark.sh gizadatateam/ModernAraBERT ./results/ner
#
#   # Run with custom model
#   ./scripts/benchmarking/run_ner_benchmark.sh ./models/my_model ./results/ner_custom
#
#   # Compare multiple models
#   for model in gizadatateam/ModernAraBERT aubmindlab/bert-base-arabertv2 bert-base-multilingual-cased; do
#       ./scripts/benchmarking/run_ner_benchmark.sh "$model" "./results/ner_$(basename $model)"
#   done

set -e  # Exit on error

# Default values
MODEL_PATH="${1:-gizadatateam/ModernAraBERT}"
OUTPUT_DIR="${2:-./results/ner}"
DATA_DIR="${DATA_DIR:-./data/benchmarking/ner/ANERCorp}"

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
echo "Model: $MODEL_PATH"
echo "Dataset: ANERCorp"
echo "Output: $OUTPUT_DIR"
echo "Data directory: $DATA_DIR"
echo -e "${BLUE}========================================${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ Data directory not found: $DATA_DIR${NC}"
    echo "Please download ANERCorp dataset first."
    exit 1
fi

echo ""
echo -e "${GREEN}Running NER benchmark...${NC}"
echo -e "${BLUE}----------------------------------------${NC}"

# Run the Python benchmarking script
python -m src.benchmarking.ner.ner_benchmark \
    --model-name "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 8 \
    --num-epochs 10 \
    --patience 8 \
    --learning-rate 2e-5 \
    --focal-alpha 0.25 \
    --focal-gamma 3.0 \
    --track-memory

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}✅ NER benchmark completed successfully!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    
    if [ -f "$OUTPUT_DIR/results.json" ]; then
        echo "Results summary:"
        echo "  - JSON: $OUTPUT_DIR/results.json"
        
        # Display key metrics if jq is available
        if command -v jq &> /dev/null; then
            echo ""
            echo "Metrics:"
            jq -r '.test_metrics | to_entries[] | "  - \(.key): \(.value)"' "$OUTPUT_DIR/results.json" 2>/dev/null || true
        fi
    fi
else
    echo -e "${RED}❌ NER benchmark failed${NC}"
    exit 1
fi

