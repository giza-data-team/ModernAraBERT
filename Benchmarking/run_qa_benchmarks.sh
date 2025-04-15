#!/bin/bash

# Check if model name is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_name>"
    echo "  model_name: 'arabert' or 'modernbert'"
    exit 1
fi

MODEL_NAME="$1"

# Validate model name
if [ "$MODEL_NAME" != "arabert" ] && [ "$MODEL_NAME" != "modernbert" ]; then
    echo "Error: model_name must be either 'arabert' or 'modernbert'"
    exit 1
fi

# Set parameters
EPOCHS=50
LEARNING_RATE=3e-5
BATCH_SIZE=8
MAX_LENGTH=512
DOC_STRIDE=128

# Create results directory
RESULTS_DIR="/gpfs/helios/home/abdelrah/ModernBERT/Benchmarking/results/qa"
mkdir -p $RESULTS_DIR

# Timestamp for the run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_LOG="${RESULTS_DIR}/benchmark_run_${MODEL_NAME}_${TIMESTAMP}.log"

echo "Starting QA benchmarking run for ${MODEL_NAME} at $(date)" | tee -a $RUN_LOG
echo "Configuration: epochs=${EPOCHS}, learning_rate=${LEARNING_RATE}, batch_size=${BATCH_SIZE}, max_length=${MAX_LENGTH}, doc_stride=${DOC_STRIDE}" | tee -a $RUN_LOG

# Metrics to benchmark with
METRICS=("rouge")

# Run benchmarks for all metrics
for METRIC in "${METRICS[@]}"; do
    echo "=====================================================" | tee -a $RUN_LOG
    echo "Starting QA benchmark for MODEL=${MODEL_NAME}, METRIC=${METRIC}" | tee -a $RUN_LOG
    echo "Started at: $(date)" | tee -a $RUN_LOG
    
    # Create checkpoint directory for this run
    CHECKPOINT_DIR="${RESULTS_DIR}/${MODEL_NAME}_${METRIC}_${TIMESTAMP}"
    mkdir -p $CHECKPOINT_DIR
    
    # Execute the benchmarking script
    echo "Running: python qa_benchmarking.py --model-name $MODEL_NAME --epochs $EPOCHS --metric $METRIC" | tee -a $RUN_LOG
    
    python qa_benchmarking.py \
    --model-name $MODEL_NAME \
    --epochs $EPOCHS \
    --metric $METRIC
    
    BENCHMARK_STATUS=$?
    
    if [ $BENCHMARK_STATUS -eq 0 ]; then
        echo "QA Benchmark for MODEL=${MODEL_NAME}, METRIC=${METRIC} completed successfully" | tee -a $RUN_LOG
    else
        echo "QA Benchmark for MODEL=${MODEL_NAME}, METRIC=${METRIC} failed with status $BENCHMARK_STATUS" | tee -a $RUN_LOG
    fi
    
    echo "Finished at: $(date)" | tee -a $RUN_LOG
    echo "=====================================================" | tee -a $RUN_LOG
    echo "" | tee -a $RUN_LOG
    
    # Free up GPU memory
    sleep 10
done

echo "All QA benchmarks completed at $(date)" | tee -a $RUN_LOG
echo "Summary of ${MODEL_NAME} QA benchmark runs:" | tee -a $RUN_LOG
echo "See detailed results in the logs directory" | tee -a $RUN_LOG
