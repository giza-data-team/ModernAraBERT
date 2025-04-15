#!/bin/bash

# Set parameters
EPOCHS=200
PATIENCE=10

# Create results directory
RESULTS_DIR="./results"
mkdir -p $RESULTS_DIR

# Timestamp for the run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_LOG="${RESULTS_DIR}/benchmark_run_arabert_${TIMESTAMP}.log"

echo "Starting benchmarking run for AraBERT at $(date)" | tee -a $RUN_LOG
echo "Configuration: epochs=${EPOCHS}, patience=${PATIENCE}, batch_size=${BATCH_SIZE}" | tee -a $RUN_LOG

# Model to benchmark - only AraBERT
MODELS=("modernbert")

# Datasets to benchmark
DATASETS=("hard" "astd" "labr" "ajgt")

# Run benchmarks for all model-dataset combinations
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo "=====================================================" | tee -a $RUN_LOG
        echo "Starting benchmark for MODEL=${MODEL}, DATASET=${DATASET}" | tee -a $RUN_LOG
        echo "Started at: $(date)" | tee -a $RUN_LOG
        
        # Execute the benchmarking script
        echo "Running: python sa_benchmarking.py --model-name $MODEL --dataset $DATASET --epochs $EPOCHS --patience $PATIENCE " | tee -a $RUN_LOG
        
        python sa_benchmarking.py \
        --model-name $MODEL \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --patience $PATIENCE
        
        BENCHMARK_STATUS=$?
        
        if [ $BENCHMARK_STATUS -eq 0 ]; then
            echo "Benchmark for MODEL=${MODEL}, DATASET=${DATASET} completed successfully" | tee -a $RUN_LOG
        else
            echo "Benchmark for MODEL=${MODEL}, DATASET=${DATASET} failed with status $BENCHMARK_STATUS" | tee -a $RUN_LOG
        fi
        
        echo "Finished at: $(date)" | tee -a $RUN_LOG
        echo "=====================================================" | tee -a $RUN_LOG
        echo "" | tee -a $RUN_LOG
        
        # Free up GPU memory
        sleep 10
    done
done

echo "All benchmarks completed at $(date)" | tee -a $RUN_LOG

# Summarize results
echo "Summary of AraBERT benchmark runs:" | tee -a $RUN_LOG
echo "See detailed results in the logs directory" | tee -a $RUN_LOG
