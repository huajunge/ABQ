#!/bin/bash
# z_abq.bat - Script for running experiments with different p-Percentiles values

echo "Starting p-Percentiles experiments..."

# Result path prefix - modify this to your preferred location
RESULT_PREFIX="/data/Projects/data/vectors/experiments"

# Dataset list
datasets=("SIFT1M" "ImageNet" "AgNews" "Glove120" "GooAQ" "Gist")

# p-Percentiles values
p_values=(0.075 0.080 0.085 0.090 0.095 0.099)

# Run tests for each p value
for p in "${p_values[@]}"; do
    echo "========================================"
    echo "Current p-Percentile: $p"
    echo "========================================"

    # Run different index types for each dataset
    for dataset in "${datasets[@]}"; do
        echo "Processing dataset: $dataset"
        
        echo "Running ABQ index..."
        ../build/nprobe 1 "$dataset" "$p" "${RESULT_PREFIX}/nprobe_z/p_${p}_D_" "ABQ" 1
        sleep 30

        echo "Dataset $dataset completed"
        echo "----------------------------------------"
    done

    echo "p-Percentile $p completed for all datasets"
    echo "========================================"
done

echo "All p-Percentiles experiments completed!"