#!/bin/bash
# nprobe.bat - Script for running nprobe experiments

echo "Starting nprobe experiments..."

# Result path prefix - modify this to your preferred location
RESULT_PREFIX="/data/Projects/data/vectors/experiments"

# Dataset list
datasets=("SIFT1M" "ImageNet" "AgNews" "Glove120" "GooAQ" "Gist")

# Run different index types for each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    echo "Running ABQ_SQ index..."
    ../build/nprobe 1 "$dataset" "0.095" "${RESULT_PREFIX}/nprobe/95_A_" "ABQ_SQ" 0
    sleep 30

    echo "Running ABQ index..."
    ../build/nprobe 1 "$dataset" "0.095" "${RESULT_PREFIX}/nprobe/95_A_" "ABQ" 0
    sleep 30

    echo "Running ABQ_SQ index..."
    ../build/nprobe 1 "$dataset" "0.095" "${RESULT_PREFIX}/nprobe/95_D_" "ABQ_SQ" 1
    sleep 30

    echo "Running ABQ index..."
    ../build/nprobe 1 "$dataset" "0.095" "${RESULT_PREFIX}/nprobe/95_D_" "ABQ" 1
    sleep 30

    echo "Running ABQ index..."
    ../build/nprobe 1 "$dataset" "0.095" "${RESULT_PREFIX}/nprobe/not_ordered_95_D" "ABQ" 1 0
    sleep 30

    # PQ index
    ../build/nprobe 1 "$dataset" "0.095" "${RESULT_PREFIX}/nprobe/" "PQ" 1
    sleep 30

    # SQ index
    ../build/nprobe 1 "$dataset" "0.095" "${RESULT_PREFIX}/nprobe/" "SQ" 1
    sleep 30

    # PQFS index
    ../build/nprobe 1 "$dataset" "0.095" "${RESULT_PREFIX}/nprobe/" "PQFS" 1
    sleep 30

    # IVFFLAT index
    ../build/nprobe 1 "$dataset" "0.095" "${RESULT_PREFIX}/nprobe/" "FLAT" 1
    sleep 30

    echo "Dataset $dataset completed"
    echo "----------------------------------------"
done

echo "All nprobe experiments completed!"