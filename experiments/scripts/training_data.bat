#!/bin/bash
# training_data.bat - Script for running training data experiments

echo "Starting training_data experiments..."

# Result path prefix - modify this to your preferred location
RESULT_PREFIX="/data/Projects/data/vectors/experiments"

# Dataset list
datasets=("SIFT1M" "ImageNet" "AgNews" "Glove120" "GooAQ" "Gist")

# Run different index types for each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    # ABQ_SQ index
    echo "Running ABQ_SQ index..."
    ../build/training_data 1 "$dataset" "0.095" "${RESULT_PREFIX}/training_data/" "ABQ_SQ" 1
    sleep 30
    
    # ABQ index
    echo "Running ABQ index..."
    ../build/training_data 1 "$dataset" "0.095" "${RESULT_PREFIX}/training_data/" "ABQ" 1
    sleep 30
    
    # PQ index
    echo "Running PQ index..."
    ../build/training_data 1 "$dataset" "0.095" "${RESULT_PREFIX}/training_data/" "PQ" 1
    sleep 30

    # SQ index
    echo "Running SQ index..."
    ../build/training_data 1 "$dataset" "0.095" "${RESULT_PREFIX}/training_data/" "SQ" 1
    sleep 30

    # PQFS index
    echo "Running PQFS index..."
    ../build/training_data 1 "$dataset" "0.095" "${RESULT_PREFIX}/training_data/" "PQFS" 1
    sleep 30

    # IVFFLAT index
    echo "Running FLAT index..."
    ../build/training_data 1 "$dataset" "0.095" "${RESULT_PREFIX}/training_data/" "FLAT" 1
    sleep 30

    echo "Dataset $dataset completed"
    echo "----------------------------------------"
done

echo "All training_data experiments completed!"