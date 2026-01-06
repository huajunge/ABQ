#!/bin/bash
# nlist.bat - Script for running nlist_abq experiments

echo "Starting nlist_abq experiments..."

# Result path prefix - modify this to your preferred location
RESULT_PREFIX="/data/Projects/data/vectors/experiments"

# Dataset list
datasets=("SIFT1M" "ImageNet" "AgNews" "Glove120" "GooAQ" "Gist")

# Run different index types for each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    # ABQ_SQ index
    echo "Running ABQ_SQ index..."
    ../build/nlist_abq 1 "$dataset" "0.095" "${RESULT_PREFIX}/nlist/95_A_" "ABQ_SQ" 0
    sleep 30
    
    # ABQ index
    echo "Running ABQ index..."
    ../build/nlist_abq 1 "$dataset" "0.095" "${RESULT_PREFIX}/nlist/95_A_" "ABQ" 0
    sleep 30
    
    ../build/nlist_abq 1 "$dataset" "0.095" "${RESULT_PREFIX}/nlist/95_D_" "ABQ_SQ" 1
    sleep 30
    
    echo "Running ABQ index..."
    ../build/nlist_abq 1 "$dataset" "0.095" "${RESULT_PREFIX}/nlist/95_D_" "ABQ" 1
    sleep 30
    
    # PQ index
    echo "Running PQ index..."
    ../build/nlist_abq 1 "$dataset" "0.095" "${RESULT_PREFIX}/nlist/95_A_" "PQ" 0
    sleep 30

    # SQ index
    echo "Running SQ index..."
    ../build/nlist_abq 1 "$dataset" "0.095" "${RESULT_PREFIX}/nlist/" "SQ" 0
    sleep 30

    # PQFS index
    echo "Running PQFS index..."
    ../build/nlist_abq 1 "$dataset" "0.095" "${RESULT_PREFIX}/nlist/" "PQFS" 0
    sleep 30

    # IVFFLAT index
    echo "Running FLAT index..."
    ../build/nlist_abq 1 "$dataset" "0.095" "${RESULT_PREFIX}/nlist/" "FLAT" 0
    sleep 30

    echo "Dataset $dataset completed"
    echo "----------------------------------------"
done

echo "All nlist experiments completed!"