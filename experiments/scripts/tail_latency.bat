#!/bin/bash
# tail_latency.bat - Script for running tail latency experiments

echo "Starting tail_latency experiments..."

# Result path prefix - modify this to your preferred location
RESULT_PREFIX="/data/Projects/data/vectors/experiments"

# Dataset list
datasets=("SIFT1M" "ImageNet" "AgNews" "Glove120" "GooAQ" "Gist")

# Run different index types for each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    # ABQ_SQ index
    ../build/tail_latency 1 "$dataset" "0.095" "${RESULT_PREFIX}/tail_latency/95_A_" "ABQ_SQ" 0
    sleep 30

    # ABQ index
    ../build/tail_latency 1 "$dataset" "0.095" "${RESULT_PREFIX}/tail_latency/95_A_" "ABQ" 0
    sleep 30

    ../build/tail_latency 1 "$dataset" "0.095" "${RESULT_PREFIX}/tail_latency/95_D_" "ABQ_SQ" 1
    sleep 30

    # ABQ index
    ../build/tail_latency 1 "$dataset" "0.095" "${RESULT_PREFIX}/tail_latency/95_D_" "ABQ" 1
    sleep 30

    # PQ index
    ../build/tail_latency 1 "$dataset" "0.095" "${RESULT_PREFIX}/tail_latency/" "PQ" 1
    sleep 30

    # SQ index
    ../build/tail_latency 1 "$dataset" "0.095" "${RESULT_PREFIX}/tail_latency/" "SQ" 1
    sleep 30

    # PQFS index
    ../build/tail_latency 1 "$dataset" "0.095" "${RESULT_PREFIX}/tail_latency/" "PQFS" 1
    sleep 30

    # IVFFLAT index
    ../build/tail_latency 1 "$dataset" "0.095" "${RESULT_PREFIX}/tail_latency/" "FLAT" 1
    sleep 30
done

echo "All tail_latency experiments completed!"