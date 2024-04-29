#!/bin/bash

# Number of times to repeat each setting
repeat_count=10

# Define arrays of sparsity and max_times values you want to iterate over
sparsities=(1)  # Example sparsity values

for ((i=1; i<=repeat_count; i++)); do
    echo "Iteration $i of $repeat_count"
    # Iterate over each sparsity
    for sparsity in "${sparsities[@]}"; do
        echo "Working on sparsity $sparsity"
        # Call your Python script with the current sparsity and max_time
        python -u "c:\iterativennsimple\scripts\07_sparse_training_neil_iterative_anomaly.py" $sparsity
    done
done