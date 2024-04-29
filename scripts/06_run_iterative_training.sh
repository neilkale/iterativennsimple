#!/bin/bash

# Number of times to repeat each setting
repeat_count=5

# Define arrays of sparsity and max_times values you want to iterate over
sparsities=(0.05 0.1 0.2 0.3 0.5)  # Example sparsity values

for ((i=1; i<=repeat_count; i++)); do
    echo "Iteration $i of $repeat_count"
    # Iterate over each sparsity
    for sparsity in "${sparsities[@]}"; do
        echo "Working on sparsity $sparsity"
        # Call your Python script with the current sparsity and max_time
        python -u "c:\iterativennsimple\scripts\06_sparse_training_neil_iterative.py" $sparsity
    done
done