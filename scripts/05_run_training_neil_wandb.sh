#!/bin/bash

# Number of times to repeat each setting
repeat_count=5

# Define arrays of sparsity and max_times values you want to iterate over
sparsities=(0.1 0.2 0.3 0.5)  # Example sparsity values
max_times=(60 300 600)       # Example max_times values

for ((i=1; i<=repeat_count; i++)); do
    echo "Iteration $i of $repeat_count"

    # Iterate over each sparsity
    for sparsity in "${sparsities[@]}"; do
        # Iterate over each max_time
        for max_time in "${max_times[@]}"; do
            echo "Working on sparsity $sparsity and max_time $max_time"
            # Call your Python script with the current sparsity and max_time
            python -u "c:\iterativennsimple\scripts\sparse_training_neil_wandb.py" $sparsity 100000 $max_time 100 0.001 25
        done
    done
done