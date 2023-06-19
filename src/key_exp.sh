#!/bin/bash

# Define the list of pruning thresholds
thresholds=(0.001 0.002 0.0025 0.0033 0.005 0.01 0.0125 0.0167 0.025 0.05 0.0667 0.1 0.2 0.5)

# Repeat the loop 10 times
for _ in {1..10}; do
  # Iterate through the thresholds and call the main.py script with the arguments
  for threshold in "${thresholds[@]}"; do
    python main.py -o 'threshold_experiment' --pruning_threshold "$threshold"
  done
done
