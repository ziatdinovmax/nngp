#!/bin/bash

# Define the path to the Python script
PYTHON_SCRIPT="./active_learning.py"

# List of synthetic functions
FUNCTION_TYPES=("piecewise1" "piecewise2" "piecewise3" "nonstationary1" "nonstationary2" "nonstationary3")

# List of model types
MODEL_TYPES=("GP" "DKL" "BNN" "VIDKL")

# Optional seed for reproducibility
SEED=42

# Loop over each function type
for FUNCTION_TYPE in "${FUNCTION_TYPES[@]}"; do
    # Loop over each model type
    for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
        # Run the Python script with the current function and model type
        echo "Running active learning for function ${FUNCTION_TYPE} with model ${MODEL_TYPE}"
        $PYTHON_SCRIPT $FUNCTION_TYPE $MODEL_TYPE --seed $SEED
        echo "Completed: ${FUNCTION_TYPE} with model ${MODEL_TYPE}"
    done
done

echo "All simulations completed."
