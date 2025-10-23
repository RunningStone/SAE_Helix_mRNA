#!/bin/bash

# Step 3: Causal Feature Analysis with ACDC
# This script runs the complete causal analysis pipeline

# Configuration
TARGET_FEATURE="gc_content"  # GC content (0-1 range)
DATA_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/DATA/transfered_dataset"
STEP1_OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/Outputs/Multi_SAE_quick_test"
OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/Outputs/Multi_SAE_quick_test/step3_causal_gc_content"

# Analysis parameters
NUM_SEQUENCE_PAIRS=100
MIN_TARGET_DIFF=0.1  # GC content difference > 0.1 (e.g., 0.4 vs 0.5)
MAX_LENGTH_RATIO=0.1  # Keep similar length
MAX_EDIT_DISTANCE=0.3
TARGET_BLOCKS="0 2"  # Only use blocks with trained SAE models (Step1 trained: 0,2,4,6)
DEVICE="cpu"  # Change to "cuda" if GPU available

# Probe parameters
PROBE_MODEL_TYPE="ridge"
PROBE_TRAIN_SPLIT=0.8
PROBE_MIN_R2=0.2  # Lowered threshold for train R²
PROBE_R2_METRIC="train"  # Use train R² instead of test R²

# Run analysis
python step3_causal_feature.py \
    --target_feature ${TARGET_FEATURE} \
    --data_dir ${DATA_DIR} \
    --step1_output_dir ${STEP1_OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_sequence_pairs ${NUM_SEQUENCE_PAIRS} \
    --min_target_diff ${MIN_TARGET_DIFF} \
    --max_length_ratio ${MAX_LENGTH_RATIO} \
    --max_edit_distance ${MAX_EDIT_DISTANCE} \
    --target_blocks ${TARGET_BLOCKS} \
    --device ${DEVICE} \
    --probe_model_type ${PROBE_MODEL_TYPE} \
    --probe_train_split ${PROBE_TRAIN_SPLIT} \
    --probe_min_r2 ${PROBE_MIN_R2} \
    --probe_r2_metric ${PROBE_R2_METRIC} \
    --baseline_methods pca random \
    --plot_format png \
    --plot_dpi 300 \
    --random_seed 42

echo ""
echo "Causal analysis completed!"
echo "Results saved to: ${OUTPUT_DIR}"
