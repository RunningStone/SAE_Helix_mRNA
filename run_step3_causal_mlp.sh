#!/bin/bash

# Step 3: Causal Feature Analysis with MLP Probe
# This script uses a neural network probe for better prediction performance

# Required parameters
TARGET_FEATURE="mfe"  # GC content (0-1 range)
DATA_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/DATA/transfered_dataset"
STEP1_OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/Outputs/Multi_SAE_quick_test"
OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/Outputs/Multi_SAE_quick_test/step3_causal_mfe"

# Analysis parameters
NUM_SEQUENCE_PAIRS=100
MIN_TARGET_DIFF=1.5  # MFE difference > 1.5 (about 0.4 std)
MAX_LENGTH_RATIO=0.2  # Allow 20% length difference
MAX_EDIT_DISTANCE=0.8  # Relaxed to 0.8 for more pairs (80% edit distance allowed)
TARGET_BLOCKS="0 2 4 6"  # Only use blocks with trained SAE models (Step1 trained: 0,2,4,6)
DEVICE="cpu"  # Change to "cuda" if GPU available

# Probe parameters - USE MLP
PROBE_MODEL_TYPE="mlp"  # Deep MLP: [256, 256, 128]
PROBE_TRAIN_SPLIT=0.8
PROBE_MIN_R2=0.2  # Lowered threshold for train R²
PROBE_R2_METRIC="test"  # Use train R² instead of test R²
# Note: MLP trains for 20 epochs with MSE loss

# Run analysis
python step3_causal_feature.py \
    --target_feature $TARGET_FEATURE \
    --data_dir $DATA_DIR \
    --step1_output_dir $STEP1_OUTPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --num_sequence_pairs $NUM_SEQUENCE_PAIRS \
    --min_target_diff $MIN_TARGET_DIFF \
    --max_length_ratio $MAX_LENGTH_RATIO \
    --max_edit_distance $MAX_EDIT_DISTANCE \
    --target_blocks $TARGET_BLOCKS \
    --batch_size 16 \
    --device $DEVICE \
    --probe_model_type $PROBE_MODEL_TYPE \
    --probe_train_split $PROBE_TRAIN_SPLIT \
    --probe_min_r2 $PROBE_MIN_R2 \
    --probe_r2_metric $PROBE_R2_METRIC \
    --baseline_methods pca random \
    --plot_format png \
    --plot_dpi 300 \
    --random_seed 42
