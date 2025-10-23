#!/bin/bash

echo "=========================================="
echo "Starting SAE Analysis with Cached Data"
echo "=========================================="
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Configuration
INPUT_DATA_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/DATA/transfered_dataset"
CACHE_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/Outputs/Multi_SAE_20251021_quick"
OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/Outputs/Multi_SAE_20251021_quick"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Using cached activations from: $CACHE_DIR"
echo "Saving new SAE models to: $OUTPUT_DIR"
echo ""

# Run the analysis with cache
python step1_SAE_analyse_db.py \
    --input_data_dir "$INPUT_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "$CACHE_DIR" \
    2>&1 | tee "$OUTPUT_DIR/step1_analysis_cached.log"

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo "Log saved to: $OUTPUT_DIR/step1_analysis_cached.log"
