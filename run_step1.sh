#!/bin/bash
# Run Step 1: SAE Analysis on Real mRNA Database

echo "=========================================="
echo "Starting SAE Analysis Pipeline"
echo "=========================================="
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Configuration
INPUT_DATA_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/DATA/transfered_dataset"
OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/Outputs/Multi_SAE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the analysis
python step1_SAE_analyse_db.py \
    --input_data_dir "$INPUT_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/step1_analysis.log"

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo "Log saved to: $OUTPUT_DIR/step1_analysis.log"
