#!/bin/bash
# Quick Test: SAE Analysis with first 2 JSON files only
# This is useful for testing the pipeline before running the full analysis

echo "=========================================="
echo "Quick Test: SAE Analysis (2 files only)"
echo "=========================================="
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Configuration
INPUT_DATA_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/DATA/transfered_dataset"
OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/Outputs/Multi_SAE_quick_test"

# Create output directory for quick test
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Input data: $INPUT_DATA_DIR"
echo "  Processing: First 2 JSON chunk files"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Run the analysis with --max-chunks 2
python step1_SAE_analyse_db.py \
    --input_data_dir "$INPUT_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max-chunks 2 \
    2>&1 | tee "$OUTPUT_DIR/step1_quick_test.log"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Quick Test Complete!"
    echo "=========================================="
    echo "Log saved to: $OUTPUT_DIR/step1_quick_test.log"
    echo ""
    echo "If the quick test succeeded, you can run the full analysis with:"
    echo "  ./run_step1.sh"
else
    echo "✗ Quick Test Failed!"
    echo "=========================================="
    echo "Check the log for errors: $OUTPUT_DIR/step1_quick_test.log"
fi

exit $EXIT_CODE
