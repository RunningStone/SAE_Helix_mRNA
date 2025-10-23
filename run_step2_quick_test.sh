#!/bin/bash
# Quick Test: Correlation Analysis with Step 1 output
# This script runs correlation analysis on the output from step1_quick_test

echo "=========================================="
echo "Quick Test: Correlation Analysis"
echo "=========================================="
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1

# Configuration
DATA_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/DATA/transfered_dataset"
SPARSE_ACTIVATION_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/Outputs/Multi_SAE_quick_test/sparse_activations"
OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_FM_explainability/Outputs/Correlation_Analysis_quick_test"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Sparse activation directory: $SPARSE_ACTIVATION_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of samples: 1000 (default)"
echo ""

# Check if sparse activation directory exists
if [ ! -d "$SPARSE_ACTIVATION_DIR" ]; then
    echo "✗ Error: Sparse activation directory not found: $SPARSE_ACTIVATION_DIR"
    echo "Please run step1_SAE_analyse_db.py first to generate sparse activations."
    exit 1
fi

# Check if sparse activation files exist
SPARSE_FILES=$(find "$SPARSE_ACTIVATION_DIR" -name "*_sparse.npz" | wc -l)
if [ "$SPARSE_FILES" -eq 0 ]; then
    echo "✗ Error: No sparse activation files found in $SPARSE_ACTIVATION_DIR"
    echo "Please complete step1_SAE_analyse_db.py first."
    exit 1
fi

echo "Found $SPARSE_FILES sparse activation files"
echo ""

# Run the correlation analysis
python step2_corr_db.py \
    --data_dir "$DATA_DIR" \
    --sparse_activation_dir "$SPARSE_ACTIVATION_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 1000 \
    --fdr_alpha 0.05 \
    --min_correlation 0.1 \
    --top_k_features 5 \
    2>&1 | tee "$OUTPUT_DIR/step2_quick_test.log"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Quick Test Complete!"
    echo "=========================================="
    echo "Log saved to: $OUTPUT_DIR/step2_quick_test.log"
    echo ""
    echo "Results:"
    echo "  All correlations: $OUTPUT_DIR/all_correlations.csv"
    echo "  Significant correlations: $OUTPUT_DIR/significant_correlations.csv"
    echo "  Best matches: $OUTPUT_DIR/best_matches.csv"
    echo "  Hierarchy plot: $OUTPUT_DIR/hierarchy_analysis.png"
    echo "  Deep validation: $OUTPUT_DIR/deep_validation_results.csv"
else
    echo "✗ Quick Test Failed!"
    echo "=========================================="
    echo "Check the log for errors: $OUTPUT_DIR/step2_quick_test.log"
fi

exit $EXIT_CODE


python step2_corr_db.py --data_dir /home/pan/Experiments/EXPs/2025_10_FM_explainability/DATA/small_data/ --sparse_activation_dir /home/pan/Experiments/EXPs/2025_10_FM_explainability/Outputs/Multi_SAE_quick_test/sparse_activations --output_dir /home/pan/Experiments/EXPs/2025_10_FM_explainability/Outputs/Multi_SAE_quick_test/step2 --num_samples 1000 --fdr_alpha 0.15 --min_correlation 0.05 --top_k_features 30