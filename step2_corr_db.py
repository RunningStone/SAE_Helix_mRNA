"""
Step 2: Correlation Analysis between SAE Features and Biological Properties

This script performs comprehensive correlation analysis to understand which biological
properties are associated with SAE-learned sparse features.

Pipeline:
1. Load biological annotations from JSON files
2. Load sparse activation matrices from Step 1
3. Compute correlations (Spearman for continuous, Mann-Whitney for discrete)
4. Filter by statistical significance (FDR correction)
5. Analyze feature hierarchy across blocks
6. Deep validation of top features

Usage:
------
python step2_corr_db.py \
    --data_dir /path/to/data \
    --sparse_activation_dir /path/to/sparse_activations \
    --output_dir /path/to/output \
    --num_samples 1000
"""

import sys
from pathlib import Path
print(f"[Project Root]: {Path(__file__).parent}")
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
from datetime import datetime

from src.pipeline.corr import (
    CorrConfig,
    CorrelationPipeline
)


def print_section_header(title: str, char: str = "="):
    """Print a formatted section header"""
    print(f"\n{char * 80}")
    print(f"{title}")
    print(f"{char * 80}")


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Step 2: Correlation Analysis between SAE Features and Biological Properties',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (1000 samples)
  python step2_corr_db.py \\
    --data_dir /path/to/data \\
    --sparse_activation_dir /path/to/sparse_activations \\
    --output_dir /path/to/output
  
  # Use all samples
  python step2_corr_db.py \\
    --data_dir /path/to/data \\
    --sparse_activation_dir /path/to/sparse_activations \\
    --output_dir /path/to/output \\
    --num_samples -1
  
  # Custom configuration
  python step2_corr_db.py \\
    --data_dir /path/to/data \\
    --sparse_activation_dir /path/to/sparse_activations \\
    --output_dir /path/to/output \\
    --num_samples 5000 \\
    --fdr_alpha 0.01 \\
    --min_correlation 0.15 \\
    --top_k_features 10
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing JSON data files with biological annotations'
    )
    
    parser.add_argument(
        '--sparse_activation_dir',
        type=str,
        required=True,
        help='Directory containing sparse activation .npz files from Step 1'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for correlation analysis results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Number of samples to use for analysis (-1 for all samples, default: 1000)'
    )
    
    parser.add_argument(
        '--fdr_alpha',
        type=float,
        default=0.05,
        help='FDR control level for multiple testing correction (default: 0.05)'
    )
    
    parser.add_argument(
        '--min_correlation',
        type=float,
        default=0.1,
        help='Minimum absolute correlation threshold (default: 0.1)'
    )
    
    parser.add_argument(
        '--top_k_features',
        type=int,
        default=5,
        help='Number of top features per block for deep validation (default: 5)'
    )
    
    parser.add_argument(
        '--high_activation_percentile',
        type=int,
        default=90,
        help='Percentile for high activation threshold (default: 90)'
    )
    
    parser.add_argument(
        '--low_activation_percentile',
        type=int,
        default=10,
        help='Percentile for low activation threshold (default: 10)'
    )
    
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Start timer
    start_time = datetime.now()
    
    print_section_header("Step 2: SAE Feature-Biology Correlation Analysis")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Sparse activation directory: {args.sparse_activation_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of samples: {args.num_samples if args.num_samples > 0 else 'ALL'}")
    print(f"  FDR alpha: {args.fdr_alpha}")
    print(f"  Min correlation: {args.min_correlation}")
    print(f"  Top K features: {args.top_k_features}")
    print(f"  High activation percentile: {args.high_activation_percentile}")
    print(f"  Low activation percentile: {args.low_activation_percentile}")
    print(f"  Random seed: {args.random_seed}")
    
    # Create configuration
    config = CorrConfig(
        num_samples=args.num_samples,
        random_seed=args.random_seed,
        fdr_alpha=args.fdr_alpha,
        min_correlation=args.min_correlation,
        top_k_features=args.top_k_features,
        high_activation_percentile=args.high_activation_percentile,
        low_activation_percentile=args.low_activation_percentile,
        save_intermediate=True,
        use_gpu=False
    )
    
    # Initialize pipeline
    print_section_header("Initializing Correlation Pipeline")
    
    try:
        pipeline = CorrelationPipeline(
            data_dir=args.data_dir,
            sparse_activation_dir=args.sparse_activation_dir,
            output_dir=args.output_dir,
            config=config
        )
        print("✓ Pipeline initialized successfully!")
        
    except Exception as e:
        print(f"✗ Error initializing pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run pipeline
    print_section_header("Running Correlation Analysis Pipeline")
    
    try:
        results = pipeline.run_full_pipeline()
        print("✓ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"✗ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print summary
    print_section_header("Analysis Summary")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\nTiming:")
    print(f"  Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total duration: {duration}")
    
    print(f"\nResults:")
    print(f"  Total correlations computed: {len(results['corr_df']):,}")
    print(f"  Significant correlations: {len(results['filtered_df']):,}")
    print(f"  Features with best matches: {len(results['best_match_df']):,}")
    print(f"  Blocks analyzed: {len(results['hierarchy_stats'])}")
    
    print(f"\nOutput files:")
    output_dir = Path(args.output_dir)
    print(f"  All correlations: {output_dir / 'all_correlations.csv'}")
    print(f"  Significant correlations: {output_dir / 'significant_correlations.csv'}")
    print(f"  Best matches: {output_dir / 'best_matches.csv'}")
    print(f"  Hierarchy stats: {output_dir / 'hierarchy_stats.json'}")
    print(f"  Hierarchy plot: {output_dir / 'hierarchy_analysis.png'}")
    print(f"  Deep validation: {output_dir / 'deep_validation_results.csv'}")
    print(f"  Validation plots: {output_dir / 'deep_validation_plots.png'}")
    
    # Save summary
    summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'configuration': {
            'num_samples': args.num_samples,
            'fdr_alpha': args.fdr_alpha,
            'min_correlation': args.min_correlation,
            'top_k_features': args.top_k_features,
            'random_seed': args.random_seed
        },
        'results': {
            'total_correlations': len(results['corr_df']),
            'significant_correlations': len(results['filtered_df']),
            'features_with_best_matches': len(results['best_match_df']),
            'blocks_analyzed': len(results['hierarchy_stats'])
        }
    }
    
    summary_path = output_dir / 'analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Analysis summary: {summary_path}")
    
    print_section_header("✓ Step 2: Correlation Analysis Complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
