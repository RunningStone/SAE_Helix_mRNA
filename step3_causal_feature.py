"""
Step 3: Causal Feature Analysis with ACDC Algorithm

This script implements a comprehensive causal analysis pipeline to verify
the causal importance of SAE features for specific prediction tasks.

Pipeline:
---------
1. Build prediction probe (regression/classification model)
2. Construct sequence pair dataset
3. Collect activations and sparse codes
4. ACDC feature importance evaluation
5. Cumulative intervention curves
6. Cross-sequence pair aggregation
7. Cross-block comparison
8. Baseline method comparison
9. Result visualization
10. Feature functional analysis

Usage:
------
python step3_causal_feature.py \
    --target_feature mrl \
    --data_dir /path/to/data \
    --step1_output_dir /path/to/step1/output \
    --output_dir /path/to/step3/output \
    --num_sequence_pairs 100 \
    --target_blocks 0 1 2 3
"""

import sys
from pathlib import Path
print(f"[Project Root]: {Path(__file__).parent}")
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
from datetime import datetime
import numpy as np
import torch

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from src.causal import (
    CausalAnalysisConfig,
    CausalDataManager,
    ProbeBuilder,
    SequencePairSelector,
    ActivationCollector,
    ACDCAnalyzer,
    InterventionAnalyzer,
    ResultAggregator,
    BlockComparator,
    BaselineComparator,
    ResultVisualizer,
    FeatureAnalyzer
)


def print_section_header(title: str, char: str = "="):
    """Print a formatted section header"""
    print(f"\n{char * 80}")
    print(f"{title}")
    print(f"{char * 80}")


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Step 3: Causal Feature Analysis with ACDC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python step3_causal_feature.py \\
    --target_feature mrl \\
    --data_dir /path/to/data \\
    --step1_output_dir /path/to/step1/output \\
    --output_dir /path/to/step3/output
  
  # Custom configuration
  python step3_causal_feature.py \\
    --target_feature mrl \\
    --data_dir /path/to/data \\
    --step1_output_dir /path/to/step1/output \\
    --output_dir /path/to/step3/output \\
    --num_sequence_pairs 200 \\
    --min_target_diff 3.0 \\
    --target_blocks 0 1 2 3 \\
    --probe_model_type ridge \\
    --device cuda
  
  # Run specific steps only
  python step3_causal_feature.py \\
    --target_feature mrl \\
    --data_dir /path/to/data \\
    --step1_output_dir /path/to/step1/output \\
    --output_dir /path/to/step3/output \\
    --steps probe pairs  # Only run probe and pair selection
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--target_feature',
        type=str,
        required=True,
        help='Target feature to predict (e.g., mrl, stability)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing original dataset JSON files'
    )
    
    parser.add_argument(
        '--step1_output_dir',
        type=str,
        required=True,
        help='Directory containing Step 1 outputs (SAE models, sparse activations)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for causal analysis results'
    )
    
    # Optional arguments - Task configuration
    parser.add_argument(
        '--task_type',
        type=str,
        default='regression',
        choices=['regression', 'classification'],
        help='Type of prediction task (default: regression)'
    )
    
    # Optional arguments - Probe configuration
    parser.add_argument(
        '--probe_model_type',
        type=str,
        default='ridge',
        choices=['ridge', 'lasso', 'mlp'],
        help='Type of probe model (default: ridge)'
    )
    
    parser.add_argument(
        '--probe_train_split',
        type=float,
        default=0.8,
        help='Train/test split ratio for probe (default: 0.8)'
    )
    
    parser.add_argument(
        '--probe_min_r2',
        type=float,
        default=0.6,
        help='Minimum R² threshold for probe validation (default: 0.6)'
    )
    
    parser.add_argument(
        '--probe_r2_metric',
        type=str,
        default='test',
        choices=['train', 'test'],
        help='Which R² to use for validation: train or test (default: test)'
    )
    
    # Optional arguments - Sequence pair selection
    parser.add_argument(
        '--num_sequence_pairs',
        type=int,
        default=100,
        help='Number of sequence pairs to analyze (default: 100)'
    )
    
    parser.add_argument(
        '--min_target_diff',
        type=float,
        default=2.0,
        help='Minimum target value difference for sequence pairs (default: 2.0)'
    )
    
    parser.add_argument(
        '--max_length_ratio',
        type=float,
        default=0.1,
        help='Maximum length ratio difference (default: 0.1, means 0.9-1.1)'
    )
    
    parser.add_argument(
        '--max_edit_distance',
        type=float,
        default=0.3,
        help='Maximum normalized edit distance (default: 0.3)'
    )
    
    # Optional arguments - Activation collection
    parser.add_argument(
        '--target_blocks',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3],
        help='List of block indices to analyze (default: 0 1 2 3)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for model inference (default: 16)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for computation (default: cuda)'
    )
    
    # Optional arguments - ACDC analysis
    parser.add_argument(
        '--max_features_to_test',
        type=int,
        default=100,
        help='Maximum number of top features to test (default: 100)'
    )
    
    # Optional arguments - Baseline comparison
    parser.add_argument(
        '--baseline_methods',
        type=str,
        nargs='+',
        default=['pca', 'random'],
        choices=['pca', 'ica', 'random'],
        help='Baseline methods to compare (default: pca random)'
    )
    
    # Optional arguments - Visualization
    parser.add_argument(
        '--plot_format',
        type=str,
        default='png',
        choices=['png', 'pdf', 'svg'],
        help='Format for saving plots (default: png)'
    )
    
    parser.add_argument(
        '--plot_dpi',
        type=int,
        default=300,
        help='DPI for rasterized plots (default: 300)'
    )
    
    # Optional arguments - General
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--no_save_intermediate',
        action='store_true',
        help='Do not save intermediate results'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    # Optional arguments - Step selection
    parser.add_argument(
        '--steps',
        type=str,
        nargs='+',
        default=None,
        help='Specific steps to run (default: all steps). Options: probe, pairs, activations, acdc, intervention, aggregation, comparison, baseline, visualization, feature_analysis'
    )
    
    args = parser.parse_args()
    
    # Start timer
    start_time = datetime.now()
    
    print_section_header("Step 3: Causal Feature Analysis with ACDC")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create configuration
    config = CausalAnalysisConfig(
        target_feature=args.target_feature,
        task_type=args.task_type,
        data_dir=args.data_dir,
        step1_output_dir=args.step1_output_dir,
        output_dir=args.output_dir,
        probe_model_type=args.probe_model_type,
        probe_train_split=args.probe_train_split,
        probe_min_r2=args.probe_min_r2,
        probe_r2_metric=args.probe_r2_metric,
        num_sequence_pairs=args.num_sequence_pairs,
        min_target_diff=args.min_target_diff,
        max_length_ratio=args.max_length_ratio,
        max_edit_distance=args.max_edit_distance,
        target_blocks=args.target_blocks,
        batch_size=args.batch_size,
        device=args.device,
        max_features_to_test=args.max_features_to_test,
        baseline_methods=args.baseline_methods,
        plot_format=args.plot_format,
        plot_dpi=args.plot_dpi,
        random_seed=args.random_seed,
        save_intermediate=not args.no_save_intermediate,
        verbose=not args.quiet
    )
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Target feature: {config.target_feature}")
    print(f"  Task type: {config.task_type}")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Step 1 output: {config.step1_output_dir}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Number of sequence pairs: {config.num_sequence_pairs}")
    print(f"  Target blocks: {config.target_blocks}")
    print(f"  Device: {config.device}")
    print(f"  Random seed: {config.random_seed}")
    
    # Save configuration
    config.save(Path(config.output_dir) / 'config.json')
    
    # Initialize data manager
    data_manager = CausalDataManager(config)
    
    # Define pipeline steps
    all_steps = {
        'probe': ProbeBuilder,
        'pairs': SequencePairSelector,
        'activations': ActivationCollector,
        'acdc': ACDCAnalyzer,
        'intervention': InterventionAnalyzer,
        'aggregation': ResultAggregator,
        'comparison': BlockComparator,
        'baseline': BaselineComparator,
        'visualization': ResultVisualizer,
        'feature_analysis': FeatureAnalyzer
    }
    
    # Determine which steps to run
    if args.steps:
        steps_to_run = {k: v for k, v in all_steps.items() if k in args.steps}
    else:
        steps_to_run = all_steps
    
    print(f"\nSteps to run: {list(steps_to_run.keys())}")
    
    # Run pipeline
    results = {}
    
    try:
        for step_name, StepClass in steps_to_run.items():
            print_section_header(f"Running: {step_name}", "-")
            
            step = StepClass(config, data_manager)
            step_results = step.run()
            results[step_name] = step_results
            
            print(f"✓ {step_name} completed successfully!")
        
        # Save final summary
        print_section_header("Saving Final Summary", "-")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        summary = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'duration_formatted': str(duration),
            'configuration': config.to_dict(),
            'steps_completed': list(steps_to_run.keys()),
            'results_summary': {
                step_name: {
                    'status': 'completed',
                    'output_files': list(data_manager.step_dirs[step_name].glob('*'))[:5]  # First 5 files
                }
                for step_name in steps_to_run.keys()
            }
        }
        
        summary_path = Path(config.output_dir) / 'analysis_summary.json'
        with open(summary_path, 'w') as f:
            # Convert Path objects to strings for JSON serialization
            summary_serializable = summary.copy()
            summary_serializable['results_summary'] = {
                k: {**v, 'output_files': [str(p) for p in v['output_files']]}
                for k, v in summary['results_summary'].items()
            }
            json.dump(summary_serializable, f, indent=2)
        
        print(f"✓ Saved analysis summary to: {summary_path}")
        
        # Print final summary
        print_section_header("Analysis Complete!", "=")
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")
        print(f"\nOutputs saved to: {config.output_dir}")
        print(f"  Configuration: {config.output_dir}/config.json")
        print(f"  Summary: {summary_path}")
        
        for step_name in steps_to_run.keys():
            step_dir = data_manager.step_dirs[step_name]
            print(f"  {step_name}: {step_dir}")
        
        print("\n" + "="*80)
        print("✓ Step 3: Causal Feature Analysis completed successfully!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
