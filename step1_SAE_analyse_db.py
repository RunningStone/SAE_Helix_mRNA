"""
Step 1: SAE Analysis on Real mRNA Database

This script performs complete SAE analysis on the real mRNA dataset:
1. Load mRNA sequences from JSON chunks
2. Extract activations from Helix mRNA model
3. Train SAE models for all transformer blocks
4. Save sparse activation matrices and checkpoints

Dataset: /home/pan/Experiments/EXPs/2025_10_FM_explainability/DATA/transfered_dataset
Output: /home/pan/Experiments/EXPs/2025_10_FM_explainability/Outputs/Multi_SAE
"""

import sys
from pathlib import Path
print(f"[Project Root]: {Path(__file__).parent}")
sys.path.insert(0, str(Path(__file__).parent))

import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from src.pipeline.sae_analyser import SAEAnalyser


def print_section_header(title: str, char: str = "="):
    """Print a formatted section header"""
    print(f"\n{char * 80}")
    print(f"{title}")
    print(f"{char * 80}")


def print_subsection_header(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'-' * 80}")
    print(f"{title}")
    print(f"{'-' * 80}")


def analyze_dataset_statistics(data_dir: Path):
    """
    Analyze and print dataset statistics
    
    Parameters:
    -----------
    data_dir : Path
        Directory containing JSON chunk files
    """
    print_subsection_header("Dataset Statistics")
    
    # Find all JSON chunk files
    json_files = sorted(data_dir.glob('mRNA_dataset_chunk_*.json'))
    
    print(f"Total chunk files: {len(json_files)}")
    
    # Load metadata if available
    metadata_file = data_dir / 'dataset_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    # Analyze first few chunks to get statistics
    total_sequences = 0
    sequence_lengths = []
    sources = set()
    cell_lines = set()
    data_types = set()
    
    print(f"\nAnalyzing sample chunks...")
    sample_chunks = json_files[:10]  # Analyze first 10 chunks
    
    for chunk_file in sample_chunks:
        with open(chunk_file, 'r') as f:
            chunk_data = json.load(f)
        
        total_sequences += len(chunk_data)
        
        for item in chunk_data:
            sequence_lengths.append(len(item['sequence']))
            
            if 'metadata' in item:
                sources.add(item['metadata'].get('source', 'unknown'))
                cell_lines.add(item['metadata'].get('cell_line', 'unknown'))
                data_types.add(item['metadata'].get('data_type', 'unknown'))
    
    # Estimate total sequences
    avg_sequences_per_chunk = total_sequences / len(sample_chunks)
    estimated_total = int(avg_sequences_per_chunk * len(json_files))
    
    print(f"\nSequence Statistics (from {len(sample_chunks)} sample chunks):")
    print(f"  Sequences analyzed: {total_sequences}")
    print(f"  Estimated total sequences: {estimated_total:,}")
    print(f"  Sequence length - Min: {min(sequence_lengths)}")
    print(f"  Sequence length - Max: {max(sequence_lengths)}")
    print(f"  Sequence length - Mean: {np.mean(sequence_lengths):.2f}")
    print(f"  Sequence length - Median: {np.median(sequence_lengths):.2f}")
    
    print(f"\nData Sources: {len(sources)}")
    for source in sorted(sources):
        print(f"  - {source}")
    
    print(f"\nCell Lines: {len(cell_lines)}")
    for cell_line in sorted(cell_lines):
        print(f"  - {cell_line}")
    
    print(f"\nData Types: {len(data_types)}")
    for data_type in sorted(data_types):
        print(f"  - {data_type}")
    
    return {
        'total_chunks': len(json_files),
        'estimated_total_sequences': estimated_total,
        'sequence_length_stats': {
            'min': min(sequence_lengths),
            'max': max(sequence_lengths),
            'mean': float(np.mean(sequence_lengths)),
            'median': float(np.median(sequence_lengths))
        },
        'sources': sorted(list(sources)),
        'cell_lines': sorted(list(cell_lines)),
        'data_types': sorted(list(data_types))
    }


def print_model_info(analyser: SAEAnalyser):
    """Print model information"""
    print_subsection_header("Model Information")
    
    model = analyser.model
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {analyser.model_name}")
    print(f"Device: {analyser.device}")
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"Frozen: {trainable_params == 0}")
    
    # Print layer structure
    print(f"\nModel Structure:")
    layer_count = 0
    for name, module in model.named_modules():
        if 'mixer' in name.lower() and name.endswith('mixer'):
            layer_count += 1
            print(f"  {layer_count}. {name}")
    
    print(f"\nTotal transformer blocks: {layer_count}")


def print_activation_statistics(layer_activations: Dict[str, torch.Tensor]):
    """Print activation statistics"""
    print_subsection_header("Activation Statistics")
    
    print(f"Extracted activations from {len(layer_activations)} layers:\n")
    
    for layer_name, activation in layer_activations.items():
        n_samples, hidden_dim = activation.shape
        
        # Compute statistics
        mean_val = activation.mean().item()
        std_val = activation.std().item()
        min_val = activation.min().item()
        max_val = activation.max().item()
        
        # Compute sparsity (percentage of near-zero values)
        threshold = 1e-6
        near_zero = (torch.abs(activation) < threshold).sum().item()
        sparsity = near_zero / activation.numel()
        
        print(f"Layer: {layer_name}")
        print(f"  Shape: {activation.shape} ({n_samples:,} samples × {hidden_dim} dims)")
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Std: {std_val:.6f}")
        print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
        print(f"  Near-zero ratio: {sparsity:.2%}")
        print()


def print_sae_training_summary(trained_saes: Dict, sae_trainer):
    """Print SAE training summary"""
    print_subsection_header("SAE Training Summary")
    
    print(f"Trained {len(trained_saes)} SAE models:\n")
    
    for layer_name, sae in trained_saes.items():
        config = sae_trainer.configs[layer_name]
        
        print(f"Layer: {layer_name}")
        print(f"  Input dim: {config.d_in}")
        print(f"  Hidden dim: {config.d_hidden}")
        print(f"  Expansion factor: {config.expansion_factor}")
        print(f"  L1 coefficient: {config.l1_coefficient}")
        print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}")
        print()


def print_sparse_activation_summary(saved_files: Dict[str, Path], sparse_activation_dir: Path):
    """Print sparse activation summary"""
    print_subsection_header("Sparse Activation Summary")
    
    import scipy.sparse as sp
    
    print(f"Saved {len(saved_files)} sparse activation matrices:\n")
    
    total_size_bytes = 0
    
    for layer_name, file_path in saved_files.items():
        file_size = file_path.stat().st_size
        total_size_bytes += file_size
        
        # Try to load as scipy sparse first, then as dense
        try:
            sparse_matrix = sp.load_npz(file_path)
            shape = sparse_matrix.shape
            total_elements = sparse_matrix.size
            nonzero_elements = sparse_matrix.nnz
            zero_elements = total_elements - nonzero_elements
            sparsity = zero_elements / total_elements
            storage_type = "scipy.sparse.csr_matrix"
        except:
            # Try loading as dense numpy array
            data = np.load(file_path)
            if 'data' in data:
                matrix = data['data']
            else:
                matrix = data['arr_0']
            shape = matrix.shape
            total_elements = matrix.size
            zero_elements = np.sum(matrix == 0)
            nonzero_elements = total_elements - zero_elements
            sparsity = zero_elements / total_elements
            storage_type = "dense numpy array"
        
        print(f"Layer: {layer_name}")
        print(f"  File: {file_path.name}")
        print(f"  Shape: {shape}")
        print(f"  Total elements: {total_elements:,}")
        print(f"  Zero elements: {zero_elements:,}")
        print(f"  Non-zero elements: {nonzero_elements:,}")
        print(f"  Sparsity (% zeros): {sparsity:.2%}")
        print(f"  Storage type: {storage_type}")
        print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
        print()
    
    print(f"Total storage: {total_size_bytes / 1024 / 1024:.2f} MB")
    
    # Print metadata
    metadata_file = sparse_activation_dir / 'metadata.json'
    if metadata_file.exists():
        print(f"\nMetadata saved to: {metadata_file}")


def save_analysis_summary(
    output_dir: Path,
    dataset_stats: Dict,
    layer_activations: Dict[str, torch.Tensor],
    saved_files: Dict[str, Path],
    start_time: datetime,
    end_time: datetime,
    mapping_file: Path = None
):
    """Save complete analysis summary"""
    
    summary = {
        'analysis_info': {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'duration_formatted': str(end_time - start_time)
        },
        'dataset_statistics': dataset_stats,
        'activation_info': {
            'num_layers': len(layer_activations),
            'layer_names': list(layer_activations.keys()),
            'layer_shapes': {name: list(act.shape) for name, act in layer_activations.items()}
        },
        'sparse_activation_files': {
            name: str(path) for name, path in saved_files.items()
        }
    }
    
    # Add mapping file info if available
    if mapping_file and mapping_file.exists():
        summary['token_to_sequence_mapping'] = str(mapping_file)
    
    summary_file = output_dir / 'analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAnalysis summary saved to: {summary_file}")


def main(input_data_dir: str, output_dir: str, max_chunks: int = None, cache_dir: str = None):
    """
    Main execution function
    
    Parameters:
    -----------
    input_data_dir : str
        Directory containing input JSON data files (required)
    output_dir : str
        Directory for all outputs (required)
    max_chunks : int, optional
        Maximum number of JSON chunk files to process. If None, process all files.
    cache_dir : str, optional
        Directory containing cached activations/embeddings. If None, will compute and save to output_dir.
        If provided, will load from cache_dir and skip model initialization.
    """
    
    start_time = datetime.now()
    
    print_section_header("Step 1: SAE Analysis on Real mRNA Database")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    INPUT_DATA_DIR = Path(input_data_dir)
    OUTPUT_DIR = Path(output_dir)
    CACHE_DIR = Path(cache_dir) if cache_dir else None
    
    # SAE training parameters (按照论文推荐设置)
    NUM_EPOCHS = 1  # 论文推荐3-5个epoch即可
    BATCH_SIZE = 256
    EXPANSION_FACTOR = 2  # 字典扩展比 R=2 (d_hidden = 2 * d_in)
    L1_COEFFICIENT = 8.6e-4  # 稀疏惩罚系数 α (论文推荐范围: 3.2e-4 到 8.6e-4)
    LEARNING_RATE = 1e-3
    
    # Model parameters
    MODEL_NAME = 'helical.models.helix_mrna.HelixmRNA'
    MAX_LENGTH = 150
    MODEL_BATCH_SIZE = 16
    DEVICE = "cpu"#'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Input data directory: {INPUT_DATA_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Cache directory: {CACHE_DIR if CACHE_DIR else 'None (will compute and save)'}")
    print(f"  Max chunks to process: {max_chunks if max_chunks else 'ALL'}")
    print(f"  Device: {DEVICE}")
    print(f"  Model batch size: {MODEL_BATCH_SIZE}")
    print(f"  SAE training epochs: {NUM_EPOCHS}")
    print(f"  SAE batch size: {BATCH_SIZE}")
    print(f"  Expansion factor: {EXPANSION_FACTOR}")
    print(f"  L1 coefficient: {L1_COEFFICIENT}")
    
    # Step 1: Analyze dataset (skip if using cache)
    if CACHE_DIR is None:
        print_section_header("Step 1: Dataset Analysis", "=")
        dataset_stats = analyze_dataset_statistics(INPUT_DATA_DIR)
    else:
        print_section_header("Step 1: Dataset Analysis (Skipped - Using Cache)", "=")
        dataset_stats = {}
    
    # Step 2: Initialize SAE Analyser
    print_section_header("Step 2: Initialize SAE Analyser", "=")
    
    try:
        analyser = SAEAnalyser(
            model_name=MODEL_NAME,
            input_data_dir=INPUT_DATA_DIR,
            output_dir=OUTPUT_DIR,
            device=DEVICE,
            max_length=MAX_LENGTH,
            batch_size=MODEL_BATCH_SIZE,
            cache_dir=CACHE_DIR
        )
        print("✓ SAE Analyser initialized successfully!")
        
        # Print model information (only if model was initialized)
        if CACHE_DIR is None:
            print_model_info(analyser)
        
    except ImportError as e:
        print(f"✗ Error: Could not import helical library: {e}")
        print("Please install helical: pip install helical")
        return
    except Exception as e:
        print(f"✗ Error initializing analyser: {e}")
        raise
    
    # Step 3: Load dataset (skip if using cached activations)
    if analyser.layer_activations is None:
        print_section_header("Step 3: Load Dataset", "=")
        sequences = analyser.load_dataset(max_chunks=max_chunks)
        print(f"✓ Loaded {len(sequences):,} sequences")
        print(f"  Example sequence (first 80 chars): {sequences[0][:80]}...")
        
        # Step 4: Extract activations
        print_section_header("Step 4: Extract Activations", "=")
        layer_activations = analyser.extract_activations(sequences)
        print(f"✓ Extracted activations from {len(layer_activations)} layers")
    else:
        print_section_header("Step 3-4: Using Cached Activations", "=")
        print(f"✓ Skipped dataset loading and activation extraction")
        print(f"✓ Using cached activations from {len(analyser.layer_activations)} layers")
        layer_activations = analyser.layer_activations
    
    # Print activation statistics
    print_activation_statistics(layer_activations)
    
    # Step 5: Train SAE models
    print_section_header("Step 5: Train SAE Models", "=")
    trained_saes = analyser.train_all_saes(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        expansion_factor=EXPANSION_FACTOR,
        l1_coefficient=L1_COEFFICIENT,
        learning_rate=LEARNING_RATE,
        validation_split=0.1,
        log_interval=10
    )
    print(f"✓ Trained {len(trained_saes)} SAE models")
    
    # Print SAE training summary
    print_sae_training_summary(trained_saes, analyser.sae_trainer)
    
    # Step 6: Extract and save sparse activations
    print_section_header("Step 6: Extract Sparse Activations", "=")
    saved_files = analyser.extract_sparse_activations(
        save_format='npz',
        chunk_size=1000
    )
    print(f"✓ Saved {len(saved_files)} sparse activation matrices")
    
    # Print sparse activation summary
    print_sparse_activation_summary(saved_files, analyser.sparse_activation_dir)
    
    # Step 7: Save analysis summary
    print_section_header("Step 7: Save Analysis Summary", "=")
    end_time = datetime.now()
    
    # Get mapping file path
    mapping_file = analyser.sparse_activation_dir / 'token_to_sequence_mapping.json'
    
    save_analysis_summary(
        output_dir=OUTPUT_DIR,
        dataset_stats=dataset_stats,
        layer_activations=layer_activations,
        saved_files=saved_files,
        start_time=start_time,
        end_time=end_time,
        mapping_file=mapping_file
    )
    
    # Final summary
    print_section_header("Analysis Complete!", "=")
    duration = end_time - start_time
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - SAE checkpoints: {analyser.sae_checkpoint_dir}")
    print(f"  - Sparse activations: {analyser.sparse_activation_dir}")
    print(f"  - Token-to-sequence mapping: {mapping_file}")
    print(f"  - Analysis summary: {OUTPUT_DIR / 'analysis_summary.json'}")
    
    print("\n" + "="*80)
    print("✓ Step 1: SAE Analysis completed successfully!")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Step 1: SAE Analysis on Real mRNA Database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First run: compute and save activations
  python step1_SAE_analyse_db.py \\
    --input_data_dir /path/to/data \\
    --output_dir /path/to/output
  
  # Quick test with first 2 files
  python step1_SAE_analyse_db.py \\
    --input_data_dir /path/to/data \\
    --output_dir /path/to/output \\
    --max-chunks 2
  
  # Use cached activations (skip model sampling)
  python step1_SAE_analyse_db.py \\
    --input_data_dir /path/to/data \\
    --output_dir /path/to/new_output \\
    --cache_dir /path/to/previous_output
        """
    )
    
    parser.add_argument(
        '--input_data_dir',
        type=str,
        required=True,
        help='Directory containing input JSON data files (required)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory for all outputs (required)'
    )
    
    parser.add_argument(
        '--max-chunks',
        type=int,
        default=None,
        help='Maximum number of JSON chunk files to process (default: all files)'
    )
    
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='Directory containing cached activations/embeddings. If provided, will load from cache and skip model initialization. If None, will compute and save to output_dir.'
    )
    
    args = parser.parse_args()
    
    try:
        main(
            input_data_dir=args.input_data_dir,
            output_dir=args.output_dir,
            max_chunks=args.max_chunks,
            cache_dir=args.cache_dir
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
