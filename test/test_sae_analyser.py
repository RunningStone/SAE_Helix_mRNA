"""
Test script for SAE Analyser Pipeline

This script demonstrates how to use the SAEAnalyser to:
1. Load a dataset from JSON chunks
2. Extract activations from mRNA-FM
3. Train SAE models for all blocks
4. Save sparse activation matrices
"""

import sys
from pathlib import Path
print(f"[root]: {Path(__file__).parent.parent}")
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
import numpy as np
import torch
from src.pipeline.sae_analyser import SAEAnalyser


def create_dummy_dataset(output_dir: Path, n_chunks: int = 3, n_samples_per_chunk: int = 50):
    """
    Create dummy JSON dataset for testing
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save JSON chunks
    n_chunks : int
        Number of chunk files to create
    n_samples_per_chunk : int
        Number of samples per chunk
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating dummy dataset:")
    print(f"  Output directory: {output_dir}")
    print(f"  Number of chunks: {n_chunks}")
    print(f"  Samples per chunk: {n_samples_per_chunk}")
    
    bases = ['A', 'C', 'G', 'U']
    
    for chunk_idx in range(n_chunks):
        chunk_data = []
        
        for sample_idx in range(n_samples_per_chunk):
            # Generate random sequence
            seq_length = random.randint(40, 80)
            sequence = ''.join(random.choices(bases, k=seq_length))
            
            # Create data item with full structure
            data_item = {
                'sequence': sequence,
                'annotations': {
                    'functional': {
                        'mrl': random.uniform(1.0, 5.0),
                        'te': random.uniform(0.5, 3.0),
                        'expression_level': random.uniform(1.0, 10.0)
                    },
                    'structural': {
                        'secondary_structure': '.' * len(sequence),  # Dummy structure
                        'mfe': random.uniform(-20.0, -5.0),
                        'gc_content': random.uniform(0.3, 0.7),
                        'length': len(sequence)
                    },
                    'regulatory': {
                        'uorf_count': random.randint(0, 3),
                        'uaug_count': random.randint(0, 2),
                        'codon_usage': {}
                    }
                },
                'metadata': {
                    'source': f'dummy_chunk_{chunk_idx}_sample_{sample_idx}',
                    'cell_line': random.choice(['HEK293', 'K562', 'MCF7']),
                    'data_type': random.choice(['endogenous', 'synthetic'])
                }
            }
            
            chunk_data.append(data_item)
        
        # Save chunk
        chunk_file = output_dir / f'chunk_{chunk_idx:03d}.json'
        with open(chunk_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        
        print(f"  Created: {chunk_file.name} ({len(chunk_data)} samples)")
    
    print(f"✓ Dummy dataset created successfully!")


def test_sae_analyser_pipeline():
    """Test the complete SAE analyser pipeline"""
    
    print("\n" + "="*80)
    print("Testing SAE Analyser Pipeline")
    print("="*80)
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Setup paths
    test_output_dir = Path('./test_outputs/sae_analyser')
    data_dir = test_output_dir / 'dummy_data'
    analysis_output_dir = test_output_dir / 'analysis_results'
    
    # Step 1: Create dummy dataset
    print("\n" + "-"*80)
    print("Step 1: Creating dummy dataset")
    print("-"*80)
    
    create_dummy_dataset(
        output_dir=data_dir,
        n_chunks=2,  # Small dataset for testing
        n_samples_per_chunk=30
    )
    
    # Step 2: Initialize SAE Analyser
    print("\n" + "-"*80)
    print("Step 2: Initializing SAE Analyser")
    print("-"*80)
    
    try:
        analyser = SAEAnalyser(
            model_name='helical.models.helix_mrna.HelixmRNA',
            data_dir=data_dir,
            output_dir=analysis_output_dir,
            device='cpu',  # Use CPU for testing
            max_length=150,
            batch_size=8
        )
        print("✓ SAE Analyser initialized successfully!")
    except ImportError as e:
        print(f"⚠ Warning: Could not import helical library: {e}")
        print("Skipping test - helical library not available")
        return
    
    # Step 3: Run full pipeline
    print("\n" + "-"*80)
    print("Step 3: Running full SAE analysis pipeline")
    print("-"*80)
    
    saved_files = analyser.run_full_pipeline(
        num_epochs=20,  # Small number for testing
        batch_size=128,
        expansion_factor=4,
        l1_coefficient=1e-3,
        save_format='npz'
    )
    
    # Step 4: Verify outputs
    print("\n" + "-"*80)
    print("Step 4: Verifying outputs")
    print("-"*80)
    
    print(f"\nSaved sparse activation files:")
    for layer_name, file_path in saved_files.items():
        file_exists = file_path.exists()
        status = "✓" if file_exists else "✗"
        print(f"  {status} {layer_name}: {file_path}")
    
    # Check metadata
    metadata_file = analysis_output_dir / 'sparse_activations' / 'metadata.json'
    if metadata_file.exists():
        print(f"\n✓ Metadata file created: {metadata_file}")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"  Model: {metadata['model_name']}")
        print(f"  Layers: {metadata['num_layers']}")
    else:
        print(f"\n✗ Metadata file not found: {metadata_file}")
    
    # Check SAE checkpoints
    checkpoint_dir = analysis_output_dir / 'sae_checkpoints'
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.ckpt'))
        print(f"\n✓ SAE checkpoints saved: {len(checkpoints)} files")
    else:
        print(f"\n✗ Checkpoint directory not found: {checkpoint_dir}")
    
    print("\n" + "="*80)
    print("✓ SAE Analyser Pipeline Test Completed!")
    print("="*80)
    
    return {
        'analyser': analyser,
        'saved_files': saved_files,
        'output_dir': analysis_output_dir
    }


def test_individual_steps():
    """Test individual steps of the pipeline"""
    
    print("\n" + "="*80)
    print("Testing Individual Pipeline Steps")
    print("="*80)
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Setup paths
    test_output_dir = Path('./test_outputs/sae_analyser_steps')
    data_dir = test_output_dir / 'dummy_data'
    analysis_output_dir = test_output_dir / 'analysis_results'
    
    # Create dummy dataset
    create_dummy_dataset(
        output_dir=data_dir,
        n_chunks=2,
        n_samples_per_chunk=20
    )
    
    try:
        # Initialize analyser
        analyser = SAEAnalyser(
            model_name='helical.models.helix_mrna.HelixmRNA',
            data_dir=data_dir,
            output_dir=analysis_output_dir,
            device='cpu',
            max_length=150,
            batch_size=8
        )
        
        # Step 1: Load dataset
        print("\n" + "-"*80)
        print("Step 1: Loading dataset")
        print("-"*80)
        sequences = analyser.load_dataset()
        print(f"✓ Loaded {len(sequences)} sequences")
        print(f"  Example sequence: {sequences[0][:50]}...")
        
        # Step 2: Extract activations
        print("\n" + "-"*80)
        print("Step 2: Extracting activations")
        print("-"*80)
        layer_activations = analyser.extract_activations(sequences)
        print(f"✓ Extracted activations from {len(layer_activations)} layers")
        
        # Step 3: Train SAEs
        print("\n" + "-"*80)
        print("Step 3: Training SAE models")
        print("-"*80)
        trained_saes = analyser.train_all_saes(
            num_epochs=10,
            batch_size=128,
            expansion_factor=4,
            l1_coefficient=1e-3
        )
        print(f"✓ Trained {len(trained_saes)} SAE models")
        
        # Step 4: Extract sparse activations
        print("\n" + "-"*80)
        print("Step 4: Extracting sparse activations")
        print("-"*80)
        saved_files = analyser.extract_sparse_activations(save_format='npz')
        print(f"✓ Saved {len(saved_files)} sparse activation files")
        
        print("\n" + "="*80)
        print("✓ Individual Steps Test Completed!")
        print("="*80)
        
        return analyser
        
    except ImportError as e:
        print(f"⚠ Warning: Could not import helical library: {e}")
        print("Skipping test - helical library not available")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SAE Analyser Pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'steps'],
        default='full',
        help='Test mode: full pipeline or individual steps'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        result = test_sae_analyser_pipeline()
    else:
        result = test_individual_steps()
    
    if result:
        print("\n✓ All tests completed successfully!")
