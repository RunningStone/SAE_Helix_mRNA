"""
Correlation Analysis Pipeline for SAE Sparse Features and Biological Properties

This module implements a comprehensive pipeline to analyze the relationship between
SAE-learned sparse features and biological properties of mRNA sequences.

Pipeline Steps:
1. Batch correlation calculation (Spearman correlation, Mann-Whitney U test)
2. Statistical significance filtering (FDR correction)
3. Cross-block feature hierarchy analysis
4. High-score feature deep validation

Author: Auto-generated for SAE interpretability analysis
Date: 2025
"""

import json
import numpy as np
import torch
import scipy.sparse as sp
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CorrConfig:
    """Configuration for correlation analysis"""
    
    # Sampling configuration
    num_samples: int = 1000  # Number of samples to use (-1 for all)
    random_seed: int = 42
    
    # Statistical testing (using mild Bonferroni correction: p * sqrt(n_tests))
    fdr_alpha: float = 0.05  # Significance threshold for corrected p-values
    min_correlation: float = 0.05  # Minimum correlation threshold (lowered from 0.1)
    
    # Feature selection
    top_k_features: int = 5  # Top K features per block for deep validation
    high_activation_percentile: int = 90  # Percentile for high activation
    low_activation_percentile: int = 10  # Percentile for low activation
    
    # Biological property categories
    continuous_properties: List[str] = field(default_factory=lambda: [
        'mrl', 'te', 'expression_level', 'mfe', 'gc_content', 'length'
    ])
    discrete_properties: List[str] = field(default_factory=lambda: [
        'uorf_count', 'uaug_count'
    ])
    
    # Output configuration
    save_intermediate: bool = True
    use_gpu: bool = False  # GPU not needed for correlation computation


class BiologicalDataLoader:
    """Load and organize biological annotations from JSON files"""
    
    def __init__(self, data_dir: Union[str, Path], max_chunks: int = None):
        self.data_dir = Path(data_dir)
        self.max_chunks = max_chunks
        self.annotations = []
        self.sequences = []
        
    def load_data(self) -> Tuple[List[str], List[Dict]]:
        """Load sequences and annotations from JSON files"""
        print(f"\nLoading biological data from: {self.data_dir}")
        
        json_files = sorted(self.data_dir.glob('mRNA_dataset_chunk_*.json'))
        
        if self.max_chunks:
            json_files = json_files[:self.max_chunks]
        
        print(f"Found {len(json_files)} JSON chunk files")
        
        sequences = []
        annotations = []
        
        for json_file in tqdm(json_files, desc="Loading JSON chunks"):
            with open(json_file, 'r') as f:
                chunk_data = json.load(f)
            
            for item in chunk_data:
                sequences.append(item['sequence'])
                annotations.append(item.get('annotations', {}))
        
        print(f"Loaded {len(sequences)} sequences with annotations")
        
        self.sequences = sequences
        self.annotations = annotations
        
        return sequences, annotations
    
    def extract_biological_features(
        self, 
        annotations: List[Dict],
        config: CorrConfig
    ) -> pd.DataFrame:
        """Extract biological features into a DataFrame"""
        print("\nExtracting biological features...")
        
        bio_data = []
        
        for idx, annot in enumerate(tqdm(annotations, desc="Processing annotations")):
            row = {'sample_idx': idx}
            
            # Functional properties
            functional = annot.get('functional', {})
            row['mrl'] = functional.get('mrl', np.nan)
            row['te'] = functional.get('te', np.nan)
            row['expression_level'] = functional.get('expression_level', np.nan)
            
            # Structural properties
            structural = annot.get('structural', {})
            row['length'] = structural.get('length', np.nan)
            row['gc_content'] = structural.get('gc_content', np.nan)
            row['mfe'] = structural.get('mfe', np.nan)
            
            # Regulatory properties
            regulatory = annot.get('regulatory', {})
            row['uorf_count'] = regulatory.get('uorf_count', 0)
            row['uaug_count'] = regulatory.get('uaug_count', 0)
            
            bio_data.append(row)
        
        bio_df = pd.DataFrame(bio_data)
        
        print(f"Extracted {len(bio_df)} samples with {len(bio_df.columns)-1} features")
        print(f"\nFeature statistics:")
        print(bio_df.describe())
        
        return bio_df


class SparseActivationLoader:
    """Load sparse activation matrices from SAE outputs"""
    
    def __init__(self, sparse_activation_dir: Union[str, Path]):
        self.sparse_activation_dir = Path(sparse_activation_dir)
        self.sparse_matrices = {}
        self.layer_names = []
        
    def load_sparse_activations(self) -> Dict[str, sp.csr_matrix]:
        """Load all sparse activation matrices"""
        print(f"\nLoading sparse activations from: {self.sparse_activation_dir}")
        
        npz_files = sorted(self.sparse_activation_dir.glob('*_sparse.npz'))
        
        if not npz_files:
            raise FileNotFoundError(f"No sparse activation files found in {self.sparse_activation_dir}")
        
        print(f"Found {len(npz_files)} sparse activation files")
        
        sparse_matrices = {}
        
        for npz_file in npz_files:
            layer_name = npz_file.stem.replace('_sparse', '')
            
            # Try to load as scipy sparse first, then as dense
            try:
                sparse_matrix = sp.load_npz(npz_file)
                
                # 计算实际稀疏度：需要转换为dense来准确计算（只对小样本）
                # 对于大矩阵，只采样前1000行
                if sparse_matrix.shape[0] > 1000:
                    sample_dense = sparse_matrix[:1000].toarray()
                    sample_zeros = np.sum(sample_dense == 0)
                    sparsity = sample_zeros / sample_dense.size
                else:
                    dense_matrix = sparse_matrix.toarray()
                    zero_elements = np.sum(dense_matrix == 0)
                    sparsity = zero_elements / dense_matrix.size
                
                storage_type = "sparse"
            except:
                # Try loading as dense numpy array
                data = np.load(npz_file)
                if 'data' in data:
                    matrix = data['data']
                else:
                    matrix = data['arr_0']
                sparse_matrix = sp.csr_matrix(matrix)  # Convert to sparse for consistency
                
                # 计算实际稀疏度
                if matrix.size > 1e6:  # 如果太大，采样
                    sample = matrix.flat[:1000000]
                    zero_elements = np.sum(sample == 0)
                    sparsity = zero_elements / len(sample)
                else:
                    zero_elements = np.sum(matrix == 0)
                    sparsity = zero_elements / matrix.size
                
                storage_type = "dense"
            
            sparse_matrices[layer_name] = sparse_matrix
            print(f"  {layer_name}: shape={sparse_matrix.shape}, sparsity={sparsity:.2%} ({storage_type})")
        
        self.sparse_matrices = sparse_matrices
        self.layer_names = list(sparse_matrices.keys())
        
        return sparse_matrices


# Import additional classes from separate files
from .corr_analyzer import CorrelationAnalyzer
from .corr_hierarchy import HierarchyAnalyzer
from .corr_validator import DeepValidator


class CorrelationPipeline:
    """Main pipeline for correlation analysis"""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        sparse_activation_dir: Union[str, Path],
        output_dir: Union[str, Path],
        config: Optional[CorrConfig] = None
    ):
        """
        Initialize correlation pipeline
        
        Parameters:
        -----------
        data_dir : str or Path
            Directory containing JSON data files
        sparse_activation_dir : str or Path
            Directory containing sparse activation .npz files
        output_dir : str or Path
            Output directory for results
        config : CorrConfig, optional
            Configuration object
        """
        self.data_dir = Path(data_dir)
        self.sparse_activation_dir = Path(sparse_activation_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config if config else CorrConfig()
        
        # Initialize components
        self.bio_loader = BiologicalDataLoader(data_dir, max_chunks=self.config.num_samples // 5000 if self.config.num_samples > 0 else None)
        self.sparse_loader = SparseActivationLoader(sparse_activation_dir)
        self.corr_analyzer = CorrelationAnalyzer(self.config)
        self.hierarchy_analyzer = HierarchyAnalyzer(self.config)
        self.deep_validator = DeepValidator(self.config)
        
    def run_full_pipeline(self) -> Dict:
        """Run the complete correlation analysis pipeline"""
        print(f"\n{'='*80}")
        print("SAE Correlation Analysis Pipeline")
        print(f"{'='*80}")
        
        # Load data
        sequences, annotations = self.bio_loader.load_data()
        bio_df = self.bio_loader.extract_biological_features(annotations, self.config)
        
        # Load sparse activations
        sparse_matrices = self.sparse_loader.load_sparse_activations()
        
        # Sample data
        sampled_matrices, sampled_bio_df = self.corr_analyzer.sample_data(sparse_matrices, bio_df)
        
        # Compute correlations
        corr_df = self.corr_analyzer.compute_correlations(sampled_matrices, sampled_bio_df)
        
        # Filter significant correlations
        filtered_df, best_match_df = self.corr_analyzer.filter_significant_correlations(corr_df)
        
        # Analyze hierarchy
        hierarchy_stats = self.hierarchy_analyzer.analyze_block_hierarchy(filtered_df, self.output_dir)
        
        # Deep validation
        sample_indices = np.arange(len(sampled_bio_df))
        validation_results = self.deep_validator.validate_top_features(
            best_match_df, sampled_matrices, sampled_bio_df, 
            sequences, sample_indices, self.output_dir
        )
        
        # Save all results
        self._save_results(corr_df, filtered_df, best_match_df, hierarchy_stats, validation_results)
        
        print(f"\n{'='*80}")
        print("Pipeline Complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}")
        
        return {
            'corr_df': corr_df,
            'filtered_df': filtered_df,
            'best_match_df': best_match_df,
            'hierarchy_stats': hierarchy_stats,
            'validation_results': validation_results
        }
    
    def _save_results(self, corr_df, filtered_df, best_match_df, hierarchy_stats, validation_results):
        """Save all results to files"""
        print("\nSaving results...")
        
        # Save correlation results
        corr_df.to_csv(self.output_dir / 'all_correlations.csv', index=False)
        filtered_df.to_csv(self.output_dir / 'significant_correlations.csv', index=False)
        best_match_df.to_csv(self.output_dir / 'best_matches.csv', index=False)
        
        # Save hierarchy stats
        with open(self.output_dir / 'hierarchy_stats.json', 'w') as f:
            json.dump(hierarchy_stats, f, indent=2)
        
        print(f"  All results saved to: {self.output_dir}")
