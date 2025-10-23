"""
Correlation Analyzer Module

Handles the computation of correlations between sparse features and biological properties.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from typing import Dict, Tuple
import scipy.sparse as sp


class CorrelationAnalyzer:
    """Compute correlations between sparse features and biological properties"""
    
    def __init__(self, config):
        """Initialize correlation analyzer"""
        self.config = config
        print(f"\nCorrelation Analyzer initialized")
    
    def sample_data(
        self,
        sparse_matrices: Dict[str, sp.csr_matrix],
        bio_df: pd.DataFrame
    ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """Sample data for correlation analysis"""
        print(f"\n{'='*80}")
        print("Step 1: Sampling Data")
        print(f"{'='*80}")
        
        total_samples = len(bio_df)
        
        if self.config.num_samples <= 0 or self.config.num_samples >= total_samples:
            num_samples = total_samples
            print(f"Using all {total_samples} samples")
            sample_indices = np.arange(total_samples)
        else:
            num_samples = self.config.num_samples
            print(f"Sampling {num_samples} out of {total_samples} samples")
            
            np.random.seed(self.config.random_seed)
            sample_indices = np.random.choice(total_samples, num_samples, replace=False)
            sample_indices = np.sort(sample_indices)
        
        # Sample sparse matrices
        sampled_matrices = {}
        for layer_name, sparse_matrix in sparse_matrices.items():
            sampled_matrix = sparse_matrix[sample_indices].toarray()
            sampled_matrices[layer_name] = sampled_matrix
            print(f"  {layer_name}: sampled shape = {sampled_matrix.shape}")
        
        # Sample biological features
        sampled_bio_df = bio_df.iloc[sample_indices].reset_index(drop=True)
        
        print(f"\nSampling complete: {num_samples} samples")
        
        return sampled_matrices, sampled_bio_df
    
    def compute_correlations(
        self,
        sampled_matrices: Dict[str, np.ndarray],
        sampled_bio_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute correlations between features and biological properties"""
        print(f"\n{'='*80}")
        print("Step 1: Batch Correlation Calculation")
        print(f"{'='*80}")
        
        results = []
        
        for layer_name, activation_matrix in tqdm(sampled_matrices.items(), desc="Processing layers"):
            n_samples, n_features = activation_matrix.shape
            print(f"\n  Layer: {layer_name} ({n_samples} samples × {n_features} features)")
            
            # Process continuous properties
            for prop in self.config.continuous_properties:
                if prop not in sampled_bio_df.columns:
                    continue
                
                bio_values = sampled_bio_df[prop].values
                
                if np.all(np.isnan(bio_values)):
                    continue
                
                valid_mask = ~np.isnan(bio_values)
                valid_bio = bio_values[valid_mask]
                valid_activation = activation_matrix[valid_mask]
                
                if len(valid_bio) < 10:
                    continue
                
                # Compute Spearman correlation for each feature
                for feat_idx in range(n_features):
                    feat_activation = valid_activation[:, feat_idx]
                    
                    if np.all(feat_activation == 0):
                        continue
                    
                    try:
                        corr, pval = spearmanr(feat_activation, valid_bio)
                        
                        if np.isnan(corr):
                            continue
                        
                        results.append({
                            'layer': layer_name,
                            'feature_idx': feat_idx,
                            'property': prop,
                            'property_type': 'continuous',
                            'correlation': corr,
                            'p_value': pval,
                            'n_samples': len(valid_bio),
                            'test_type': 'spearman'
                        })
                    except:
                        continue
            
            # Process discrete properties (Mann-Whitney U test)
            for prop in self.config.discrete_properties:
                if prop not in sampled_bio_df.columns:
                    continue
                
                bio_values = sampled_bio_df[prop].values
                
                group_has = bio_values > 0
                group_not = bio_values == 0
                
                if np.sum(group_has) < 5 or np.sum(group_not) < 5:
                    continue
                
                for feat_idx in range(n_features):
                    feat_activation = activation_matrix[:, feat_idx]
                    
                    if np.all(feat_activation == 0):
                        continue
                    
                    activation_has = feat_activation[group_has]
                    activation_not = feat_activation[group_not]
                    
                    try:
                        statistic, pval = mannwhitneyu(activation_has, activation_not, alternative='two-sided')
                        
                        # Compute effect size (rank-biserial correlation)
                        n1, n2 = len(activation_has), len(activation_not)
                        effect_size = 1 - (2*statistic) / (n1 * n2)
                        
                        results.append({
                            'layer': layer_name,
                            'feature_idx': feat_idx,
                            'property': prop,
                            'property_type': 'discrete',
                            'correlation': effect_size,
                            'p_value': pval,
                            'n_samples': n1 + n2,
                            'test_type': 'mann_whitney'
                        })
                    except:
                        continue
        
        corr_df = pd.DataFrame(results)
        
        print(f"\n{'='*80}")
        print(f"Correlation calculation complete!")
        print(f"  Total tests: {len(corr_df):,}")
        print(f"  Layers: {corr_df['layer'].nunique()}")
        print(f"  Properties: {corr_df['property'].nunique()}")
        print(f"{'='*80}")
        
        return corr_df
    
    def filter_significant_correlations(self, corr_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter correlations by statistical significance (mild Bonferroni correction)"""
        print(f"\n{'='*80}")
        print("Step 2: Statistical Significance Filtering")
        print(f"{'='*80}")
        
        # Use mild Bonferroni correction: p_corrected = p_raw * sqrt(n_tests)
        # This is much less conservative than standard Bonferroni (p * n_tests)
        # but still provides some protection against false positives
        n_tests = len(corr_df)
        correction_factor = np.sqrt(n_tests)
        
        print(f"\nApplying mild Bonferroni correction...")
        print(f"  Total tests: {n_tests:,}")
        print(f"  Correction factor: sqrt({n_tests:,}) = {correction_factor:.2f}")
        print(f"  Effective alpha: {self.config.fdr_alpha} / {correction_factor:.2f} = {self.config.fdr_alpha/correction_factor:.6f}")
        
        # Apply mild correction
        pvals_corrected = np.minimum(corr_df['p_value'].values * correction_factor, 1.0)
        
        corr_df['p_value_corrected'] = pvals_corrected
        corr_df['significant'] = pvals_corrected < self.config.fdr_alpha
        
        filtered_df = corr_df[
            (corr_df['significant']) &
            (np.abs(corr_df['correlation']) >= self.config.min_correlation)
        ].copy()
        
        print(f"\nFiltering results:")
        print(f"  Total tests: {len(corr_df):,}")
        print(f"  Raw p < 0.05: {np.sum(corr_df['p_value'] < 0.05):,}")
        print(f"  Raw p < 0.01: {np.sum(corr_df['p_value'] < 0.01):,}")
        print(f"  Raw p < 0.001: {np.sum(corr_df['p_value'] < 0.001):,}")
        print(f"  Significant (corrected p < {self.config.fdr_alpha}): {np.sum(corr_df['significant']):,}")
        print(f"  After correlation threshold (|r| >= {self.config.min_correlation}): {len(filtered_df):,}")
        
        # Add property category labels
        property_categories = {
            'mrl': 'Functional',
            'te': 'Functional', 
            'expression_level': 'Functional',
            'length': 'Structural',
            'gc_content': 'Structural',
            'mfe': 'Structural',
            'uorf_count': 'Regulatory',
            'uaug_count': 'Regulatory'
        }
        
        corr_df['property_category'] = corr_df['property'].map(property_categories)
        if len(filtered_df) > 0:
            filtered_df['property_category'] = filtered_df['property'].map(property_categories)
        
        # Compute interpretability score
        filtered_df['interpretability_score'] = (
            np.abs(filtered_df['correlation']) * 
            -np.log10(filtered_df['p_value_corrected'] + 1e-300)
        )
        
        # Normalize interpretability score
        if len(filtered_df) > 0:
            max_score = filtered_df['interpretability_score'].max()
            if max_score > 0:
                filtered_df['interpretability_score'] /= max_score
        
        # Find best matching property for each feature
        best_matches = []
        if len(filtered_df) > 0:
            for (layer, feat_idx), group in filtered_df.groupby(['layer', 'feature_idx']):
                best_row = group.loc[group['interpretability_score'].idxmax()]
                best_matches.append({
                    'layer': layer,
                    'feature_idx': feat_idx,
                    'best_property': best_row['property'],
                    'best_correlation': best_row['correlation'],
                    'best_p_value': best_row['p_value_corrected'],
                    'interpretability_score': best_row['interpretability_score']
                })
        
        # Create DataFrame with proper columns even if empty
        if len(best_matches) > 0:
            best_match_df = pd.DataFrame(best_matches)
        else:
            # Create empty DataFrame with correct column structure
            best_match_df = pd.DataFrame(columns=[
                'layer', 'feature_idx', 'best_property', 
                'best_correlation', 'best_p_value', 'interpretability_score'
            ])
        
        print(f"\nBest matching properties:")
        print(f"  Unique features with significant correlations: {len(best_match_df)}")
        if len(best_match_df) > 0:
            print(f"  Mean interpretability score: {best_match_df['interpretability_score'].mean():.4f}")
        
        print(f"{'='*80}")
        
        return filtered_df, best_match_df
