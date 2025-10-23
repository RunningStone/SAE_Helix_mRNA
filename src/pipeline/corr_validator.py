"""
Deep Validator Module

Performs deep validation of high-score features.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


class DeepValidator:
    """Deep validation of high-score features"""
    
    def __init__(self, config):
        """Initialize deep validator"""
        self.config = config
    
    def validate_top_features(
        self,
        best_match_df: pd.DataFrame,
        sampled_matrices: Dict[str, np.ndarray],
        sampled_bio_df: pd.DataFrame,
        sequences: List[str],
        sample_indices: np.ndarray,
        output_dir: Path
    ) -> Dict:
        """Validate top features by comparing high vs low activation sequences"""
        print(f"\n{'='*80}")
        print("Step 4: High-Score Feature Deep Validation")
        print(f"{'='*80}")
        
        validation_results = {}
        
        # Check if best_match_df is empty or missing required columns
        if len(best_match_df) == 0:
            print("⚠ No features to validate (best_match_df is empty)")
            print(f"{'='*80}")
            return validation_results
        
        if 'layer' not in best_match_df.columns:
            print("⚠ Error: 'layer' column not found in best_match_df")
            print(f"  Available columns: {list(best_match_df.columns)}")
            print(f"{'='*80}")
            return validation_results
        
        # Select top K features per block
        for layer in sorted(best_match_df['layer'].unique()):
            layer_df = best_match_df[best_match_df['layer'] == layer]
            
            # Sort by interpretability score
            top_features = layer_df.nlargest(self.config.top_k_features, 'interpretability_score')
            
            print(f"\n  Layer: {layer}")
            print(f"  Analyzing top {len(top_features)} features...")
            
            layer_results = []
            
            for _, row in top_features.iterrows():
                feat_idx = row['feature_idx']
                best_prop = row['best_property']
                
                # Get activation values for this feature
                activation = sampled_matrices[layer][:, feat_idx]
                
                # Get high and low activation sequences
                nonzero_activation = activation[activation > 0]
                if len(nonzero_activation) < 20:
                    continue
                
                high_threshold = np.percentile(nonzero_activation, 
                                              self.config.high_activation_percentile)
                low_threshold = np.percentile(nonzero_activation, 
                                             self.config.low_activation_percentile)
                
                high_mask = activation >= high_threshold
                low_mask = activation <= low_threshold
                
                # Get biological property values
                bio_values = sampled_bio_df[best_prop].values
                
                # Remove NaN
                valid_high = high_mask & ~np.isnan(bio_values)
                valid_low = low_mask & ~np.isnan(bio_values)
                
                if np.sum(valid_high) < 10 or np.sum(valid_low) < 10:
                    continue
                
                high_bio = bio_values[valid_high]
                low_bio = bio_values[valid_low]
                
                # Statistical test
                if row['best_property'] in self.config.continuous_properties:
                    stat, pval = stats.mannwhitneyu(high_bio, low_bio, alternative='two-sided')
                    test_type = 'Mann-Whitney U'
                else:
                    # For discrete properties, use chi-square test
                    try:
                        contingency_table = pd.crosstab(
                            pd.Series([1]*np.sum(valid_high) + [0]*np.sum(valid_low)),
                            pd.Series(list(high_bio) + list(low_bio))
                        )
                        stat, pval, _, _ = stats.chi2_contingency(contingency_table)
                        test_type = 'Chi-square'
                    except:
                        stat, pval = stats.mannwhitneyu(high_bio, low_bio, alternative='two-sided')
                        test_type = 'Mann-Whitney U'
                
                # Compute effect size
                effect_size = (np.mean(high_bio) - np.mean(low_bio)) / (np.std(bio_values) + 1e-10)
                
                result = {
                    'layer': layer,
                    'feature_idx': feat_idx,
                    'property': best_prop,
                    'interpretability_score': row['interpretability_score'],
                    'n_high_activation': np.sum(valid_high),
                    'n_low_activation': np.sum(valid_low),
                    'mean_bio_high': np.mean(high_bio),
                    'mean_bio_low': np.mean(low_bio),
                    'std_bio_high': np.std(high_bio),
                    'std_bio_low': np.std(low_bio),
                    'effect_size': effect_size,
                    'test_statistic': stat,
                    'p_value': pval,
                    'test_type': test_type
                }
                
                layer_results.append(result)
                
                print(f"    Feature {feat_idx} ({best_prop}):")
                print(f"      High activation mean: {np.mean(high_bio):.4f}")
                print(f"      Low activation mean: {np.mean(low_bio):.4f}")
                print(f"      Effect size: {effect_size:.4f}")
                print(f"      p-value: {pval:.4e}")
            
            validation_results[layer] = layer_results
        
        # Save validation results
        self._save_validation_results(validation_results, output_dir)
        
        print(f"{'='*80}")
        
        return validation_results
    
    def _save_validation_results(self, validation_results: Dict, output_dir: Path):
        """Save validation results to file"""
        print("\nSaving validation results...")
        
        # Convert to DataFrame
        all_results = []
        for layer, layer_results in validation_results.items():
            all_results.extend(layer_results)
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Save to CSV
            csv_path = output_dir / 'deep_validation_results.csv'
            results_df.to_csv(csv_path, index=False)
            print(f"  Saved to: {csv_path}")
            
            # Create visualization
            self._plot_validation_results(results_df, output_dir)
    
    def _plot_validation_results(self, results_df: pd.DataFrame, output_dir: Path):
        """Plot validation results"""
        print("  Generating validation plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Effect sizes
        ax = axes[0]
        layers = results_df['layer'].unique()
        for layer in layers:
            layer_df = results_df[results_df['layer'] == layer]
            ax.scatter(layer_df['feature_idx'], layer_df['effect_size'], 
                      label=layer, alpha=0.7, s=100)
        
        ax.set_xlabel('Feature Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Effect Size', fontsize=12, fontweight='bold')
        ax.set_title('Effect Sizes of Top Features', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot 2: P-values
        ax = axes[1]
        for layer in layers:
            layer_df = results_df[results_df['layer'] == layer]
            ax.scatter(layer_df['feature_idx'], -np.log10(layer_df['p_value'] + 1e-300), 
                      label=layer, alpha=0.7, s=100)
        
        ax.set_xlabel('Feature Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('-log10(p-value)', fontsize=12, fontweight='bold')
        ax.set_title('Statistical Significance of Top Features', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(y=-np.log10(0.05), color='r', linestyle='--', alpha=0.5, label='p=0.05')
        
        plt.tight_layout()
        
        plot_path = output_dir / 'deep_validation_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plots to: {plot_path}")
