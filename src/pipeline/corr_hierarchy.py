"""
Hierarchy Analyzer Module

Analyzes feature hierarchy across transformer blocks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


class HierarchyAnalyzer:
    """Analyze feature hierarchy across blocks"""
    
    def __init__(self, config):
        """Initialize hierarchy analyzer"""
        self.config = config
    
    def analyze_block_hierarchy(
        self,
        filtered_df: pd.DataFrame,
        output_dir: Path
    ) -> Dict:
        """Analyze how features in different blocks relate to different biological properties"""
        print(f"\n{'='*80}")
        print("Step 3: Cross-Block Feature Hierarchy Analysis")
        print(f"{'='*80}")
        
        # Define property categories
        structural_props = ['gc_content', 'mfe', 'length']
        functional_props = ['mrl', 'te', 'expression_level']
        regulatory_props = ['uorf_count', 'uaug_count']
        
        def categorize_property(prop):
            if prop in structural_props:
                return 'Structural'
            elif prop in functional_props:
                return 'Functional'
            elif prop in regulatory_props:
                return 'Regulatory'
            else:
                return 'Other'
        
        filtered_df['property_category'] = filtered_df['property'].apply(categorize_property)
        
        # Count features per block and category
        hierarchy_stats = {}
        
        for layer in sorted(filtered_df['layer'].unique()):
            layer_df = filtered_df[filtered_df['layer'] == layer]
            
            # Count unique features per category
            category_counts = {}
            for category in ['Structural', 'Functional', 'Regulatory']:
                category_df = layer_df[layer_df['property_category'] == category]
                n_features = category_df['feature_idx'].nunique()
                category_counts[category] = n_features
            
            hierarchy_stats[layer] = category_counts
            
            print(f"\n  {layer}:")
            print(f"    Structural: {category_counts['Structural']} features")
            print(f"    Functional: {category_counts['Functional']} features")
            print(f"    Regulatory: {category_counts['Regulatory']} features")
        
        # Create stacked bar plot
        self._plot_hierarchy(hierarchy_stats, output_dir)
        
        print(f"{'='*80}")
        
        return hierarchy_stats
    
    def _plot_hierarchy(self, hierarchy_stats: Dict, output_dir: Path):
        """Plot stacked bar chart of feature distribution across blocks"""
        print("\nGenerating hierarchy plot...")
        
        # Prepare data for plotting
        layers = sorted(hierarchy_stats.keys())
        categories = ['Structural', 'Functional', 'Regulatory']
        
        data = {cat: [hierarchy_stats[layer][cat] for layer in layers] for cat in categories}
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(layers))
        width = 0.6
        
        bottom = np.zeros(len(layers))
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, (category, color) in enumerate(zip(categories, colors)):
            values = data[category]
            ax.bar(x, values, width, label=category, bottom=bottom, color=color, alpha=0.8)
            bottom += values
        
        ax.set_xlabel('Transformer Block', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
        ax.set_title('Feature Distribution Across Blocks by Biological Property Category', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.legend(loc='upper left', frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        plot_path = output_dir / 'hierarchy_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot to: {plot_path}")
