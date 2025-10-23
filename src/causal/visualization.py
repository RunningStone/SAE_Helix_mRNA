"""
Step 9: Result Visualization

Generate comprehensive visualizations of causal analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from pathlib import Path

from .base import BaseCausalStep, CausalAnalysisConfig, CausalDataManager


class ResultVisualizer(BaseCausalStep):
    """Visualize causal analysis results"""
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Generate visualizations"""
        self.log("="*80)
        self.log("Step 9: Generating Visualizations")
        self.log("="*80)
        
        # Load all results
        aggregated = self.data_manager.get_data('aggregated_results')
        comparison = self.data_manager.get_data('comparison_results')
        baseline = self.data_manager.get_data('baseline_results')
        
        if aggregated is None:
            aggregated = self.load_results('resultaggregator_results')
        if comparison is None:
            comparison = self.load_results('blockcomparator_results')
        
        # Generate plots
        plot_paths = {}
        
        # Plot 1: Intervention efficiency comparison
        plot_paths['efficiency'] = self._plot_efficiency(aggregated)
        
        # Plot 2: Convergence comparison
        plot_paths['convergence'] = self._plot_convergence(aggregated)
        
        # Plot 3: Block comparison
        if comparison:
            plot_paths['block_comparison'] = self._plot_block_comparison(comparison)
        
        results = {'plot_paths': plot_paths}
        
        if self.config.save_intermediate:
            self.save_results(results)
        
        self.data_manager.set_data('visualization_results', results)
        
        self.log("="*80)
        self.log("✓ Visualization completed!", level='success')
        self.log("="*80)
        
        return results
    
    def _plot_efficiency(self, aggregated: Dict) -> Path:
        """Plot intervention efficiency curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        curves = aggregated['aggregated_curves']
        
        for block_name, curve_data in curves.items():
            k_values = np.arange(1, len(curve_data['mean_error']) + 1)
            
            # Plot error curve
            ax1.plot(k_values, curve_data['mean_error'], label=block_name, marker='o')
            ax1.fill_between(
                k_values,
                np.array(curve_data['mean_error']) - np.array(curve_data['std_error']),
                np.array(curve_data['mean_error']) + np.array(curve_data['std_error']),
                alpha=0.2
            )
            
            # Plot magnitude curve
            ax2.plot(curve_data['mean_magnitude'], curve_data['mean_error'], 
                    label=block_name, marker='o')
        
        ax1.set_xlabel('Number of Features Intervened')
        ax1.set_ylabel('Prediction Error')
        ax1.set_title('Intervention Efficiency: Error vs Feature Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Cumulative Intervention Magnitude')
        ax2.set_ylabel('Prediction Error')
        ax2.set_title('Intervention Efficiency: Error vs Magnitude')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.data_manager.step_dirs['visualization'] / f'intervention_efficiency.{self.config.plot_format}'
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        self.log(f"Saved: {save_path.name}")
        return save_path
    
    def _plot_convergence(self, aggregated: Dict) -> Path:
        """Plot convergence statistics"""
        conv_stats = aggregated['convergence_statistics']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        blocks = list(conv_stats.keys())
        medians = [conv_stats[b]['median'] for b in blocks]
        q25 = [conv_stats[b]['q25'] for b in blocks]
        q75 = [conv_stats[b]['q75'] for b in blocks]
        
        x = np.arange(len(blocks))
        ax.bar(x, medians, alpha=0.7, label='Median')
        ax.errorbar(x, medians, 
                   yerr=[np.array(medians) - np.array(q25), 
                         np.array(q75) - np.array(medians)],
                   fmt='none', color='black', capsize=5, label='IQR')
        
        ax.set_xticks(x)
        ax.set_xticklabels(blocks, rotation=45, ha='right')
        ax.set_ylabel('Number of Features to Converge')
        ax.set_title('Convergence Comparison Across Blocks')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        save_path = self.data_manager.step_dirs['visualization'] / f'convergence_comparison.{self.config.plot_format}'
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        self.log(f"Saved: {save_path.name}")
        return save_path
    
    def _plot_block_comparison(self, comparison: Dict) -> Path:
        """Plot block comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        efficiency_ranking = comparison['block_comparison']['efficiency_ranking']
        
        blocks = [item[0] for item in efficiency_ranking]
        efficiencies = [item[1] for item in efficiency_ranking]
        
        ax.barh(blocks, efficiencies, alpha=0.7)
        ax.set_xlabel('Intervention Efficiency')
        ax.set_title('Block Efficiency Ranking')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        save_path = self.data_manager.step_dirs['visualization'] / f'block_efficiency_ranking.{self.config.plot_format}'
        plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        self.log(f"Saved: {save_path.name}")
        return save_path
