"""
Step 7: Cross-Block Comparison

Compare feature importance and intervention efficiency across different blocks.
"""

import numpy as np
from typing import Dict, Any, List

from .base import BaseCausalStep, CausalAnalysisConfig, CausalDataManager


class BlockComparator(BaseCausalStep):
    """Compare results across blocks"""
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Compare blocks"""
        self.log("="*80)
        self.log("Step 7: Cross-Block Comparison")
        self.log("="*80)
        
        aggregated_results = self.data_manager.get_data('aggregated_results')
        if aggregated_results is None:
            aggregated_results = self.load_results('resultaggregator_results')
        
        # Compare convergence across blocks
        comparison = self._compare_blocks(aggregated_results)
        
        results = {'block_comparison': comparison}
        
        if self.config.save_intermediate:
            self.save_results(results)
        
        self.data_manager.set_data('comparison_results', results)
        
        self.log("="*80)
        self.log("✓ Block comparison completed!", level='success')
        self.log("="*80)
        
        return results
    
    def _compare_blocks(self, aggregated_results: Dict) -> Dict:
        """Compare blocks"""
        curves = aggregated_results['aggregated_curves']
        conv_stats = aggregated_results['convergence_statistics']
        
        comparison = {
            'convergence_comparison': conv_stats,
            'efficiency_ranking': self._rank_by_efficiency(curves)
        }
        
        return comparison
    
    def _rank_by_efficiency(self, curves: Dict) -> List:
        """Rank blocks by intervention efficiency"""
        efficiency = []
        
        for block_name, curve_data in curves.items():
            # Efficiency = error reduction per unit magnitude
            mean_error = np.array(curve_data['mean_error'])
            mean_mag = np.array(curve_data['mean_magnitude'])
            
            if len(mean_error) > 0 and mean_mag[-1] > 0:
                eff = (mean_error[0] - mean_error[-1]) / mean_mag[-1]
                efficiency.append((block_name, float(eff)))
        
        efficiency.sort(key=lambda x: x[1], reverse=True)
        return efficiency
