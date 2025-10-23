"""
Step 6: Cross-Sequence Pair Aggregation

Aggregate results across all sequence pairs to get population-level statistics.
"""

import numpy as np
from typing import Dict, Any
from scipy import stats

from .base import BaseCausalStep, CausalAnalysisConfig, CausalDataManager


class ResultAggregator(BaseCausalStep):
    """Aggregate results across sequence pairs"""
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Aggregate intervention results"""
        self.log("="*80)
        self.log("Step 6: Aggregating Results Across Pairs")
        self.log("="*80)
        
        intervention_results = self.data_manager.get_data('intervention_results')
        if intervention_results is None:
            intervention_results = self.load_results('interventionanalyzer_results')
        
        # Aggregate curves by block
        aggregated = self._aggregate_by_block(intervention_results['intervention_curves'])
        
        # Compute convergence statistics
        conv_stats = self._convergence_statistics(intervention_results['convergence_points'])
        
        results = {
            'aggregated_curves': aggregated,
            'convergence_statistics': conv_stats
        }
        
        if self.config.save_intermediate:
            self.save_results(results)
        
        self.data_manager.set_data('aggregated_results', results)
        
        self.log("="*80)
        self.log("✓ Aggregation completed!", level='success')
        self.log("="*80)
        
        return results
    
    def _aggregate_by_block(self, curves: Dict) -> Dict:
        """Aggregate curves by block"""
        block_curves = {}
        
        for (pair_id, block_name), curve_data in curves.items():
            if block_name not in block_curves:
                block_curves[block_name] = {'error': [], 'magnitude': []}
            
            block_curves[block_name]['error'].append(curve_data['error_curve'])
            block_curves[block_name]['magnitude'].append(curve_data['magnitude_curve'])
        
        # Compute mean and std
        aggregated = {}
        for block_name, data in block_curves.items():
            error_curves = np.array(data['error'])
            mag_curves = np.array(data['magnitude'])
            
            aggregated[block_name] = {
                'mean_error': error_curves.mean(axis=0).tolist(),
                'std_error': error_curves.std(axis=0).tolist(),
                'mean_magnitude': mag_curves.mean(axis=0).tolist(),
                'std_magnitude': mag_curves.std(axis=0).tolist()
            }
        
        return aggregated
    
    def _convergence_statistics(self, convergence_points: Dict) -> Dict:
        """Compute convergence statistics"""
        block_conv = {}
        
        for (pair_id, block_name), k in convergence_points.items():
            if block_name not in block_conv:
                block_conv[block_name] = []
            if k is not None:
                block_conv[block_name].append(k)
        
        stats_dict = {}
        for block_name, k_values in block_conv.items():
            if k_values:
                stats_dict[block_name] = {
                    'median': float(np.median(k_values)),
                    'mean': float(np.mean(k_values)),
                    'q25': float(np.percentile(k_values, 25)),
                    'q75': float(np.percentile(k_values, 75))
                }
        
        return stats_dict
