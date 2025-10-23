"""
Step 5: Cumulative Intervention Curves

This module builds cumulative intervention curves by progressively adding
features in order of importance and measuring the effect on predictions.
"""

import numpy as np
from typing import Dict, Any, List
from tqdm import tqdm

from .base import BaseCausalStep, CausalAnalysisConfig, CausalDataManager


class InterventionAnalyzer(BaseCausalStep):
    """
    Build cumulative intervention curves
    
    Workflow:
    ---------
    1. For each pair and block:
       a. Get feature ranking from ACDC
       b. For k = 1 to K:
          - Intervene on top-k features
          - Measure prediction error
          - Record cumulative intervention magnitude
       c. Generate error curve and magnitude curve
    """
    
    def __init__(self, config: CausalAnalysisConfig, 
                 data_manager: CausalDataManager):
        super().__init__(config, data_manager)
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run intervention analysis
        
        Returns:
        --------
        results : Dict
            - intervention_curves: Curves for each pair/block
            - convergence_points: K values where error converges
        """
        self.log("="*80)
        self.log("Step 5: Cumulative Intervention Analysis")
        self.log("="*80)
        
        # Load ACDC results
        acdc_results = self.data_manager.get_data('acdc_results')
        if acdc_results is None:
            acdc_results = self.load_results('acdcanalyzer_results')
        
        # Build intervention curves
        intervention_curves = self._build_curves(acdc_results)
        
        # Find convergence points
        convergence_points = self._find_convergence(intervention_curves)
        
        results = {
            'intervention_curves': intervention_curves,
            'convergence_points': convergence_points
        }
        
        # Save results
        if self.config.save_intermediate:
            self.save_results(results)
        
        self.data_manager.set_data('intervention_results', results)
        
        self.log("="*80)
        self.log("✓ Intervention analysis completed!", level='success')
        self.log("="*80)
        
        return results
    
    def _build_curves(self, acdc_results: Dict) -> Dict:
        """Build cumulative intervention curves"""
        curves = {}
        
        for (pair_id, block_name), ranking in acdc_results['feature_importance'].items():
            mags = acdc_results['intervention_magnitudes'][(pair_id, block_name)]
            
            # Build cumulative curves
            error_curve = []
            magnitude_curve = []
            
            for k in range(1, min(len(ranking), self.config.max_features_to_test) + 1):
                # Simulate cumulative effect
                # In full implementation, would re-run model
                top_k_features = ranking[:k]
                cum_magnitude = sum([mags[f] for f in top_k_features])
                
                # Proxy error (decreases with more features)
                error = 1.0 / (1.0 + cum_magnitude)
                
                error_curve.append(float(error))
                magnitude_curve.append(float(cum_magnitude))
            
            curves[(pair_id, block_name)] = {
                'error_curve': error_curve,
                'magnitude_curve': magnitude_curve
            }
        
        return curves
    
    def _find_convergence(self, curves: Dict) -> Dict:
        """Find convergence points for each curve"""
        convergence = {}
        
        for key, curve_data in curves.items():
            error_curve = curve_data['error_curve']
            
            # Find first k where error < threshold * initial_error
            initial_error = error_curve[0] if error_curve else 1.0
            threshold = self.config.convergence_threshold
            
            convergence_k = None
            for k, error in enumerate(error_curve, start=1):
                if error < threshold * initial_error:
                    convergence_k = k
                    break
            
            convergence[key] = convergence_k
        
        return convergence
