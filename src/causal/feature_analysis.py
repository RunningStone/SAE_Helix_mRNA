"""
Step 10: Feature Functional Analysis

Analyze the biological functions of key causal features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
from collections import Counter

from .base import BaseCausalStep, CausalAnalysisConfig, CausalDataManager


class FeatureAnalyzer(BaseCausalStep):
    """Analyze biological functions of key features"""
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run feature functional analysis"""
        self.log("="*80)
        self.log("Step 10: Feature Functional Analysis")
        self.log("="*80)
        
        # Load ACDC results
        acdc_results = self.data_manager.get_data('acdc_results')
        if acdc_results is None:
            acdc_results = self.load_results('acdcanalyzer_results')
        
        # Load feature annotations from Step 2
        feature_annotations = self._load_feature_annotations()
        
        # Identify key features
        key_features = self._identify_key_features(acdc_results)
        
        # Analyze biological functions
        functional_analysis = self._analyze_functions(key_features, feature_annotations)
        
        results = {
            'key_features': key_features,
            'functional_analysis': functional_analysis
        }
        
        if self.config.save_intermediate:
            self.save_results(results)
            self._save_summary(results)
        
        self.data_manager.set_data('feature_analysis_results', results)
        
        self.log("="*80)
        self.log("✓ Feature analysis completed!", level='success')
        self.log("="*80)
        
        return results
    
    def _load_feature_annotations(self) -> Dict:
        """Load feature annotations from Step 2 correlation analysis"""
        step2_dir = Path(self.config.step1_output_dir).parent / 'step2'
        
        # Try to load best matches
        best_matches_file = step2_dir / 'best_matches.csv'
        
        if best_matches_file.exists():
            df = pd.read_csv(best_matches_file)
            
            # Convert to dictionary
            annotations = {}
            for _, row in df.iterrows():
                feature_id = row['feature_id']
                annotations[feature_id] = {
                    'property': row['property'],
                    'correlation': row['correlation'],
                    'p_value': row['p_value']
                }
            
            self.log(f"Loaded annotations for {len(annotations)} features")
            return annotations
        else:
            self.log("No feature annotations found", level='warning')
            return {}
    
    def _identify_key_features(self, acdc_results: Dict) -> Dict:
        """Identify frequently important features"""
        feature_importance = acdc_results['feature_importance']
        
        # Count how often each feature appears in top-K
        top_k = 10
        feature_counts = {}
        
        for (pair_id, block_name), ranking in feature_importance.items():
            if block_name not in feature_counts:
                feature_counts[block_name] = Counter()
            
            top_features = ranking[:top_k]
            feature_counts[block_name].update(top_features)
        
        # Get most common features per block
        key_features = {}
        for block_name, counts in feature_counts.items():
            most_common = counts.most_common(20)
            key_features[block_name] = [
                {'feature_id': f, 'frequency': count}
                for f, count in most_common
            ]
        
        return key_features
    
    def _analyze_functions(self, key_features: Dict, 
                          annotations: Dict) -> Dict:
        """Analyze biological functions of key features"""
        functional_analysis = {}
        
        for block_name, features in key_features.items():
            # Collect properties
            properties = []
            for feat in features:
                feat_id = feat['feature_id']
                if feat_id in annotations:
                    properties.append(annotations[feat_id]['property'])
            
            # Count property categories
            property_counts = Counter(properties)
            
            functional_analysis[block_name] = {
                'property_distribution': dict(property_counts),
                'num_annotated': len(properties),
                'num_total': len(features)
            }
        
        return functional_analysis
    
    def _save_summary(self, results: Dict):
        """Save human-readable summary"""
        summary = {
            'key_features_per_block': {
                block: len(features)
                for block, features in results['key_features'].items()
            },
            'functional_analysis': results['functional_analysis']
        }
        
        self.data_manager.save_data(
            step_name='feature_analysis',
            data=summary,
            filename='feature_analysis_summary',
            format='json'
        )
