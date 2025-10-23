"""
Causal Feature Analysis Module

This module implements ACDC (Automatic Circuit Discovery with Causality) algorithm
to verify the causal importance of SAE features for specific prediction tasks.

Modules:
--------
- base: Base classes for data management and common utilities
- probe: Step 1 - Build prediction probe (regression/classification)
- sequence_pairs: Step 2 - Construct sequence pair dataset
- activations: Step 3 - Forward propagation and activation collection
- acdc: Step 4 - ACDC feature importance evaluation
- intervention: Step 5 - Cumulative intervention curves
- aggregation: Step 6 - Cross-sequence pair aggregation
- comparison: Step 7 - Cross-block comparison
- baseline: Step 8 - Baseline method comparison (PCA, ICA)
- visualization: Step 9 - Result visualization
- feature_analysis: Step 10 - Feature functional analysis
"""

from .base import CausalAnalysisConfig, CausalDataManager, BaseCausalStep
from .probe import ProbeBuilder
from .sequence_pairs import SequencePairSelector
from .activations import ActivationCollector
from .acdc import ACDCAnalyzer
from .intervention import InterventionAnalyzer
from .aggregation import ResultAggregator
from .comparison import BlockComparator
from .baseline import BaselineComparator
from .visualization import ResultVisualizer
from .feature_analysis import FeatureAnalyzer

__all__ = [
    'CausalAnalysisConfig',
    'CausalDataManager',
    'BaseCausalStep',
    'ProbeBuilder',
    'SequencePairSelector',
    'ActivationCollector',
    'ACDCAnalyzer',
    'InterventionAnalyzer',
    'ResultAggregator',
    'BlockComparator',
    'BaselineComparator',
    'ResultVisualizer',
    'FeatureAnalyzer',
]
