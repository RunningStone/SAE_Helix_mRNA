"""
SAE Helix mRNA - 稀疏自编码器分析工具
"""

from .model import (
    SparseAutoencoder,
    TopKSparseAutoencoder,
    SAEConfig,
    ActivationExtractor,
    GradientExtractor,
    SAETrainer,
    MultiLayerSAETrainer
)

from .pipeline import SAEAnalyser

__version__ = '0.1.0'

__all__ = [
    'SparseAutoencoder',
    'TopKSparseAutoencoder',
    'SAEConfig',
    'ActivationExtractor',
    'GradientExtractor',
    'SAETrainer',
    'MultiLayerSAETrainer',
    'SAEAnalyser',
]
