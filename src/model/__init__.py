"""
SAE 模型模块
"""

from .sparse_autoencoder import (
    SparseAutoencoder,
    TopKSparseAutoencoder,
    SAEConfig
)
from .activation_extractor import (
    ActivationExtractor,
    GradientExtractor,
    LayerInfo,
    get_all_layer_names,
    extract_activations_from_model
)
from .sae_lightning import (
    SAELightningModule,
    SAEDataModule,
    SAETrainer
)
from .multi_sae_lightning import (
    MultiLayerSAEModule,
    MultiLayerDataModule,
    MultiLayerSAETrainer
)

__all__ = [
    # Sparse Autoencoder models
    'SparseAutoencoder',
    'TopKSparseAutoencoder',
    'SAEConfig',
    
    # Activation extraction
    'ActivationExtractor',
    'GradientExtractor',
    'LayerInfo',
    'get_all_layer_names',
    'extract_activations_from_model',
    
    # Single-layer SAE training (PyTorch Lightning)
    'SAELightningModule',
    'SAEDataModule',
    'SAETrainer',
    
    # Multi-layer SAE training (PyTorch Lightning)
    'MultiLayerSAEModule',
    'MultiLayerDataModule',
    'MultiLayerSAETrainer',
]