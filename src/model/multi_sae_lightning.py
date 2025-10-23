"""
Multi-Layer SAE PyTorch Lightning Implementation

Extends the single SAE trainer to support training multiple SAEs in parallel
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

# Import from the single SAE implementation
from src.model.sae_lightning import SAELightningModule, SAEDataModule, SAETrainer
from src.model.sparse_autoencoder import SAEConfig, SparseAutoencoder


class MultiLayerSAEModule(pl.LightningModule):
    """
    Multi-layer SAE Lightning module - trains multiple SAEs in parallel
    
    Example:
    --------
    >>> layer_configs = {
    >>>     'layer_0': SAEConfig(d_in=512, expansion_factor=4),
    >>>     'layer_1': SAEConfig(d_in=512, expansion_factor=4),
    >>> }
    >>> layer_saes = {name: SparseAutoencoder(cfg) for name, cfg in layer_configs.items()}
    >>> module = MultiLayerSAEModule(layer_saes, layer_configs)
    """
    
    def __init__(
        self,
        layer_saes: Dict[str, 'SparseAutoencoder'],
        layer_configs: Dict[str, 'SAEConfig']
    ):
        super().__init__()
        self.layer_names = list(layer_saes.keys())
        
        # Create safe module names (replace '.' with '_')
        # torch.nn.ModuleDict doesn't allow '.' in keys
        self.name_mapping = {name: name.replace('.', '_') for name in self.layer_names}
        self.reverse_mapping = {safe: orig for orig, safe in self.name_mapping.items()}
        
        # Register SAEs as submodules with safe names
        safe_saes = {self.name_mapping[name]: sae for name, sae in layer_saes.items()}
        self.saes = torch.nn.ModuleDict(safe_saes)
        self.configs = layer_configs
        
        self.save_hyperparameters(ignore=['layer_saes'])
    
    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple]:
        """Forward pass for all layers"""
        outputs = {}
        for layer_name in self.layer_names:
            x = batch_dict[layer_name]
            safe_name = self.name_mapping[layer_name]
            outputs[layer_name] = self.saes[safe_name](x, return_loss=True)
        return outputs
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step - compute loss for all layers"""
        total_loss = 0
        
        for layer_name in self.layer_names:
            x = batch[layer_name]
            safe_name = self.name_mapping[layer_name]
            _, _, loss_dict = self.saes[safe_name](x, return_loss=True)
            
            # Accumulate loss
            total_loss += loss_dict['total_loss']
            
            # Log per-layer metrics (use safe name for logging to avoid issues)
            self.log(f'{safe_name}/train/total_loss', loss_dict['total_loss'])
            self.log(f'{safe_name}/train/reconstruction_loss', loss_dict['reconstruction_loss'])
            self.log(f'{safe_name}/train/l0_norm', loss_dict['l0_norm'])
        
        # Log average loss
        self.log('train/avg_total_loss', total_loss / len(self.layer_names), prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        total_loss = 0
        
        for layer_name in self.layer_names:
            x = batch[layer_name]
            safe_name = self.name_mapping[layer_name]
            _, _, loss_dict = self.saes[safe_name](x, return_loss=True)
            
            total_loss += loss_dict['total_loss']
            
            self.log(f'{safe_name}/val/total_loss', loss_dict['total_loss'])
            self.log(f'{safe_name}/val/reconstruction_loss', loss_dict['reconstruction_loss'])
            self.log(f'{safe_name}/val/l0_norm', loss_dict['l0_norm'])
        
        self.log('val/avg_total_loss', total_loss / len(self.layer_names), prog_bar=True)
        
        return total_loss
    
    def on_after_backward(self):
        """Normalize decoder weights for all SAEs"""
        for sae in self.saes.values():
            sae.normalize_decoder_step()
    
    def configure_optimizers(self):
        """Configure optimizer for all SAEs"""
        # Use the first config's learning rate (or you can make this configurable)
        first_config = next(iter(self.configs.values()))
        
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=first_config.learning_rate,
            weight_decay=first_config.weight_decay
        )
        return optimizer


class MultiLayerDataModule(pl.LightningDataModule):
    """
    Data module for multi-layer SAE training
    
    Example:
    --------
    >>> layer_activations = {
    >>>     'layer_0': torch.randn(10000, 512),
    >>>     'layer_1': torch.randn(10000, 512),
    >>> }
    >>> data_module = MultiLayerDataModule(layer_activations, batch_size=256)
    """
    
    def __init__(
        self,
        layer_activations: Dict[str, torch.Tensor],
        batch_size: int = 256,
        validation_split: float = 0.1,
        num_workers: int = 0
    ):
        super().__init__()
        self.layer_activations = layer_activations
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_workers = num_workers
        
        self.train_datasets = {}
        self.val_datasets = {}
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each layer"""
        # Get number of samples (assume all layers have same number)
        n_samples = next(iter(self.layer_activations.values())).shape[0]
        n_val = int(n_samples * self.validation_split)
        n_train = n_samples - n_val
        
        # Create indices for train/val split (shared across layers)
        indices = torch.randperm(n_samples, generator=torch.Generator().manual_seed(42))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Split data for each layer
        for layer_name, activations in self.layer_activations.items():
            self.train_datasets[layer_name] = activations[train_indices]
            self.val_datasets[layer_name] = activations[val_indices] if n_val > 0 else None
    
    def train_dataloader(self):
        """Create training dataloader that yields dict of tensors"""
        # Create dataset that returns dict
        class MultiLayerDataset(torch.utils.data.Dataset):
            def __init__(self, layer_data_dict):
                self.layer_data = layer_data_dict
                self.layer_names = list(layer_data_dict.keys())
                self.length = len(next(iter(layer_data_dict.values())))
            
            def __len__(self):
                return self.length
            
            def __getitem__(self, idx):
                return {name: data[idx] for name, data in self.layer_data.items()}
        
        dataset = MultiLayerDataset(self.train_datasets)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self):
        """Create validation dataloader"""
        if all(v is None for v in self.val_datasets.values()):
            return None
        
        class MultiLayerDataset(torch.utils.data.Dataset):
            def __init__(self, layer_data_dict):
                self.layer_data = layer_data_dict
                self.layer_names = list(layer_data_dict.keys())
                self.length = len(next(iter(layer_data_dict.values())))
            
            def __len__(self):
                return self.length
            
            def __getitem__(self, idx):
                return {name: data[idx] for name, data in self.layer_data.items()}
        
        dataset = MultiLayerDataset(self.val_datasets)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )


class MultiLayerSAETrainer:
    """
    Multi-layer SAE trainer with PyTorch Lightning
    
    Example:
    --------
    >>> layer_activations = {
    >>>     'layer_0': torch.randn(10000, 512),
    >>>     'layer_1': torch.randn(10000, 512),
    >>>     'layer_2': torch.randn(10000, 512),
    >>> }
    >>> 
    >>> trainer = MultiLayerSAETrainer(
    >>>     layer_activations=layer_activations,
    >>>     expansion_factor=4,
    >>>     l1_coefficient=1e-3
    >>> )
    >>> trainer.train_all(num_epochs=100, batch_size=256)
    """
    
    def __init__(
        self,
        layer_activations: Dict[str, torch.Tensor],
        expansion_factor: int = 4,
        l1_coefficient: float = 1e-3,
        learning_rate: float = 1e-3,
        accelerator: str = 'auto',
        devices: int = 1
    ):
        """
        Parameters:
        -----------
        layer_activations : Dict[str, torch.Tensor]
            Activation data for each layer
        expansion_factor : int
            Expansion factor for all SAEs
        l1_coefficient : float
            L1 coefficient for all SAEs
        learning_rate : float
            Learning rate
        accelerator : str
            Accelerator type
        devices : int
            Number of devices
        """
        self.layer_activations = layer_activations
        self.accelerator = accelerator
        self.devices = devices
        
        # Create SAEs and configs for each layer
        self.saes = {}
        self.configs = {}
        
        for layer_name, activations in layer_activations.items():
            d_in = activations.shape[1]
            
            config = SAEConfig(
                d_in=d_in,
                expansion_factor=expansion_factor,
                l1_coefficient=l1_coefficient,
                learning_rate=learning_rate
            )
            
            sae = SparseAutoencoder(config)
            
            self.configs[layer_name] = config
            self.saes[layer_name] = sae
        
        # Create Lightning module
        self.lightning_module = MultiLayerSAEModule(self.saes, self.configs)
        
        print(f"Initialized {len(self.saes)} SAEs for layers: {list(self.saes.keys())}")
    
    def train_all(
        self,
        num_epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.1,
        log_interval: int = 10,
        save_dir: Optional[Path] = None,
        **trainer_kwargs
    ) -> Dict[str, Dict[str, list]]:
        """
        Train all SAEs in parallel
        
        Parameters:
        -----------
        num_epochs : int
            Number of epochs
        batch_size : int
            Batch size
        validation_split : float
            Validation split ratio
        log_interval : int
            Logging interval
        save_dir : Path, optional
            Directory to save checkpoints
        **trainer_kwargs : dict
            Additional trainer arguments
        
        Returns:
        --------
        all_histories : Dict[str, Dict[str, list]]
            Training history for all layers
        """
        # Create data module
        data_module = MultiLayerDataModule(
            self.layer_activations,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        # Setup callbacks
        callbacks = []
        
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=save_dir,
                filename='multilayer-sae-{epoch:02d}-{val/avg_total_loss:.4f}',
                monitor='val/avg_total_loss',
                mode='min',
                save_top_k=1,
                save_last=True
            )
            callbacks.append(checkpoint_callback)
        
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='step'))
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            callbacks=callbacks,
            log_every_n_steps=log_interval,
            enable_progress_bar=True,
            enable_model_summary=True,
            **trainer_kwargs
        )
        
        # Train
        trainer.fit(self.lightning_module, data_module)
        
        print("Training completed!")
        
        return {}  # History can be extracted from logger if needed
    
    def get_sae(self, layer_name: str) -> 'SparseAutoencoder':
        """
        Get SAE for specific layer
        
        Parameters:
        -----------
        layer_name : str
            Original layer name (e.g., 'layers.0.mixer')
        
        Returns:
        --------
        SparseAutoencoder
            The trained SAE for the specified layer
        """
        # Get the safe name and retrieve from lightning module
        safe_name = self.lightning_module.name_mapping[layer_name]
        return self.lightning_module.saes[safe_name]
    
    def get_all_saes(self) -> Dict[str, 'SparseAutoencoder']:
        """
        Get all trained SAEs
        
        Returns:
        --------
        Dict[str, SparseAutoencoder]
            Dictionary mapping original layer names to trained SAEs
        """
        # Return SAEs with original layer names
        all_saes = {}
        for layer_name in self.lightning_module.layer_names:
            safe_name = self.lightning_module.name_mapping[layer_name]
            all_saes[layer_name] = self.lightning_module.saes[safe_name]
        return all_saes
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load from checkpoint"""
        self.lightning_module = MultiLayerSAEModule.load_from_checkpoint(
            checkpoint_path,
            layer_saes=self.saes,
            layer_configs=self.configs
        )
        print(f"Loaded checkpoint: {checkpoint_path}")



# Example usage
if __name__ == "__main__":
    from sparse_autoencoder import SAEConfig
    
    # Generate sample multi-layer activations
    # Simulating activations from different layers of a neural network
    layer_activations = {
        'layer_0': torch.randn(10000, 512),  # First layer: 512 dimensions
        'layer_1': torch.randn(10000, 768),  # Second layer: 768 dimensions
        'layer_2': torch.randn(10000, 1024), # Third layer: 1024 dimensions
    }
    
    # Create multi-layer SAE trainer
    trainer = MultiLayerSAETrainer(
        layer_activations=layer_activations,
        expansion_factor=4,
        l1_coefficient=1e-3,
        learning_rate=1e-3,
        accelerator='gpu',
        devices=1
    )
    
    # Train all layers
    histories = trainer.train_all(
        num_epochs=10,
        batch_size=256,
        validation_split=0.1,
        save_dir=Path('./multi_layer_checkpoints')
    )
    
    # Get trained SAEs
    layer_0_sae = trainer.get_sae('layer_0')
    layer_1_sae = trainer.get_sae('layer_1')
    layer_2_sae = trainer.get_sae('layer_2')
    all_saes = trainer.get_all_saes()
    
    print("Multi-layer SAE training completed!")
    print(f"Trained {len(all_saes)} SAE layers")
    
    # Example: Use trained SAE to encode/decode activations
    with torch.no_grad():
        test_activation = layer_activations['layer_0'][:10]  # Take 10 samples
        encoded = layer_0_sae.encode(test_activation)
        decoded = layer_0_sae.decode(encoded)
        reconstruction_error = torch.mean((test_activation - decoded) ** 2)
        print(f"Layer 0 reconstruction error: {reconstruction_error:.6f}")