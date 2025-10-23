"""
Sparse Auto-Encoder (SAE) PyTorch Lightning Implementation
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

from src.model.sparse_autoencoder import SparseAutoencoder, SAEConfig, TopKSparseAutoencoder

class SAELightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for Sparse Autoencoder
    
    Example:
    --------
    >>> # prepare data
    >>> activations = torch.randn(10000, 512)
    >>> 
    >>> # configure and create model
    >>> config = SAEConfig(d_in=512, expansion_factor=4, l1_coefficient=1e-3)
    >>> sae_model = SparseAutoencoder(config)
    >>> lightning_module = SAELightningModule(sae_model, config)
    >>> 
    >>> # create data module
    >>> data_module = SAEDataModule(activations, batch_size=256, validation_split=0.1)
    >>> 
    >>> # create trainer
    >>> trainer = pl.Trainer(max_epochs=100, log_every_n_steps=10)
    >>> trainer.fit(lightning_module, data_module)
    """
    
    def __init__(
        self,
        sae: 'SparseAutoencoder',
        config: 'SAEConfig'
    ):
        """
        Parameters:
        -----------
        sae : SparseAutoencoder
            Sparse autoencoder model
        config : SAEConfig
            Configuration object
        """
        super().__init__()
        self.sae = sae
        self.config = config
        
        # Save hyperparameters (will be logged automatically)
        self.save_hyperparameters(ignore=['sae'])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass"""
        return self.sae(x, return_loss=True)
    
    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        x = batch[0]
        
        # Forward pass
        x_reconstructed, features, loss_dict = self.sae(x, return_loss=True)
        
        # Log metrics
        self.log('train/total_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('train/reconstruction_loss', loss_dict['reconstruction_loss'])
        self.log('train/sparsity_loss', loss_dict['sparsity_loss'])
        self.log('train/l0_norm', loss_dict['l0_norm'], prog_bar=True)
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        x = batch[0]
        
        # Forward pass
        x_reconstructed, features, loss_dict = self.sae(x, return_loss=True)
        
        # Log metrics
        self.log('val/total_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('val/reconstruction_loss', loss_dict['reconstruction_loss'])
        self.log('val/sparsity_loss', loss_dict['sparsity_loss'])
        self.log('val/l0_norm', loss_dict['l0_norm'], prog_bar=True)
        
        return loss_dict['total_loss']
    
    def on_after_backward(self):
        """Called after backward pass - normalize decoder weights"""
        self.sae.normalize_decoder_step()
    
    def configure_optimizers(self):
        """Configure optimizers"""
        optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return optimizer


class SAEDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for SAE training
    
    Example:
    --------
    >>> activations = torch.randn(10000, 512)
    >>> data_module = SAEDataModule(activations, batch_size=256, validation_split=0.1)
    """
    
    def __init__(
        self,
        activations: torch.Tensor,
        batch_size: int = 256,
        validation_split: float = 0.1,
        num_workers: int = 0
    ):
        """
        Parameters:
        -----------
        activations : torch.Tensor, shape (n_samples, d_in)
            Training data (activation vectors)
        batch_size : int
            Batch size
        validation_split : float
            Validation split ratio (0.0 to 1.0)
        num_workers : int
            Number of dataloader workers
        """
        super().__init__()
        self.activations = activations
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""
        # Create full dataset
        full_dataset = TensorDataset(self.activations)
        
        # Split into train and validation
        n_samples = len(full_dataset)
        n_val = int(n_samples * self.validation_split)
        n_train = n_samples - n_val
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
    
    def train_dataloader(self):
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self):
        """Create validation dataloader"""
        if len(self.val_dataset) == 0:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )


class SAETrainer:
    """
    High-level wrapper for SAE training with PyTorch Lightning
    
    Example:
    --------
    >>> # prepare activation data
    >>> activations = torch.randn(10000, 512)
    >>> 
    >>> # configure SAE
    >>> config = SAEConfig(d_in=512, expansion_factor=4, l1_coefficient=1e-3)
    >>> sae = SparseAutoencoder(config)
    >>> 
    >>> # train
    >>> trainer = SAETrainer(sae, config)
    >>> history = trainer.train(
    >>>     activations, 
    >>>     num_epochs=100, 
    >>>     batch_size=256,
    >>>     save_dir=Path('./checkpoints')
    >>> )
    """
    
    def __init__(
        self,
        sae: 'SparseAutoencoder',
        config: 'SAEConfig',
        accelerator: str = 'auto',
        devices: int = 1
    ):
        """
        Parameters:
        -----------
        sae : SparseAutoencoder
            Sparse autoencoder model
        config : SAEConfig
            Configuration object
        accelerator : str
            Accelerator type ('auto', 'gpu', 'cpu', 'tpu', etc.)
        devices : int
            Number of devices to use
        """
        self.sae = sae
        self.config = config
        self.accelerator = accelerator
        self.devices = devices
        
        # Create Lightning module
        self.lightning_module = SAELightningModule(sae, config)
    
    def train(
        self,
        activations: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.1,
        log_interval: int = 10,
        save_dir: Optional[Path] = None,
        **trainer_kwargs
    ) -> Dict[str, list]:
        """
        Train SAE
        
        Parameters:
        -----------
        activations : torch.Tensor, shape (n_samples, d_in)
            Training data (activation vectors)
        num_epochs : int
            Number of epochs
        batch_size : int
            Batch size
        validation_split : float
            Validation split ratio
        log_interval : int
            Logging interval (steps)
        save_dir : Path, optional
            Directory to save model checkpoints
        **trainer_kwargs : dict
            Additional arguments for pl.Trainer
        
        Returns:
        --------
        history : Dict[str, list]
            Training history (can be extracted from logger)
        """
        # Create data module
        data_module = SAEDataModule(
            activations,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        # Setup callbacks
        callbacks = []
        
        # Add checkpoint callback if save_dir is provided
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=save_dir,
                filename='sae-{epoch:02d}-{val/total_loss:.4f}',
                monitor='val/total_loss',
                mode='min',
                save_top_k=1,
                save_last=True
            )
            callbacks.append(checkpoint_callback)
        
        # Add learning rate monitor
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
        
        # Train model
        trainer.fit(self.lightning_module, data_module)
        
        # Extract history from logger
        history = self._extract_history_from_logger(trainer)
        
        # Save training history
        if save_dir is not None:
            self.save_history(save_dir / 'training_history.json', history)
        
        return history
    
    def _extract_history_from_logger(self, trainer: pl.Trainer) -> Dict[str, list]:
        """Extract training history from Lightning logger"""
        if trainer.logger is None:
            return {}
        
        # For CSVLogger, read the metrics.csv file
        import pandas as pd
        from pytorch_lightning.loggers import CSVLogger
        
        if isinstance(trainer.logger, CSVLogger):
            metrics_file = Path(trainer.logger.log_dir) / 'metrics.csv'
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                
                # Organize into history dict
                history = {}
                
                # Extract all metric columns (excluding epoch and step)
                metric_cols = [col for col in df.columns if col not in ['epoch', 'step']]
                
                for col in metric_cols:
                    # Remove NaN values and convert to list
                    values = df[col].dropna().tolist()
                    if values:
                        # Clean column name (remove '/' and replace with '_')
                        clean_name = col.replace('/', '_')
                        history[clean_name] = values
                
                return history
        
        # Fallback: return empty dict if logger type is not supported
        return {}
    
    def save_history(self, path: Path, history: Dict[str, list]):
        """Save training history to JSON"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Saved training history: {path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint"""
        self.lightning_module = SAELightningModule.load_from_checkpoint(
            checkpoint_path,
            sae=self.sae,
            config=self.config
        )
        print(f"Loaded checkpoint: {checkpoint_path}")


# Example usage
if __name__ == "__main__":
    # Assume SAEConfig and SparseAutoencoder are defined as in your original code
    
    # Generate sample data
    activations = torch.randn(10000, 512)
    
    # Configure SAE
    config = SAEConfig(
        d_in=512,
        expansion_factor=4,
        l1_coefficient=1e-3,
        learning_rate=1e-3
    )
    
    # Create model
    sae = SparseAutoencoder(config)
    
    # Create trainer
    trainer = SAETrainer(sae, config)
    
    # Train
    history = trainer.train(
        activations,
        num_epochs=50,
        batch_size=256,
        validation_split=0.1,
        save_dir=Path('./checkpoints')
    )
    
    print("Training completed!")