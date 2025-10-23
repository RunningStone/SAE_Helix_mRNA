"""
Neural Network Probe for Regression

A more powerful probe model using PyTorch neural networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Tuple
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class MLPProbe(nn.Module):
    """
    Multi-layer Perceptron for regression
    
    Architecture:
    -------------
    Input -> Hidden1 -> ReLU -> Dropout -> Hidden2 -> ReLU -> Dropout -> Output
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256], 
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class SimpleMLPProbe(nn.Module):
    """
    Simple 2-layer MLP for regression (lightweight version)
    
    Architecture:
    -------------
    Input -> Linear -> ReLU -> Linear -> Output
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class NeuralProbeTrainer:
    """
    Trainer for neural network probe
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256],
                 dropout: float = 0.2, learning_rate: float = 1e-3,
                 batch_size: int = 256, epochs: int = 100,
                 early_stopping_patience: int = 10,
                 device: str = 'cpu', model_type: str = 'simple'):
        """
        Parameters:
        -----------
        input_dim : int
            Input feature dimension
        hidden_dims : list or int
            Hidden layer dimensions (list for MLPProbe, int for SimpleMLPProbe)
        dropout : float
            Dropout rate (only for MLPProbe)
        learning_rate : float
            Learning rate
        batch_size : int
            Batch size for training
        epochs : int
            Maximum number of epochs
        early_stopping_patience : int
            Patience for early stopping
        device : str
            Device to use ('cpu' or 'cuda')
        model_type : str
            'simple' for 2-layer MLP, 'deep' for multi-layer MLP
        """
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.model_type = model_type
        
        # Create model based on type
        if model_type == 'simple':
            # Simple 2-layer MLP
            hidden_dim = hidden_dims[0] if isinstance(hidden_dims, list) else hidden_dims
            self.model = SimpleMLPProbe(input_dim, hidden_dim).to(device)
        else:
            # Deep MLP with dropout
            self.model = MLPProbe(input_dim, hidden_dims, dropout).to(device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # For tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Train the probe model
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation labels
        verbose : bool
            Whether to print progress
        
        Returns:
        --------
        history : Dict
            Training history
        """
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            val_loader = None
        
        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * len(batch_X)
            
            train_loss /= len(train_dataset)
            self.train_losses.append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item() * len(batch_X)
                
                val_loss /= len(val_dataset)
                self.val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}")
                
                # Early stopping check
                if self.patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - "
                          f"Train Loss: {train_loss:.4f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'final_epoch': epoch + 1
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        
        Returns:
        --------
        predictions : np.ndarray
            Predicted values
        """
        self.model.eval()
        
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_X, in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        y : np.ndarray
            True labels
        
        Returns:
        --------
        metrics : Dict
            Performance metrics
        """
        predictions = self.predict(X)
        
        return {
            'r2': r2_score(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions)
        }
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
