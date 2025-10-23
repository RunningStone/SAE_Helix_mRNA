"""
Sparse Auto-Encoder (SAE) Implementation
from paper: https://arxiv.org/pdf/2309.08600

Sparse autoencoder for learning interpretable feature dictionary from transformers block outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SAEConfig:
    """SAE config class"""
    
    d_in: int  # input dimension (activation vector dimension)
    expansion_factor: int = 4  # overcomplete ratio R (hidden layer dimension = R * d_in)
    l1_coefficient: float = 1e-3  # L1 sparsity penalty coefficient α
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    normalize_decoder: bool = True  # whether to normalize decoder weights
    tied_weights: bool = True  # whether to use weight tying (encoder and decoder share weights)
    
    @property
    def d_hidden(self) -> int:
        """hidden layer dimension"""
        return self.expansion_factor * self.d_in


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder [basiclly from https://arxiv.org/pdf/2309.08600]
    
    Architecture:
        Encoding: c = ReLU(Wx + b)
        Decoding: x̂ = W^T c  (if tied_weights=True)
    
    Loss:
        L = ||x - x̂||² + α||c||₁
    
    Parameters:
    -----------
    config : SAEConfig
        SAE config object
    
    Example:
    --------
    >>> config = SAEConfig(d_in=512, expansion_factor=4, l1_coefficient=1e-3)
    >>> sae = SparseAutoencoder(config)
    >>> x = torch.randn(32, 512)  # batch_size=32, d_in=512
    >>> x_reconstructed, features, loss_dict = sae(x)
    >>> print(f"Sparsity: {(features == 0).float().mean():.2%}")
    """
    
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        
        # encoder: W ∈ R^{d_hidden × d_in}
        self.encoder = nn.Linear(config.d_in, config.d_hidden, bias=True)
        
        # decoder
        if config.tied_weights:
            # weight tying: decoder uses encoder's transpose
            self.decoder = None
        else:
            self.decoder = nn.Linear(config.d_hidden, config.d_in, bias=False)
        
        # initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """initialize weights"""
        # encoder weights: Xavier initialization
        nn.init.xavier_uniform_(self.encoder.weight)
        
        # encoder bias: initialized to negative value, to raise activation threshold
        # 更负的偏置可以产生更强的稀疏性（只有强信号才能激活）
        nn.init.constant_(self.encoder.bias, -1.0)
        
        # decoder weights (if not using weight tying)
        if not self.config.tied_weights:
            nn.init.xavier_uniform_(self.decoder.weight)
        
        # normalize decoder weights
        if self.config.normalize_decoder:
            self._normalize_decoder_weights()
    
    def _normalize_decoder_weights(self):
        """
        normalize decoder weights (row normalization)
        ensure ||f_i||₂ = 1, prevent model from avoiding sparsity penalty by amplifying feature vector norm
        """
        with torch.no_grad():
            if self.config.tied_weights:
                # normalize encoder weights row by row
                weight = self.encoder.weight
                weight.div_(weight.norm(dim=1, keepdim=True) + 1e-8)
            else:
                # normalize decoder weights column by column
                weight = self.decoder.weight
                weight.div_(weight.norm(dim=0, keepdim=True) + 1e-8)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        encode: map input to sparse feature space
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, d_in)
            input activation vector
        
        Returns:
        --------
        features : torch.Tensor, shape (batch_size, d_hidden)
            sparse feature coefficients c = ReLU(Wx + b)
        """
        features = F.relu(self.encoder(x))
        return features
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        decode: reconstruct input from sparse features
        
        Parameters:
        -----------
        features : torch.Tensor, shape (batch_size, d_hidden)
            sparse feature coefficients
        
        Returns:
        --------
        x_reconstructed : torch.Tensor, shape (batch_size, d_in)
            reconstructed activation vector x̂ = W^T c
        """
        if self.config.tied_weights:
            # use encoder weights' transpose
            x_reconstructed = F.linear(features, self.encoder.weight.t())
        else:
            x_reconstructed = self.decoder(features)
        
        return x_reconstructed
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        forward pass
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, d_in)
            input activation vector
        return_loss : bool
            whether to compute and return loss
        
        Returns:
        --------
        x_reconstructed : torch.Tensor, shape (batch_size, d_in)
            reconstructed activation vector
        features : torch.Tensor, shape (batch_size, d_hidden)
            sparse feature coefficients
        loss_dict : Dict[str, torch.Tensor] or None
            loss dictionary, containing:
                - 'total_loss': total loss
                - 'reconstruction_loss': reconstruction loss
                - 'sparsity_loss': sparsity penalty
                - 'l0_norm': L0 norm (number of non-zero features)
        """
        # encode
        features = self.encode(x)
        
        # decode
        x_reconstructed = self.decode(features)
        
        # compute loss
        loss_dict = None
        if return_loss:
            loss_dict = self.compute_loss(x, x_reconstructed, features)
        
        return x_reconstructed, features, loss_dict
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        x_reconstructed: torch.Tensor, 
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        compute loss
        
        L = ||x - x̂||² + α||c||₁
        
        Parameters:
        -----------
        x : torch.Tensor
            original input
        x_reconstructed : torch.Tensor
            reconstructed output
        features : torch.Tensor
            sparse feature coefficients
        
        Returns:
        --------
        loss_dict : Dict[str, torch.Tensor]
            loss dictionary
        """
        # reconstruction loss: MSE
        reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction='mean')
        
        # sparsity penalty: L1 norm
        sparsity_loss = features.abs().mean()
        
        # total loss
        total_loss = reconstruction_loss + self.config.l1_coefficient * sparsity_loss
        
        # L0 norm (number of non-zero features)
        l0_norm = (features != 0).float().sum(dim=1).mean()
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'sparsity_loss': sparsity_loss,
            'l0_norm': l0_norm,
        }
    
    def get_feature_dictionary(self) -> torch.Tensor:
        """
        get feature dictionary {f_0, f_1, ..., f_{d_hidden-1}}
        
        Returns:
        --------
        feature_dict : torch.Tensor, shape (d_hidden, d_in)
            feature dictionary matrix, each row is a feature vector
        """
        if self.config.tied_weights:
            return self.encoder.weight.data
        else:
            return self.decoder.weight.data.t()
    
    @torch.no_grad()
    def normalize_decoder_step(self):
        """
        normalize decoder weights after each training step
        called after each optimization step to keep ||f_i||₂ = 1
        """
        if self.config.normalize_decoder:
            self._normalize_decoder_weights()
    
    def get_sparsity_stats(self, features: torch.Tensor) -> Dict[str, float]:
        """
        compute sparsity statistics
        
        Parameters:
        -----------
        features : torch.Tensor, shape (batch_size, d_hidden)
            feature activations
        
        Returns:
        --------
        stats : Dict[str, float]
            sparsity statistics dictionary
        """
        with torch.no_grad():
            l0_norm = (features != 0).float().sum(dim=1).mean().item()
            l1_norm = features.abs().mean().item()
            max_activation = features.max().item()
            mean_activation = features[features > 0].mean().item() if (features > 0).any() else 0.0
            
            return {
                'l0_norm': l0_norm,  # 平均激活特征数
                'l1_norm': l1_norm,
                'max_activation': max_activation,
                'mean_activation': mean_activation,
                'sparsity': 1.0 - (l0_norm / self.config.d_hidden),  # 稀疏度百分比
            }


class TopKSparseAutoencoder(SparseAutoencoder):
    """
    Top-K sparse autoencoder  from Sup-G
    
    Use Top-K activation instead of ReLU + L1 penalty, directly control the number of activated features
    
    Parameters:
    -----------
    config : SAEConfig
        SAE config object
    k : int
        Number of top-k activations to keep
    """
    
    def __init__(self, config: SAEConfig, k: int = 50):
        super().__init__(config)
        self.k = k
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Top-K encoding: only keep the largest k activations
        """
        # through linear layer
        pre_activation = self.encoder(x)
        
        # Top-K selection
        topk_values, topk_indices = torch.topk(pre_activation, self.k, dim=1)
        
        # create sparse feature vector
        features = torch.zeros_like(pre_activation)
        features.scatter_(1, topk_indices, F.relu(topk_values))
        
        return features
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        x_reconstructed: torch.Tensor, 
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Top-K SAE only uses reconstruction loss (no L1 penalty)
        """
        reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction='mean')
        l0_norm = (features != 0).float().sum(dim=1).mean()
        
        return {
            'total_loss': reconstruction_loss,
            'reconstruction_loss': reconstruction_loss,
            'sparsity_loss': torch.tensor(0.0, device=x.device),
            'l0_norm': l0_norm,
        }



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
    
    # example forward
    x = torch.randn(1, 512)
    x_reconstructed, features, loss_dict = sae(x)
    print(loss_dict)
    
    # create TopK ver
    topk_sae = TopKSparseAutoencoder(config, k=50)
    
    # example forward
    x = torch.randn(1, 512)
    x_reconstructed, features, loss_dict = topk_sae(x)
    print(loss_dict)
    