"""
SAE Analyser Pipeline for mRNA Foundation Models

This module provides a complete pipeline for:
1. Loading mRNA-FM models (e.g., Helix mRNA)
2. Loading datasets from JSON files
3. Training SAE models for each transformer block
4. Extracting and saving sparse activation matrices

Usage:
------
>>> from src.pipeline.sae_analyser import SAEAnalyser
>>> 
>>> analyser = SAEAnalyser(
>>>     model_name='helical.models.helix_mrna.HelixmRNA',
>>>     data_dir='./data/processed_chunks',
>>>     output_dir='./outputs/sae_analysis',
>>>     device='cuda'
>>> )
>>> 
>>> # Train SAE models for all blocks
>>> analyser.train_all_saes(
>>>     num_epochs=100,
>>>     batch_size=256,
>>>     expansion_factor=4,
>>>     l1_coefficient=1e-3
>>> )
>>> 
>>> # Extract and save sparse activations
>>> analyser.extract_sparse_activations(
>>>     save_format='npz'  # or 'pt' for PyTorch
>>> )
"""

import sys
from pathlib import Path
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import importlib
from scipy import sparse as sp
from torch.utils.data import DataLoader

# Import local modules
from src.model.activation_extractor import ActivationExtractor
from src.model.multi_sae_lightning import MultiLayerSAETrainer
from src.model.sparse_autoencoder import SparseAutoencoder


class SAEAnalyser:
    """
    Complete pipeline for SAE analysis on mRNA foundation models
    
    Workflow:
    ---------
    1. Initialize mRNA-FM model
    2. Load dataset from JSON chunks
    3. Extract activations from all transformer blocks
    4. Train SAE models for each block
    5. Extract and save sparse activation matrices

    [check code blocks]:
    Run pipeline :: main function to run 
    Foundation Model waiting for analysis :: load model, load data 
    infer main model with hooks :: main model inference to get activations for SAE training 
    Train SAE and extract SAE emb :: train a list of SAE models for each block, extract and save sparse activation matrices
    load and save data :: utils functions fo saving loading etc.
    Parameters:
    -----------
    model_name : str
        Full import path to the model class (e.g., 'helical.models.helix_mrna.HelixmRNA')
    data_dir : str or Path
        Directory containing JSON chunk files
    output_dir : str or Path
        Directory to save SAE models and sparse activations
    device : str
        Device to use ('cuda' or 'cpu')
    max_length : int
        Maximum sequence length for the model
    batch_size : int
        Batch size for data processing
    """
    
    def __init__(
        self,
        model_name: str,
        input_data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        device: str = 'cuda',
        max_length: int = 150,
        batch_size: int = 8,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize SAE Analyser
        
        Parameters:
        -----------
        model_name : str
            Model name to use
        input_data_dir : str or Path
            Directory containing input JSON data files
        output_dir : str or Path
            Directory for all outputs (SAE checkpoints, sparse activations, etc.)
        device : str
            Device to use ('cuda' or 'cpu')
        max_length : int
            Maximum sequence length
        batch_size : int
            Batch size for processing
        cache_dir : str or Path, optional
            Directory containing cached activations/embeddings. If None:
            - Will compute activations and save to output_dir
            If provided:
            - Will load cached data from cache_dir (must exist)
            - Will skip model initialization
        """
        self.model_name = model_name
        self.input_data_dir = Path(input_data_dir)
        self.output_dir = Path(output_dir)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Create output directories
        self.sae_checkpoint_dir = self.output_dir / 'sae_checkpoints'
        self.sparse_activation_dir = self.output_dir / 'sparse_activations'
        self.activation_cache_dir = self.output_dir / 'activation_cache'
        self.embedding_dir = self.output_dir / 'embeddings'
        self.sae_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_activation_dir.mkdir(parents=True, exist_ok=True)
        self.activation_cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine if we should use cache
        use_cache = self.cache_dir is not None
        
        # Initialize model (skip if using cache)
        self.model = None
        self.model_wrapper = None
        
        if not use_cache:
            # No cache provided, need to initialize model for computation
            print(f"Initializing model: {model_name}")
            self.model, self.model_wrapper = self._initialize_model()
            
            # Freeze model parameters
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"Model initialized and frozen on device: {self.device}")
        else:
            # Cache provided, skip model initialization
            print(f"Using cached data from: {self.cache_dir}")
            print("Skipping model initialization")
        
        # Storage for activations and SAEs
        self.layer_activations: Optional[Dict[str, torch.Tensor]] = None
        self.sae_trainer: Optional[MultiLayerSAETrainer] = None
        self.trained_saes: Optional[Dict[str, SparseAutoencoder]] = None
        
        # Storage for sequence metadata (for mapping)
        self.sequence_metadata: Optional[List[Dict]] = None
        self.token_to_sequence_mapping: Optional[List[Dict]] = None
        
        # Storage for embeddings
        self.embeddings: Optional[torch.Tensor] = None
        
        # Try to load from cache if cache_dir is provided
        if use_cache:
            self._load_from_cache()

    ########################################################################################
    #    Run pipeline 
    ########################################################################################

    def run_full_pipeline(
        self,
        num_epochs: int = 100,
        batch_size: int = 256,
        expansion_factor: int = 4,
        l1_coefficient: float = 1e-3,
        save_format: str = 'npz'
    ) -> Dict[str, Path]:
        """
        Run the complete SAE analysis pipeline
        
        Parameters:
        -----------
        num_epochs : int
            Number of SAE training epochs
        batch_size : int
            Batch size for SAE training
        expansion_factor : int
            SAE expansion factor
        l1_coefficient : float
            L1 sparsity coefficient
        save_format : str
            Format to save sparse matrices
        
        Returns:
        --------
        saved_files : Dict[str, Path]
            Dictionary mapping layer names to saved sparse activation files
        """
        print(f"\n{'='*80}")
        print("Running Full SAE Analysis Pipeline")
        print(f"{'='*80}")
        
        # Step 1: Load dataset
        sequences = self.load_dataset()
        
        # Step 2: Extract activations
        self.extract_activations(sequences)
        
        # Step 3: Train SAEs
        self.train_all_saes(
            num_epochs=num_epochs,
            batch_size=batch_size,
            expansion_factor=expansion_factor,
            l1_coefficient=l1_coefficient
        )
        
        # Step 4: Extract and save sparse activations
        saved_files = self.extract_sparse_activations(save_format=save_format)
        
        print(f"\n{'='*80}")
        print("✓ Full pipeline completed successfully!")
        print(f"{'='*80}")
        
        return saved_files
    ########################################################################################
    #    Foundation Model waiting for analysis
    ########################################################################################

    def _initialize_model(self) -> Tuple:
        """
        Dynamically import and initialize the mRNA-FM model
        
        Returns:
        --------
        model : torch.nn.Module
            The underlying model
        wrapper : object
            The model wrapper (e.g., HelixmRNA instance)
        """
        # Parse model name
        parts = self.model_name.rsplit('.', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid model_name format: {self.model_name}")
        
        module_path, class_name = parts
        
        # Import module
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Failed to import {module_path}: {e}")
        
        # Get model class
        if not hasattr(module, class_name):
            raise AttributeError(f"Module {module_path} has no class {class_name}")
        
        ModelClass = getattr(module, class_name)
        
        # Get config class (assume it's named {ModelClass}Config)
        config_class_name = f"{class_name}Config"
        if not hasattr(module, config_class_name):
            raise AttributeError(f"Module {module_path} has no class {config_class_name}")
        
        ConfigClass = getattr(module, config_class_name)
        
        # Initialize model
        config = ConfigClass(
            batch_size=self.batch_size,
            max_length=self.max_length,
            device=self.device
        )
        
        wrapper = ModelClass(configurer=config)
        
        # Get the underlying model
        if hasattr(wrapper, 'model'):
            model = wrapper.model
        else:
            model = wrapper
        
        return model, wrapper

    def load_dataset(self, max_chunks: int = None) -> List[str]:
        """
        Load mRNA sequences from JSON chunk files
        
        Parameters:
        -----------
        max_chunks : int, optional
            Maximum number of chunk files to load. If None, load all files.
        
        Returns:
        --------
        sequences : List[str]
            List of mRNA sequences
        """
        print(f"\nLoading dataset from: {self.input_data_dir}")
        
        # Find all JSON files
        json_files = sorted(self.input_data_dir.glob('*.json'))
        
        # Limit number of chunks if specified
        if max_chunks is not None:
            json_files = json_files[:max_chunks]
            print(f"Limiting to first {max_chunks} chunk files")
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.input_data_dir}")
        
        print(f"Found {len(json_files)} JSON chunk files")
        
        sequences = []
        sequence_metadata = []  # Store metadata for mapping
        has_dna = False
        
        for json_file in tqdm(json_files, desc="Loading JSON chunks"):
            with open(json_file, 'r') as f:
                chunk_data = json.load(f)
            
            # Extract sequences from each data item
            for item_idx, item in enumerate(chunk_data):
                if 'sequence' in item:
                    seq = item['sequence']
                    # Check if this is DNA (contains T)
                    if 'T' in seq:
                        has_dna = True
                        # Convert DNA to RNA (T -> U)
                        seq = seq.replace('T', 'U')
                    sequences.append(seq)
                    
                    # Store metadata for mapping
                    sequence_metadata.append({
                        'sequence_idx': len(sequences) - 1,
                        'source_file': json_file.name,
                        'item_idx_in_file': item_idx,
                        'sequence_length': len(seq)
                    })
                else:
                    print(f"Warning: No 'sequence' field in item from {json_file.name}")
        
        print(f"Loaded {len(sequences)} sequences")
        
        if has_dna:
            print("✓ Converted DNA sequences (T) to RNA sequences (U)")
        
        if len(sequences) == 0:
            raise ValueError("No sequences found in dataset")
        
        # Store metadata for later use
        self.sequence_metadata = sequence_metadata
        
        return sequences


    ########################################################################################
    #    infer main model with hooks
    ########################################################################################
    def extract_activations(self, sequences: List[str], save_embeddings: bool = True) -> Dict[str, torch.Tensor]:
        """
        Extract activations from all transformer blocks and optionally save embeddings
        
        Parameters:
        -----------
        sequences : List[str]
            List of mRNA sequences
        save_embeddings : bool
            Whether to extract and save final embeddings
        
        Returns:
        --------
        layer_activations : Dict[str, torch.Tensor]
            Dictionary mapping layer names to activation tensors
        """
        # Check if model is initialized
        if self.model is None or self.model_wrapper is None:
            raise RuntimeError("Model not initialized. Cannot extract activations without model. "
                             "If you want to use cached activations, provide cache_dir parameter.")
        
        print(f"\nExtracting activations from {len(sequences)} sequences")
        
        # Process sequences with the model
        dataset = self.model_wrapper.process_data(sequences)
        print(f"Dataset size: {len(dataset)}")
        
        # Create activation extractor
        extractor = ActivationExtractor(self.model)
        
        # Register hooks for mixer modules (transformer blocks)
        extractor.register_hooks(
            layer_filter=lambda name, m: (
                'mixer' in name.lower() and 
                name.endswith('mixer')  # Only the mixer itself, not submodules
            )
        )
        
        print(f"Registered {len(extractor.hooks)} hooks for activation extraction")
        
        # Create DataLoader
        config = self.model_wrapper.configurer.config
        dataloader = DataLoader(
            dataset,
            collate_fn=self.model_wrapper._collate_fn,
            batch_size=config["batch_size"],
            shuffle=False,
        )
        
        # Collect activations, embeddings, and build token-to-sequence mapping
        all_layer_activations = {}
        all_embeddings = [] if save_embeddings else None
        token_to_sequence_mapping = []
        current_token_idx = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting activations")):
                extractor.clear_activations()
                
                input_ids = batch["input_ids"].to(self.device)
                special_tokens_mask = batch["special_tokens_mask"].to(self.device)
                attention_mask = 1 - special_tokens_mask
                
                # Forward pass - get final embeddings
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Extract embeddings if requested
                if save_embeddings:
                    # Get the final hidden states (embeddings)
                    # For most models, this is the last hidden state
                    if hasattr(outputs, 'last_hidden_state'):
                        embeddings = outputs.last_hidden_state
                    elif isinstance(outputs, torch.Tensor):
                        embeddings = outputs
                    else:
                        # Try to get from tuple/list output
                        embeddings = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                    
                    # Flatten and store: (batch_size, seq_len, hidden_dim) -> (batch_size * seq_len, hidden_dim)
                    batch_size, seq_len, hidden_dim = embeddings.shape
                    flattened_emb = embeddings.reshape(-1, hidden_dim).cpu()
                    all_embeddings.append(flattened_emb)
                
                # Get batch activations
                batch_activations = extractor.get_activations()
                
                # Get batch size and sequence length
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                
                # Build token-to-sequence mapping for this batch
                for i in range(batch_size):
                    sequence_idx = batch_idx * config["batch_size"] + i
                    if sequence_idx < len(sequences):
                        # Get actual sequence length (before padding)
                        actual_seq_length = self.sequence_metadata[sequence_idx]['sequence_length']
                        
                        # Record mapping for all tokens (including padding)
                        token_to_sequence_mapping.append({
                            'sequence_idx': sequence_idx,
                            'token_start': current_token_idx,
                            'token_end': current_token_idx + seq_len,
                            'num_tokens': seq_len,
                            'actual_sequence_length': actual_seq_length,
                            'source_file': self.sequence_metadata[sequence_idx]['source_file'],
                            'item_idx_in_file': self.sequence_metadata[sequence_idx]['item_idx_in_file']
                        })
                        current_token_idx += seq_len
                
                # Accumulate activations
                for layer_name, activation in batch_activations.items():
                    # activation shape: (batch_size, seq_len, hidden_dim)
                    # Flatten to (batch_size * seq_len, hidden_dim)
                    batch_size, seq_len, hidden_dim = activation.shape
                    flattened = activation.reshape(-1, hidden_dim).cpu()
                    
                    if layer_name not in all_layer_activations:
                        all_layer_activations[layer_name] = []
                    
                    all_layer_activations[layer_name].append(flattened)
        
        # Concatenate all batches
        layer_activations = {}
        for layer_name, activation_list in all_layer_activations.items():
            layer_activations[layer_name] = torch.cat(activation_list, dim=0)
        
        extractor.remove_hooks()
        
        print(f"\nExtracted activations from {len(layer_activations)} layers:")
        for layer_name, activation in layer_activations.items():
            print(f"  {layer_name:60s} | Shape: {activation.shape}")
        
        # Store mapping information
        self.token_to_sequence_mapping = token_to_sequence_mapping
        
        # Save mapping to JSON
        mapping_file = self.sparse_activation_dir / 'token_to_sequence_mapping.json'
        with open(mapping_file, 'w') as f:
            json.dump({
                'total_sequences': len(sequences),
                'total_tokens': current_token_idx,
                'max_length': self.max_length,
                'mapping': token_to_sequence_mapping
            }, f, indent=2)
        
        print(f"\n✓ Saved token-to-sequence mapping to: {mapping_file}")
        print(f"  Total sequences: {len(sequences)}")
        print(f"  Total tokens: {current_token_idx}")
        
        # Save embeddings if extracted
        if save_embeddings and all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)
            self.embeddings = embeddings
            self._save_embeddings(embeddings)
        
        # Save activations to cache
        self._save_activations_cache(layer_activations)
        
        self.layer_activations = layer_activations
        return layer_activations
    
    ########################################################################################
    #    Train SAE and extract SAE emb
    ########################################################################################
    def train_all_saes(
        self,
        num_epochs: int = 100,
        batch_size: int = 256,
        expansion_factor: int = 4,
        l1_coefficient: float = 1e-3,
        learning_rate: float = 1e-3,
        validation_split: float = 0.1,
        log_interval: int = 10
    ) -> Dict[str, SparseAutoencoder]:
        """
        Train SAE models for all transformer blocks
        
        Parameters:
        -----------
        num_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for SAE training
        expansion_factor : int
            SAE expansion factor (d_hidden = expansion_factor * d_in)
        l1_coefficient : float
            L1 sparsity coefficient
        learning_rate : float
            Learning rate
        validation_split : float
            Validation split ratio
        log_interval : int
            Logging interval
        
        Returns:
        --------
        trained_saes : Dict[str, SparseAutoencoder]
            Dictionary of trained SAE models
        """
        if self.layer_activations is None:
            raise ValueError("No activations found. Run extract_activations() first.")
        
        print(f"\n{'='*80}")
        print("Training SAE models for all layers")
        print(f"{'='*80}")
        
        print(f"\nSAE Configuration:")
        print(f"  Expansion factor: {expansion_factor}")
        print(f"  L1 coefficient: {l1_coefficient}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Num epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        
        # Create Multi-Layer SAE Trainer
        self.sae_trainer = MultiLayerSAETrainer(
            layer_activations=self.layer_activations,
            expansion_factor=expansion_factor,
            l1_coefficient=l1_coefficient,
            learning_rate=learning_rate,
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            devices=1
        )
        
        # Display SAE configurations
        print(f"\nSAE configurations for {len(self.sae_trainer.configs)} layers:")
        for layer_name, config in self.sae_trainer.configs.items():
            print(f"  {layer_name:40s} | d_in: {config.d_in:4d} | d_hidden: {config.d_hidden:5d}")
        
        # Train
        print(f"\nStarting training...")
        print(f"Checkpoint directory: {self.sae_checkpoint_dir}")
        
        histories = self.sae_trainer.train_all(
            num_epochs=num_epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            log_interval=log_interval,
            save_dir=self.sae_checkpoint_dir
        )
        
        # Get trained SAEs
        self.trained_saes = self.sae_trainer.get_all_saes()
        
        print(f"\n{'='*80}")
        print("✓ SAE training completed!")
        print(f"{'='*80}")
        print(f"Trained {len(self.trained_saes)} SAE models")
        
        return self.trained_saes

    def extract_sparse_activations(
        self,
        save_format: str = 'npz',
        chunk_size: int = 1000
    ) -> Dict[str, Path]:
        """
        Extract sparse activation matrices from trained SAEs
        
        Parameters:
        -----------
        save_format : str
            Format to save sparse matrices ('npz' for scipy sparse, 'pt' for PyTorch)
        chunk_size : int
            Process activations in chunks to save memory
        
        Returns:
        --------
        saved_files : Dict[str, Path]
            Dictionary mapping layer names to saved file paths
        """
        if self.trained_saes is None:
            raise ValueError("No trained SAEs found. Run train_all_saes() first.")
        
        if self.layer_activations is None:
            raise ValueError("No activations found. Run extract_activations() first.")
        
        print(f"\n{'='*80}")
        print("Extracting sparse activations")
        print(f"{'='*80}")
        
        saved_files = {}
        
        for layer_name, sae in tqdm(self.trained_saes.items(), desc="Processing layers"):
            print(f"\nProcessing layer: {layer_name}")
            
            # Get activations for this layer
            activations = self.layer_activations[layer_name]
            n_samples = activations.shape[0]
            
            print(f"  Activation shape: {activations.shape}")
            print(f"  Processing in chunks of {chunk_size}")
            
            # Process in chunks
            all_sparse_features = []
            
            sae.eval()
            with torch.no_grad():
                for i in range(0, n_samples, chunk_size):
                    chunk = activations[i:i+chunk_size]
                    
                    # Encode to get sparse features
                    sparse_features = sae.encode(chunk)
                    all_sparse_features.append(sparse_features.cpu())
            
            # Concatenate all chunks
            sparse_features = torch.cat(all_sparse_features, dim=0)
            
            # Convert to numpy for analysis
            sparse_features_np = sparse_features.numpy()
            
            # Calculate actual sparsity (ReLU should make many zeros)
            total_elements = sparse_features_np.size
            zero_elements = np.sum(sparse_features_np == 0)
            nonzero_elements = total_elements - zero_elements
            actual_sparsity = zero_elements / total_elements
            
            print(f"  Sparse feature shape: {sparse_features.shape}")
            print(f"  Total elements: {total_elements:,}")
            print(f"  Zero elements: {zero_elements:,}")
            print(f"  Non-zero elements: {nonzero_elements:,}")
            print(f"  Actual sparsity (zeros): {actual_sparsity:.2%}")
            
            # Create scipy sparse matrix (CSR format) only if truly sparse
            if actual_sparsity > 0.5:  # Only use sparse format if >50% zeros
                sparse_matrix = sp.csr_matrix(sparse_features_np)
                print(f"  Storage: scipy.sparse.csr_matrix")
                print(f"  Compression ratio: {sparse_matrix.data.nbytes / sparse_features_np.nbytes:.2%}")
            else:
                # Keep as dense if not sparse enough
                sparse_matrix = sparse_features_np
                print(f"  Storage: dense numpy array (not sparse enough for CSR)")
            
            # Save sparse matrix
            safe_layer_name = layer_name.replace('.', '_')
            
            if save_format == 'npz':
                save_path = self.sparse_activation_dir / f"{safe_layer_name}_sparse.npz"
                if isinstance(sparse_matrix, np.ndarray):
                    # Save as dense array
                    np.savez_compressed(save_path, data=sparse_matrix)
                else:
                    # Save as scipy sparse
                    sp.save_npz(save_path, sparse_matrix)
            elif save_format == 'pt':
                save_path = self.sparse_activation_dir / f"{safe_layer_name}_sparse.pt"
                torch.save({
                    'sparse_features': sparse_features,
                    'shape': sparse_features.shape,
                    'layer_name': layer_name,
                    'sparsity': actual_sparsity,
                    'nonzero_count': nonzero_elements
                }, save_path)
            else:
                raise ValueError(f"Unsupported save_format: {save_format}")
            
            saved_files[layer_name] = save_path
            file_size_mb = save_path.stat().st_size / 1024 / 1024
            print(f"  Saved to: {save_path}")
            print(f"  File size: {file_size_mb:.2f} MB")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_layers': len(self.trained_saes),
            'layer_names': list(self.trained_saes.keys()),
            'save_format': save_format,
            'files': {name: str(path) for name, path in saved_files.items()}
        }
        
        metadata_path = self.sparse_activation_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*80}")
        print("✓ Sparse activation extraction completed!")
        print(f"{'='*80}")
        print(f"Saved {len(saved_files)} sparse activation matrices")
        print(f"Metadata saved to: {metadata_path}")
        
        return saved_files

    ########################################################################################
    #    load and save data
    ########################################################################################
    
    def _load_from_cache(self):
        """Load cached data from cache_dir (activations, embeddings, mapping)"""
        print(f"\n{'='*80}")
        print("Loading Cached Data")
        print(f"{'='*80}")
        
        if self.cache_dir is None:
            raise ValueError("cache_dir is None, cannot load cache")
        
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory does not exist: {self.cache_dir}")
        
        # Load activations from cache_dir
        cache_file = self.cache_dir / 'activation_cache' / 'layer_activations.pt'
        if not cache_file.exists():
            raise FileNotFoundError(f"Activation cache not found: {cache_file}")
        
        try:
            cache_data = torch.load(cache_file, map_location='cpu')
            self.layer_activations = cache_data['layer_activations']
            print(f"✓ Loaded activation cache from: {cache_file}")
            print(f"  Layers: {len(self.layer_activations)}")
            for layer_name, activation in self.layer_activations.items():
                print(f"  {layer_name:60s} | Shape: {activation.shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load activation cache: {e}")
        
        # Load embeddings from cache_dir
        embedding_file = self.cache_dir / 'embeddings' / 'embeddings.pt'
        if embedding_file.exists():
            try:
                data = torch.load(embedding_file, map_location='cpu')
                self.embeddings = data['embeddings']
                print(f"✓ Loaded embeddings from: {embedding_file}")
                print(f"  Shape: {self.embeddings.shape}")
            except Exception as e:
                print(f"⚠ Warning: Failed to load embeddings: {e}")
        else:
            print(f"  No embeddings found in cache (optional)")
        
        # Load mapping from cache_dir
        mapping_file = self.cache_dir / 'sparse_activations' / 'token_to_sequence_mapping.json'
        if mapping_file.exists():
            try:
                with open(mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                self.token_to_sequence_mapping = mapping_data['mapping']
                print(f"✓ Loaded token-to-sequence mapping from: {mapping_file}")
                print(f"  Sequences: {len(self.token_to_sequence_mapping)}")
            except Exception as e:
                print(f"⚠ Warning: Failed to load mapping: {e}")
        else:
            print(f"  No token-to-sequence mapping found in cache (optional)")
        
    def _save_activations_cache(self, layer_activations: Dict[str, torch.Tensor]):
        """
        Save layer activations to cache for future use
        
        Parameters:
        -----------
        layer_activations : Dict[str, torch.Tensor]
            Dictionary mapping layer names to activation tensors
        """
        print(f"\n{'='*80}")
        print("Saving Activation Cache")
        print(f"{'='*80}")
        
        cache_file = self.activation_cache_dir / 'layer_activations.pt'
        
        # Save activations
        torch.save({
            'layer_activations': layer_activations,
            'layer_names': list(layer_activations.keys()),
            'shapes': {name: list(act.shape) for name, act in layer_activations.items()}
        }, cache_file)
        
        # Calculate total size
        total_size = sum(act.numel() * act.element_size() for act in layer_activations.values())
        
        print(f"✓ Saved activation cache to: {cache_file}")
        print(f"  Layers: {len(layer_activations)}")
        print(f"  Total size: {total_size / 1024 / 1024:.2f} MB")
        
        # Save metadata
        metadata = {
            'num_layers': len(layer_activations),
            'layer_names': list(layer_activations.keys()),
            'shapes': {name: list(act.shape) for name, act in layer_activations.items()},
            'total_size_mb': total_size / 1024 / 1024
        }
        
        metadata_file = self.activation_cache_dir / 'cache_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved cache metadata to: {metadata_file}")
    
    
    def _save_embeddings(self, embeddings: torch.Tensor):
        """
        Save final embeddings to file
        
        Parameters:
        -----------
        embeddings : torch.Tensor
            Final embeddings tensor (num_tokens, hidden_dim)
        """
        print(f"\n{'='*80}")
        print("Saving Embeddings")
        print(f"{'='*80}")
        
        embedding_file = self.embedding_dir / 'embeddings.pt'
        
        # Save embeddings
        torch.save({
            'embeddings': embeddings,
            'shape': list(embeddings.shape)
        }, embedding_file)
        
        # Calculate size
        size_mb = embeddings.numel() * embeddings.element_size() / 1024 / 1024
        
        print(f"✓ Saved embeddings to: {embedding_file}")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Size: {size_mb:.2f} MB")
        
        # Save metadata
        metadata = {
            'shape': list(embeddings.shape),
            'dtype': str(embeddings.dtype),
            'size_mb': size_mb
        }
        
        metadata_file = self.embedding_dir / 'embedding_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved embedding metadata to: {metadata_file}")
    
    
    


# Example usage
if __name__ == "__main__":
    # Example configuration
    analyser = SAEAnalyser(
        model_name='helical.models.helix_mrna.HelixmRNA',
        data_dir='./data/processed_chunks',
        output_dir='./outputs/sae_analysis',
        device='cuda',
        max_length=150,
        batch_size=8
    )
    
    # Run full pipeline
    saved_files = analyser.run_full_pipeline(
        num_epochs=100,
        batch_size=256,
        expansion_factor=4,
        l1_coefficient=1e-3,
        save_format='npz'
    )
    
    print("\nSaved sparse activation files:")
    for layer_name, file_path in saved_files.items():
        print(f"  {layer_name}: {file_path}")
