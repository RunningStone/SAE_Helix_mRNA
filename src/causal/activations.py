"""
Step 3: Forward Propagation and Activation Collection

This module runs the mRNA-FM model on sequence pairs and collects:
1. Raw intermediate activations at target blocks
2. SAE sparse encodings at target blocks
3. Final embeddings for probe prediction
"""

import numpy as np
import torch
import scipy.sparse as sp
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm

from .base import BaseCausalStep, CausalAnalysisConfig, CausalDataManager


class ActivationCollector(BaseCausalStep):
    """
    Collect activations and sparse codes for sequence pairs
    
    Workflow:
    ---------
    1. Load mRNA-FM model and SAE models
    2. For each sequence pair:
       a. Run forward pass through mRNA-FM
       b. Collect raw activations at target blocks
       c. Encode activations with SAE to get sparse codes
       d. Collect final embeddings
    3. Save activation data for each pair
    """
    
    def __init__(self, config: CausalAnalysisConfig, 
                 data_manager: CausalDataManager):
        super().__init__(config, data_manager)
        self.model = None
        self.model_wrapper = None
        self.sae_models = {}
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run activation collection
        
        Returns:
        --------
        results : Dict
            - activations: List of activation data for each pair
            - collection_metadata: Metadata about collection process
        """
        self.log("="*80)
        self.log("Step 3: Collecting Activations and Sparse Codes")
        self.log("="*80)
        
        # Load sequence pairs
        sequence_pairs = self.data_manager.get_data('sequence_pairs')
        if sequence_pairs is None:
            pair_results = self.load_results('sequencepair_results')
            sequence_pairs = pair_results['sequence_pairs']
        
        self.log(f"Processing {len(sequence_pairs)} sequence pairs")
        
        # Initialize models
        self._initialize_models()
        
        # Collect activations for all pairs
        all_activations = []
        
        for pair in tqdm(sequence_pairs, desc="Collecting activations"):
            pair_activations = self._process_pair(pair)
            all_activations.append(pair_activations)
        
        # Compute metadata
        metadata = self._compute_metadata(all_activations)
        
        # Prepare results
        results = {
            'activations': all_activations,
            'collection_metadata': metadata
        }
        
        # Save results
        if self.config.save_intermediate:
            self.save_results(results)
        
        # Store in data manager
        self.data_manager.set_data('activations', all_activations)
        
        self.log("="*80)
        self.log("✓ Activation collection completed!", level='success')
        self.log("="*80)
        
        return results
    
    def _initialize_models(self):
        """Initialize mRNA-FM model and SAE models"""
        self.log("Initializing models...")
        
        # Load mRNA-FM model (reuse from SAEAnalyser)
        from src.pipeline.sae_analyser import SAEAnalyser
        
        analyser = SAEAnalyser(
            model_name='helical.models.helix_mrna.HelixmRNA',
            input_data_dir=self.config.data_dir,
            output_dir=self.config.step1_output_dir,
            device=self.config.device,
            max_length=150,
            batch_size=self.config.batch_size
        )
        
        self.model = analyser.model
        self.model_wrapper = analyser.model_wrapper
        self.model.eval()
        
        # Load SAE models - CRITICAL: Must succeed for causal analysis
        try:
            self._load_sae_models()
        except Exception as e:
            error_msg = (
                f"❌ CRITICAL ERROR: Failed to load SAE models!\n"
                f"Error: {str(e)}\n\n"
                f"Causal analysis REQUIRES SAE models to:\n"
                f"  1. Extract sparse feature activations\n"
                f"  2. Perform feature interventions\n"
                f"  3. Analyze causal relationships\n\n"
                f"Please ensure:\n"
                f"  - Step 1 SAE training completed successfully\n"
                f"  - SAE checkpoints exist in: {self.config.step1_output_dir}/sae_checkpoints/\n"
                f"  - Checkpoint format is correct (PyTorch Lightning format expected)\n"
            )
            self.log(error_msg, level='error')
            raise RuntimeError(error_msg) from e
        
        self.log("✓ Models initialized", level='success')
    
    def _load_sae_models(self):
        """Load trained SAE models for target blocks"""
        from src.model.sparse_autoencoder import SparseAutoencoder, SAEConfig
        
        step1_dir = Path(self.config.step1_output_dir)
        checkpoint_dir = step1_dir / 'sae_checkpoints'
        
        # Find latest checkpoint
        checkpoints = list(checkpoint_dir.glob('*.ckpt'))
        if not checkpoints:
            raise FileNotFoundError(f"No SAE checkpoints found in {checkpoint_dir}")
        
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        self.log(f"Loading SAE checkpoint: {latest_checkpoint.name}")
        
        # Load checkpoint (weights_only=False for custom classes)
        checkpoint = torch.load(latest_checkpoint, map_location=self.config.device, weights_only=False)
        
        # Get state_dict from PyTorch Lightning checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Extract SAE models for target blocks
        self.sae_models = {}
        for block_idx in self.config.target_blocks:
            layer_name = f"layers.{block_idx}.mixer"
            # PyTorch Lightning format: saes.layers_X_mixer.encoder.weight
            sae_prefix = f"saes.layers_{block_idx}_mixer."
            
            # Extract SAE weights for this layer
            sae_state = {}
            for key, value in state_dict.items():
                if key.startswith(sae_prefix):
                    # Remove prefix: saes.layers_0_mixer.encoder.weight -> encoder.weight
                    new_key = key[len(sae_prefix):]
                    sae_state[new_key] = value
            
            if sae_state:
                # Create SAE model
                d_in = sae_state['encoder.weight'].shape[1]
                d_hidden = sae_state['encoder.weight'].shape[0]
                expansion_factor = d_hidden // d_in
                
                # Create SAE config
                sae_config = SAEConfig(
                    d_in=d_in,
                    expansion_factor=expansion_factor,
                    l1_coefficient=1e-3,  # Default value
                    normalize_decoder=True,
                    tied_weights=True
                )
                
                # Create and load SAE model
                sae = SparseAutoencoder(sae_config)
                sae.load_state_dict(sae_state)
                sae.to(self.config.device)
                sae.eval()
                
                self.sae_models[layer_name] = sae
                self.log(f"  ✓ Loaded SAE for {layer_name}: {d_in} -> {d_hidden} (expansion={expansion_factor}x)")
            else:
                self.log(f"  ✗ No SAE found for {layer_name}", level='warning')
        
        # Validate that all required SAE models were loaded
        if not self.sae_models:
            raise ValueError(
                f"❌ No SAE models were loaded!\n"
                f"Expected SAE models for blocks: {self.config.target_blocks}\n"
                f"Checkpoint file: {latest_checkpoint}\n"
                f"Available keys in state_dict: {list(state_dict.keys())[:10]}...\n"
                f"Please check if SAE training completed successfully."
            )
        
        # Verify all target blocks have SAE models
        missing_blocks = []
        for block_idx in self.config.target_blocks:
            layer_name = f"layers.{block_idx}.mixer"
            if layer_name not in self.sae_models:
                missing_blocks.append(block_idx)
        
        if missing_blocks:
            raise ValueError(
                f"❌ Missing SAE models for blocks: {missing_blocks}\n"
                f"Loaded SAE models for: {list(self.sae_models.keys())}\n"
                f"Required blocks: {self.config.target_blocks}\n"
                f"This will cause sparse activations to be None, breaking causal analysis."
            )
        
        self.log(f"✓ Successfully loaded {len(self.sae_models)} SAE models", level='success')
    
    def _process_pair(self, pair: Dict) -> Dict:
        """
        Process a single sequence pair
        
        Parameters:
        -----------
        pair : Dict
            Sequence pair data
        
        Returns:
        --------
        pair_activations : Dict
            Activation data for this pair
        """
        from src.model.activation_extractor import ActivationExtractor
        
        pair_id = pair['pair_id']
        source_seq = pair['source_sequence']
        target_seq = pair['target_sequence']
        
        # Process both sequences
        source_data = self._process_sequence(source_seq)
        target_data = self._process_sequence(target_seq)
        
        # Combine results
        pair_activations = {
            'pair_id': pair_id,
            'source': source_data,
            'target': target_data,
            'source_label': pair['source_label'],
            'target_label': pair['target_label']
        }
        
        return pair_activations
    
    def _process_sequence(self, sequence: str) -> Dict:
        """
        Process a single sequence through model
        
        Parameters:
        -----------
        sequence : str
            RNA sequence
        
        Returns:
        --------
        seq_data : Dict
            Activation data for this sequence
        """
        from src.model.activation_extractor import ActivationExtractor
        
        # Prepare data
        dataset = self.model_wrapper.process_data([sequence])
        
        # Create activation extractor
        extractor = ActivationExtractor(self.model)
        extractor.register_hooks(
            layer_filter=lambda name, m: (
                'mixer' in name.lower() and 
                name.endswith('mixer')
            )
        )
        
        # Forward pass
        with torch.no_grad():
            batch = dataset[0]
            # Convert to tensors if they are lists
            if isinstance(batch["input_ids"], list):
                input_ids = torch.tensor([batch["input_ids"]]).to(self.config.device)
                special_tokens_mask = torch.tensor([batch["special_tokens_mask"]]).to(self.config.device)
            else:
                input_ids = batch["input_ids"].unsqueeze(0).to(self.config.device)
                special_tokens_mask = batch["special_tokens_mask"].unsqueeze(0).to(self.config.device)
            attention_mask = 1 - special_tokens_mask
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get final embedding
            if hasattr(outputs, 'last_hidden_state'):
                embedding = outputs.last_hidden_state
            elif isinstance(outputs, torch.Tensor):
                embedding = outputs
            else:
                embedding = outputs[0]
            
            # Mean pooling over sequence length
            embedding = embedding.mean(dim=1).squeeze(0).cpu().numpy()
            
            # Get activations
            activations = extractor.get_activations()
        
        extractor.remove_hooks()
        
        # Process activations for target blocks
        block_data = {}
        
        for block_idx in self.config.target_blocks:
            layer_name = f"layers.{block_idx}.mixer"
            
            if layer_name in activations:
                raw_activation = activations[layer_name].cpu()
                
                # Encode with SAE - CRITICAL: SAE must exist
                if layer_name not in self.sae_models:
                    raise RuntimeError(
                        f"❌ CRITICAL: SAE model not found for {layer_name}!\n"
                        f"This should have been caught during initialization.\n"
                        f"Available SAE models: {list(self.sae_models.keys())}\n"
                        f"Cannot proceed with causal analysis without SAE models."
                    )
                
                sae = self.sae_models[layer_name]
                with torch.no_grad():
                    sparse_code = sae.encode(raw_activation.to(self.config.device))
                    sparse_code = sparse_code.cpu().numpy()
                
                # Validate sparse code
                if sparse_code is None or sparse_code.size == 0:
                    raise RuntimeError(
                        f"❌ CRITICAL: SAE encoding failed for {layer_name}!\n"
                        f"Sparse code is None or empty.\n"
                        f"Raw activation shape: {raw_activation.shape}\n"
                        f"This breaks causal analysis."
                    )
                
                block_data[f'block_{block_idx}'] = {
                    'raw': raw_activation.numpy(),
                    'sparse': sparse_code
                }
        
        seq_data = {
            'blocks': block_data,
            'final_embedding': embedding
        }
        
        return seq_data
    
    def _compute_metadata(self, all_activations: List[Dict]) -> Dict:
        """Compute metadata about collected activations"""
        if not all_activations:
            return {}
        
        # Get shapes from first pair
        first_pair = all_activations[0]
        source_blocks = first_pair['source']['blocks']
        
        shapes = {}
        for block_name, block_data in source_blocks.items():
            shapes[block_name] = {
                'raw_shape': block_data['raw'].shape,
                'sparse_shape': block_data['sparse'].shape if block_data['sparse'] is not None else None
            }
        
        metadata = {
            'num_pairs': len(all_activations),
            'num_blocks': len(self.config.target_blocks),
            'block_shapes': shapes,
            'embedding_dim': first_pair['source']['final_embedding'].shape[0]
        }
        
        return metadata
