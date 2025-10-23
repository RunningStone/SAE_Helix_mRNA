"""
Step 1: Build Prediction Probe

This module trains a regression/classification model to predict target features
from mRNA-FM embeddings. The probe serves as a "readout layer" to measure
the effect of interventions on model predictions.
"""

import numpy as np
import torch
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

from .base import BaseCausalStep, CausalAnalysisConfig, CausalDataManager
from .neural_probe import NeuralProbeTrainer


class ProbeBuilder(BaseCausalStep):
    """
    Build and train prediction probe model
    
    Workflow:
    ---------
    1. Load embeddings from Step 1 output
    2. Load target labels from original dataset
    3. Split data into train/test sets
    4. Train probe model with cross-validation
    5. Evaluate baseline performance
    6. Save trained probe model
    """
    
    def __init__(self, config: CausalAnalysisConfig, 
                 data_manager: CausalDataManager):
        super().__init__(config, data_manager)
        self.scaler = StandardScaler()
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run probe building pipeline
        
        Returns:
        --------
        results : Dict
            - probe_model: Trained model
            - train_metrics: Training performance metrics
            - test_metrics: Test performance metrics
            - train_indices: Indices of training samples
            - test_indices: Indices of test samples
        """
        self.log("="*80)
        self.log("Step 1: Building Prediction Probe")
        self.log("="*80)
        
        # Load embeddings and labels
        embeddings, labels, metadata = self._load_data()
        
        # Split data
        train_idx, test_idx = self._split_data(len(embeddings))
        X_train = embeddings[train_idx]
        X_test = embeddings[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        
        self.log(f"Train samples: {len(train_idx)}")
        self.log(f"Test samples: {len(test_idx)}")
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Train probe model
        self.log("Training probe model...")
        probe_model, best_params = self._train_probe(X_train, y_train)
        
        # Evaluate on train set
        train_pred = probe_model.predict(X_train)
        train_metrics = self._compute_metrics(y_train, train_pred)
        self.log(f"Train R²: {train_metrics['r2']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
        
        # Evaluate on test set
        test_pred = probe_model.predict(X_test)
        test_metrics = self._compute_metrics(y_test, test_pred)
        self.log(f"Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}", 
                level='success')
        
        # Validate performance - STRICT CHECK
        # Choose which metric to use for validation
        if self.config.probe_r2_metric == 'train':
            r2_to_check = train_metrics['r2']
            metric_name = 'Train'
        else:
            r2_to_check = test_metrics['r2']
            metric_name = 'Test'
        
        if r2_to_check < self.config.probe_min_r2:
            error_msg = (f"❌ PROBE FAILED: {metric_name} R² ({r2_to_check:.4f}) is below "
                        f"the required threshold ({self.config.probe_min_r2}).\n"
                        f"The embeddings cannot predict the target feature '{self.config.target_feature}'.\n"
                        f"Causal analysis cannot proceed with an unreliable probe.\n"
                        f"Suggestions:\n"
                        f"  1. Use a more powerful probe model (e.g., 'mlp' instead of 'ridge')\n"
                        f"  2. Check if the target feature is learnable from embeddings\n"
                        f"  3. Lower the threshold (not recommended)\n"
                        f"  4. Change probe_r2_metric to 'train' if test performance is poor due to overfitting")
            self.log(error_msg, level='error')
            raise ValueError(error_msg)
        else:
            self.log(f"✓ Probe validation passed: {metric_name} R² ({r2_to_check:.4f}) >= threshold ({self.config.probe_min_r2})", 
                    level='success')
        
        # Prepare results
        results = {
            'probe_model': probe_model,
            'scaler': self.scaler,
            'best_params': best_params,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_indices': train_idx,
            'test_indices': test_idx,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'metadata': metadata
        }
        
        # Save results
        if self.config.save_intermediate:
            self.save_results(results)
            self._save_summary(results)
        
        # Store in data manager
        self.data_manager.set_data('probe_model', probe_model)
        self.data_manager.set_data('probe_results', results)
        
        self.log("="*80)
        self.log("✓ Probe building completed!", level='success')
        self.log("="*80)
        
        return results
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load embeddings and target labels
        
        Returns:
        --------
        embeddings : np.ndarray
            Shape (n_samples, embedding_dim)
        labels : np.ndarray
            Shape (n_samples,)
        metadata : Dict
            Additional metadata
        """
        self.log("Loading embeddings and labels...")
        
        # Load embeddings from Step 1
        step1_dir = Path(self.config.step1_output_dir)
        embedding_file = step1_dir / 'embeddings' / 'embeddings.pt'
        
        if not embedding_file.exists():
            raise FileNotFoundError(f"Embeddings not found: {embedding_file}")
        
        data = torch.load(embedding_file, map_location='cpu')
        embeddings = data['embeddings'].numpy()
        
        self.log(f"Loaded embeddings: {embeddings.shape}")
        
        # Load labels from original dataset
        labels, metadata = self._load_labels()
        
        self.log(f"Loaded labels: {labels.shape}")
        self.log(f"Label range: [{labels.min():.2f}, {labels.max():.2f}]")
        
        # Verify alignment
        if len(embeddings) != len(labels):
            raise ValueError(f"Embedding count ({len(embeddings)}) != "
                           f"label count ({len(labels)})")
        
        return embeddings, labels, metadata
    
    def _load_labels(self) -> Tuple[np.ndarray, Dict]:
        """
        Load target labels from original dataset
        
        Returns:
        --------
        labels : np.ndarray
            Target values
        metadata : Dict
            Metadata about labels
        """
        data_dir = Path(self.config.data_dir)
        
        # Load token-to-sequence mapping
        step1_dir = Path(self.config.step1_output_dir)
        mapping_file = step1_dir / 'sparse_activations' / 'token_to_sequence_mapping.json'
        
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        mapping = mapping_data['mapping']
        total_tokens = mapping_data['total_tokens']
        
        # Load JSON files and extract target feature
        json_files = sorted(data_dir.glob('*.json'))
        
        all_labels = []
        sequence_metadata = []
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                chunk_data = json.load(f)
            
            for item in chunk_data:
                # Handle nested structure for different feature paths
                if isinstance(item, dict):
                    label = None
                    
                    # Try different paths based on feature name
                    if 'annotations' in item:
                        annotations = item['annotations']
                        
                        # Functional features: annotations.functional.{feature}
                        if 'functional' in annotations:
                            label = annotations['functional'].get(self.config.target_feature)
                        
                        # Structural features: annotations.structural.{feature}
                        if label is None and 'structural' in annotations:
                            label = annotations['structural'].get(self.config.target_feature)
                        
                        # Regulatory features: annotations.regulatory.{feature}
                        if label is None and 'regulatory' in annotations:
                            label = annotations['regulatory'].get(self.config.target_feature)
                    
                    # Direct access
                    if label is None and self.config.target_feature in item:
                        label = item[self.config.target_feature]
                    
                    if label is None:
                        label = 0.0  # Default value
                    
                    all_labels.append(float(label))
                    
                    sequence = item.get('sequence', '')
                    sequence_metadata.append({
                        'source_file': json_file.name,
                        'sequence': sequence[:50] + '...' if len(sequence) > 50 else sequence
                    })
                else:
                    # If item is not a dict (shouldn't happen), use default
                    all_labels.append(0.0)
                    sequence_metadata.append({
                        'source_file': json_file.name,
                        'sequence': str(item)[:50] + '...'
                    })
        
        labels = np.array(all_labels, dtype=np.float32)
        
        # Expand labels to token level (repeat for each token in sequence)
        token_labels = []
        for seq_idx, map_entry in enumerate(mapping):
            if seq_idx < len(labels):
                num_tokens = map_entry['num_tokens']
                token_labels.extend([labels[seq_idx]] * num_tokens)
        
        token_labels = np.array(token_labels, dtype=np.float32)
        
        metadata = {
            'num_sequences': len(labels),
            'num_tokens': len(token_labels),
            'sequence_metadata': sequence_metadata
        }
        
        return token_labels, metadata
    
    def _split_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into train/test sets
        
        Parameters:
        -----------
        n_samples : int
            Total number of samples
        
        Returns:
        --------
        train_idx : np.ndarray
            Training indices
        test_idx : np.ndarray
            Test indices
        """
        indices = np.arange(n_samples)
        train_idx, test_idx = train_test_split(
            indices,
            train_size=self.config.probe_train_split,
            random_state=self.config.random_seed,
            shuffle=True
        )
        return train_idx, test_idx
    
    def _train_probe(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, Dict]:
        """
        Train probe model with cross-validation
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        
        Returns:
        --------
        best_model : model
            Trained model with best hyperparameters
        best_params : Dict
            Best hyperparameters
        """
        if self.config.probe_model_type == 'mlp':
            # Neural network probe - Simple 2-layer MLP
            self.log("Training 2-layer MLP probe (200 epochs)...")
            
            # Split train into train/val for monitoring
            val_split = 0.1
            n_val = int(len(X_train) * val_split)
            indices = np.random.permutation(len(X_train))
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            
            X_train_split = X_train[train_idx]
            y_train_split = y_train[train_idx]
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]
            
            # Create trainer with simple 2-layer architecture
            trainer = NeuralProbeTrainer(
                input_dim=X_train.shape[1],
                hidden_dims=[256,256,128],  # Single hidden layer with 128 units
                dropout=0.0,  # No dropout for simple model
                learning_rate=1e-3,
                batch_size=256,
                epochs=200,  # Fixed 50 epochs for convergence
                early_stopping_patience=50,  # Disable early stopping
                device=self.config.device,
                model_type='deep'  # "simple" Use simple 2-layer MLP, "deep" Use deep MLP
            )
            
            # Train
            history = trainer.train(
                X_train_split, y_train_split,
                X_val, y_val,
                verbose=self.config.verbose
            )
            
            best_params = {
                'architecture': '2-layer MLP',
                'hidden_dim': 128,
                'learning_rate': 1e-3,
                'epochs': 50,
                'final_epoch': history['final_epoch'],
                'best_val_loss': history['best_val_loss']
            }
            
            self.log(f"Final validation loss: {history['best_val_loss']:.4f}")
            self.log(f"Training completed at epoch: {history['final_epoch']}")
            
            return trainer, best_params
            
        elif self.config.probe_model_type == 'ridge':
            model = Ridge()
            param_grid = {'alpha': self.config.probe_alpha_range}
        elif self.config.probe_model_type == 'lasso':
            model = Lasso()
            param_grid = {'alpha': self.config.probe_alpha_range}
        else:
            raise ValueError(f"Unsupported probe model type: {self.config.probe_model_type}")
        
        # Grid search with cross-validation (for linear models)
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.config.probe_cv_folds,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        self.log(f"Best hyperparameters: {best_params}")
        self.log(f"Best CV R²: {grid_search.best_score_:.4f}")
        
        return best_model, best_params
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        
        Returns:
        --------
        metrics : Dict
            Dictionary of metrics
        """
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        
        metrics = {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae)
        }
        
        return metrics
    
    def _save_summary(self, results: Dict):
        """Save human-readable summary"""
        summary = {
            'model_type': self.config.probe_model_type,
            'best_params': results['best_params'],
            'train_metrics': results['train_metrics'],
            'test_metrics': results['test_metrics'],
            'num_train_samples': len(results['train_indices']),
            'num_test_samples': len(results['test_indices'])
        }
        
        self.data_manager.save_data(
            step_name='probe',
            data=summary,
            filename='probe_summary',
            format='json'
        )
