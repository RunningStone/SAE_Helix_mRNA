"""
Step 2: Construct Sequence Pair Dataset

This module selects pairs of sequences with significant functional differences
but high sequence similarity for causal intervention analysis.
"""

import numpy as np
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Try to import Levenshtein, fall back to difflib if not available
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    import difflib
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Fallback implementation using difflib"""
        return sum(1 for _ in difflib.ndiff(s1, s2) if _[0] in ['+', '-']) // 2

from .base import BaseCausalStep, CausalAnalysisConfig, CausalDataManager


class SequencePairSelector(BaseCausalStep):
    """
    Select sequence pairs for causal analysis
    
    Workflow:
    ---------
    1. Load sequences and target labels
    2. Compute pairwise differences and similarities
    3. Filter pairs based on criteria:
       - Significant target difference (Δtarget > threshold)
       - Similar length (ratio within range)
       - Similar sequence (edit distance < threshold)
    4. Sample N pairs for analysis
    5. Label high/low target sequences as source/target
    """
    
    def __init__(self, config: CausalAnalysisConfig, 
                 data_manager: CausalDataManager):
        super().__init__(config, data_manager)
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run sequence pair selection
        
        Returns:
        --------
        results : Dict
            - sequence_pairs: List of selected pairs
            - pair_statistics: Statistics about selected pairs
        """
        self.log("="*80)
        self.log("Step 2: Constructing Sequence Pair Dataset")
        self.log("="*80)
        
        # Load probe results to get train/test indices
        probe_results = self.data_manager.get_data('probe_results')
        if probe_results is None:
            probe_results = self.load_results('probe_results')
        
        # Choose which set to use based on probe_r2_metric
        if self.config.probe_r2_metric == 'train':
            selected_indices = probe_results['train_indices']
            set_name = 'train'
            self.log("Using TRAIN set for sequence pair selection (matching probe_r2_metric='train')")
        else:
            selected_indices = probe_results['test_indices']
            set_name = 'test'
            self.log("Using TEST set for sequence pair selection (matching probe_r2_metric='test')")
        
        # Load sequences and labels
        sequences, labels, metadata = self._load_sequences()
        
        # Filter to selected set
        selected_sequences = [sequences[i] for i in selected_indices]
        selected_labels = labels[selected_indices]
        selected_metadata = [metadata[i] for i in selected_indices]
        
        self.log(f"{set_name.capitalize()} set size: {len(selected_sequences)}")
        
        # Find candidate pairs
        self.log("Finding candidate sequence pairs...")
        candidate_pairs = self._find_candidate_pairs(
            selected_sequences, selected_labels, selected_metadata
        )
        
        self.log(f"Found {len(candidate_pairs)} candidate pairs")
        
        # Sample pairs
        selected_pairs = self._sample_pairs(candidate_pairs)
        
        self.log(f"Selected {len(selected_pairs)} pairs for analysis", level='success')
        
        # Compute statistics
        pair_stats = self._compute_statistics(selected_pairs)
        
        # Prepare results
        results = {
            'sequence_pairs': selected_pairs,
            'pair_statistics': pair_stats,
            'selected_indices': selected_indices,
            'set_name': set_name  # 'train' or 'test'
        }
        
        # Save results
        if self.config.save_intermediate:
            self.save_results(results)
            self._save_summary(results)
        
        # Store in data manager
        self.data_manager.set_data('sequence_pairs', selected_pairs)
        
        self.log("="*80)
        self.log("✓ Sequence pair selection completed!", level='success')
        self.log("="*80)
        
        return results
    
    def _load_sequences(self) -> Tuple[List[str], np.ndarray, List[Dict]]:
        """
        Load sequences and labels from original dataset
        
        Returns:
        --------
        sequences : List[str]
            RNA sequences
        labels : np.ndarray
            Target labels
        metadata : List[Dict]
            Sequence metadata
        """
        data_dir = Path(self.config.data_dir)
        json_files = sorted(data_dir.glob('*.json'))
        
        sequences = []
        labels = []
        metadata = []
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                chunk_data = json.load(f)
            
            for item in chunk_data:
                if isinstance(item, dict):
                    seq = item.get('sequence', '')
                    # Convert DNA to RNA if needed
                    if 'T' in seq:
                        seq = seq.replace('T', 'U')
                    
                    sequences.append(seq)
                    
                    # Handle nested structure for different feature paths
                    label = None
                    
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
                        label = 0.0
                    
                    labels.append(float(label))
                    
                    metadata.append({
                        'source_file': json_file.name,
                        'length': len(seq)
                    })
                else:
                    # Skip non-dict items
                    continue
        
        labels = np.array(labels, dtype=np.float32)
        
        return sequences, labels, metadata
    
    def _find_candidate_pairs(self, sequences: List[str], labels: np.ndarray,
                             metadata: List[Dict]) -> List[Dict]:
        """
        Find candidate sequence pairs meeting criteria
        
        Parameters:
        -----------
        sequences : List[str]
            Sequences to pair
        labels : np.ndarray
            Target labels
        metadata : List[Dict]
            Sequence metadata
        
        Returns:
        --------
        candidate_pairs : List[Dict]
            List of candidate pairs with metadata
        """
        n = len(sequences)
        candidate_pairs = []
        
        # Compute pairwise comparisons (sample if too large)
        max_comparisons = 500000  # Increased from 100000 to 500000
        if n * (n - 1) // 2 > max_comparisons:
            # Sample pairs randomly
            n_samples = int(np.sqrt(2 * max_comparisons))
            sample_indices = np.random.choice(n, size=min(n_samples, n), replace=False)
            self.log(f"Sampling {n_samples} sequences from {n} total (to limit comparisons)")
        else:
            sample_indices = np.arange(n)
        
        self.log(f"Comparing {len(sample_indices)} sequences...")
        
        for i, idx_a in enumerate(sample_indices):
            if i % 100 == 0:
                self.log(f"  Progress: {i}/{len(sample_indices)}")
            
            for idx_b in sample_indices[i+1:]:
                # Check target difference
                delta_target = abs(labels[idx_a] - labels[idx_b])
                if delta_target < self.config.min_target_diff:
                    continue
                
                # Check length similarity
                len_a = metadata[idx_a]['length']
                len_b = metadata[idx_b]['length']
                length_ratio = min(len_a, len_b) / max(len_a, len_b)
                
                if length_ratio < (1 - self.config.max_length_ratio):
                    continue
                
                # Check sequence similarity (edit distance)
                seq_a = sequences[idx_a]
                seq_b = sequences[idx_b]
                edit_dist = levenshtein_distance(seq_a, seq_b)
                normalized_dist = edit_dist / max(len_a, len_b)
                
                if normalized_dist > self.config.max_edit_distance:
                    continue
                
                # Determine source (high target) and target (low target)
                if labels[idx_a] > labels[idx_b]:
                    source_idx, target_idx = idx_a, idx_b
                else:
                    source_idx, target_idx = idx_b, idx_a
                
                candidate_pairs.append({
                    'source_idx': int(source_idx),
                    'target_idx': int(target_idx),
                    'source_sequence': sequences[source_idx],
                    'target_sequence': sequences[target_idx],
                    'source_label': float(labels[source_idx]),
                    'target_label': float(labels[target_idx]),
                    'delta_target': float(delta_target),
                    'length_ratio': float(length_ratio),
                    'edit_distance': float(normalized_dist),
                    'source_metadata': metadata[source_idx],
                    'target_metadata': metadata[target_idx]
                })
        
        return candidate_pairs
    
    def _sample_pairs(self, candidate_pairs: List[Dict]) -> List[Dict]:
        """
        Sample N pairs from candidates
        
        Parameters:
        -----------
        candidate_pairs : List[Dict]
            Candidate pairs
        
        Returns:
        --------
        selected_pairs : List[Dict]
            Sampled pairs
        """
        n_candidates = len(candidate_pairs)
        n_to_sample = min(self.config.num_sequence_pairs, n_candidates)
        
        if n_candidates == 0:
            raise ValueError("No candidate pairs found! Try relaxing the selection criteria.")
        
        # Sample randomly
        np.random.seed(self.config.random_seed)
        sample_indices = np.random.choice(n_candidates, size=n_to_sample, replace=False)
        
        selected_pairs = [candidate_pairs[i] for i in sample_indices]
        
        # Add pair IDs
        for i, pair in enumerate(selected_pairs):
            pair['pair_id'] = i
        
        return selected_pairs
    
    def _compute_statistics(self, pairs: List[Dict]) -> Dict[str, Any]:
        """Compute statistics about selected pairs"""
        delta_targets = [p['delta_target'] for p in pairs]
        length_ratios = [p['length_ratio'] for p in pairs]
        edit_distances = [p['edit_distance'] for p in pairs]
        
        stats = {
            'num_pairs': len(pairs),
            'delta_target': {
                'mean': float(np.mean(delta_targets)),
                'std': float(np.std(delta_targets)),
                'min': float(np.min(delta_targets)),
                'max': float(np.max(delta_targets))
            },
            'length_ratio': {
                'mean': float(np.mean(length_ratios)),
                'std': float(np.std(length_ratios)),
                'min': float(np.min(length_ratios)),
                'max': float(np.max(length_ratios))
            },
            'edit_distance': {
                'mean': float(np.mean(edit_distances)),
                'std': float(np.std(edit_distances)),
                'min': float(np.min(edit_distances)),
                'max': float(np.max(edit_distances))
            }
        }
        
        return stats
    
    def _save_summary(self, results: Dict):
        """Save human-readable summary"""
        summary = {
            'num_pairs': len(results['sequence_pairs']),
            'statistics': results['pair_statistics']
        }
        
        self.data_manager.save_data(
            step_name='pairs',
            data=summary,
            filename='pairs_summary',
            format='json'
        )
