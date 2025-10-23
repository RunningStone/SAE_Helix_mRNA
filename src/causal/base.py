"""
Base Classes for Causal Feature Analysis

This module provides:
1. CausalAnalysisConfig: Configuration dataclass for all analysis parameters
2. CausalDataManager: Centralized data storage and I/O management
3. BaseCausalStep: Abstract base class for all analysis steps
"""

import json
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import numpy as np
import torch


@dataclass
class CausalAnalysisConfig:
    """
    Configuration for causal feature analysis
    
    Parameters:
    -----------
    # Task configuration
    target_feature: str
        Name of the target feature to predict (e.g., 'mrl', 'stability')
    task_type: str
        Type of task: 'regression' or 'classification'
    
    # Data paths
    data_dir: str
        Directory containing original dataset JSON files
    step1_output_dir: str
        Directory containing Step 1 outputs (SAE models, sparse activations)
    output_dir: str
        Output directory for causal analysis results
    
    # Probe configuration (Step 1)
    probe_model_type: str
        Type of probe model: 'ridge', 'lasso', 'mlp'
    probe_train_split: float
        Train/test split ratio for probe
    probe_cv_folds: int
        Number of cross-validation folds for hyperparameter tuning
    probe_alpha_range: List[float]
        Range of regularization parameters to search
    probe_min_r2: float
        Minimum R² threshold for probe validation
    probe_r2_metric: str
        Which R² to use for validation: 'train' or 'test'
    
    # Sequence pair selection (Step 2)
    num_sequence_pairs: int
        Number of sequence pairs to analyze
    min_target_diff: float
        Minimum target value difference for sequence pairs
    max_length_ratio: float
        Maximum length ratio difference (e.g., 0.1 means 0.9-1.1)
    max_edit_distance: float
        Maximum normalized edit distance for sequence similarity
    
    # Activation collection (Step 3)
    target_blocks: List[int]
        List of block indices to analyze (e.g., [0, 1, 2, 3])
    batch_size: int
        Batch size for model inference
    device: str
        Device for computation: 'cuda' or 'cpu'
    
    # ACDC analysis (Step 4)
    max_features_to_test: int
        Maximum number of top features to test in cumulative intervention
    
    # Aggregation (Step 6)
    confidence_level: float
        Confidence level for error bars (e.g., 0.95)
    convergence_threshold: float
        Threshold for detecting convergence (fraction of initial error)
    
    # Baseline comparison (Step 8)
    baseline_methods: List[str]
        List of baseline methods: ['pca', 'ica', 'random']
    
    # Visualization (Step 9)
    plot_format: str
        Format for saving plots: 'png', 'pdf', 'svg'
    plot_dpi: int
        DPI for rasterized plots
    
    # General
    random_seed: int
        Random seed for reproducibility
    save_intermediate: bool
        Whether to save intermediate results
    verbose: bool
        Whether to print detailed progress
    """
    
    # Task configuration
    target_feature: str = 'mrl'
    task_type: str = 'regression'
    
    # Data paths
    data_dir: str = ''
    step1_output_dir: str = ''
    output_dir: str = ''
    
    # Probe configuration
    probe_model_type: str = 'ridge'
    probe_train_split: float = 0.8
    probe_cv_folds: int = 5
    probe_alpha_range: List[float] = None
    probe_min_r2: float = 0.6
    probe_r2_metric: str = 'test'  # 'train' or 'test'
    
    # Sequence pair selection
    num_sequence_pairs: int = 100
    min_target_diff: float = 2.0
    max_length_ratio: float = 0.1
    max_edit_distance: float = 0.3
    
    # Activation collection
    target_blocks: List[int] = None
    batch_size: int = 16
    device: str = 'cuda'
    
    # ACDC analysis
    max_features_to_test: int = 100
    
    # Aggregation
    confidence_level: float = 0.95
    convergence_threshold: float = 0.1
    
    # Baseline comparison
    baseline_methods: List[str] = None
    
    # Visualization
    plot_format: str = 'png'
    plot_dpi: int = 300
    
    # General
    random_seed: int = 42
    save_intermediate: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """Set default values for mutable fields"""
        if self.probe_alpha_range is None:
            self.probe_alpha_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        if self.target_blocks is None:
            self.target_blocks = [0, 1, 2, 3]
        if self.baseline_methods is None:
            self.baseline_methods = ['pca', 'random']
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return asdict(self)
    
    def save(self, path: Union[str, Path]):
        """Save config to JSON file"""
        path = Path(path)
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'CausalAnalysisConfig':
        """Load config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'CausalAnalysisConfig':
        """Load config from JSON file"""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class CausalDataManager:
    """
    Centralized data storage and I/O management for causal analysis
    
    This class manages all data flow between analysis steps, including:
    - Loading and caching data from disk
    - Storing intermediate results in memory
    - Saving results to disk
    - Providing unified access to all data
    """
    
    def __init__(self, config: CausalAnalysisConfig):
        """
        Initialize data manager
        
        Parameters:
        -----------
        config : CausalAnalysisConfig
            Configuration object
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each step
        self.step_dirs = {}
        for step_name in ['probe', 'pairs', 'activations', 'acdc', 
                          'intervention', 'aggregation', 'comparison',
                          'baseline', 'visualization', 'feature_analysis']:
            step_dir = self.output_dir / step_name
            step_dir.mkdir(exist_ok=True)
            self.step_dirs[step_name] = step_dir
        
        # In-memory data storage
        self.data = {
            'probe_model': None,
            'probe_results': None,
            'sequence_pairs': None,
            'activations': None,
            'acdc_results': None,
            'intervention_results': None,
            'aggregated_results': None,
            'comparison_results': None,
            'baseline_results': None,
            'visualization_results': None,
            'feature_analysis_results': None,
        }
        
        # Cache for loaded data
        self._cache = {}
    
    def save_data(self, step_name: str, data: Any, filename: str, 
                  format: str = 'pkl') -> Path:
        """
        Save data to disk
        
        Parameters:
        -----------
        step_name : str
            Name of the analysis step
        data : Any
            Data to save
        filename : str
            Filename (without extension)
        format : str
            Format: 'pkl', 'json', 'npy', 'pt'
        
        Returns:
        --------
        save_path : Path
            Path to saved file
        """
        if step_name not in self.step_dirs:
            raise ValueError(f"Unknown step name: {step_name}")
        
        step_dir = self.step_dirs[step_name]
        
        if format == 'pkl':
            save_path = step_dir / f"{filename}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
        elif format == 'json':
            save_path = step_dir / f"{filename}.json"
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'npy':
            save_path = step_dir / f"{filename}.npy"
            np.save(save_path, data)
        elif format == 'pt':
            save_path = step_dir / f"{filename}.pt"
            torch.save(data, save_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if self.config.verbose:
            print(f"✓ Saved {step_name}/{filename}.{format}")
        
        return save_path
    
    def load_data(self, step_name: str, filename: str, 
                  format: str = 'pkl', use_cache: bool = True) -> Any:
        """
        Load data from disk
        
        Parameters:
        -----------
        step_name : str
            Name of the analysis step
        filename : str
            Filename (without extension)
        format : str
            Format: 'pkl', 'json', 'npy', 'pt'
        use_cache : bool
            Whether to use cached data if available
        
        Returns:
        --------
        data : Any
            Loaded data
        """
        cache_key = f"{step_name}/{filename}.{format}"
        
        if use_cache and cache_key in self._cache:
            if self.config.verbose:
                print(f"✓ Loaded {cache_key} from cache")
            return self._cache[cache_key]
        
        if step_name not in self.step_dirs:
            raise ValueError(f"Unknown step name: {step_name}")
        
        step_dir = self.step_dirs[step_name]
        
        if format == 'pkl':
            load_path = step_dir / f"{filename}.pkl"
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
        elif format == 'json':
            load_path = step_dir / f"{filename}.json"
            with open(load_path, 'r') as f:
                data = json.load(f)
        elif format == 'npy':
            load_path = step_dir / f"{filename}.npy"
            data = np.load(load_path, allow_pickle=True)
        elif format == 'pt':
            load_path = step_dir / f"{filename}.pt"
            data = torch.load(load_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if use_cache:
            self._cache[cache_key] = data
        
        if self.config.verbose:
            print(f"✓ Loaded {cache_key}")
        
        return data
    
    def set_data(self, key: str, value: Any):
        """Store data in memory"""
        self.data[key] = value
    
    def get_data(self, key: str) -> Any:
        """Retrieve data from memory"""
        return self.data.get(key)
    
    def clear_cache(self):
        """Clear cached data"""
        self._cache.clear()


class BaseCausalStep(ABC):
    """
    Abstract base class for all causal analysis steps
    
    Each step should:
    1. Inherit from this class
    2. Implement the run() method
    3. Use data_manager for I/O operations
    4. Return results in a standardized format
    """
    
    def __init__(self, config: CausalAnalysisConfig, 
                 data_manager: CausalDataManager):
        """
        Initialize analysis step
        
        Parameters:
        -----------
        config : CausalAnalysisConfig
            Configuration object
        data_manager : CausalDataManager
            Data manager for I/O operations
        """
        self.config = config
        self.data_manager = data_manager
        self.step_name = self.__class__.__name__
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the analysis step
        
        Returns:
        --------
        results : Dict[str, Any]
            Dictionary containing step results
        """
        pass
    
    def log(self, message: str, level: str = 'info'):
        """Print log message if verbose"""
        if self.config.verbose:
            prefix = {
                'info': '  ',
                'success': '✓ ',
                'warning': '⚠ ',
                'error': '✗ '
            }.get(level, '  ')
            print(f"{prefix}{message}")
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """
        Save step results
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Results to save
        filename : str, optional
            Custom filename (default: step_name_results)
        """
        if filename is None:
            filename = f"{self.step_name.lower()}_results"
        
        # Map class names to step directory names
        class_to_dir = {
            'ProbeBuilder': 'probe',
            'SequencePairSelector': 'pairs',
            'ActivationCollector': 'activations',
            'ACDCAnalyzer': 'acdc',
            'InterventionAnalyzer': 'intervention',
            'ResultAggregator': 'aggregation',
            'BlockComparator': 'comparison',
            'BaselineComparator': 'baseline',
            'ResultVisualizer': 'visualization',
            'FeatureAnalyzer': 'feature_analysis'
        }
        
        step_dir_name = class_to_dir.get(self.step_name, 'general')
        
        self.data_manager.save_data(
            step_name=step_dir_name,
            data=results,
            filename=filename,
            format='pkl'
        )
    
    def load_results(self, filename: str = None) -> Dict[str, Any]:
        """
        Load step results
        
        Parameters:
        -----------
        filename : str, optional
            Custom filename (default: step_name_results)
        
        Returns:
        --------
        results : Dict[str, Any]
            Loaded results
        """
        if filename is None:
            filename = f"{self.step_name.lower()}_results"
        
        # Map class names to step directory names
        class_to_dir = {
            'ProbeBuilder': 'probe',
            'SequencePairSelector': 'pairs',
            'ActivationCollector': 'activations',
            'ACDCAnalyzer': 'acdc',
            'InterventionAnalyzer': 'intervention',
            'ResultAggregator': 'aggregation',
            'BlockComparator': 'comparison',
            'BaselineComparator': 'baseline',
            'ResultVisualizer': 'visualization',
            'FeatureAnalyzer': 'feature_analysis'
        }
        
        step_dir_name = class_to_dir.get(self.step_name, 'general')
        
        return self.data_manager.load_data(
            step_name=step_dir_name,
            filename=filename,
            format='pkl'
        )
