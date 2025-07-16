"""
Configuration utilities for training scripts.
This module provides functions for handling configuration management.
"""

import json
import yaml
import itertools
from pathlib import Path
from typing import Dict, Any, Optional, List
import datetime
import argparse


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for AKOrN models.
    
    Returns:
        Default configuration dictionary
    """
    return {
        # Data
        'batch_size': 128,
        'num_workers': 4,
        'num_classes': 10,
        
        # Model architecture
        'n': 2,              # Oscillator dimension (2D for complex oscillators)
        'ch': 64,            # Base number of channels
        'L': 3,              # Number of layers
        'T': 3,              # Number of time steps per layer
        'gamma': 1.0,        # Integration step size
        'J': 'conv',         # Connectivity type
        'J_bias': False,     # Connectivity bias
        'ksizes': [9, 7, 5], # Kernel sizes for each layer
        'ro_ksize': 3,       # Readout kernel size
        'ro_N': 2,           # Readout N parameter
        'norm': 'bn',        # Normalization type
        'c_norm': 'gn',      # C normalization type
        'use_omega': True,   # Use natural frequencies
        'init_omg': 1.0,     # Initial omega value
        'global_omg': True,  # Global omega parameter
        'learn_omg': True,   # Learn omega parameters
        'ensemble': 1,       # Ensemble size
        
        # Training
        'epochs': 100,
        'lr': 1e-4,
        'weight_decay': 0.0,
        'scheduler': 'constant',
        
        # Logging
        'log_interval': 100,
        'eval_interval': 5,
        'save_interval': 20,
        
        # Experiment
        'seed': 42,
        'experiment_name': 'akorn_cifar10',
        'save_dir': None,
    }


def load_yaml_config(yaml_file: str, index: Optional[int] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with optional grid search support.
    
    Args:
        yaml_file: Path to YAML configuration file
        index: Index for grid search (if applicable)
        
    Returns:
        Configuration dictionary
    """
    with open(yaml_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Handle grid search configuration
    if 'grid' in yaml_config and index is not None:
        grid_config = yaml_config['grid']
        keys, lists = zip(*[(k, v if isinstance(v, list) else [v]) for k, v in grid_config.items()])
        combos = [dict(zip(keys, values)) for values in itertools.product(*lists)]
        
        if index >= len(combos):
            raise ValueError(f"Index {index} out of range for grid with {len(combos)} combinations")
        
        return combos[index]
    
    return yaml_config


def update_config_from_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Update configuration dictionary with command line arguments.
    
    Args:
        config: Base configuration dictionary
        args: Parsed command line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Get all argument names and their values
    arg_dict = vars(args)
    
    # Update config with non-None values from args
    for key, value in arg_dict.items():
        if value is not None and key in config:
            config[key] = value
    
    return config


def create_save_directory(config: Dict[str, Any]) -> Path:
    """
    Create and return save directory path.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to save directory
    """
    if config['save_dir'] is None:
        current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        save_dir = Path(f"./results/{config['experiment_name']}_{current_time}")
    else:
        save_dir = Path(config['save_dir'])
    
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def save_config(config: Dict[str, Any], save_dir: Path, filename: str = 'parameters.json') -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save configuration
        filename: Name of the configuration file
    """
    save_path = save_dir / filename
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {save_path}")


def print_config(config: Dict[str, Any], title: str = "Configuration") -> None:
    """
    Print configuration in a formatted way.
    
    Args:
        config: Configuration dictionary
        title: Title for the configuration display
    """
    print(f"\n{title}:")
    print("-" * (len(title) + 1))
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add common arguments to argument parser.
    
    Args:
        parser: ArgumentParser instance
        
    Returns:
        Updated ArgumentParser instance
    """
    # WandB arguments
    parser.add_argument('--wandb-project', type=str, help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    
    # Data arguments
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    # Model architecture arguments
    parser.add_argument('--n', type=int, default=2, help='Oscillator dimension')
    parser.add_argument('--ch', type=int, default=64, help='Base number of channels')
    parser.add_argument('--L', type=int, default=3, help='Number of layers')
    parser.add_argument('--T', type=int, default=3, help='Number of time steps per layer')
    parser.add_argument('--gamma', type=float, default=1.0, help='Integration step size')
    parser.add_argument('--J', type=str, default='conv', help='Connectivity type')
    parser.add_argument('--ksizes', type=int, nargs='+', default=[9, 7, 5], help='Kernel sizes')
    parser.add_argument('--ro-ksize', type=int, default=3, help='Readout kernel size')
    parser.add_argument('--ro-N', type=int, default=2, help='Readout N parameter')
    parser.add_argument('--norm', type=str, default='bn', help='Normalization type')
    parser.add_argument('--c-norm', type=str, default='gn', help='C normalization type')
    parser.add_argument('--use-omega', type=bool, default=True, help='Use natural frequencies')
    parser.add_argument('--init-omg', type=float, default=1.0, help='Initial omega value')
    parser.add_argument('--global-omg', type=bool, default=True, help='Global omega parameter')
    parser.add_argument('--learn-omg', type=bool, default=True, help='Learn omega parameters')
    parser.add_argument('--ensemble', type=int, default=1, help='Ensemble size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='constant', help='Learning rate scheduler')
    
    # Logging arguments
    parser.add_argument('--log-interval', type=int, default=100, help='Log every N batches')
    parser.add_argument('--eval-interval', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--save-interval', type=int, default=20, help='Save checkpoint every N epochs')
    
    # Experiment arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--experiment-name', type=str, default='akorn_cifar10', help='Experiment name')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--wandb-id', type=str, default=None, help='WandB run ID to resume')
    
    # Grid search arguments
    parser.add_argument('--param-file', type=str, default=None, help='YAML file with parameter grid')
    parser.add_argument('--index', type=int, default=None, help='Grid search index')
    
    return parser