"""
Training utilities for classification models.
This module provides common training functions used across different classification scripts.
"""

import random
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pathlib import Path
import json
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': total_params * 4 / 1e6  # Assuming float32
    }


def save_checkpoint_with_config(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    config: Dict[str, Any],
    filepath: Path
) -> None:
    """
    Save model checkpoint with configuration.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        loss: Current loss value
        config: Configuration dictionary
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")



def save_parameters(config: Dict[str, Any], save_dir: Path, filename: str = 'parameters.json') -> None:
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
    print(f"Parameters saved to: {save_path}")

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    config: Dict[str, Any],
    log_wandb: bool = True,
    ema: Optional[Any] = None
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        config: Configuration dictionary
        log_wandb: Whether to log to wandb
        ema: Exponential moving average model (optional)
        
    Returns:
        Tuple of (epoch_loss, epoch_acc)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Update EMA if provided
        if ema is not None:
            ema.update()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Log to wandb
        if log_wandb and HAS_WANDB and batch_idx % config.get('log_interval', 100) == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_acc': 100. * correct / total,
                'epoch': epoch,
                'batch': batch_idx + epoch * len(train_loader)
            })
            
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def print_model_info(model: nn.Module, model_name: str = "Model") -> None:
    """
    Print model information including parameter counts.
    
    Args:
        model: Model to analyze
        model_name: Name of the model for printing
    """
    param_info = count_parameters(model)
    
    print(f"\n{model_name} Information:")
    print(f"Total parameters: {param_info['total_params']:,}")
    print(f"Trainable parameters: {param_info['trainable_params']:,}")
    print(f"Model size: {param_info['size_mb']:.2f} MB (float32)")


def setup_device() -> str:
    """
    Setup and return the appropriate device.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return str(device)


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration dictionary
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = config.get('scheduler', 'constant')
    
    if scheduler_type == 'constant':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['epochs'],
            eta_min=config['lr'] * 0.01
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('decay_gamma', 0.1)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def initialize_wandb(config: Dict[str, Any], project_name: str, entity: str = None, 
                    run_name: str = None, resume_id: str = None) -> None:
    """
    Initialize wandb logging.
    
    Args:
        config: Configuration dictionary
        project_name: W&B project name
        entity: W&B entity name
        run_name: Custom run name
        resume_id: Resume ID for continuing runs
    """
    if not HAS_WANDB:
        raise ImportError("wandb is not installed")
    
    if run_name is None:
        run_name = f"{config['experiment_name']}_{int(time.time())}"
    
    wandb.init(
        project=project_name,
        entity=entity,
        config=config,
        name=run_name,
        id=resume_id,
        resume="allow" if resume_id else None
    )