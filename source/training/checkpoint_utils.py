"""
Checkpoint utilities for training scripts.
This module provides functions for saving and loading model checkpoints.
"""

import os
import json
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch.nn as nn
import torch.optim as optim


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


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    filepath: str,
    device: str = 'cuda'
) -> int:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        filepath: Path to checkpoint file
        device: Device to load on
        
    Returns:
        Starting epoch number
    """
    if os.path.isfile(filepath):
        print(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch
    else:
        print(f"No checkpoint found at {filepath}")
        return 0


def save_final_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: Dict[str, Any],
    history: Dict[str, List[float]],
    final_accuracy: float,
    best_accuracy: float,
    class_accuracies: Dict[str, float],
    training_time: float,
    save_dir: Path,
    model_name: str = 'final_model.pth'
) -> None:
    """
    Save final trained model with comprehensive information.
    
    Args:
        model: Trained model
        optimizer: Final optimizer state
        config: Configuration used for training
        history: Training history
        final_accuracy: Final test accuracy
        best_accuracy: Best test accuracy achieved
        class_accuracies: Per-class accuracies
        training_time: Total training time
        save_dir: Directory to save model
        model_name: Name of the model file
    """
    final_model_path = save_dir / model_name
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'history': history,
        'final_accuracy': final_accuracy,
        'best_accuracy': best_accuracy,
        'class_accuracies': class_accuracies,
        'training_time': training_time
    }, final_model_path)
    
    print(f"Final model saved to: {final_model_path}")


def save_training_history(
    history: Dict[str, List[float]],
    save_dir: Path,
    config: Dict[str, Any],
    filename: str = 'training_history.csv'
) -> None:
    """
    Save training history to CSV file.
    
    Args:
        history: Training history dictionary
        save_dir: Directory to save history
        config: Configuration dictionary
        filename: Name of the CSV file
    """
    # Pad test metrics to match training length
    test_loss_padded = []
    test_acc_padded = []
    eval_interval = config.get('eval_interval', 5)
    
    for i in range(len(history['train_loss'])):
        if (i % eval_interval == eval_interval - 1 and 
            i // eval_interval < len(history.get('test_loss', []))):
            test_loss_padded.append(history['test_loss'][i // eval_interval])
            test_acc_padded.append(history['test_acc'][i // eval_interval])
        else:
            test_loss_padded.append(None)
            test_acc_padded.append(None)
    
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'test_loss': test_loss_padded,
        'test_acc': test_acc_padded,
        'learning_rate': history.get('lr', [None] * len(history['train_loss']))
    })
    
    history_path = save_dir / filename
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")


def save_results_summary(
    model_name: str,
    final_accuracy: float,
    best_accuracy: float,
    total_params: int,
    trainable_params: int,
    training_time: float,
    epochs_trained: int,
    class_accuracies: Dict[str, float],
    config: Dict[str, Any],
    save_dir: Path,
    filename: str = 'results_summary.json'
) -> None:
    """
    Save results summary to JSON file.
    
    Args:
        model_name: Name of the model
        final_accuracy: Final test accuracy
        best_accuracy: Best test accuracy achieved
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
        training_time: Total training time in seconds
        epochs_trained: Number of epochs trained
        class_accuracies: Per-class accuracies
        config: Configuration dictionary
        save_dir: Directory to save summary
        filename: Name of the JSON file
    """
    results_summary = {
        'model': model_name,
        'dataset': 'CIFAR-10',
        'final_test_accuracy': final_accuracy,
        'best_test_accuracy': best_accuracy,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'training_time_hours': training_time / 3600,
        'epochs_trained': epochs_trained,
        'class_accuracies': class_accuracies,
        'config': config
    }
    
    results_summary_path = save_dir / filename
    with open(results_summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"Results summary saved to: {results_summary_path}")


def print_final_results(
    model_name: str,
    final_accuracy: float,
    best_accuracy: float,
    total_params: int,
    training_time: float,
    class_accuracies: Dict[str, float]
) -> None:
    """
    Print final training results in a formatted way.
    
    Args:
        model_name: Name of the model
        final_accuracy: Final test accuracy
        best_accuracy: Best test accuracy achieved
        total_params: Total number of parameters
        training_time: Total training time in seconds
        class_accuracies: Per-class accuracies
    """
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"Total Parameters: {total_params:,}")
    print(f"Training Time: {training_time/3600:.2f} hours")
    print("="*60)
    
    print("\nPer-class Accuracies:")
    for class_name, acc in class_accuracies.items():
        print(f"  {class_name}: {acc:.2f}%")
    print("="*60)