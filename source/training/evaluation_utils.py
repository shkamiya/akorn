"""
Evaluation utilities for classification models.
This module provides functions for evaluating model performance.
"""

from typing import Dict, Tuple, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    class_names: Optional[Tuple[str, ...]] = None
) -> Tuple[float, float, Dict[str, float]]:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to use
        class_names: Names of classes (optional)
        
    Returns:
        Tuple of (test_loss, test_accuracy, class_accuracies)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    num_classes = len(class_names) if class_names else 10
    class_correct = [0.0] * num_classes
    class_total = [0.0] * num_classes
    
    if class_names is None:
        class_names = tuple(f'class_{i}' for i in range(num_classes))
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Per-class accuracy
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    # Calculate per-class accuracies
    class_accuracies = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies[class_names[i]] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracies[class_names[i]] = 0
    
    return test_loss, test_acc, class_accuracies


def log_evaluation_results(
    test_loss: float,
    test_acc: float,
    class_accuracies: Dict[str, float],
    epoch: int,
    log_wandb: bool = True,
    prefix: str = "test/"
) -> None:
    """
    Log evaluation results to wandb and console.
    
    Args:
        test_loss: Test loss value
        test_acc: Test accuracy value
        class_accuracies: Per-class accuracies
        epoch: Current epoch
        log_wandb: Whether to log to wandb
        prefix: Prefix for wandb logging
    """
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    if log_wandb and HAS_WANDB:
        log_dict = {
            f'{prefix}loss': test_loss,
            f'{prefix}acc': test_acc,
            'epoch': epoch
        }
        
        # Add per-class accuracies
        for class_name, acc in class_accuracies.items():
            log_dict[f'{prefix}class_acc/{class_name}'] = acc
        
        wandb.log(log_dict)


def print_class_accuracies(class_accuracies: Dict[str, float], title: str = "Per-class Accuracies") -> None:
    """
    Print per-class accuracies in a formatted way.
    
    Args:
        class_accuracies: Dictionary of class accuracies
        title: Title for the output
    """
    print(f"\n{title}:")
    for class_name, acc in class_accuracies.items():
        print(f"  {class_name}: {acc:.2f}%")


def evaluate_with_adversarial_attacks(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    eps: float = 8/255,
    attack_method: str = "fgsm"
) -> float:
    """
    Evaluate model with adversarial attacks.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to use
        eps: Attack epsilon
        attack_method: Type of attack ('fgsm', 'pgd', 'random')
        
    Returns:
        Adversarial accuracy
    """
    try:
        from ..evals.classification.adv_attacks import (
            fgsm_attack, pgd_linf_attack, random_attack
        )
    except ImportError:
        raise ImportError("Adversarial attack functions not available")
    
    correct = 0
    total = 0
    model.eval()
    
    for data, target in tqdm(test_loader, desc=f'Evaluating {attack_method.upper()}'):
        data, target = data.to(device), target.to(device)
        
        if attack_method == "fgsm":
            data = fgsm_attack(model, data, target, eps, criterion=criterion)
        elif attack_method == "pgd":
            data = pgd_linf_attack(
                model, data, target, eps,
                alpha=eps/3, num_iter=20, criterion=criterion
            )
        elif attack_method == "random":
            data = random_attack(data, eps)
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")
        
        with torch.no_grad():
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    acc = 100 * correct / total
    print(f"{attack_method.upper()} Adversarial Accuracy: {acc:.2f}%, eps: {255*eps:.1f}/255")
    return acc


def comprehensive_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    class_names: Optional[Tuple[str, ...]] = None,
    log_wandb: bool = True,
    eval_adversarial: bool = False
) -> Dict[str, Any]:
    """
    Perform comprehensive model evaluation.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to use
        epoch: Current epoch
        class_names: Names of classes
        log_wandb: Whether to log to wandb
        eval_adversarial: Whether to include adversarial evaluation
        
    Returns:
        Dictionary containing all evaluation results
    """
    # Clean evaluation
    test_loss, test_acc, class_accuracies = evaluate_model(
        model, test_loader, criterion, device, class_names
    )
    
    results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'class_accuracies': class_accuracies
    }
    
    # Log clean results
    log_evaluation_results(test_loss, test_acc, class_accuracies, epoch, log_wandb)
    
    # Adversarial evaluation if requested
    if eval_adversarial:
        try:
            fgsm_acc = evaluate_with_adversarial_attacks(
                model, test_loader, criterion, device, eps=8/255, attack_method="fgsm"
            )
            results['fgsm_acc'] = fgsm_acc
            
            if log_wandb and HAS_WANDB:
                wandb.log({'test/fgsm_acc': fgsm_acc, 'epoch': epoch})
                
        except ImportError:
            print("Adversarial attack functions not available, skipping adversarial evaluation")
    
    return results