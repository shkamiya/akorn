"""
Sweep Analysis Utilities for AKOrN Models

This module provides comprehensive utilities for analyzing parameter sweep results
from AKOrN (Artificial Kuramoto Oscillator Network) models. It includes functions
for loading models, evaluating performance, extracting energy dynamics, and
conducting systematic experiments with different T patterns.

Extracted from notebooks/e2025_0712_J_comprehensive_counterfactual_analysis.ipynb
"""

import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .my_knet import MyAKOrN
from .analysis_utils import AKOrNStaticAnalyzer, AKOrNDynamicalAnalyzer

# T mapping for extending T values following 2^n-1 pattern
T_MAPPING = {
    3: 7,    # 2^2-1 -> 2^3-1  
    7: 15,   # 2^3-1 -> 2^4-1
    15: 31,  # 2^4-1 -> 2^5-1
    31: 63,  # 2^5-1 -> 2^6-1
    63: 127  # 2^6-1 -> 2^7-1
}

def load_model_from_sweep(sweep_dir: str, results_dir: Path, device: torch.device) -> Optional[Tuple[MyAKOrN, Dict]]:
    """
    Load a trained model from sweep results.
    
    Args:
        sweep_dir: Directory name containing the sweep results
        results_dir: Path to the results directory
        device: Device to load model on
        
    Returns:
        Tuple of (model, config) or None if loading fails
    """
    # Load config
    config_path = results_dir / sweep_dir / "parameters.json"
    if not config_path.exists():
        print(f"Config not found for {sweep_dir}")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check if model exists
    model_path = results_dir / sweep_dir / "my_akorn_cifar10_final.pth"
    if not model_path.exists():
        print(f"Model not found for {sweep_dir}")
        return None
    
    try:
        # Create model
        model = MyAKOrN(
            n=config['n'],
            ch=config['ch'], 
            out_classes=config['num_classes'],
            L=config['L'],
            T=config['T'],
            J=config['J'],
            J_bias=config['J_bias'],
            ksizes=config['ksizes'],
            ro_ksize=config['ro_ksize'],
            ro_N=config['ro_N'],
            norm=config['norm'],
            c_norm=config['c_norm'],
            gamma=config['gamma'],
            use_omega=config['use_omega'],
            init_omg=config['init_omg'],
            global_omg=config['global_omg'],
            learn_omg=config['learn_omg'],
            ensemble=config['ensemble']
        ).to(device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Successfully loaded {sweep_dir} (gamma={config['gamma']}, T={config['T']})")
        return model, config
        
    except Exception as e:
        print(f"Error loading {sweep_dir}: {e}")
        return None


def load_all_sweep_models(sweep_dirs: List[str], results_dir: Path, device: torch.device) -> Dict:
    """
    Load all models from a list of sweep directories.
    
    Args:
        sweep_dirs: List of sweep directory names
        results_dir: Path to results directory
        device: Device to load models on
        
    Returns:
        Dictionary mapping case names to model data
    """
    loaded_models = {}
    parameter_summary = []
    
    for i, sweep_dir in enumerate(sweep_dirs):
        result = load_model_from_sweep(sweep_dir, results_dir, device)
        if result is not None:
            model, config = result
            case_name = f"Case {i}"
            loaded_models[case_name] = {
                "model": model,
                "config": config,
                "gamma": config["gamma"],
                "T": config["T"],
                "sweep_dir": sweep_dir,
                "index": i
            }
            
            # Add to parameter summary
            parameter_summary.append({
                "case": case_name,
                "index": i,
                "gamma": config["gamma"],
                "T": config["T"],
                "sweep_dir": sweep_dir
            })
    
    print(f"Successfully loaded {len(loaded_models)} models for analysis")
    return loaded_models, parameter_summary


def evaluate_model_performance(model: MyAKOrN, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model performance on given data loader.
    
    Args:
        model: The model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, accuracy


def create_test_loader(batch_size: int = 64, data_dir: str = './data') -> DataLoader:
    """
    Create test data loader for CIFAR10.
    
    Args:
        batch_size: Batch size for data loader
        data_dir: Directory to store/load data
        
    Returns:
        DataLoader for test data
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform_test
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return test_loader


def create_train_loader(batch_size: int = 64, data_dir: str = './data') -> DataLoader:
    """
    Create training data loader for CIFAR10.
    
    Args:
        batch_size: Batch size for data loader
        data_dir: Directory to store/load data
        
    Returns:
        DataLoader for training data
    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform_train
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Don't shuffle for consistent analysis
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader


def extract_energy_dynamics_batch(model: MyAKOrN, input_batch: torch.Tensor, layer_indices: List[int] = [0, 1, 2]) -> Dict:
    """
    Extract energy dynamics for a batch of inputs across specified layers.
    
    Args:
        model: The AKOrN model
        input_batch: Batch of input tensors
        layer_indices: List of layer indices to analyze
        
    Returns:
        Dictionary mapping layer indices to lists of energy trajectories
    """
    model.eval()
    
    batch_trajectories = {layer_idx: [] for layer_idx in layer_indices}
    
    with torch.no_grad():
        # Process each sample in the batch separately to get individual trajectories
        for sample_idx in range(input_batch.size(0)):
            sample_input = input_batch[sample_idx:sample_idx+1]  # Keep batch dimension
            
            # Get intermediate states and energies for all layers
            _, _, xs, es = model.feature(sample_input)
            
            # Extract energy trajectories for specified layers
            for layer_idx in layer_indices:
                if layer_idx < len(es):
                    layer_energies = es[layer_idx]
                    if layer_energies is not None and len(layer_energies) > 0:
                        energy_values = [float(e.item()) for e in layer_energies]
                        batch_trajectories[layer_idx].append(energy_values)
    
    return batch_trajectories


def create_model_with_custom_T_pattern(original_model: MyAKOrN, original_config: Dict, 
                                     T_pattern: List[int], device: torch.device) -> MyAKOrN:
    """
    Create a model with custom T pattern across layers.
    
    Args:
        original_model: The original trained model
        original_config: Original model configuration
        T_pattern: List of T values for each layer [T1, T2, T3]
        device: Device to run on
        
    Returns:
        New model with specified T pattern
    """
    modified_config = original_config.copy()
    
    print(f"  Creating model with T={T_pattern} for layers [1, 2, 3]")
    
    new_model = MyAKOrN(
        n=modified_config['n'],
        ch=modified_config['ch'], 
        out_classes=modified_config['num_classes'],
        L=modified_config['L'],
        T=T_pattern,  # Pass custom T pattern
        J=modified_config['J'],
        J_bias=modified_config['J_bias'],
        ksizes=modified_config['ksizes'],
        ro_ksize=modified_config['ro_ksize'],
        ro_N=modified_config['ro_N'],
        norm=modified_config['norm'],
        c_norm=modified_config['c_norm'],
        gamma=modified_config['gamma'],
        use_omega=modified_config['use_omega'],
        init_omg=modified_config['init_omg'],
        global_omg=modified_config['global_omg'],
        learn_omg=modified_config['learn_omg'],
        ensemble=modified_config['ensemble']
    ).to(device)
    
    # Copy weights from original model
    new_model.load_state_dict(original_model.state_dict())
    return new_model


def evaluate_T_pattern_experiment(loaded_models: Dict, T_pattern_func: Callable, 
                                experiment_name: str, device: torch.device,
                                train_loader: DataLoader, test_loader: DataLoader) -> Dict:
    """
    Generic function to evaluate T pattern experiments.
    
    Args:
        loaded_models: Dictionary of loaded models
        T_pattern_func: Function that takes (original_T, longer_T) and returns T_pattern list
        experiment_name: Name for this experiment
        device: Device to run on
        train_loader: Training data loader
        test_loader: Test data loader
        
    Returns:
        Dictionary with performance results
    """
    print(f"\n{experiment_name} Experiment:")
    print("=" * 60)
    
    performance_results = {}
    
    for name, model_data in loaded_models.items():
        gamma = model_data['gamma']
        T = model_data['T']
        case_index = model_data['index']
        
        print(f"\nProcessing {name} (γ={gamma}, T={T})...")
        
        original_model = model_data["model"]
        original_T = T
        longer_T = T_MAPPING.get(original_T, original_T * 2)
        T_pattern = T_pattern_func(original_T, longer_T)
        
        try:
            # Create model with custom T pattern
            modified_model = create_model_with_custom_T_pattern(
                original_model, model_data["config"], T_pattern, device
            )
            
            print(f"  Evaluating T pattern: {T_pattern}")
            
            # Evaluate both models
            mod_train_loss, mod_train_acc = evaluate_model_performance(modified_model, train_loader, device)
            mod_test_loss, mod_test_acc = evaluate_model_performance(modified_model, test_loader, device)
            
            orig_train_loss, orig_train_acc = evaluate_model_performance(original_model, train_loader, device)
            orig_test_loss, orig_test_acc = evaluate_model_performance(original_model, test_loader, device)
            
            performance_results[name] = {
                'gamma': gamma,
                'original_T': original_T,
                'longer_T': longer_T,
                'T_pattern': T_pattern,
                'case_index': case_index,
                'original': {
                    'train_loss': orig_train_loss,
                    'train_accuracy': orig_train_acc,
                    'test_loss': orig_test_loss,
                    'test_accuracy': orig_test_acc
                },
                'modified': {
                    'train_loss': mod_train_loss,
                    'train_accuracy': mod_train_acc,
                    'test_loss': mod_test_loss,
                    'test_accuracy': mod_test_acc
                }
            }
            
            print(f"  Original T={original_T}: Train Acc={orig_train_acc:.3f}, Test Acc={orig_test_acc:.3f}")
            print(f"  Modified {T_pattern}: Train Acc={mod_train_acc:.3f}, Test Acc={mod_test_acc:.3f}")
            
        except Exception as e:
            print(f"  Error creating modified model for {name}: {e}")
    
    print(f"\nCompleted {experiment_name} experiment for {len(performance_results)} models")
    return performance_results


def plot_experiment_comparison(performance_data: Dict, experiment_name: str, colormap: str = 'plasma'):
    """
    Generic function to plot experiment comparisons.
    
    Args:
        performance_data: Results from evaluate_T_pattern_experiment
        experiment_name: Name for the experiment
        colormap: Matplotlib colormap to use
    """
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data for plotting
    plot_data = {
        'gamma': [],
        'T': [],
        'case_index': [],
        'orig_train_acc': [],
        'mod_train_acc': [],
        'orig_test_acc': [],
        'mod_test_acc': [],
        'orig_train_loss': [],
        'mod_train_loss': [],
        'orig_test_loss': [],
        'mod_test_loss': []
    }
    
    for name, data in performance_data.items():
        plot_data['gamma'].append(data['gamma'])
        plot_data['T'].append(data['original_T'])
        plot_data['case_index'].append(data['case_index'])
        plot_data['orig_train_acc'].append(data['original']['train_accuracy'])
        plot_data['mod_train_acc'].append(data['modified']['train_accuracy'])
        plot_data['orig_test_acc'].append(data['original']['test_accuracy'])
        plot_data['mod_test_acc'].append(data['modified']['test_accuracy'])
        plot_data['orig_train_loss'].append(data['original']['train_loss'])
        plot_data['mod_train_loss'].append(data['modified']['train_loss'])
        plot_data['orig_test_loss'].append(data['original']['test_loss'])
        plot_data['mod_test_loss'].append(data['modified']['test_loss'])
    
    # Plot 1: Training Accuracy
    axes[0, 0].scatter(plot_data['orig_train_acc'], plot_data['mod_train_acc'], 
                      c=plot_data['case_index'], cmap=colormap, s=100, alpha=0.7, edgecolors='black')
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.8, linewidth=2)
    axes[0, 0].set_xlabel('Original - Training Accuracy', fontsize=12)
    axes[0, 0].set_ylabel(f'{experiment_name} - Training Accuracy', fontsize=12)
    axes[0, 0].set_title(f'Training Accuracy: Original vs {experiment_name}', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy
    axes[0, 1].scatter(plot_data['orig_test_acc'], plot_data['mod_test_acc'], 
                      c=plot_data['case_index'], cmap=colormap, s=100, alpha=0.7, edgecolors='black')
    axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.8, linewidth=2)
    axes[0, 1].set_xlabel('Original - Test Accuracy', fontsize=12)
    axes[0, 1].set_ylabel(f'{experiment_name} - Test Accuracy', fontsize=12)
    axes[0, 1].set_title(f'Test Accuracy: Original vs {experiment_name}', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training Loss
    axes[1, 0].scatter(plot_data['orig_train_loss'], plot_data['mod_train_loss'], 
                      c=plot_data['case_index'], cmap=colormap, s=100, alpha=0.7, edgecolors='black')
    min_loss = min(min(plot_data['orig_train_loss']), min(plot_data['mod_train_loss']))
    max_loss = max(max(plot_data['orig_train_loss']), max(plot_data['mod_train_loss']))
    axes[1, 0].plot([min_loss, max_loss], [min_loss, max_loss], 'r--', alpha=0.8, linewidth=2)
    axes[1, 0].set_xlabel('Original - Training Loss', fontsize=12)
    axes[1, 0].set_ylabel(f'{experiment_name} - Training Loss', fontsize=12)
    axes[1, 0].set_title(f'Training Loss: Original vs {experiment_name}', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Test Loss
    axes[1, 1].scatter(plot_data['orig_test_loss'], plot_data['mod_test_loss'], 
                      c=plot_data['case_index'], cmap=colormap, s=100, alpha=0.7, edgecolors='black')
    min_loss = min(min(plot_data['orig_test_loss']), min(plot_data['mod_test_loss']))
    max_loss = max(max(plot_data['orig_test_loss']), max(plot_data['mod_test_loss']))
    axes[1, 1].plot([min_loss, max_loss], [min_loss, max_loss], 'r--', alpha=0.8, linewidth=2)
    axes[1, 1].set_xlabel('Original - Test Loss', fontsize=12)
    axes[1, 1].set_ylabel(f'{experiment_name} - Test Loss', fontsize=12)
    axes[1, 1].set_title(f'Test Loss: Original vs {experiment_name}', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add case annotations to all plots
    for ax in axes.flat:
        for i, case_idx in enumerate(plot_data['case_index']):
            if 'Accuracy' in ax.get_title():
                if 'Training' in ax.get_title():
                    x, y = plot_data['orig_train_acc'][i], plot_data['mod_train_acc'][i]
                else:
                    x, y = plot_data['orig_test_acc'][i], plot_data['mod_test_acc'][i]
            else:  # Loss plots
                if 'Training' in ax.get_title():
                    x, y = plot_data['orig_train_loss'][i], plot_data['mod_train_loss'][i]
                else:
                    x, y = plot_data['orig_test_loss'][i], plot_data['mod_test_loss'][i]
            
            ax.annotate(f"{case_idx}", (x, y), xytext=(3, 3), textcoords='offset points', 
                       fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    plt.suptitle(f'{experiment_name} Experiment: Performance Comparison', fontsize=16, y=1.02)
    plt.show()


def print_experiment_summary(performance_data: Dict, experiment_name: str):
    """
    Print detailed summary of experiment results.
    
    Args:
        performance_data: Results from evaluate_T_pattern_experiment
        experiment_name: Name for the experiment
    """
    print(f"\n{experiment_name} - Detailed Performance Comparison:")
    print("=" * 130)
    print(f"{'Case':<8} {'γ':<6} {'T_Pattern':<15} {'Train Acc (Orig)':<15} {'Train Acc (Mod)':<15} {'Test Acc (Orig)':<14} {'Test Acc (Mod)':<14}")
    print("-" * 130)
    
    for name in sorted(performance_data.keys(), key=lambda x: performance_data[x]['case_index']):
        data = performance_data[name]
        T_pattern_str = str(data['T_pattern'])
        print(f"{data['case_index']:<8} {data['gamma']:<6} {T_pattern_str:<15} "
              f"{data['original']['train_accuracy']:<15.3f} {data['modified']['train_accuracy']:<15.3f} "
              f"{data['original']['test_accuracy']:<14.3f} {data['modified']['test_accuracy']:<14.3f}")
    
    # Summary statistics
    orig_train_accs = [data['original']['train_accuracy'] for data in performance_data.values()]
    mod_train_accs = [data['modified']['train_accuracy'] for data in performance_data.values()]
    orig_test_accs = [data['original']['test_accuracy'] for data in performance_data.values()]
    mod_test_accs = [data['modified']['test_accuracy'] for data in performance_data.values()]
    
    print(f"\n{experiment_name} - Summary Statistics:")
    print("-" * 60)
    print(f"Training Accuracy - Original: {np.mean(orig_train_accs):.3f} ± {np.std(orig_train_accs):.3f}")
    print(f"Training Accuracy - Modified: {np.mean(mod_train_accs):.3f} ± {np.std(mod_train_accs):.3f}")
    print(f"Test Accuracy - Original: {np.mean(orig_test_accs):.3f} ± {np.std(orig_test_accs):.3f}")
    print(f"Test Accuracy - Modified: {np.mean(mod_test_accs):.3f} ± {np.std(mod_test_accs):.3f}")
    
    # Performance changes
    train_acc_changes = [mod - orig for mod, orig in zip(mod_train_accs, orig_train_accs)]
    test_acc_changes = [mod - orig for mod, orig in zip(mod_test_accs, orig_test_accs)]
    
    print(f"\n{experiment_name} - Performance Changes:")
    print("-" * 60)
    print(f"Training Accuracy Change: {np.mean(train_acc_changes):.3f} ± {np.std(train_acc_changes):.3f}")
    print(f"Test Accuracy Change: {np.mean(test_acc_changes):.3f} ± {np.std(test_acc_changes):.3f}")
    
    improved_train = sum(1 for change in train_acc_changes if change > 0)
    improved_test = sum(1 for change in test_acc_changes if change > 0)
    
    print(f"\nModels with improved performance ({experiment_name}):")
    print(f"Training accuracy improved: {improved_train}/{len(train_acc_changes)} models")
    print(f"Test accuracy improved: {improved_test}/{len(test_acc_changes)} models")


def extract_connectivity_decomposition(model: MyAKOrN, layer_idx: int = 0, device: torch.device = torch.device('cpu')) -> Optional[Dict]:
    """
    Extract connectivity decomposition results for network analysis.
    
    Args:
        model: The AKOrN model
        layer_idx: Layer index to analyze
        device: Device to run on
        
    Returns:
        Dictionary with decomposition results or None if extraction fails
    """
    try:
        # Create static analyzer for the specified layer
        static_analyzer = AKOrNStaticAnalyzer(model, layer_idx, device=device)
        
        # Extract connectivity blocks
        connectivity_blocks = static_analyzer.extract_connectivity_blocks()
        if connectivity_blocks is None:
            return None
        
        # Compute decomposition metrics
        results = {}
        
        # 1. Frobenius norms
        frob_norms = np.linalg.norm(connectivity_blocks, axis=(1, 2))
        results['frob_norms'] = frob_norms
        
        # 2. Rotation/Symmetric decomposition
        c_R, c_S, alpha, beta = static_analyzer.decompose_rotation_symmetric(connectivity_blocks)
        results['c_R'] = c_R
        results['c_S'] = c_S
        results['alpha'] = alpha
        results['beta'] = beta
        
        # 3. Symmetric/Skew-symmetric decomposition  
        p1, p2, p3, q, sym_frob, skew_frob = static_analyzer.decompose_symmetric_skew(connectivity_blocks)
        results['sym_frob'] = sym_frob
        results['skew_frob'] = skew_frob
        
        return results
        
    except Exception as e:
        print(f"Error in connectivity decomposition: {e}")
        return None


# T pattern functions for different experiments
def all_layers_extended_T_pattern(original_T: int, longer_T: int) -> List[int]:
    """Create T pattern with extended T in all layers."""
    return [longer_T, longer_T, longer_T]


def first_layer_extended_T_pattern(original_T: int, longer_T: int) -> List[int]:
    """Create T pattern with extended T in first layer only."""
    return [longer_T, original_T, original_T]


def second_third_layers_T_pattern(original_T: int, longer_T: int) -> List[int]:
    """Create T pattern with extended T in layers 2 and 3, original T in layer 1."""
    return [original_T, longer_T, longer_T]


def analyze_energy_dynamics_statistics(loaded_models: Dict, data_loader: DataLoader, 
                                     device: torch.device, max_batches: int = 10, 
                                     max_samples_per_batch: int = 8) -> Dict:
    """
    Analyze energy dynamics statistics across multiple models and data samples.
    
    Args:
        loaded_models: Dictionary of loaded models
        data_loader: Data loader for analysis
        device: Device to run on
        max_batches: Maximum number of batches to process
        max_samples_per_batch: Maximum samples per batch
        
    Returns:
        Dictionary with energy dynamics statistics for each model
    """
    all_dynamics = {}
    
    for name, model_data in loaded_models.items():
        print(f"\nProcessing {name} (γ={model_data['gamma']}, T={model_data['T']})...")
        model = model_data["model"]
        
        # Collect all trajectories for this model
        all_trajectories = {0: [], 1: [], 2: []}
        
        batch_count = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            # Take subset of batch for computational efficiency
            data_subset = data[:max_samples_per_batch].to(device)
            
            try:
                batch_trajectories = extract_energy_dynamics_batch(model, data_subset)
                
                # Accumulate trajectories
                for layer_idx in [0, 1, 2]:
                    if layer_idx in batch_trajectories:
                        all_trajectories[layer_idx].extend(batch_trajectories[layer_idx])
                
                batch_count += 1
                if batch_count % 5 == 0:
                    print(f"  Processed {batch_count} batches...")
                    
            except Exception as e:
                print(f"  Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate statistics for each layer
        layer_stats = {}
        for layer_idx in [0, 1, 2]:
            if all_trajectories[layer_idx]:
                # Convert to numpy array for easier computation
                trajectories_array = np.array(all_trajectories[layer_idx])
                
                # Calculate mean and std across all samples
                mean_trajectory = np.mean(trajectories_array, axis=0)
                std_trajectory = np.std(trajectories_array, axis=0)
                
                layer_stats[layer_idx] = {
                    'mean': mean_trajectory,
                    'std': std_trajectory,
                    'n_samples': len(all_trajectories[layer_idx]),
                    'trajectory_length': len(mean_trajectory)
                }
                
                print(f"  Layer {layer_idx}: {len(all_trajectories[layer_idx])} samples, "
                      f"trajectory length {len(mean_trajectory)}")
        
        all_dynamics[name] = {
            'layer_stats': layer_stats,
            'gamma': model_data['gamma'],
            'T': model_data['T'],
            'case_index': model_data['index']
        }
    
    print(f"\nCompleted energy dynamics analysis for {len(all_dynamics)} models")
    return all_dynamics


# ==================================================================================
# ENERGY DYNAMICS PLOTTING UTILITIES
# ==================================================================================

def _setup_parameter_space():
    """Setup standard parameter space for plotting (backward compatibility)."""
    gamma_values = [0.01, 0.1, 1.0]
    T_values = [3, 7, 15, 31, 63]
    return gamma_values, T_values


def _create_case_mapping(dynamics_data: Dict, X_label: str = "gamma", Y_label: str = "T"):
    """
    Create mapping from case indices to parameter combinations.
    
    Args:
        dynamics_data: Dictionary with dynamics data
        X_label: Name of X parameter in data (e.g., "gamma")  
        Y_label: Name of Y parameter in data (e.g., "T")
    """
    case_to_params = {}
    for name, data in dynamics_data.items():
        case_idx = data['case_index']
        case_to_params[case_idx] = (data[X_label], data[Y_label])
    return case_to_params


def _find_matching_case(X_val: Union[float, int], Y_val: Union[float, int], case_to_params: Dict):
    """Find the case index that matches the given parameter combination."""
    for case_idx, (x, y) in case_to_params.items():
        # Handle both float and int comparisons
        if isinstance(x, float) and isinstance(X_val, float):
            x_match = abs(x - X_val) < 1e-6
        else:
            x_match = x == X_val
        
        if isinstance(y, float) and isinstance(Y_val, float):
            y_match = abs(y - Y_val) < 1e-6
        else:
            y_match = y == Y_val
            
        if x_match and y_match:
            return case_idx
    return None


def _format_subplot(ax, X_val: Union[float, int], Y_val: Union[float, int], 
                   X_label: str = "γ", Y_label: str = "T", 
                   case_idx: Optional[int] = None, 
                   additional_info: str = "", layer_info: str = ""):
    """Apply standard formatting to subplot."""
    if case_idx is not None:
        title = f'{X_label}={X_val}, {Y_label}={Y_val}\n(Case {case_idx}{additional_info})'
    else:
        title = f'{X_label}={X_val}, {Y_label}={Y_val}{additional_info}'
    
    if layer_info:
        title += f'\n{layer_info}'
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_ylabel('Energy', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))


def _handle_no_data_case(ax, X_val: Union[float, int], Y_val: Union[float, int], 
                        X_label: str = "γ", Y_label: str = "T"):
    """Handle case where no data is available."""
    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
           ha='center', va='center', fontsize=14, color='red')
    ax.set_title(f'{X_label}={X_val}, {Y_label}={Y_val}', fontsize=12)


def plot_energy_dynamics(
        dynamics_data: Dict,
        X_values: List = None,
        Y_values: List = None,
        X_label: str = "γ",
        Y_label: str = "T",
        data_type: str = "Training",
        title: str = None
        ):
    """
    Plot energy dynamics for all three layers across parameter combinations.
    Automatically detects whether to plot single trajectories or statistical data (mean ± std).
    
    Args:
        dynamics_data: Dictionary with energy dynamics data
        X_values: List of X parameter values (rows in subplot grid)
        Y_values: List of Y parameter values (columns in subplot grid)
        X_label: Label for X parameter (e.g., "γ", "alpha")
        Y_label: Label for Y parameter (e.g., "T", "beta")
        data_type: Type of data ("Training" or "Test") - used for statistical plots
        title: Title for the overall figure
    """
    # Use default values if not provided
    if X_values is None or Y_values is None:
        X_values, Y_values = _setup_parameter_space()
        X_label = "γ"
        Y_label = "T"
    
    # Auto-detect data type by checking first case
    is_statistical = False
    if dynamics_data:
        first_case = next(iter(dynamics_data.values()))
        is_statistical = 'layer_stats' in first_case
    
    if title is None:
        if is_statistical:
            title = f'Energy Dynamics on {data_type} Data: Mean ± Std for All Three Layers'
        else:
            title = f"Energy Dynamics: All Three Layers by {X_label}-{Y_label} Combinations"
    
    fig, axes = plt.subplots(len(X_values), len(Y_values), figsize=(25, 18))
    
    # Handle case where we have only one row or column
    if len(X_values) == 1 or len(Y_values) == 1:
        axes = axes.reshape(len(X_values), len(Y_values))
    
    # Get X_label and Y_label from data keys if not explicitly provided
    X_data_key = X_label.replace("γ", "gamma") if X_label == "γ" else X_label
    Y_data_key = Y_label.replace("T", "T") if Y_label == "T" else Y_label
    
    case_to_params = _create_case_mapping(dynamics_data, X_data_key, Y_data_key)
    
    # Layer colors and names
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    layer_names = ['Layer 0', 'Layer 1', 'Layer 2']
    
    for X_idx, X_val in enumerate(X_values):
        for Y_idx, Y_val in enumerate(Y_values):
            ax = axes[X_idx, Y_idx]
            
            matching_case = _find_matching_case(X_val, Y_val, case_to_params)
            
            if matching_case is not None:
                case_name = f"Case {matching_case}"
                if case_name in dynamics_data:
                    data = dynamics_data[case_name]
                    
                    if is_statistical:
                        # Plot statistical data (mean ± std)
                        for layer_idx in range(3):
                            if layer_idx in data['layer_stats']:
                                stats = data['layer_stats'][layer_idx]
                                mean_traj = stats['mean']
                                std_traj = stats['std']
                                n_samples = stats['n_samples']
                                
                                time_steps = np.arange(len(mean_traj))
                                
                                # Plot mean trajectory
                                ax.plot(time_steps, mean_traj, color=colors[layer_idx], 
                                       linewidth=2.5, alpha=0.9, 
                                       label=f'{layer_names[layer_idx]} (n={n_samples})')
                                
                                # Plot confidence interval (mean ± std)
                                ax.fill_between(time_steps, 
                                               mean_traj - std_traj, 
                                               mean_traj + std_traj,
                                               color=colors[layer_idx], alpha=0.2)
                        ax.legend(loc='best', fontsize=8)
                    else:
                        # Plot basic trajectory data
                        for layer_idx in range(3):
                            if layer_idx in data['layers']:
                                trajectory = data['layers'][layer_idx]['trajectory']
                                ax.plot(trajectory, color=colors[layer_idx], 
                                       linewidth=2.5, alpha=0.8, label=layer_names[layer_idx])
                        ax.legend(loc='best', fontsize=9)
                    
                    _format_subplot(ax, X_val, Y_val, X_label, Y_label, matching_case)
                else:
                    _handle_no_data_case(ax, X_val, Y_val, X_label, Y_label)
            else:
                _handle_no_data_case(ax, X_val, Y_val, X_label, Y_label)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_extended_T_comparison(dynamics_data: Dict, data_type: str = "Training", 
                              title: str = None):
    """
    Plot comparison between original T and extended T dynamics.
    
    Args:
        dynamics_data: Dictionary with extended T dynamics data
        data_type: Type of data ("Training" or "Test")
        title: Title for the overall figure
    """
    if title is None:
        title = f'Energy Dynamics with {data_type} Data: Original T vs Extended T (Mean ± Std)\n(Solid: Original T, Dashed: Extended T)'
    
    fig, axes = plt.subplots(3, 5, figsize=(25, 18))
    
    gamma_values, T_values = _setup_parameter_space()
    case_to_params = _create_case_mapping(dynamics_data)
    
    # Layer colors and names
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    layer_names = ['Layer 0', 'Layer 1', 'Layer 2']
    
    for gamma_idx, gamma in enumerate(gamma_values):
        for T_idx, T in enumerate(T_values):
            ax = axes[gamma_idx, T_idx]
            
            matching_case = _find_matching_case(gamma, T, case_to_params)
            
            if matching_case is not None:
                case_name = f"Case {matching_case}"
                if case_name in dynamics_data:
                    data = dynamics_data[case_name]
                    
                    # Plot all three layers with original vs extended T comparison
                    for layer_idx in range(3):
                        color = colors[layer_idx]
                        layer_name = layer_names[layer_idx]
                        
                        # Plot original T dynamics (solid line)
                        if layer_idx in data['original_stats']:
                            orig_stats = data['original_stats'][layer_idx]
                            orig_mean = orig_stats['mean']
                            orig_std = orig_stats['std']
                            orig_time = np.arange(len(orig_mean))
                            
                            ax.plot(orig_time, orig_mean, color=color, linewidth=2.5, alpha=0.9, 
                                   label=f'{layer_name} T={data["original_T"]}')
                            ax.fill_between(orig_time, orig_mean - orig_std, orig_mean + orig_std,
                                           color=color, alpha=0.15)
                        
                        # Plot extended T dynamics (dashed line)
                        if layer_idx in data['longer_stats']:
                            longer_stats = data['longer_stats'][layer_idx]
                            longer_mean = longer_stats['mean']
                            longer_std = longer_stats['std']
                            longer_time = np.arange(len(longer_mean))
                            
                            ax.plot(longer_time, longer_mean, color=color, linewidth=2, alpha=0.7, 
                                   linestyle='--', label=f'{layer_name} T={data["longer_T"]}')
                            ax.fill_between(longer_time, longer_mean - longer_std, longer_mean + longer_std,
                                           color=color, alpha=0.1)
                    
                    n_samples = data['original_stats'].get(0, {}).get('n_samples', 0)
                    additional_info = f', n={n_samples}'
                    _format_subplot(ax, gamma, T, matching_case, 
                                   f'→{data["longer_T"]}{additional_info}')
                    ax.legend(loc='best', fontsize=7)
                else:
                    _handle_no_data_case(ax, gamma, T)
            else:
                _handle_no_data_case(ax, gamma, T)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_train_test_comparison(train_dynamics: Dict, test_dynamics: Dict, 
                              title: str = "Training vs Test Data Comparison: Energy Dynamics (Layer 0 Only)"):
    """
    Plot comparison between training and test data dynamics.
    
    Args:
        train_dynamics: Dictionary with training dynamics data
        test_dynamics: Dictionary with test dynamics data
        title: Title for the overall figure
    """
    fig, axes = plt.subplots(3, 5, figsize=(25, 18))
    
    gamma_values, T_values = _setup_parameter_space()
    case_to_params = _create_case_mapping(train_dynamics)
    
    # Colors for train vs test
    colors = ['#1f77b4', '#ff7f0e']  # Blue for train, Orange for test
    
    for gamma_idx, gamma in enumerate(gamma_values):
        for T_idx, T in enumerate(T_values):
            ax = axes[gamma_idx, T_idx]
            
            matching_case = _find_matching_case(gamma, T, case_to_params)
            
            if matching_case is not None:
                case_name = f"Case {matching_case}"
                
                # Check if we have both training and test data
                if case_name in train_dynamics and case_name in test_dynamics:
                    train_data = train_dynamics[case_name]
                    test_data = test_dynamics[case_name]
                    
                    # Plot comparison for Layer 0 only (to avoid clutter)
                    layer_idx = 0
                    
                    if (layer_idx in train_data['layer_stats'] and 
                        layer_idx in test_data['layer_stats']):
                        
                        # Training data
                        train_stats = train_data['layer_stats'][layer_idx]
                        train_mean = train_stats['mean']
                        train_std = train_stats['std']
                        train_time = np.arange(len(train_mean))
                        
                        # Test data  
                        test_stats = test_data['layer_stats'][layer_idx]
                        test_mean = test_stats['mean']
                        test_std = test_stats['std']
                        test_time = np.arange(len(test_mean))
                        
                        # Plot training data
                        ax.plot(train_time, train_mean, color=colors[0], 
                               linewidth=2.5, alpha=0.9, 
                               label=f'Train (n={train_stats["n_samples"]})')
                        ax.fill_between(train_time, train_mean - train_std, train_mean + train_std,
                                       color=colors[0], alpha=0.2)
                        
                        # Plot test data
                        ax.plot(test_time, test_mean, color=colors[1], 
                               linewidth=2.5, alpha=0.9, 
                               label=f'Test (n={test_stats["n_samples"]})')
                        ax.fill_between(test_time, test_mean - test_std, test_mean + test_std,
                                       color=colors[1], alpha=0.2)
                        
                        _format_subplot(ax, gamma, T, matching_case, layer_info="Layer 0")
                        ax.legend(loc='best', fontsize=8)
                    else:
                        _handle_no_data_case(ax, gamma, T)
                else:
                    _handle_no_data_case(ax, gamma, T)
            else:
                _handle_no_data_case(ax, gamma, T)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_energy_dynamics_comparison_grid(dynamics_data_list: List[Dict], 
                                        data_labels: List[str],
                                        colors: List[str] = None,
                                        title: str = "Energy Dynamics Comparison"):
    """
    Plot comparison of multiple energy dynamics datasets in a grid.
    
    Args:
        dynamics_data_list: List of dynamics data dictionaries
        data_labels: List of labels for each dataset
        colors: List of colors for each dataset
        title: Title for the overall figure
    """
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig, axes = plt.subplots(3, 5, figsize=(25, 18))
    
    gamma_values, T_values = _setup_parameter_space()
    case_to_params = _create_case_mapping(dynamics_data_list[0])
    
    for gamma_idx, gamma in enumerate(gamma_values):
        for T_idx, T in enumerate(T_values):
            ax = axes[gamma_idx, T_idx]
            
            matching_case = _find_matching_case(gamma, T, case_to_params)
            
            if matching_case is not None:
                case_name = f"Case {matching_case}"
                
                # Plot data from all provided datasets
                for data_idx, (dynamics_data, label) in enumerate(zip(dynamics_data_list, data_labels)):
                    if case_name in dynamics_data:
                        data = dynamics_data[case_name]
                        color = colors[data_idx % len(colors)]
                        
                        # Plot Layer 0 only for comparison
                        layer_idx = 0
                        if 'layer_stats' in data and layer_idx in data['layer_stats']:
                            stats = data['layer_stats'][layer_idx]
                            mean_traj = stats['mean']
                            std_traj = stats['std']
                            n_samples = stats['n_samples']
                            
                            time_steps = np.arange(len(mean_traj))
                            
                            ax.plot(time_steps, mean_traj, color=color, 
                                   linewidth=2.5, alpha=0.9, 
                                   label=f'{label} (n={n_samples})')
                            ax.fill_between(time_steps, mean_traj - std_traj, mean_traj + std_traj,
                                           color=color, alpha=0.2)
                        elif 'layers' in data and layer_idx in data['layers']:
                            # Handle basic dynamics data
                            trajectory = data['layers'][layer_idx]['trajectory']
                            ax.plot(trajectory, color=color, linewidth=2.5, alpha=0.8, label=label)
                
                _format_subplot(ax, gamma, T, matching_case, layer_info="Layer 0")
                ax.legend(loc='best', fontsize=8)
            else:
                _handle_no_data_case(ax, gamma, T)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()