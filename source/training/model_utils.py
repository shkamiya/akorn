"""
Model utilities for classification tasks.
This module provides functions for creating and configuring models.
"""

from typing import Dict, Any, Union
import torch
import torch.nn as nn
import torch.optim as optim


def create_myakorn_model(config: Dict[str, Any], device: str) -> nn.Module:
    """
    Create MyAKOrN model from configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
        
    Returns:
        MyAKOrN model
    """
    from ..models.classification.my_knet import MyAKOrN
    
    model = MyAKOrN(
        n=config['n'],
        ch=config['ch'],
        out_classes=config['num_classes'],
        L=config['L'],
        T=config['T'],
        J=config['J'],
        J_bias=config.get('J_bias', False),
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
        ensemble=config['ensemble'],
    ).to(device)
    
    return model


def create_akorn_model(config: Dict[str, Any], device: str) -> nn.Module:
    """
    Create AKOrN model from configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
        
    Returns:
        AKOrN model
    """
    from ..models.classification.knet import AKOrN
    
    model = AKOrN(
        n=config['n'],
        ch=config['ch'],
        out_classes=config['num_classes'],
        L=config['L'],
        T=config['T'],
        J=config['J'],
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
        ensemble=config['ensemble'],
    ).to(device)
    
    return model


def create_akorn_resnet_model(config: Dict[str, Any], device: str) -> nn.Module:
    """
    Create AKOrN+ResNet model from configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
        
    Returns:
        AKOrN+ResNet model
    """
    from ..models.classification.my_knet import AKOrNResNet
    
    model = AKOrNResNet(
        n=config['n'],
        ch=config['ch'],
        out_classes=config['num_classes'],
        L=config['L'],
        T=config['T'],
        ksizes=config['ksizes'],
        gamma=config['gamma'],
        bp_steps=config.get('bp_steps', None),
    ).to(device)
    
    return model


def create_model(model_type: str, config: Dict[str, Any], device: str) -> nn.Module:
    """
    Create model based on type and configuration.
    
    Args:
        model_type: Type of model ('myakorn', 'akorn', 'akorn_resnet')
        config: Configuration dictionary
        device: Device to place model on
        
    Returns:
        Created model
    """
    if model_type == 'myakorn':
        return create_myakorn_model(config, device)
    elif model_type == 'akorn':
        return create_akorn_model(config, device)
    elif model_type == 'akorn_resnet':
        return create_akorn_resnet_model(config, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Create optimizer from configuration.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        Optimizer
    """
    optimizer_type = config.get('optimizer', 'adam')
    
    if optimizer_type == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.0)
        )
    elif optimizer_type == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_criterion(config: Dict[str, Any]) -> nn.Module:
    """
    Create loss criterion from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Loss criterion
    """
    criterion_type = config.get('criterion', 'ce')
    
    if criterion_type == 'ce':
        return nn.CrossEntropyLoss()
    elif criterion_type == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown criterion type: {criterion_type}")


def setup_ema(model: nn.Module, config: Dict[str, Any]) -> Any:
    """
    Setup Exponential Moving Average (EMA) for model.
    
    Args:
        model: Model to apply EMA to
        config: Configuration dictionary
        
    Returns:
        EMA wrapper or None if not available
    """
    if not config.get('use_ema', False):
        return None
    
    try:
        from ema_pytorch import EMA
        
        ema = EMA(
            model,
            beta=config.get('ema_beta', 0.99),
            update_every=config.get('ema_update_every', 10),
            update_after_step=config.get('ema_update_after_step', 200)
        )
        
        return ema
    except ImportError:
        print("EMA not available, skipping EMA setup")
        return None


def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str,
    device: str,
    ignore_size_mismatch: bool = False
) -> None:
    """
    Load pretrained weights into model.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        ignore_size_mismatch: Whether to ignore size mismatches
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    if ignore_size_mismatch:
        from ..utils import load_state_dict_ignore_size_mismatch
        load_state_dict_ignore_size_mismatch(model, state_dict)
    else:
        model.load_state_dict(state_dict, strict=False)
    
    print(f"Loaded pretrained weights from {checkpoint_path}")