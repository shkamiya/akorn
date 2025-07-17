"""
CIFAR-10 dataset utilities.
This module provides functions for loading and preparing CIFAR-10 datasets.
"""

from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from .augs import augmentation_strong


def get_cifar10_transforms(use_augmentation: bool = True) -> Dict[str, transforms.Compose]:
    """
    Get CIFAR-10 transforms for training and testing.
    
    Args:
        use_augmentation: Whether to use strong augmentation for training
        
    Returns:
        Dictionary containing 'train' and 'test' transforms
    """
    if use_augmentation:
        transform_train = augmentation_strong(imsize=32)
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    return {
        'train': transform_train,
        'test': transform_test
    }


def create_cifar10_dataloaders(
    config: Dict[str, Any],
    use_augmentation: bool = True,
    data_root: str = './data'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 train and test data loaders.
    
    Args:
        config: Configuration dictionary containing batch_size, num_workers, etc.
        use_augmentation: Whether to use data augmentation for training
        data_root: Root directory for dataset
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    transforms_dict = get_cifar10_transforms(use_augmentation)
    
    # Load datasets
    train_dataset = CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transforms_dict['train']
    )
    
    test_dataset = CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transforms_dict['test']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 128),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 128),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_cifar10_classes() -> Tuple[str, ...]:
    """
    Get CIFAR-10 class names.
    
    Returns:
        Tuple of class names
    """
    return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def print_dataset_info(train_loader: DataLoader, test_loader: DataLoader) -> None:
    """
    Print dataset information.
    
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    print(f"\nDataset Information:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {len(get_cifar10_classes())}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Number of batches - Train: {len(train_loader)}, Test: {len(test_loader)}")