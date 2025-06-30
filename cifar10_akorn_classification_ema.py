#!/usr/bin/env python3
"""
CIFAR-10 Classification with AKOrN (Artificial Kuramoto Oscillator Networks)

This script trains an AKOrN model for CIFAR-10 image classification with wandb logging.
AKOrN is based on the dynamics of Kuramoto oscillators and provides an alternative to traditional neural networks.

References:
- Miyato et al., "Artificial Kuramoto Oscillatory Neurons", ICLR 2025
"""

import os
import sys
import time
import random
import json
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
#from sched import scheduler 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from ema_pytorch import EMA

import wandb

# Add source directory to path
sys.path.append('source')

from models.classification.knet import AKOrN
from data.augs import augmentation_strong
from training_utils import save_checkpoint, save_model
from utils import str2bool


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_config():
    """Get training configuration"""

    config = {
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
        'J': 'conv',         # Connectivity type ('conv' or 'attn')
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
        #'warmup_epochs': 5,

        # EMA
        'ema_beta': 0.99,
        'ema_update_every': 10,
        'ema_update_after_step': 200,

        # Logging
        'log_interval': 100,
        'eval_interval': 5,
        'save_interval': 20,
        
        # Experiment
        'seed': 42,
        'experiment_name': 'akorn_cifar10',
        'save_dir': None,
    }
    return config


def create_data_loaders(config):
    """Create train and test data loaders"""
    # Data transforms
    transform_train = augmentation_strong(imsize=32)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load datasets
    train_dataset = CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )

    test_dataset = CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )

    return train_loader, test_loader


def create_model(config, device):
    """Create AKOrN model"""
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


def train_epoch(model, ema, train_loader, criterion, optimizer, device, epoch, config):
    """Train for one epoch"""
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
        
        # Update EMA
        ema.update()

        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Log to wandb
        if batch_idx % config['log_interval'] == 0:
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


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
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
    
    # Per-class accuracies
    class_accuracies = {}
    for i in range(10):
        if class_total[i] > 0:
            class_accuracies[classes[i]] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracies[classes[i]] = 0
    
    return test_loss, test_acc, class_accuracies


def save_checkpoint_with_config(model, optimizer, epoch, loss, config, filename, scheduler=None):
    """Save model checkpoint with configuration"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def save_parameters(config, save_dir):
    """Save training parameters to JSON file"""
    save_path = Path(save_dir) / 'parameters.json'
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Parameters saved to: {save_path}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='CIFAR-10 AKOrN Classification')
    
    # Wandb arguments
    parser.add_argument('--wandb-project', default='akorn-cifar10', help='W&B project name')
    parser.add_argument('--wandb-entity', default=None, help='W&B entity name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    
    # Data arguments
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    # Model architecture arguments
    parser.add_argument('--n', type=int, default=2, help='Oscillator dimension (2D for complex oscillators)')
    parser.add_argument('--ch', type=int, default=16, help='Base number of channels')
    parser.add_argument('--L', type=int, default=3, help='Number of layers')
    parser.add_argument('--T', type=int, default=3, help='Number of time steps per layer')
    parser.add_argument('--gamma', type=float, default=1.0, help='Integration step size')
    parser.add_argument('--J', type=str, default='conv', choices=['conv', 'attn'], help='Connectivity type')
    parser.add_argument('--ksizes', type=int, nargs='+', default=[9, 7, 5], help='Kernel sizes for each layer')
    parser.add_argument('--ro-ksize', type=int, default=3, help='Readout kernel size')
    parser.add_argument('--ro-N', type=int, default=2, help='Readout N parameter')
    parser.add_argument('--norm', type=str, default='bn', choices=['bn', 'gn', 'ln'], help='Normalization type')
    parser.add_argument('--c-norm', type=str, default='gn', choices=['bn', 'gn', 'ln'], help='C normalization type')
    parser.add_argument('--use-omega', type=str2bool, default=True, help='Use natural frequencies')
    parser.add_argument('--init-omg', type=float, default=1.0, help='Initial omega value')
    parser.add_argument('--global-omg', type=str2bool, default=True, help='Global omega parameter')
    parser.add_argument('--learn-omg', type=str2bool, default=True, help='Learn omega parameters')
    parser.add_argument('--ensemble', type=int, default=1, help='Ensemble size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    #parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs')

    # EMA arguments
    parser.add_argument('--ema-beta', type=float, default=0.99, help='EMA decay rate')
    parser.add_argument('--ema-update-every', type=int, default=10, help='EMA update frequency')
    parser.add_argument('--ema-update-after-step', type=int, default=200, help='Start EMA updates after this many steps')
    
    # Logging arguments
    parser.add_argument('--log-interval', type=int, default=100, help='Log every N batches')
    parser.add_argument('--eval-interval', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--save-interval', type=int, default=20, help='Save checkpoint every N epochs')
    
    # Experiment arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--experiment-name', type=str, default='akorn_cifar10', help='Experiment name')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save results. Defaults to [experiment_name]_[timestamp]')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--wandb-id', type=str, default=None, help='WandB run ID to resume') # この行を追加
    
    args = parser.parse_args()
    
    if args.save_dir is None:
        current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        args.save_dir = f"./results/{args.experiment_name}_{current_time}"

    # Get configuration and override with command line arguments
    config = get_config()
    
    # Override config with command line arguments
    config.update({
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'n': args.n,
        'ch': args.ch,
        'L': args.L,
        'T': args.T,
        'gamma': args.gamma,
        'J': args.J,
        'ksizes': args.ksizes,
        'ro_ksize': args.ro_ksize,
        'ro_N': args.ro_N,
        'norm': args.norm,
        'c_norm': args.c_norm,
        'use_omega': args.use_omega,
        'init_omg': args.init_omg,
        'global_omg': args.global_omg,
        'learn_omg': args.learn_omg,
        'ensemble': args.ensemble,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        #'warmup_epochs': args.warmup_epochs,
        'ema_beta': args.ema_beta,
        'ema_update_every': args.ema_update_every,
        'ema_update_after_step': args.ema_update_after_step,
        'log_interval': args.log_interval,
        'eval_interval': args.eval_interval,
        'save_interval': args.save_interval,
        'seed': args.seed,
        'experiment_name': args.experiment_name,
        'save_dir': args.save_dir,
    })
    
    # Set seed
    set_seed(config['seed'])
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(exist_ok=True)
    
    # Save parameters
    save_parameters(config, save_dir)
    
    # Initialize wandb
    # main関数の中
    if not args.no_wandb:
        # 再開用のIDを設定
        resume_id = args.wandb_id if args.resume else None

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
            name=f"{config['experiment_name']}_{int(time.time())}",
            id=resume_id, # この行を追加
            resume="allow" # この行を追加
        )
    
    # Print configuration
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, test_loader = create_data_loaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, device)
    
    # EMA setup
    ema = EMA(
        model,
        beta=config['ema_beta'],
        update_every=config['ema_update_every'],
        update_after_step=config['ema_update_after_step']
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1e6:.2f} MB (float32)")
    
    # Log model info to wandb
    if not args.no_wandb:
        wandb.log({
            'model/total_params': total_params,
            'model/trainable_params': trainable_params,
            'model/size_mb': total_params * 4 / 1e6
        })
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max=config['epochs'],
    #     eta_min=config['lr'] * 0.01
    # )

    start_epoch = 0 # 開始エポックを初期化

    # ▼▼▼ このブロックをここに追加 ▼▼▼
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # if 'scheduler_state_dict' in checkpoint: # 古いチェックポイントとの互換性のため
            #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'ema_test_loss': [],
        'ema_test_acc': [],
        'lr': []
    }
    
    best_acc = 0
    best_ema_acc = 0
    start_time = time.time()
    
    print(f"\nStarting training for {config['epochs']} epochs...")
    print("-" * 60)
    
    for epoch in range(start_epoch, config['epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, ema, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # Update learning rate
        # scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['lr'].append(current_lr)
        
        # Log to wandb
        if not args.no_wandb:
            wandb.log({
                'train/epoch_loss': train_loss,
                'train/epoch_acc': train_acc,
                'train/lr': current_lr,
                'epoch': epoch
            })
        
        # Evaluate
        if (epoch + 1) % config['eval_interval'] == 0:
            test_loss, test_acc, class_accs = evaluate(model, test_loader, criterion, device)
            ema_test_loss, ema_test_acc, ema_class_accs = evaluate(ema.ema_model, test_loader, criterion, device)

            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['ema_test_loss'].append(ema_test_loss)
            history['ema_test_acc'].append(ema_test_acc)
            
            print(f"\nEpoch {epoch+1}/{config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print(f"  EMA Test Loss: {ema_test_loss:.4f}, EMA Test Acc: {ema_test_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Log to wandb
            if not args.no_wandb:
                log_dict = {
                    'test/loss': test_loss,
                    'test/acc': test_acc,
                    'ema_test/loss': ema_test_loss,
                    'ema_test/acc': ema_test_acc,
                    'epoch': epoch
                }
                # Add per-class accuracies
                for class_name, acc in class_accs.items():
                    log_dict[f'test/class_acc/{class_name}'] = acc
                for class_name, acc in ema_class_accs.items():
                    log_dict[f'ema_test/class_acc/{class_name}'] = acc
                
                wandb.log(log_dict)
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_model_path = save_dir / f'best_model_acc_{best_acc:.2f}.pth'
                save_checkpoint_with_config(
                    model, optimizer, epoch, test_loss, config, best_model_path
                )
                
                # Log best accuracy to wandb
                if not args.no_wandb:
                    wandb.log({'best_acc': best_acc})

            if ema_test_acc > best_ema_acc:
                best_ema_acc = ema_test_acc
                best_ema_model_path = save_dir / f'best_ema_model_acc_{best_ema_acc:.2f}.pth'
                save_checkpoint_with_config(
                    ema.ema_model, optimizer, epoch, ema_test_loss, config, best_ema_model_path
                )

                if not args.no_wandb:
                    wandb.log({'best_ema_acc': best_ema_acc})
            
            print("-" * 60)
        
        # Save periodic checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint_with_config(
                model, optimizer, epoch, train_loss, config, checkpoint_path
            )
            ema_checkpoint_path = save_dir / f'ema_checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint_with_config(
                ema.ema_model, optimizer, epoch, train_loss, config, ema_checkpoint_path
            )
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/3600:.2f} hours")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Best EMA test accuracy: {best_ema_acc:.2f}%")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_test_loss, final_test_acc, final_class_accs = evaluate(model, test_loader, criterion, device)
    final_ema_test_loss, final_ema_test_acc, final_ema_class_accs = evaluate(ema.ema_model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"Final Test Loss: {final_test_loss:.4f}")
    print(f"Final EMA Test Accuracy: {final_ema_test_acc:.2f}%")
    print(f"Final EMA Test Loss: {final_ema_test_loss:.4f}")
    
    # Per-class accuracies
    print("\nPer-class Accuracies:")
    for class_name, acc in final_class_accs.items():
        print(f"  {class_name}: {acc:.2f}%")

    print("\nEMA Per-class Accuracies:")
    for class_name, acc in final_ema_class_accs.items():
        print(f"  {class_name}: {acc:.2f}%")
    
    # Save final model and results
    final_model_path = save_dir / 'akorn_cifar10_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'history': history,
        'final_accuracy': final_test_acc,
        'best_accuracy': best_acc,
        'class_accuracies': final_class_accs,
        'training_time': total_time
    }, final_model_path)
    
    final_ema_model_path = save_dir / 'akorn_cifar10_ema_final.pth'
    torch.save({
        'model_state_dict': ema.ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'history': history,
        'final_accuracy': final_ema_test_acc,
        'best_accuracy': best_ema_acc,
        'class_accuracies': final_ema_class_accs,
        'training_time': total_time
    }, final_ema_model_path)

    print(f"Final model saved to: {final_model_path}")
    print(f"Final EMA model saved to: {final_ema_model_path}")
    
    # Save training history
    import pandas as pd
    
    # Pad test metrics to match training length
    test_loss_padded = []
    test_acc_padded = []
    ema_test_loss_padded = []
    ema_test_acc_padded = []
    for i in range(len(history['train_loss'])):
        if i % config['eval_interval'] == config['eval_interval'] - 1 and i // config['eval_interval'] < len(history['test_loss']):
            test_loss_padded.append(history['test_loss'][i // config['eval_interval']])
            test_acc_padded.append(history['test_acc'][i // config['eval_interval']])
            ema_test_loss_padded.append(history['ema_test_loss'][i // config['eval_interval']])
            ema_test_acc_padded.append(history['ema_test_acc'][i // config['eval_interval']])
        else:
            test_loss_padded.append(None)
            test_acc_padded.append(None)
            ema_test_loss_padded.append(None)
            ema_test_acc_padded.append(None)

    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'test_loss': test_loss_padded,
        'test_acc': test_acc_padded,
        'ema_test_loss': ema_test_loss_padded,
        'ema_test_acc': ema_test_acc_padded,
        'learning_rate': history['lr']
    })

    history_path = save_dir / 'training_history.csv'
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")
    
    # Save final results summary
    results_summary = {
        'model': 'AKOrN',
        'dataset': 'CIFAR-10',
        'final_test_accuracy': final_test_acc,
        'best_test_accuracy': best_acc,
        'final_ema_test_accuracy': final_ema_test_acc,
        'best_ema_test_accuracy': best_ema_acc,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'training_time_hours': total_time / 3600,
        'epochs_trained': len(history['train_loss']),
        'class_accuracies': final_class_accs,
        'ema_class_accuracies': final_ema_class_accs,
        'config': config
    }
    
    results_summary_path = save_dir / 'results_summary.json'
    with open(results_summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
        
    print(f"Results summary saved to: {results_summary_path}")
    
    # Final wandb logging
    if not args.no_wandb:
        wandb.log({
            'final/test_acc': final_test_acc,
            'final/test_loss': final_test_loss,
            'final/ema_test_acc': final_ema_test_acc,
            'final/ema_test_loss': final_ema_test_loss,
            'final/training_time_hours': total_time / 3600
        })
        
        # Log final per-class accuracies
        for class_name, acc in final_class_accs.items():
            wandb.log({f'final/class_acc/{class_name}': acc})
        for class_name, acc in final_ema_class_accs.items():
            wandb.log({f'final/ema_class_acc/{class_name}': acc})
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Final EMA Test Accuracy: {final_ema_test_acc:.2f}%")
    print(f"Best EMA Test Accuracy: {best_ema_acc:.2f}%")
    print(f"Total Parameters: {total_params:,}")
    print(f"Training Time: {total_time/3600:.2f} hours")
    print("="*60)
    
    # Finish wandb run
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()