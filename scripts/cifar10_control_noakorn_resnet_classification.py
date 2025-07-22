#!/usr/bin/env python3
"""
CIFAR-10 Classification with Control version (no AKOrN iterations) using AKOrNResNet architecture but skipping Kuramoto dynamics.

This script is a control version of cifar10_akorn_resnet_classification.py that removes AKOrN iterations
but keeps the AKOrN readout structure for fair comparison. The Kuramoto oscillator dynamics (T iterations)
are bypassed, but the readout processing is preserved.

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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

import wandb
import yaml, copy
import itertools

from source.models.classification.my_knet import MyAKOrN, AKOrNResNet
from source.data.augs import augmentation_strong
from source.my_training_utils import (
    set_seed, count_parameters, save_checkpoint_with_config, save_parameters
)
from source.utils import str2bool

# Import layers we need for creating the control model
from source.layers.klayer import KLayer
from source.layers.common_layers import (
    RGBNormalize,
    ReadOutConv,
    Reshape,
    ResBlock,
    FF,
    BNReLUConv2d,
)
from torchvision.models.resnet import BasicBlock

class ControlKLayer(nn.Module):
    """
    Control version of KLayer that skips AKOrN iterations but keeps the structure for readout.
    This bypasses the Kuramoto oscillator dynamics (T iterations) but maintains the same
    input/output interface as the original KLayer.
    """
    
    def __init__(
        self,
        n,
        ch,
        J="conv",
        J_bias=False,
        c_norm="gn",
        use_omega=True,
        init_omg=1.0,
        ksize=3,
        gta=False,
        hw=None,
        global_omg=False,
        heads=8,
        learn_omg=True,
        apply_proj=True,
        bp_steps=None,
    ):
        super().__init__()
        assert (ch % n) == 0
        self.n = n
        self.ch = ch
        
        # Keep the same normalization as original for readout compatibility
        if c_norm == "gn":
            self.c_norm = nn.GroupNorm(ch // n, ch, affine=True)
        elif c_norm == "sandb":
            from source.layers.common_layers import ScaleAndBias
            self.c_norm = ScaleAndBias(ch, token_input=False)
        elif c_norm is None or c_norm == "none":
            self.c_norm = nn.Identity()
        else:
            raise NotImplementedError
            
        # Simple linear transformation to replace Kuramoto dynamics
        # This maintains the same input/output dimensions but removes oscillator iterations
        self.control_transform = nn.Conv2d(ch, ch, 1, 1, 0, bias=False)
        
    def forward(self, x: torch.Tensor, c: torch.Tensor, T: int, gamma):
        """
        Control forward pass: Skip AKOrN iterations, apply simple transformation.
        
        Returns the same structure as KLayer (xs, es) but without oscillator dynamics:
        - xs: List of states (here just the transformed input repeated T times)
        - es: List of energies (here just zeros)
        """
        xs, es = [], []
        c = self.c_norm(c)
        
        # Apply control transformation instead of Kuramoto iterations
        x_transformed = self.control_transform(x)
        
        # Return the same structure as original KLayer
        # Repeat the transformed state T times to match expected output structure
        for t in range(T):
            xs.append(x_transformed)
            es.append(torch.zeros(x_transformed.shape[0]).to(x_transformed.device))
            
        return xs, es

class ControlMyAKOrN(nn.Module):
    """
    Control version of MyAKOrN that uses ControlKLayer instead of KLayer.
    This removes AKOrN iterations but keeps the readout processing intact.
    """

    def __init__(
        self,
        n=4,
        ch=64,
        out_classes=10,
        L=3,
        T=3,
        J="conv",
        J_bias=False,
        ksizes=[9, 7, 5],
        ro_ksize=3,
        ro_N=2,
        norm="bn",
        c_norm="gn",
        gamma=1.0,
        use_omega=True,
        init_omg=1.0,
        global_omg=True,
        learn_omg=True,
        ensemble=1,
        bp_steps=None,
    ):
        super().__init__()
        
        self.fixed_noise = False
        self.n = n
        self.L = L
        self.ensemble = ensemble
        self.gamma = nn.Parameter(torch.tensor([gamma]), requires_grad=False)
        
        # Expand parameters to match number of layers
        self.ns = self._expand_param(n, L)
        self.T = self._expand_param(T, L)
        self.bp_steps = self._expand_param(bp_steps,L)
        J = self._expand_param(J, L)
        ksizes = self._expand_param(ksizes, L)
        ro_N = self._expand_param(ro_N, L)
                
        # Calculate layer dimensions
        strides = [2, 2, 2] + [1] * (L - 3)
        channels = [ch * (2**i) for i in range(L)]
        hw_sizes = [[32 // (2**i), 32 // (2**i)] if i < 3 else [8, 8] for i in range(L)]
        
        # Build network components
        self.rgb_normalize = RGBNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.conv0 = self._create_initial_conv(channels[0])
        self.layers = self._create_control_layers(L, channels, strides, hw_sizes, J, J_bias, ksizes, ro_N, ro_ksize, norm, c_norm, use_omega, init_omg, global_omg, learn_omg)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), Reshape(-1, channels[-1]))
        self.out = nn.Linear(channels[-1], out_classes)
    
    def _expand_param(self, param, length):
        """Expand parameter to match the number of layers."""
        if isinstance(param, (list, tuple)):
            return list(param) if len(param) == length else [param[0]] * length
        return [param] * length
    
    def _create_initial_conv(self, out_channels):
        """Create the initial convolutional layer."""
        return nn.Conv2d(
            3, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1
        )
    
    def _create_strided_conv(self, in_ch, out_ch, stride, kernel_size, padding):
        """Create a strided convolution layer."""
        return nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=1)
    
    def _create_readout_block(self, channels, ro_N, ro_ksize, norm):
        """Create a readout block - KEPT UNCHANGED for fair comparison."""
        padding = ro_ksize // 2
        return nn.Sequential(
            ReadOutConv(
                inch=channels,
                outch=channels,
                ro_N=ro_N,
                kernel_size=ro_ksize,
                stride=1,
                padding=padding,
            ),
            ResBlock(
                FF(
                    inch=channels,
                    outch=channels,
                    hidch=channels,
                    kernel_size=ro_ksize,
                    stride=1,
                    padding=padding,
                    norm=norm,
                )
            ),
            BNReLUConv2d(
                inch=channels,
                outch=channels,
                kernel_size=ro_ksize,
                stride=1,
                padding=padding,
                norm=norm,
            )
        )
    
    def _create_control_layers(self, L, channels, strides, hw_sizes, J, J_bias, ksizes, ro_N, ro_ksize, norm, c_norm, use_omega, init_omg, global_omg, learn_omg):
        """Create all network layers using ControlKLayer instead of KLayer."""
        layers = nn.ModuleList()
        padding = ro_ksize // 2
        
        for l in range(L):
            # Create transition layers
            if l == 0:
                transition_layers = nn.ModuleList([nn.Identity(), nn.Identity()])
            else:
                conv_layer = self._create_strided_conv(
                    channels[l-1], channels[l], strides[l], ro_ksize, padding
                )
                transition_layers = nn.ModuleList([conv_layer, conv_layer])
            
            # Create Control K-layer (NO AKORN ITERATIONS)
            control_k_layer = ControlKLayer(
                n=self.ns[l],
                ch=channels[l],
                J=J[l],
                J_bias=J_bias,
                c_norm=c_norm,
                use_omega=use_omega,
                init_omg=init_omg,
                global_omg=global_omg,
                learn_omg=learn_omg,
                ksize=ksizes[l],
                hw=hw_sizes[l],
                bp_steps=self.bp_steps[l],
            )
            
            # Create readout block - KEPT UNCHANGED
            readout_block = self._create_readout_block(channels[l], ro_N[l], ro_ksize, norm)
            
            layers.append(nn.ModuleList([
                transition_layers,
                nn.Identity(),
                control_k_layer,  # Using ControlKLayer instead of KLayer
                readout_block,
                nn.Identity()
            ]))
        
        return layers
    
    def feature(self, inp):
        """Extract features from input through the network layers."""
        # Initial processing
        c = self.conv0(self.rgb_normalize(inp))
        x = torch.randn_like(c)
        xs, es = [], []

        # Process through each layer (same structure as original)
        for l, (transition_layer, _, control_k_layer, readout_layer, _) in enumerate(self.layers):
            x, c = transition_layer[0](x), transition_layer[1](c)
            layer_xs, layer_es = control_k_layer(x, c, self.T[l], self.gamma)  # ControlKLayer call
            xs.append(layer_xs)
            es.append(layer_es)
            x = layer_xs[-1]
            c = readout_layer(x)  # AKORN READOUT PRESERVED
            
        # Final pooling
        x, c = map(self.pool, (x, c))
        return c, x, xs, es

    def forward(self, inp, return_xs=False, return_es=False):
        """Forward pass through the network."""
        if self.ensemble == 1:
            c, x, xs, es = self.feature(inp)
            logits = self.out(c)
            
            # Build return value based on flags
            result = [logits]
            if return_xs:
                result.append(xs)
            if return_es:
                result.append(es)
            
            return result[0] if len(result) == 1 else result
        else:
            # Ensemble prediction
            assert not return_xs and not return_es, "Ensemble mode doesn't support returning xs/es"
            logits_list = []
            for _ in range(self.ensemble):
                features, _, _, _ = self.feature(inp)
                logits = self.out(features)
                logits_list.append(logits)
            return torch.stack(logits_list).mean(0)


class ControlAKOrNResNet(nn.Module):
    """
    Control version of AKOrNResNet that uses ControlMyAKOrN instead of MyAKOrN.
    This removes AKOrN iterations but preserves the readout and ResNet components.
    """

    def __init__(
            self,
            n=2,
            ch=128,
            T=15,
            J="conv",
            gamma=0.01,
            ksizes=3,
            L=1,
            out_classes=10,
            transform_to_theta=False,
            bp_steps=3,
            ):
        
        super().__init__()
        # Use ControlMyAKOrN instead of MyAKOrN
        self.kur1 = ControlMyAKOrN(
            n=n,
            ch=ch,
            L=L,
            J=J,
            T=T,
            ksizes=ksizes,
            gamma=gamma,
            use_omega=True,
            init_omg=1.0,
            global_omg=False,
            learn_omg=True,
            out_classes=out_classes,
            bp_steps=bp_steps,
        )
        self.n  = n
        self.ch = ch
        self.transform_to_theta = transform_to_theta
        
        # For debugging
        self.c, self.x, self.xs, self.es = None, None, None, None
        
        def make_layer(in_ch, out_ch, blocks, stride):
            # ① 先頭ブロック用の downsample を準備
            downsample = None
            if stride != 1 or in_ch != out_ch:
                downsample = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

            # ② ブロックを積む
            layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
            for _ in range(1, blocks):
                layers.append(BasicBlock(out_ch, out_ch))  # 2 枚目以降 stride=1
            return nn.Sequential(*layers)

        if self.transform_to_theta and self.n == 2:
            dim = ch // n
        else:
            dim = ch

        # ResNet layers - KEPT UNCHANGED
        self.layer1 = make_layer(dim, 2*dim, 2, 2)
        self.layer2 = make_layer(2*dim, 4*dim, 2, 2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(4*dim, out_classes)

    def forward(self, inp):
        _, _, xs, es = self.kur1.feature(inp)  # ControlMyAKOrN call
        self.xs = xs
        self.es = es

        self.x = xs[-1][-1]
        self.c = self.kur1.layers[-1][3](self.x)  # AKORN READOUT PRESERVED
        
        if self.transform_to_theta and self.n==2:
            y = torch.atan2(self.c[:,1::2,:,:], self.c[:,0::2,:,:])
        else:
            y = self.c
        
        # ResNet processing - KEPT UNCHANGED
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.pool(y).flatten(1)
        return self.fc(y)


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
        'L': 1,              # Number of layers
        'T': 3,              # Number of time steps per layer (NO EFFECT in control)
        'gamma': 1.0,        # Integration step size (NO EFFECT in control)
        'J': 'conv',         # Connectivity type ('conv', 'attn' or 'conv_repeated_const')
        'J_bias': False,     # Connectivity bias turned off! by SK on Jul 4, 2025
        'ksizes': 3, # Kernel sizes for each layer
        'ro_ksize': 3,       # Readout kernel size
        'ro_N': 2,           # Readout N parameter
        'norm': 'bn',        # Normalization type
        'c_norm': 'gn',      # C normalization type
        'use_omega': True,   # Use natural frequencies (NO EFFECT in control)
        'init_omg': 1.0,     # Initial omega value (NO EFFECT in control)
        'global_omg': False,  # Global omega parameter (NO EFFECT in control)
        'learn_omg': True,   # Learn omega parameters (NO EFFECT in control)
        'ensemble': 1,       # Ensemble size
        'bp_steps': None,    # Steps to apply BP for each layer (NO EFFECT in control)
        
        # Training
        'epochs': 100,
        'lr': 1e-4,
        'weight_decay': 0.0,
        #'warmup_epochs': 5,

        # Logging
        'log_interval': 100,
        'eval_interval': 5,
        'save_interval': 20,
        
        # Experiment
        'seed': 42,
        'experiment_name': 'control_noakorn_resnet_cifar10',  # Changed experiment name
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
    """Create Control AKOrNResNet model (no AKOrN iterations)"""
    model = ControlAKOrNResNet(  # Using ControlAKOrNResNet
        n=config['n'],
        ch=config['ch'],
        out_classes=config['num_classes'],
        J=config['J'],
        L=config['L'],
        T=config['T'],  # This parameter has no effect in control version
        ksizes=config['ksizes'],
        gamma=config['gamma'],  # This parameter has no effect in control version
        bp_steps=config['bp_steps'],  # This parameter has no effect in control version
    ).to(device)
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
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

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='CIFAR-10 Control (No AKOrN Iterations) Classification')
    
    # Wandb arguments
    parser.add_argument('--wandb-project', default='control_noakorn_resnet_cifar10', help='W&B project name')  # Changed default
    parser.add_argument('--wandb-entity', default=None, help='W&B entity name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    
    # Data arguments
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    # Model architecture arguments
    parser.add_argument('--n', type=int, default=2, help='Oscillator dimension (2D for complex oscillators)')
    parser.add_argument('--ch', type=int, default=16, help='Base number of channels')
    parser.add_argument('--L', type=int, default=1, help='Number of layers')
    parser.add_argument('--T', type=int, default=15, help='Number of time steps per layer (NO EFFECT in control)')
    parser.add_argument('--gamma', type=float, default=.01, help='Integration step size (NO EFFECT in control)')
    parser.add_argument('--J', type=str, default='conv', choices=['conv', 'attn', 'conv_repeated_const'], help='Connectivity type')
    parser.add_argument('--J_bias', type=str2bool, default=False, help='Bias of connection convolutions, no bias as default')
    parser.add_argument('--ksizes', type=int, nargs='+', default=3, help='Kernel sizes for each layer')
    parser.add_argument('--ro-ksize', type=int, default=3, help='Readout kernel size')
    parser.add_argument('--ro-N', type=int, default=2, help='Readout N parameter')
    parser.add_argument('--norm', type=str, default='bn', choices=['bn', 'gn', 'ln'], help='Normalization type')
    parser.add_argument('--c-norm', type=str, default='gn', choices=['bn', 'gn', 'ln'], help='C normalization type')
    parser.add_argument('--use-omega', type=str2bool, default=True, help='Use natural frequencies (NO EFFECT in control)')
    parser.add_argument('--init-omg', type=float, default=1.0, help='Initial omega value (NO EFFECT in control)')
    parser.add_argument('--global-omg', type=str2bool, default=True, help='Global omega parameter (NO EFFECT in control)')
    parser.add_argument('--learn-omg', type=str2bool, default=True, help='Learn omega parameters (NO EFFECT in control)')
    parser.add_argument('--ensemble', type=int, default=1, help='Ensemble size')
    parser.add_argument('--bp_steps', type=int, default=None, help='Steps of back propagations in each layer (NO EFFECT in control)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    #parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs')
    
    # Logging arguments
    parser.add_argument('--log-interval', type=int, default=100, help='Log every N batches')
    parser.add_argument('--eval-interval', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--save-interval', type=int, default=20, help='Save checkpoint every N epochs')
    
    # Experiment arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--experiment-name', type=str, default='control_noakorn_resnet_cifar10', help='Experiment name')  # Changed default
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save results. Defaults to [experiment_name]_[timestamp]')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--wandb-id', type=str, default=None, help='WandB run ID to resume')
    parser.add_argument("--run-name", type=str, default=None, help="If set, use this as wandb run name")
    parser.add_argument('--param-file', type=str, default=None,
                     help='YAML file listing hyper-param sets')
    parser.add_argument('--index', type=int, default=None,
                        help='Which entry of YAML to load (0-based)')

    
    args = parser.parse_args()
    
    # ── ★ Grid 展開ロジック ────────────────────────
    if args.param_file is not None and args.index is not None:
        with open(args.param_file) as f:
            cfg = yaml.safe_load(f)['grid']
        keys, lists = zip(*[(k, v if isinstance(v, list) else [v]) for k, v in cfg.items()])
        combos = [dict(zip(keys, values)) for values in itertools.product(*lists)]
        sel = combos[args.index]                      # PBS_ARRAYID 番目を選択
        for k, v in sel.items(): setattr(args, k, v)  # argparse を上書き

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
        'T': args.T,  # NO EFFECT in control version
        'gamma': args.gamma,  # NO EFFECT in control version
        'J': args.J,
        'J_bias': args.J_bias,
        'ksizes': args.ksizes,
        'ro_ksize': args.ro_ksize,
        'ro_N': args.ro_N,
        'norm': args.norm,
        'c_norm': args.c_norm,
        'use_omega': args.use_omega,  # NO EFFECT in control version
        'init_omg': args.init_omg,  # NO EFFECT in control version
        'global_omg': args.global_omg,  # NO EFFECT in control version
        'learn_omg': args.learn_omg,  # NO EFFECT in control version
        'ensemble': args.ensemble,
        'bp_steps': args.bp_steps,  # NO EFFECT in control version
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
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

        job_id = os.environ.get("PBS_JOBID", "local")
        if args.run_name is not None:
            run_name = args.run_name.format(**vars(args), job_id=job_id)
        else:
            run_name = f"{config['experiment_name']}_job{job_id}"

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
            name=run_name,
            id=resume_id,
            resume="allow"
        )
    
    # Print configuration
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\\n*** CONTROL VERSION: AKOrN iterations are DISABLED ***")
    print(f"*** AKOrN readout processing is PRESERVED ***\\n")
    
    # Create data loaders
    print("\\nCreating data loaders...")
    train_loader, test_loader = create_data_loaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\\nCreating control model...")
    model = create_model(config, device)
    
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
        wandb.watch(model, log='all', log_freq=config['log_interval'])
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0        # ← 何 epoch たっても lr を変えない
    )

    start_epoch = 0 # 開始エポックを初期化

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint: # 古いチェックポイントとの互換性のため
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
        'lr': []
    }
    
    best_acc = 0
    start_time = time.time()
    
    print(f"\\nStarting training for {config['epochs']} epochs...")
    print("-" * 60)
    
    for epoch in range(start_epoch, config['epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # Update learning rate
        scheduler.step()
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
            
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            print(f"\\nEpoch {epoch+1}/{config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Log to wandb
            if not args.no_wandb:
                log_dict = {
                    'test/loss': test_loss,
                    'test/acc': test_acc,
                    'epoch': epoch
                }
                # Add per-class accuracies
                for class_name, acc in class_accs.items():
                    log_dict[f'test/class_acc/{class_name}'] = acc
                
                wandb.log(log_dict)
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_model_path = save_dir / f'best_model_acc_{best_acc:.2f}.pth'
                save_checkpoint_with_config(
                    model, optimizer, scheduler, epoch, test_loss, config, best_model_path
                )
                
                # Log best accuracy to wandb
                if not args.no_wandb:
                    wandb.log({'best_acc': best_acc})
            
            print("-" * 60)
        
        # Save periodic checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint_with_config(
                model, optimizer, scheduler, epoch, train_loss, config, checkpoint_path
            )
    
    total_time = time.time() - start_time
    print(f"\\nTraining completed in {total_time/3600:.2f} hours")
    print(f"Best test accuracy: {best_acc:.2f}%")
    
    # Final evaluation
    print("\\nFinal Evaluation:")
    final_test_loss, final_test_acc, final_class_accs = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"Final Test Loss: {final_test_loss:.4f}")
    
    # Per-class accuracies
    print("\\nPer-class Accuracies:")
    for class_name, acc in final_class_accs.items():
        print(f"  {class_name}: {acc:.2f}%")
    
    # Save final model and results
    final_model_path = save_dir / 'control_noakorn_resnet_cifar10_final.pth'  # Changed filename
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
    
    print(f"Final model saved to: {final_model_path}")
    
    # Save training history
    import pandas as pd
    
    # Pad test metrics to match training length
    test_loss_padded = []
    test_acc_padded = []
    for i in range(len(history['train_loss'])):
        if i % config['eval_interval'] == config['eval_interval'] - 1 and i // config['eval_interval'] < len(history['test_loss']):
            test_loss_padded.append(history['test_loss'][i // config['eval_interval']])
            test_acc_padded.append(history['test_acc'][i // config['eval_interval']])
        else:
            test_loss_padded.append(None)
            test_acc_padded.append(None)

    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'test_loss': test_loss_padded,
        'test_acc': test_acc_padded,
        'learning_rate': history['lr']
    })

    history_path = save_dir / 'training_history.csv'
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")
    
    # Save final results summary
    results_summary = {
        'model': 'Control (No AKOrN Iterations) + ResNet',  # Changed model name
        'dataset': 'CIFAR-10',
        'final_test_accuracy': final_test_acc,
        'best_test_accuracy': best_acc,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'training_time_hours': total_time / 3600,
        'epochs_trained': len(history['train_loss']),
        'class_accuracies': final_class_accs,
        'config': config,
        'control_note': 'AKOrN iterations removed, readout preserved'  # Added note
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
            'final/training_time_hours': total_time / 3600
        })
        
        # Log final per-class accuracies
        for class_name, acc in final_class_accs.items():
            wandb.log({f'final/class_acc/{class_name}': acc})
    
    print("\\n" + "="*60)
    print("CONTROL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Total Parameters: {total_params:,}")
    print(f"Training Time: {total_time/3600:.2f} hours")
    print("="*60)
    
    # Finish wandb run
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()