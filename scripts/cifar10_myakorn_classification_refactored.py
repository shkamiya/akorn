#!/usr/bin/env python3
"""
CIFAR-10 Classification with AKOrN (Artificial Kuramoto Oscillator Networks)
using MyAKOrN - REFACTORED VERSION.

This script demonstrates how to use the new utility modules for training.
This is a refactored version of the original cifar10_myakorn_classification.py
showing how to use the new utility modules.

References:
- Miyato et al., "Artificial Kuramoto Oscillatory Neurons", ICLR 2025
"""

import os
import sys
import time
import argparse
import datetime
import yaml
import itertools

# Add source directory to path
sys.path.append('source')

# Import utility modules
from source.training import (
    set_seed, get_default_config, train_epoch, 
    save_checkpoint_with_config, save_config, print_config,
    setup_device, print_model_info, create_save_directory,
    save_final_model, save_training_history, save_results_summary,
    print_final_results, initialize_wandb, create_myakorn_model,
    create_optimizer, create_criterion, get_scheduler, 
    comprehensive_evaluation, count_parameters, load_yaml_config,
    update_config_from_args, add_common_arguments
)
from source.data.cifar10_utils import (
    create_cifar10_dataloaders, get_cifar10_classes, print_dataset_info
)
from source.utils import str2bool


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='CIFAR-10 AKOrN Classification using MyAKOrN - Refactored')
    
    # Add common arguments using utility function
    parser = add_common_arguments(parser)
    
    # Add MyAKOrN specific arguments
    parser.add_argument('--J_bias', type=str2bool, default=False, 
                       help='Bias of connection convolutions')
    
    args = parser.parse_args()
    
    # Handle grid search configuration
    if args.param_file is not None and args.index is not None:
        grid_config = load_yaml_config(args.param_file, args.index)
        for key, value in grid_config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Get configuration and update with arguments
    config = get_default_config()
    config = update_config_from_args(config, args)
    
    # Add specific config for MyAKOrN
    config['J_bias'] = args.J_bias
    
    # Set seed
    set_seed(config['seed'])
    
    # Setup device
    device = setup_device()
    
    # Create save directory and save config
    save_dir = create_save_directory(config)
    save_config(config, save_dir)
    print_config(config)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, test_loader = create_cifar10_dataloaders(config)
    print_dataset_info(train_loader, test_loader)
    
    # Create model
    print("\nCreating model...")
    model = create_myakorn_model(config, device)
    print_model_info(model, "MyAKOrN")
    
    # Create optimizer, criterion, and scheduler
    optimizer = create_optimizer(model, config)
    criterion = create_criterion(config)
    scheduler = get_scheduler(optimizer, config)
    
    # Initialize wandb if enabled
    if not args.no_wandb:
        run_name = f"{config['experiment_name']}_T{config['T']}_gamma{config['gamma']}"
        initialize_wandb(config, args.wandb_project, args.wandb_entity, 
                        run_name=run_name, resume_id=args.wandb_id)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        from source.training.checkpoint_utils import load_checkpoint
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, device)
    
    # Training setup
    history = {
        'train_loss': [],
        'train_acc': [], 
        'test_loss': [],
        'test_acc': [],
        'lr': []
    }
    
    best_acc = 0
    start_time = time.time()
    
    print(f"\nStarting training for {config['epochs']} epochs...")
    print("-" * 60)
    
    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config,
            log_wandb=not args.no_wandb
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['lr'].append(current_lr)
        
        # Log training to wandb
        if not args.no_wandb:
            import wandb
            wandb.log({
                'train/epoch_loss': train_loss,
                'train/epoch_acc': train_acc,
                'train/lr': current_lr,
                'epoch': epoch
            })
        
        # Evaluate
        if (epoch + 1) % config['eval_interval'] == 0:
            print(f"\nEvaluating at epoch {epoch + 1}...")
            results = comprehensive_evaluation(
                model, test_loader, criterion, device, epoch,
                class_names=get_cifar10_classes(),
                log_wandb=not args.no_wandb
            )
            
            # Update history
            history['test_loss'].append(results['test_loss'])
            history['test_acc'].append(results['test_acc'])
            
            print(f"Epoch {epoch+1}/{config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {results['test_loss']:.4f}, Test Acc: {results['test_acc']:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if results['test_acc'] > best_acc:
                best_acc = results['test_acc']
                best_model_path = save_dir / f'best_model_acc_{best_acc:.2f}.pth'
                save_checkpoint_with_config(
                    model, optimizer, scheduler, epoch, 
                    results['test_loss'], config, best_model_path
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
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_results = comprehensive_evaluation(
        model, test_loader, criterion, device, config['epochs'],
        class_names=get_cifar10_classes(),
        log_wandb=not args.no_wandb
    )
    
    # Get parameter info
    param_info = count_parameters(model)
    
    # Save final model and results
    save_final_model(
        model, optimizer, config, history,
        final_results['test_acc'], best_acc,
        final_results['class_accuracies'], total_time,
        save_dir, 'my_akorn_cifar10_final.pth'
    )
    
    # Save training history
    save_training_history(history, save_dir, config)
    
    # Save results summary
    save_results_summary(
        'MyAKOrN', final_results['test_acc'], best_acc,
        param_info['total_params'], param_info['trainable_params'],
        total_time, len(history['train_loss']),
        final_results['class_accuracies'], config, save_dir
    )
    
    # Print final results
    print_final_results(
        'MyAKOrN', final_results['test_acc'], best_acc,
        param_info['total_params'], total_time,
        final_results['class_accuracies']
    )
    
    # Final wandb logging
    if not args.no_wandb:
        import wandb
        wandb.log({
            'final/test_acc': final_results['test_acc'],
            'final/test_loss': final_results['test_loss'],
            'final/training_time_hours': total_time / 3600
        })
        
        # Log final per-class accuracies
        for class_name, acc in final_results['class_accuracies'].items():
            wandb.log({f'final/class_acc/{class_name}': acc})
        
        wandb.finish()


if __name__ == '__main__':
    main()