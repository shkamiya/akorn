# Classification Scripts Refactoring Guide

This guide explains how to refactor the existing classification scripts to use the new utility modules.

## New Utility Modules Created

### 1. `source/training/` - Training utilities package
- `training_utils.py` - Core training functions
- `config_utils.py` - Configuration management
- `checkpoint_utils.py` - Checkpoint saving/loading
- `evaluation_utils.py` - Model evaluation functions
- `model_utils.py` - Model creation and setup

### 2. `source/data/cifar10_utils.py` - CIFAR-10 data utilities

## How to Refactor Each Script

### 1. For `scripts/cifar10_myakorn_classification.py`

**Original imports (remove these):**
```python
# Remove these functions that are now in utilities
def set_seed(seed=42):
def get_config():
def create_data_loaders(config):
def create_model(config, device):
def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
def evaluate(model, test_loader, criterion, device):
def save_checkpoint_with_config(model, optimizer, scheduler, epoch, loss, config, filename):
def save_parameters(config, save_dir):
```

**New imports (add these):**
```python
# Add these imports at the top
from source.training import (
    set_seed, get_default_config, train_epoch, evaluate_model,
    save_checkpoint_with_config, save_config, print_config,
    setup_device, print_model_info, create_save_directory,
    save_final_model, save_training_history, save_results_summary,
    print_final_results, initialize_wandb, create_myakorn_model,
    create_optimizer, create_criterion, get_scheduler, comprehensive_evaluation
)
from source.data.cifar10_utils import create_cifar10_dataloaders, get_cifar10_classes, print_dataset_info
```

**Modified main function:**
```python
def main():
    # Parse arguments (keep existing argument parsing)
    args = parser.parse_args()
    
    # Get configuration
    config = get_default_config()
    config = update_config_from_args(config, args)
    
    # Set seed
    set_seed(config['seed'])
    
    # Setup device
    device = setup_device()
    
    # Create save directory
    save_dir = create_save_directory(config)
    save_config(config, save_dir)
    print_config(config)
    
    # Create data loaders
    train_loader, test_loader = create_cifar10_dataloaders(config)
    print_dataset_info(train_loader, test_loader)
    
    # Create model
    model = create_myakorn_model(config, device)
    print_model_info(model, "MyAKOrN")
    
    # Create optimizer and criterion
    optimizer = create_optimizer(model, config)
    criterion = create_criterion(config)
    scheduler = get_scheduler(optimizer, config)
    
    # Initialize wandb
    if not args.no_wandb:
        initialize_wandb(config, args.wandb_project, args.wandb_entity, 
                        resume_id=args.wandb_id)
    
    # Training loop (simplified)
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'lr': []}
    best_acc = 0
    
    for epoch in range(config['epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # Evaluate
        if (epoch + 1) % config['eval_interval'] == 0:
            results = comprehensive_evaluation(
                model, test_loader, criterion, device, epoch, 
                class_names=get_cifar10_classes()
            )
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(results['test_loss'])
            history['test_acc'].append(results['test_acc'])
            
            # Save best model
            if results['test_acc'] > best_acc:
                best_acc = results['test_acc']
                save_checkpoint_with_config(
                    model, optimizer, scheduler, epoch, 
                    results['test_loss'], config, save_dir / 'best_model.pth'
                )
    
    # Save final results
    param_info = count_parameters(model)
    save_final_model(model, optimizer, config, history, 
                    results['test_acc'], best_acc, 
                    results['class_accuracies'], total_time, 
                    save_dir, 'my_akorn_cifar10_final.pth')
    
    save_training_history(history, save_dir, config)
    save_results_summary('MyAKOrN', results['test_acc'], best_acc,
                        param_info['total_params'], param_info['trainable_params'],
                        total_time, len(history['train_loss']), 
                        results['class_accuracies'], config, save_dir)
    
    print_final_results('MyAKOrN', results['test_acc'], best_acc,
                       param_info['total_params'], total_time, 
                       results['class_accuracies'])
```

### 2. For `scripts/cifar10_akorn_classification.py`

**Changes:**
- Replace `create_model()` with `create_akorn_model()`
- Same pattern as above but use `create_akorn_model` instead of `create_myakorn_model`
- Update final model filename to `'akorn_cifar10_final.pth'`

### 3. For `scripts/cifar10_akorn_resnet_classification.py`

**Changes:**
- Replace `create_model()` with `create_akorn_resnet_model()`
- Same pattern as above but use `create_akorn_resnet_model` instead of `create_myakorn_model`
- Update final model filename to `'akorn_resnet_cifar10_final.pth'`

### 4. For `scripts/train_classification.py`

**Changes:**
- Replace `evaluate_model()` with `evaluate_with_adversarial_attacks()` for adversarial evaluation
- Replace `train_epoch()` with the utility version
- Use `create_akorn_model()` for model creation
- Use `setup_ema()` for EMA setup

**Modified imports:**
```python
from source.training import (
    train_epoch, evaluate_with_adversarial_attacks, create_akorn_model,
    create_optimizer, create_criterion, setup_ema, count_parameters
)
from source.data.cifar10_utils import create_cifar10_dataloaders
```

## Benefits of This Refactoring

1. **Code Reusability**: Common functions are now reusable across all scripts
2. **Consistency**: All scripts use the same evaluation and training logic
3. **Maintainability**: Bug fixes and improvements only need to be made once
4. **Modularity**: Each utility module has a specific purpose
5. **Extensibility**: Easy to add new model types or evaluation methods

## Testing the Refactored Code

After refactoring, test each script to ensure:
1. Models train correctly
2. Evaluation metrics are computed properly
3. Checkpoints are saved and loaded correctly
4. WandB logging works as expected
5. All configuration options are preserved

## Migration Steps

1. **Create the utility modules** (already done)
2. **Update imports** in each script
3. **Replace function calls** with utility functions
4. **Remove duplicated code** from original scripts
5. **Test thoroughly** to ensure functionality is preserved
6. **Update documentation** and comments as needed

## Example Usage

```python
# Simple example using the new utilities
from source.training import *
from source.data.cifar10_utils import *

# Setup
config = get_default_config()
device = setup_device()
set_seed(config['seed'])

# Data and model
train_loader, test_loader = create_cifar10_dataloaders(config)
model = create_myakorn_model(config, device)
optimizer = create_optimizer(model, config)
criterion = create_criterion(config)

# Training
for epoch in range(config['epochs']):
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device, epoch, config
    )
    
    if epoch % config['eval_interval'] == 0:
        results = comprehensive_evaluation(
            model, test_loader, criterion, device, epoch
        )
```

This refactoring significantly reduces code duplication while maintaining all existing functionality and making the codebase more maintainable and extensible.