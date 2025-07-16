"""
Training utilities package for AKOrN classification models.
This package provides common functionality for training, evaluation, and configuration management.
"""

from .training_utils import (
    set_seed,
    count_parameters,
    train_epoch,
    print_model_info,
    setup_device,
    get_scheduler,
    initialize_wandb
)

from .config_utils import (
    get_default_config,
    load_yaml_config,
    update_config_from_args,
    create_save_directory,
    save_config,
    print_config,
    add_common_arguments
)

from .checkpoint_utils import (
    save_checkpoint_with_config,
    load_checkpoint,
    save_final_model,
    save_training_history,
    save_results_summary,
    print_final_results
)

from .evaluation_utils import (
    evaluate_model,
    log_evaluation_results,
    print_class_accuracies,
    evaluate_with_adversarial_attacks,
    comprehensive_evaluation
)

from .model_utils import (
    create_myakorn_model,
    create_akorn_model,
    create_akorn_resnet_model,
    create_model,
    create_optimizer,
    create_criterion,
    setup_ema,
    load_pretrained_weights
)

__all__ = [
    # Training utilities
    'set_seed',
    'count_parameters', 
    'train_epoch',
    'print_model_info',
    'setup_device',
    'get_scheduler',
    'initialize_wandb',
    
    # Config utilities
    'get_default_config',
    'load_yaml_config',
    'update_config_from_args',
    'create_save_directory',
    'save_config',
    'print_config',
    'add_common_arguments',
    
    # Checkpoint utilities
    'save_checkpoint_with_config',
    'load_checkpoint',
    'save_final_model',
    'save_training_history',
    'save_results_summary',
    'print_final_results',
    
    # Evaluation utilities
    'evaluate_model',
    'log_evaluation_results',
    'print_class_accuracies',
    'evaluate_with_adversarial_attacks',
    'comprehensive_evaluation',
    
    # Model utilities
    'create_myakorn_model',
    'create_akorn_model',
    'create_akorn_resnet_model',
    'create_model',
    'create_optimizer',
    'create_criterion',
    'setup_ema',
    'load_pretrained_weights'
]