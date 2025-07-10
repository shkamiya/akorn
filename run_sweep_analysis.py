#!/usr/bin/env python3
"""
Run sweep connectivity analysis using existing notebook setup.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Set working directory
os.chdir('/Users/shunsukekamiya/Desktop/1_Lab/_6_asymmetry/akorn')

# Setup like in the notebook
sys.path.append('source')
from models.classification.my_knet import MyAKOrN
from models.classification.analysis_utils import AKOrNStaticAnalyzer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define sweep directories
sweep_dirs = [
    "sweep_20250708_581384.opbs_0",
    "sweep_20250708_581385.opbs_1", 
    "sweep_20250708_581386.opbs_2",
    "sweep_20250708_581387.opbs_3",
    "sweep_20250708_581388.opbs_4",
    "sweep_20250708_581389.opbs_5",
    "sweep_20250708_581390.opbs_6",
    "sweep_20250708_581391.opbs_7",
    "sweep_20250708_581392.opbs_8",
    "sweep_20250708_581393.opbs_9",
    "sweep_20250708_581394.opbs_10",
    "sweep_20250708_581395.opbs_11",
    "sweep_20250708_581396.opbs_12",
    "sweep_20250708_581397.opbs_13",
    "sweep_20250708_581398.opbs_14",
]

def load_and_analyze_sweep(sweep_dir, layer_idx=0):
    """Load and analyze a single sweep configuration."""
    results_dir = Path("results")
    
    # Load config
    config_path = results_dir / sweep_dir / "parameters.json"
    if not config_path.exists():
        print(f"Config not found for {sweep_dir}")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    gamma = config.get('gamma', 'unknown')
    T = config.get('T', 'unknown')
    print(f"Processing {sweep_dir}: gamma={gamma}, T={T}")
    
    # Check if model exists
    model_path = results_dir / sweep_dir / "my_akorn_cifar10_final.pth"
    if not model_path.exists():
        print(f"  Model not found for {sweep_dir}")
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
        
        # Create analyzer
        analyzer = AKOrNStaticAnalyzer(model, device)
        
        # Perform analysis
        analyzer.analyze_full_connectivity(layer_idx)
        
        # Create output directory
        output_dir = Path(f"results/sweep_connectivity_analysis/gamma{gamma}_T{T}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plots without showing them
        plt.ioff()  # Turn off interactive mode
        
        # Generate and save plots
        plot_functions = [
            ("omega_distributions", analyzer.plot_omega_distributions),
            ("connectivity_statistics", lambda: analyzer.plot_connectivity_statistics(layer_idx)),
            ("frob_norms_kernel", lambda: analyzer.plot_Frob_norms_summary(layer_idx, 'kernel')),
            ("frob_norms_channels", lambda: analyzer.plot_Frob_norms_summary(layer_idx, 'channels')),
            ("decomposition_symmetric_skew", lambda: analyzer.plot_decomposition_results(layer_idx, 'symmetric_skew')),
            ("decomposition_rotation_symmetric", lambda: analyzer.plot_decomposition_results(layer_idx, 'rotation_symmetric')),
            ("torus_visualization", lambda: analyzer.plot_torus_visualization(layer_idx))
        ]
        
        for plot_name, plot_func in plot_functions:
            try:
                plot_func()
                plt.savefig(output_dir / f"{plot_name}.png", dpi=300, bbox_inches='tight')
                plt.close('all')
                print(f"  Saved {plot_name}.png")
            except Exception as e:
                print(f"  Error saving {plot_name}: {e}")
                plt.close('all')
        
        plt.ion()  # Turn interactive mode back on
        return analyzer
        
    except Exception as e:
        print(f"  Error processing {sweep_dir}: {e}")
        return None

def main():
    """Main function."""
    print("Starting sweep connectivity analysis...")
    
    analyzers = {}
    
    # Process each sweep directory
    for sweep_dir in sweep_dirs:
        analyzer = load_and_analyze_sweep(sweep_dir, layer_idx=0)
        if analyzer:
            analyzers[sweep_dir] = analyzer
    
    print(f"\\nSuccessfully processed {len(analyzers)} sweep configurations")
    print("Individual plots saved to: results/sweep_connectivity_analysis/gamma{gamma}_T{T}/")
    
    # Generate summary
    print("\\n=== Summary ===")
    for sweep_dir, analyzer in analyzers.items():
        if analyzer.decomposition_results:
            results = analyzer.decomposition_results
            sym_mean = results['symmetric_skew']['sym_frob'].mean()
            skew_mean = results['symmetric_skew']['skew_frob'].mean()
            print(f"{sweep_dir}: sym_frob_mean={sym_mean:.4f}, skew_frob_mean={skew_mean:.4f}")

if __name__ == "__main__":
    main()