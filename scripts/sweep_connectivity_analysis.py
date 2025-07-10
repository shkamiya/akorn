#!/usr/bin/env python3
"""
Sweep Connectivity Analysis for AKOrN Models

This script analyzes the connectivity patterns across different gamma and T values
from the parameter sweep results.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Add source directory to path
sys.path.append('source')
from models.classification.my_knet import MyAKOrN
from models.classification.analysis_utils import AKOrNStaticAnalyzer

def load_and_analyze_sweep(sweep_dir, results_dir="results"):
    """Load and analyze a single sweep configuration."""
    results_dir = Path(results_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    config_path = results_dir / sweep_dir / "parameters.json"
    if not config_path.exists():
        print(f"Config not found for {sweep_dir}")
        return None, None, None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    gamma = config.get('gamma', 'unknown')
    T = config.get('T', 'unknown')
    print(f"Processing {sweep_dir}: gamma={gamma}, T={T}")
    
    # Check if model exists
    model_path = results_dir / sweep_dir / "my_akorn_cifar10_final.pth"
    if not model_path.exists():
        print(f"  Model not found for {sweep_dir}")
        return None, gamma, T
    
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
        
        print(f"  Successfully loaded {sweep_dir}")
        return model, gamma, T
        
    except Exception as e:
        print(f"  Error processing {sweep_dir}: {e}")
        return None, gamma, T

def save_plots_for_sweep(sweep_dir, model, gamma, T, layer_idx=0):
    """Generate and save all plots for a sweep configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create analyzer for specific layer
    analyzer = AKOrNStaticAnalyzer(model, layer_idx, device)
    
    # Create output directory
    output_dir = Path(f"results/e2025_0710_sweep_connectivity_analysis/gamma{gamma}_T{T}/layer{layer_idx}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving plots for {sweep_dir} (gamma={gamma}, T={T}, layer={layer_idx})...")
    
    try:
        # 1. Omega distributions
        try:
            analyzer.plot_omega_distributions(show=False)
            plt.savefig(output_dir / "omega_distributions.png", bbox_inches='tight')
            plt.close('all')
            print(f"  Saved omega_distributions.png")
        except Exception as e:
            print(f"  Error saving omega_distributions: {e}")
            plt.close('all')
        
        # 2. Connectivity statistics  
        try:
            analyzer.plot_connectivity_statistics(show=False)
            plt.savefig(output_dir / "connectivity_statistics.png", bbox_inches='tight')
            plt.close('all')
            print(f"  Saved connectivity_statistics.png")
        except Exception as e:
            print(f"  Error saving connectivity_statistics: {e}")
            plt.close('all')
        
        # 3. Frobenius norms summary (kernel)
        try:
            analyzer.plot_Frob_norms_summary('kernel', show=False)
            plt.savefig(output_dir / "frob_norms_kernel.png", bbox_inches='tight')
            plt.close('all')
            print(f"  Saved frob_norms_kernel.png")
        except Exception as e:
            print(f"  Error saving frob_norms_kernel: {e}")
            plt.close('all')
        
        # 4. Frobenius norms summary (channels)
        try:
            analyzer.plot_Frob_norms_summary('channels', show=False)
            plt.savefig(output_dir / "frob_norms_channels.png", bbox_inches='tight')
            plt.close('all')
            print(f"  Saved frob_norms_channels.png")
        except Exception as e:
            print(f"  Error saving frob_norms_channels: {e}")
            plt.close('all')
        
        # 5. Decomposition results (symmetric_skew)
        try:
            analyzer.plot_decomposition_results('symmetric_skew', show=False)
            plt.savefig(output_dir / "decomposition_symmetric_skew.png", bbox_inches='tight')
            plt.close('all')
            print(f"  Saved decomposition_symmetric_skew.png")
        except Exception as e:
            print(f"  Error saving decomposition_symmetric_skew: {e}")
            plt.close('all')
        
        # 6. Decomposition results (rotation_symmetric)
        try:
            analyzer.plot_decomposition_results('rotation_symmetric', show=False)
            plt.savefig(output_dir / "decomposition_rotation_symmetric.png", bbox_inches='tight')
            plt.close('all')
            print(f"  Saved decomposition_rotation_symmetric.png")
        except Exception as e:
            print(f"  Error saving decomposition_rotation_symmetric: {e}")
            plt.close('all')
        
        # 7. Torus visualization
        try:
            analyzer.plot_torus_visualization(show=False)
            plt.savefig(output_dir / "torus_visualization.png", bbox_inches='tight')
            plt.close('all')
            print(f"  Saved torus_visualization.png")
        except Exception as e:
            print(f"  Error saving torus_visualization: {e}")
            plt.close('all')
    
    except Exception as e:
        print(f"  General error in save_plots_for_sweep: {e}")
        plt.close('all')
    
    return analyzer

def main():
    """Main analysis function."""
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Will analyze {len(sweep_dirs)} sweep directories")
    
    # Process all sweep directories
    models = {}
    configs = {}
    
    for sweep_dir in sweep_dirs:
        model, gamma, T = load_and_analyze_sweep(sweep_dir)
        if model is not None:
            models[sweep_dir] = model
            configs[sweep_dir] = {'gamma': gamma, 'T': T}
    
    print(f"\\nSuccessfully processed {len(models)} sweep configurations")
    
    # Save plots for all analyzed sweeps (all 3 layers)
    layer0_analyzers = {}  # Store layer 0 analyzers for summary statistics
    
    for layer_idx in range(3):
        for sweep_dir, model in models.items():
            config = configs[sweep_dir]
            analyzer = save_plots_for_sweep(sweep_dir, model, config['gamma'], config['T'], layer_idx=layer_idx)
            
            # Store layer 0 analyzer for summary statistics
            if layer_idx == 0:
                layer0_analyzers[sweep_dir] = analyzer
    
    # Generate summary statistics
    print("\\n=== Summary Statistics ===")
    summary_data = []
    
    for sweep_dir, analyzer in layer0_analyzers.items():
        # Compute decomposition for layer 0 to get summary statistics
        connectivity_blocks = analyzer.extract_connectivity_blocks()
        p1, p2, p3, q, sym_frob, skew_frob = analyzer.decompose_symmetric_skew(connectivity_blocks)
        c_R, c_S, alpha, beta = analyzer.decompose_rotation_symmetric(connectivity_blocks)
        
        config = configs[sweep_dir]
        
        summary = {
            'sweep_dir': sweep_dir,
            'gamma': config['gamma'],
            'T': config['T'],
            'sym_frob_mean': sym_frob.mean(),
            'sym_frob_std': sym_frob.std(),
            'skew_frob_mean': skew_frob.mean(),
            'skew_frob_std': skew_frob.std(),
            'c_R_mean': c_R.mean(),
            'c_R_std': c_R.std(),
            'c_S_mean': c_S.mean(),
            'c_S_std': c_S.std(),
            'alpha_std': alpha.std(),
            'beta_std': beta.std()
        }
        summary_data.append(summary)
        
        print(f"{sweep_dir}: gamma={config['gamma']}, T={config['T']}")
        print(f"  sym_frob: {summary['sym_frob_mean']:.4f} ± {summary['sym_frob_std']:.4f}")
        print(f"  skew_frob: {summary['skew_frob_mean']:.4f} ± {summary['skew_frob_std']:.4f}")
        print(f"  c_R: {summary['c_R_mean']:.4f} ± {summary['c_R_std']:.4f}")
        print(f"  c_S: {summary['c_S_mean']:.4f} ± {summary['c_S_std']:.4f}")
    
    print(f"\\nAnalysis complete! Individual plots saved to: results/e2025_0710_sweep_connectivity_analysis/gamma{{gamma}}_T{{T}}/")
    
    # Save summary data as JSON
    summary_path = Path("results/e2025_0710_sweep_connectivity_analysis/summary_statistics.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    json_summary = []
    for item in summary_data:
        json_item = {}
        for key, value in item.items():
            if isinstance(value, np.floating):
                json_item[key] = float(value)
            elif isinstance(value, np.integer):
                json_item[key] = int(value)
            else:
                json_item[key] = value
        json_summary.append(json_item)
    
    with open(summary_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    print(f"Summary statistics saved to {summary_path}")

if __name__ == "__main__":
    main()