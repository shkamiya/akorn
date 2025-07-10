#!/usr/bin/env python3
"""Test script for sweep connectivity analysis."""

import sys
sys.path.append('source')

from sweep_connectivity_analysis import SweepConnectivityAnalyzer

def test_single_sweep():
    """Test analysis of a single sweep."""
    analyzer = SweepConnectivityAnalyzer()
    
    # Test with first sweep directory
    test_dirs = ["sweep_20250708_581384.opbs_0"]
    
    configs = analyzer.load_sweep_configs(test_dirs)
    print(f"Loaded {len(configs)} configurations")
    
    # Test model loading
    model = analyzer.load_model(test_dirs[0])
    if model:
        print("Model loaded successfully")
        
        # Test analysis
        sweep_analyzer = analyzer.analyze_sweep_model(test_dirs[0], layer_idx=0)
        if sweep_analyzer:
            print("Analysis completed successfully")
            
            # Test saving plots
            analyzer.save_plots_for_sweep(test_dirs[0], layer_idx=0)
            print("Plots saved successfully")
        else:
            print("Analysis failed")
    else:
        print("Model loading failed")

if __name__ == "__main__":
    test_single_sweep()