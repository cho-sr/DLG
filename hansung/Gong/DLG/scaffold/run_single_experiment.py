#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick-start script for running a single DLG experiment
Run with: python run_single_experiment.py
"""
import subprocess
import sys
import os

def print_banner(text):
    """Print a formatted banner."""
    width = 70
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")

def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"▶ {description}")
    print(f"  Command: {' '.join(cmd)}\n")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {description} failed!")
        print(f"  Exit code: {e.returncode}\n")
        return False
    except FileNotFoundError:
        print(f"✗ Error: Command not found. Is Python installed?")
        return False

def main():
    """Main execution function."""
    print_banner("DLG Attack on SCAFFOLD - Quick Start")
    
    # Check if we're in the right directory
    if not os.path.exists('main.py'):
        print("✗ Error: main.py not found!")
        print("  Please run this script from the scaffold directory.")
        sys.exit(1)
    
    print("This script will run a basic DLG attack experiment.")
    print("You can modify the parameters in this script or run main.py directly.\n")
    
    # Experiment configuration
    config = {
        'image_index': 25,
        'local_epochs': 1,
        'learning_rate': 0.01,
        'dlg_iterations': 300,
        'dlg_lr': 0.1
    }
    
    print("Experiment Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Ask for confirmation
    response = input("Press Enter to start the experiment (or 'q' to quit): ")
    if response.lower() == 'q':
        print("Experiment cancelled.")
        sys.exit(0)
    
    print_banner("Starting Experiment")
    
    # Build command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        'main.py',
        '--index', str(config['image_index']),
        '--local_epochs', str(config['local_epochs']),
        '--lr', str(config['learning_rate']),
        '--dlg_iterations', str(config['dlg_iterations']),
        '--dlg_lr', str(config['dlg_lr'])
    ]
    
    # Run experiment
    success = run_command(cmd, "DLG Attack Experiment")
    
    if success:
        print_banner("Experiment Completed Successfully!")
        print("Generated files:")
        print("  - ground_truth.png: Original image")
        print("  - initial_dummy.png: Random initialization")
        print("  - dlg_reconstruction_progress.png: Reconstruction steps")
        print("  - dlg_loss_curve.png: Convergence curve")
        print("  - dlg_final_comparison.png: Final results")
        print("\nTip: Try different parameters for varied results:")
        print("  python main.py --index 42 --local_epochs 2 --dlg_iterations 500")
    else:
        print_banner("Experiment Failed")
        print("Common issues:")
        print("  1. Missing dependencies: pip install -r requirements.txt")
        print("  2. CUDA errors: Try adding --device cpu")
        print("  3. Dataset not downloaded: Will download automatically on first run")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Experiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

