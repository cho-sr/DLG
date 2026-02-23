# -*- coding: utf-8 -*-
"""
Batch Experiments: MIA with Various Sparsification Ratios

다양한 스파시티 비율에서 Model Inversion Attack 성능을 비교합니다.

Usage:
    python batch_mia_experiments.py --index 25
    python batch_mia_experiments.py --index 42 --mia_iters 500
"""

import argparse
import subprocess
import json
import os
from pathlib import Path

# ============================================================================
# Argument Parser
# ============================================================================
parser = argparse.ArgumentParser(description='Batch MIA Experiments')
parser.add_argument('--index', type=int, default=25,
                    help='Target image index')
parser.add_argument('--sparsities', type=float, nargs='+', 
                    default=[1.0, 0.5, 0.1, 0.05, 0.01],
                    help='List of sparsity ratios to test')
parser.add_argument('--mia_iters', type=int, default=300,
                    help='MIA iterations')
parser.add_argument('--local_epochs', type=int, default=1,
                    help='Local training epochs')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed')
args = parser.parse_args()

# ============================================================================
# Batch Experiments
# ============================================================================
print("="*70)
print("BATCH MODEL INVERSION ATTACK EXPERIMENTS")
print("="*70)
print(f"\nExperiment Setup:")
print(f"  Target Image Index: {args.index}")
print(f"  Sparsity Ratios: {args.sparsities}")
print(f"  MIA Iterations: {args.mia_iters}")
print(f"  Local Epochs: {args.local_epochs}")
print(f"\nStarting experiments...\n")

results = {}

for sparsity in args.sparsities:
    print("-" * 70)
    print(f"Running MIA with Sparsity = {sparsity * 100}%")
    print("-" * 70)
    
    cmd = [
        'python', 'fedavg_mia.py',
        '--index', str(args.index),
        '--sparsity', str(sparsity),
        '--mia_iters', str(args.mia_iters),
        '--local_epochs', str(args.local_epochs),
        '--seed', str(args.seed)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"✓ Completed successfully (sparsity={sparsity*100}%)")
            results[f"{sparsity*100}%"] = "SUCCESS"
        else:
            print(f"✗ Failed (sparsity={sparsity*100}%)")
            print(f"Error: {result.stderr}")
            results[f"{sparsity*100}%"] = "FAILED"
    
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout (sparsity={sparsity*100}%)")
        results[f"{sparsity*100}%"] = "TIMEOUT"
    
    except Exception as e:
        print(f"✗ Error (sparsity={sparsity*100}%): {str(e)}")
        results[f"{sparsity*100}%"] = f"ERROR: {str(e)}"
    
    print()

# ============================================================================
# Summary
# ============================================================================
print("="*70)
print("BATCH EXPERIMENTS SUMMARY")
print("="*70)
print("\nResults:")
for sparsity, status in results.items():
    status_symbol = "✓" if status == "SUCCESS" else "✗"
    print(f"  {status_symbol} Sparsity {sparsity}: {status}")

print("\n" + "="*70)
print("All experiments completed!")
print("="*70)
