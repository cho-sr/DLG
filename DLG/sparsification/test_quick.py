#!/usr/bin/env python3
"""Quick test script (10 iterations only)"""
import subprocess
import sys

print("Running quick test (10 iterations)...")
result = subprocess.run([
    sys.executable, "dlg_fedavg.py",
    "--index", "25",
    "--sparsity", "1.0",
    "--dlg_iters", "10",
    "--local_epochs", "1"
], capture_output=False, text=True)

sys.exit(result.returncode)
