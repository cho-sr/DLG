#!/bin/bash

# Sparsification vs FL Performance & DLG Attack Experiment
# Run this script to execute the full experiment

echo "=========================================="
echo "Starting Experiment..."
echo "=========================================="
echo ""

# Check dependencies
echo "Checking dependencies..."
python -c "import torch, torchvision, matplotlib, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing missing dependencies..."
    pip install -r requirements.txt -q
fi

echo "✅ Dependencies OK"
echo ""

# Run experiment
echo "Running experiment (this may take 5-10 minutes on GPU, 20-30 minutes on CPU)..."
echo ""

python main.py

# Check results
echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "📊 Generated visualizations:"
ls -lh results/*.png
echo ""
echo "📁 Check the following files:"
echo "  - results/comprehensive_metrics.png"
echo "  - results/dlg_convergence.png"
echo "  - results/reconstruction_comparison.png"
echo "  - RESULTS.md (detailed analysis)"
echo ""
echo "✅ Done!"
