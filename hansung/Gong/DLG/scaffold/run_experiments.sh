#!/bin/bash
# Quick-start script for running DLG experiments on SCAFFOLD

echo "=========================================="
echo "DLG Attack on SCAFFOLD - Experiments"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Experiment 1: Basic DLG Attack (1 epoch)"
echo "=========================================="
python main.py --index 25 --local_epochs 1 --dlg_iterations 300

echo ""
echo "=========================================="
echo "Experiment 2: Advanced DLG with TV Regularization"
echo "=========================================="
python dlg_advanced.py --index 42 --local_epochs 1 --use_tv --tv_weight 0.001 --dlg_iterations 300

echo ""
echo "=========================================="
echo "Experiment 3: Label Inference Attack"
echo "=========================================="
python dlg_advanced.py --index 100 --local_epochs 1 --infer_label --dlg_iterations 500

echo ""
echo "=========================================="
echo "Experiment 4: Harder Case (More Epochs)"
echo "=========================================="
python main.py --index 77 --local_epochs 3 --dlg_iterations 500

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Check the dlg_results directory for outputs"
echo "=========================================="

