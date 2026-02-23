# Usage Examples for DLG Attack on SCAFFOLD

This document provides practical examples for running DLG attacks on SCAFFOLD federated learning.

## Quick Start

### 1. Basic Attack (Recommended for First Try)

```bash
python main.py
```

This runs with default parameters:
- Image index: 25
- Local epochs: 1
- Learning rate: 0.01
- DLG iterations: 300

**Expected Results:**
- Execution time: ~2-5 minutes (CPU), ~30 seconds (GPU)
- MSE: ~0.001-0.01
- PSNR: >25 dB
- Visual quality: Good reconstruction

### 2. Custom Image Selection

```bash
# Try different CIFAR-10 images
python main.py --index 0    # First image
python main.py --index 42   # Different image
python main.py --index 999  # Another image
```

### 3. Using Your Own Image

```bash
python main.py --image /path/to/your/image.png
```

**Note:** Image will be resized to 32x32 pixels to match CIFAR-10 format.

## Advanced Usage

### 4. High-Quality Reconstruction (More Iterations)

```bash
python main.py --dlg_iterations 500
```

More iterations generally improve reconstruction quality but take longer.

### 5. Harder Attack Scenario (Multiple Epochs)

```bash
python main.py --local_epochs 5 --dlg_iterations 500
```

More local epochs make gradient inversion harder. You'll see:
- Lower reconstruction quality
- Higher MSE
- Longer convergence time

### 6. Different Learning Rates

```bash
# Lower learning rate (easier to attack)
python main.py --lr 0.001 --dlg_iterations 200

# Higher learning rate (harder to attack)
python main.py --lr 0.1 --dlg_iterations 500
```

### 7. Advanced Attack with TV Regularization

```bash
python dlg_advanced.py --use_tv --tv_weight 0.001
```

Total Variation regularization produces smoother images with less noise.

**Comparison:**
- Without TV: Sharp but potentially noisy
- With TV: Smoother but may lose fine details

### 8. Label Inference Attack

When the attacker doesn't know the label:

```bash
python dlg_advanced.py --infer_label --dlg_iterations 500
```

This simultaneously reconstructs the image AND infers the label.

**Note:** This is significantly harder and may require more iterations.

### 9. Different Initialization Strategies

```bash
# Random Gaussian noise (default)
python dlg_advanced.py --init_strategy random

# Zero initialization
python dlg_advanced.py --init_strategy zeros

# Mean initialization (gray image)
python dlg_advanced.py --init_strategy mean
```

### 10. Different Optimizers

```bash
# LBFGS (default, usually best)
python dlg_advanced.py --optimizer LBFGS --dlg_lr 0.1

# Adam optimizer
python dlg_advanced.py --optimizer Adam --dlg_lr 0.01

# SGD optimizer
python dlg_advanced.py --optimizer SGD --dlg_lr 1.0
```

## Comparison Experiments

### 11. Compare Across FL Algorithms

```bash
python compare_algorithms.py --index 25
```

This compares DLG effectiveness on:
- FedAvg
- SCAFFOLD
- FedProx

**Expected Output:**
- Side-by-side reconstruction comparison
- MSE and PSNR metrics for each
- Convergence curves
- Bar chart comparison

### 12. Run Multiple Experiments

```bash
# Make script executable
chmod +x run_experiments.sh

# Run all experiments
./run_experiments.sh
```

This runs 4 different experiments:
1. Basic attack (1 epoch)
2. Advanced with TV regularization
3. Label inference
4. Harder case (3 epochs)

## Practical Scenarios

### Scenario A: Maximum Reconstruction Quality

When you want the best possible reconstruction:

```bash
python dlg_advanced.py \
    --index 25 \
    --local_epochs 1 \
    --lr 0.001 \
    --dlg_iterations 1000 \
    --optimizer LBFGS \
    --dlg_lr 0.1 \
    --use_tv \
    --tv_weight 0.0001
```

**Expected:** Near-perfect reconstruction with PSNR > 30 dB

### Scenario B: Realistic Federated Learning Settings

More realistic with multiple epochs:

```bash
python main.py \
    --local_epochs 5 \
    --lr 0.01 \
    --dlg_iterations 500
```

**Expected:** Partial reconstruction, recognizable but with artifacts

### Scenario C: Privacy-Preserving Defense Evaluation

Test attack resistance with different configurations:

```bash
# Easy target (no defense)
python main.py --local_epochs 1 --lr 0.01

# Medium difficulty
python main.py --local_epochs 3 --lr 0.05

# Hard target (strong implicit defense)
python main.py --local_epochs 10 --lr 0.1
```

### Scenario D: Batch Processing Multiple Images

```bash
# Process images 0-9
for i in {0..9}; do
    python main.py --index $i --dlg_iterations 300
done
```

**Note:** Each run will overwrite previous output files. Use `dlg_advanced.py` with `--output_dir` for separate outputs.

### Scenario E: Research Experiment with Full Logging

```bash
python dlg_advanced.py \
    --index 42 \
    --local_epochs 1 \
    --dlg_iterations 500 \
    --use_tv \
    --tv_weight 0.001 \
    --save_interval 10 \
    --output_dir ./research_results
```

This saves:
- Reconstruction at every 10 iterations
- All metrics and plots
- Summary report
- Organized in timestamped directory

## Troubleshooting Examples

### Issue: Poor Reconstruction Quality

**Try:**
```bash
# Reduce local epochs
python main.py --local_epochs 1

# Reduce learning rate
python main.py --lr 0.001

# Increase DLG iterations
python main.py --dlg_iterations 1000

# Use TV regularization
python dlg_advanced.py --use_tv --tv_weight 0.001
```

### Issue: DLG Not Converging

**Try:**
```bash
# Different optimizer
python dlg_advanced.py --optimizer Adam --dlg_lr 0.01

# Different learning rate
python main.py --dlg_lr 0.5

# More iterations
python main.py --dlg_iterations 1000
```

### Issue: Out of Memory

**Try:**
```bash
# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
python main.py

# Or check available memory
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}')"
```

### Issue: Slow Execution

**Try:**
```bash
# Reduce iterations
python main.py --dlg_iterations 100

# Use GPU if available
python main.py  # Will automatically use CUDA/MPS if available

# Check device being used - look for "Running on device:" in output
```

## Parameter Guidelines

### Local Epochs (`--local_epochs`)
- **1**: Easy to attack, good reconstruction
- **2-3**: Moderate difficulty
- **5+**: Hard to attack, poor reconstruction
- **10+**: Very difficult, privacy protection

### Learning Rate (`--lr`)
- **0.001**: Very easy to attack
- **0.01**: Easy to attack (default)
- **0.1**: Moderate difficulty
- **0.5+**: Hard to attack

### DLG Iterations (`--dlg_iterations`)
- **100**: Quick test, poor quality
- **200-300**: Good balance (default)
- **500**: Better quality, longer time
- **1000+**: Best quality, slow

### DLG Learning Rate (`--dlg_lr`)
- **0.01**: Slow but stable (for Adam/SGD)
- **0.1**: Good default (for LBFGS)
- **0.5**: Fast but may diverge
- **1.0**: Very aggressive

### TV Weight (`--tv_weight`)
- **0.0001**: Subtle smoothing
- **0.001**: Moderate smoothing (recommended)
- **0.01**: Strong smoothing, may blur
- **0.1**: Too strong, over-smoothed

## Benchmarking

Run systematic benchmarks:

```bash
# Create benchmark script
cat > benchmark.sh << 'EOF'
#!/bin/bash
for epochs in 1 2 5 10; do
    for lr in 0.001 0.01 0.1; do
        echo "Testing: epochs=$epochs, lr=$lr"
        python main.py \
            --local_epochs $epochs \
            --lr $lr \
            --dlg_iterations 300 \
            --index 25
        # Save results with descriptive names
        mv ground_truth.png "results_e${epochs}_lr${lr}_gt.png"
        mv dlg_final_comparison.png "results_e${epochs}_lr${lr}_comparison.png"
    done
done
EOF

chmod +x benchmark.sh
./benchmark.sh
```

## Python API Usage

You can also import and use functions programmatically:

```python
import torch
from models.vision import LeNet, weights_init
from utils import label_to_onehot, cross_entropy_for_onehot
from torchvision import datasets, transforms

# Load data
dst = datasets.CIFAR10("~/.torch", download=True)
tp = transforms.ToTensor()
gt_data = tp(dst[25][0]).view(1, 3, 32, 32)
gt_label = torch.tensor([dst[25][1]])

# Initialize model
net = LeNet()
net.apply(weights_init)

# Perform SCAFFOLD training
# ... (see main.py for full code)

# Perform DLG attack
# ... (see main.py for full code)
```

## Tips and Best Practices

1. **Start Simple**: Begin with default parameters before experimenting
2. **Save Results**: Use `dlg_advanced.py` with `--output_dir` to organize outputs
3. **Iterate**: Try different parameters to understand their effects
4. **Compare**: Use `compare_algorithms.py` to see differences
5. **Document**: Keep notes on which parameters work best for your use case
6. **GPU**: Use GPU if available for faster experimentation
7. **Seed**: Set `--seed` for reproducible results

## Getting Help

```bash
# View all available options
python main.py --help
python dlg_advanced.py --help
python compare_algorithms.py --help

# Check versions
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
```

## Next Steps

After running these examples:

1. **Analyze Results**: Check MSE, PSNR, and visual quality
2. **Experiment**: Try different parameters combinations
3. **Research**: Read the papers referenced in README.md
4. **Defend**: Implement defense mechanisms (differential privacy, secure aggregation)
5. **Extend**: Try other datasets, models, or FL algorithms

---

For more information, see:
- [README.md](README.md) - Overview and algorithm details
- [main.py](main.py) - Basic implementation
- [dlg_advanced.py](dlg_advanced.py) - Advanced features
- [compare_algorithms.py](compare_algorithms.py) - Cross-algorithm comparison

