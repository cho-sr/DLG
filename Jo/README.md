# Sparsification vs FL Performance & DLG Attack Experiment

Privacy-preserving machine learning experiment comparing Federated Learning (FL) performance and Deep Leakage from Gradients (DLG) reconstruction under different sparsification levels.

## 📋 Overview

This experiment validates the hypothesis:
> **"FL performance is maintained even with 10% gradient retention, while DLG reconstruction quality degrades dramatically with minimal gradient information loss."**

## 🎯 Experimental Setup

### Dataset
- **MNIST**: 28×28 grayscale handwritten digits
- **Training subset**: 10,000 samples
- **Test set**: Full 10,000 samples

### Model
- **LeNet-style CNN**
  - Conv1: 1→12 channels (5×5 kernel)
  - Conv2: 12→12 channels (5×5 kernel)
  - FC1: 588→100
  - FC2: 100→10 (output)

### Sparsification Cases
1. **100% (No Sparsification)**: Baseline
2. **Top 10%**: Keep only 10% largest gradients by absolute value
3. **Top 1%**: Keep only 1% largest gradients

### Attack Method
- **DLG (Deep Leakage from Gradients)** [Zhu et al., NeurIPS 2019]
- Optimizer: L-BFGS
- Iterations: 300
- Metric: Mean Squared Error (MSE) between original and reconstructed images

## 🚀 Usage

### Install Dependencies
```bash
pip install torch torchvision matplotlib numpy
```

### Run Experiment
```bash
cd /root/Jo
python main.py
```

### Expected Runtime
- **GPU**: ~5-10 minutes
- **CPU**: ~20-30 minutes

## 📊 Outputs

The script generates three visualizations in the `results/` directory:

1. **`performance_comparison.png`**
   - FL Test Accuracy vs Sparsification
   - DLG Reconstruction MSE vs Sparsification

2. **`dlg_convergence.png`**
   - DLG optimization convergence for each case

3. **`reconstruction_comparison.png`**
   - Visual comparison: Original vs Reconstructed images
   - Difference heatmaps with MSE values

## 📈 Expected Results

### FL Performance (Test Accuracy)
- **100%**: ~98-99% accuracy
- **10%**: ~97-98% accuracy (minimal degradation)
- **1%**: ~90-95% accuracy (acceptable for privacy gains)

### DLG Reconstruction (MSE)
- **100%**: ~0.001-0.01 (near-perfect reconstruction)
- **10%**: ~0.1-0.5 (significant degradation)
- **1%**: ~1.0+ (reconstruction fails)

### Key Finding
**Sparsification provides strong privacy protection (high DLG MSE) while maintaining federated learning utility (high accuracy).**

## 🔬 Implementation Details

### Sparsification Algorithm
```python
def sparsify_gradients(model, ratio):
    1. Collect all gradient values
    2. Compute threshold = kth largest absolute value (k = ratio × total)
    3. Zero out gradients below threshold
    4. Keep only top-k% gradients
```

### FL Training Process
1. Initialize model
2. Local training (5 epochs on client data)
3. Compute weight updates (Δw = w_new - w_old)
4. Apply sparsification to updates
5. Return sparsified gradients

### DLG Attack Process
1. Initialize dummy data and labels randomly
2. Compute gradients from dummy data
3. Minimize gradient matching loss: ||∇_dummy - ∇_target||²
4. Optimize using L-BFGS
5. Track reconstruction MSE over iterations

## 📚 References

1. **Deep Leakage from Gradients**
   - Zhu, L., Liu, Z., & Han, S. (2019)
   - NeurIPS 2019
   - https://arxiv.org/abs/1906.08935

2. **Federated Learning**
   - McMahan, B., et al. (2017)
   - Communication-Efficient Learning of Deep Networks from Decentralized Data

## 🛠️ Customization

### Change Dataset to CIFAR-10
```python
# In run_experiment():
train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

# Update model input channels:
self.conv1 = nn.Conv2d(3, 12, kernel_size=5, padding=2)  # 1 → 3
```

### Adjust Sparsification Levels
```python
SPARSITY_CASES = [
    {"name": "100%", "ratio": 1.0},
    {"name": "50%", "ratio": 0.5},
    {"name": "25%", "ratio": 0.25},
    {"name": "10%", "ratio": 0.1},
    {"name": "5%", "ratio": 0.05},
    {"name": "1%", "ratio": 0.01},
]
```

### Increase DLG Attack Strength
```python
DLG_ITERATIONS = 500  # More iterations for better reconstruction
```

## 🎓 Educational Value

This experiment demonstrates:
- ✅ **Privacy-Utility Trade-off**: Sparsification reduces privacy leakage while maintaining model performance
- ✅ **Gradient Leakage Vulnerability**: Full gradients leak significant private information
- ✅ **Defense Effectiveness**: Simple sparsification provides strong protection
- ✅ **FL Robustness**: Federated learning tolerates gradient compression well

## 📝 Notes

- The experiment uses a single sample for DLG attack (batch size = 1)
- Larger batches make DLG attacks harder (batch size > 1 increases difficulty)
- Real-world FL systems should combine multiple defenses (sparsification + DP + secure aggregation)

## 🤝 Contributing

Feel free to extend this experiment:
- Test different architectures (ResNet, Transformer)
- Implement advanced attacks (iDLG, GradInversion)
- Add differential privacy mechanisms
- Test on real federated datasets

## 📄 License

MIT License - Free to use for research and education.
