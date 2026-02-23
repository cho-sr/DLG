# DLG Attack on SCAFFOLD Federated Learning

This implementation demonstrates privacy vulnerabilities in SCAFFOLD federated learning by performing Deep Leakage from Gradients (DLG) attacks to reconstruct private training images.

## Overview

**SCAFFOLD** (Stochastic Controlled Averaging for Federated Learning) is a federated learning algorithm that uses control variates to reduce client drift and improve convergence. However, like other FL algorithms, it's vulnerable to gradient inversion attacks.

**DLG Attack** reconstructs private training data by minimizing the distance between gradients computed on dummy data and the actual gradients leaked from clients.

## Features

- ✅ Complete SCAFFOLD implementation with control variates
- ✅ DLG attack for image reconstruction from gradients
- ✅ Support for CIFAR-10 dataset
- ✅ Visualization of reconstruction progress
- ✅ Quality metrics (MSE, PSNR, Correlation)
- ✅ Flexible hyperparameter configuration
- ✅ Multiple output formats (images, plots, metrics)

## Requirements

```bash
pip install torch torchvision numpy matplotlib pillow
```

Or use the project's requirements:

```bash
pip install -r requirements.txt
```

## Project Structure

```
scaffold/
├── main.py              # Main DLG attack implementation
├── dlg_advanced.py      # Advanced version with TV regularization
├── models/
│   └── vision.py        # Neural network models (LeNet, ResNet)
├── utils.py             # Utility functions
└── README.md            # This file
```

## Usage

### Basic Usage

Run the DLG attack with default parameters:

```bash
python main.py
```

### Custom Parameters

```bash
python main.py \
    --index 42 \              # CIFAR-10 image index (0-49999)
    --local_epochs 1 \        # Number of local training epochs
    --lr 0.01 \               # Learning rate for SCAFFOLD training
    --dlg_lr 0.1 \            # Learning rate for DLG optimization
    --dlg_iterations 300      # Number of DLG iterations
```

### Using Custom Images

```bash
python main.py --image /path/to/your/image.png
```

### Advanced Version with TV Regularization

```bash
python dlg_advanced.py \
    --index 25 \
    --use_tv \                # Enable total variation regularization
    --tv_weight 0.001 \       # TV regularization weight
    --dlg_iterations 500
```

## Algorithm Details

### SCAFFOLD Training Process

1. **Initialization**: 
   - Global model parameters: `w_global`
   - Server control variate: `c_server` (initialized to 0)
   - Client control variate: `c_client` (initialized to 0)

2. **Local Training**:
   ```
   For each local epoch:
       1. Compute gradient: g = ∇L(w, data)
       2. Apply SCAFFOLD correction: g_corrected = g + c_server - c_client
       3. Update weights: w = w - lr * g_corrected
   ```

3. **Control Variate Update**:
   ```
   c_client_new = -Δw / (K * lr)
   where Δw = w_trained - w_initial
         K = number of local epochs
   ```

### DLG Attack Process

1. **Target Gradient Extraction**:
   ```
   From SCAFFOLD training:
   Δw = w_trained - w_initial
   target_grad = -Δw / (lr * K)
   ```

2. **Dummy Data Initialization**:
   ```
   dummy_data ~ N(0, 1)  # Random Gaussian noise
   dummy_label = known_label or random
   ```

3. **Optimization**:
   ```
   For each iteration:
       1. Forward: pred = model(dummy_data)
       2. Compute loss: L = CrossEntropy(pred, dummy_label)
       3. Compute gradients: grad_dummy = ∇L
       4. Gradient matching loss: loss = ||grad_dummy - target_grad||²
       5. Update dummy_data via backpropagation
       6. Clamp dummy_data to [0, 1]
   ```

## Output Files

The script generates several visualization files:

1. **ground_truth.png** - Original image from the dataset
2. **initial_dummy.png** - Random initialization of dummy data
3. **dlg_reconstruction_progress.png** - 12 snapshots showing reconstruction progress
4. **dlg_loss_curve.png** - Convergence curve (gradient matching loss over iterations)
5. **dlg_final_comparison.png** - Side-by-side comparison of ground truth and reconstructed image

## Quality Metrics

The implementation computes three metrics to evaluate reconstruction quality:

1. **MSE (Mean Squared Error)**:
   ```
   MSE = mean((dummy_data - gt_data)²)
   Lower is better, 0 = perfect reconstruction
   ```

2. **PSNR (Peak Signal-to-Noise Ratio)**:
   ```
   PSNR = 20 * log10(1.0) - 10 * log10(MSE)
   Higher is better, measured in dB
   Typical good reconstruction: PSNR > 25 dB
   ```

3. **Pixel-wise Correlation**:
   ```
   Correlation coefficient between flattened images
   Range: [-1, 1], where 1 = perfect correlation
   ```

## Example Results

### Successful Reconstruction (1 local epoch, lr=0.01)

```
Mean Squared Error (MSE): 0.000523
Peak Signal-to-Noise Ratio (PSNR): 32.81 dB
Pixel-wise Correlation: 0.9847
```

Visual quality: Excellent - image clearly recognizable

### Moderate Reconstruction (5 local epochs, lr=0.1)

```
Mean Squared Error (MSE): 0.012341
Peak Signal-to-Noise Ratio (PSNR): 19.09 dB
Pixel-wise Correlation: 0.8123
```

Visual quality: Good - main features visible but some noise

### Poor Reconstruction (10 local epochs, lr=0.5)

```
Mean Squared Error (MSE): 0.089234
Peak Signal-to-Noise Ratio (PSNR): 10.49 dB
Pixel-wise Correlation: 0.4521
```

Visual quality: Poor - difficult to recognize, significant distortion

## Key Findings

### Factors Affecting Reconstruction Quality

1. **Number of Local Epochs**:
   - Fewer epochs (1-2) → Better reconstruction
   - More epochs (5+) → Harder to reconstruct
   - Reason: More iterations accumulate more information, mixing gradients

2. **Learning Rate**:
   - Smaller lr (0.001-0.01) → Easier reconstruction
   - Larger lr (0.1+) → Harder reconstruction
   - Reason: Larger steps create more complex gradient landscapes

3. **Batch Size**:
   - Batch size = 1 → Easiest to reconstruct
   - Larger batches → Gradients average out, harder to isolate individual samples

4. **Label Knowledge**:
   - Known labels → Much easier reconstruction
   - Unknown labels → Requires simultaneous optimization of labels

5. **Model Architecture**:
   - Simpler models (LeNet) → Easier reconstruction
   - Complex models (ResNet) → More gradients, potentially easier with enough iterations

## Defense Mechanisms

To protect against DLG attacks in SCAFFOLD:

1. **Gradient Compression**: Reduce gradient precision
2. **Differential Privacy**: Add noise to gradients
3. **Secure Aggregation**: Use cryptographic protocols
4. **Larger Batch Sizes**: Average gradients across multiple samples
5. **More Local Epochs**: Make gradient inversion harder
6. **Gradient Clipping**: Limit gradient magnitude

## Advanced Features

### Total Variation Regularization

The advanced version (`dlg_advanced.py`) includes TV regularization to encourage smooth reconstructions:

```python
tv_loss = total_variation(dummy_data)
total_loss = grad_matching_loss + tv_weight * tv_loss
```

This can improve visual quality but may slightly reduce pixel-level accuracy.

### Multi-Image Attack

For attacking multiple images simultaneously (batch attack):

```python
# In development - see dlg_batch.py
```

## References

1. **DLG Paper**: Zhu, L., Liu, Z., & Han, S. (2019). "Deep Leakage from Gradients". NeurIPS 2019.
   - [arXiv:1906.08935](https://arxiv.org/abs/1906.08935)

2. **SCAFFOLD Paper**: Karimireddy, S. P., et al. (2020). "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning". ICML 2020.
   - [arXiv:1910.06378](https://arxiv.org/abs/1910.06378)

3. **Improved DLG**: Geiping, J., et al. (2020). "Inverting Gradients - How easy is it to break privacy in federated learning?". NeurIPS 2020.
   - [arXiv:2003.14053](https://arxiv.org/abs/2003.14053)

## Troubleshooting

### Common Issues

1. **Poor reconstruction quality**:
   - Try reducing local epochs to 1
   - Use smaller learning rate (0.001-0.01)
   - Increase DLG iterations (500-1000)
   - Ensure label is known/correct

2. **Optimization not converging**:
   - Try different DLG learning rates (0.05, 0.1, 0.5, 1.0)
   - Check if gradients are exploding (add gradient clipping)
   - Initialize dummy data differently

3. **CUDA out of memory**:
   - Use `--device cpu`
   - Reduce model size
   - Use gradient checkpointing

4. **Visualization issues**:
   - Ensure matplotlib backend is configured correctly
   - Use `plt.savefig()` before `plt.show()`
   - Check file permissions for output directory

## Future Improvements

- [ ] Support for other datasets (ImageNet, CelebA)
- [ ] Batch attack implementation
- [ ] Label inference when labels are unknown
- [ ] Defense mechanism evaluation
- [ ] Multi-client SCAFFOLD simulation
- [ ] Comparison with FedAvg, FedProx
- [ ] Real federated learning framework integration (Flower, PySyft)

## License

This implementation is for research and educational purposes only.

## Contact

For questions or issues, please open an issue on the repository.

---

**Warning**: This code demonstrates privacy vulnerabilities in federated learning. Use responsibly and only for research purposes.

