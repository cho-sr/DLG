# Quick Reference Card - DLG Attack on SCAFFOLD

## 🚀 Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run basic attack (recommended first try)
python main.py

# 3. Run advanced attack with TV regularization
python dlg_advanced.py --use_tv --tv_weight 0.001

# 4. Compare algorithms
python compare_algorithms.py

# 5. Interactive mode
python run_single_experiment.py

# 6. Batch experiments
./run_experiments.sh
```

## 📝 Most Common Parameters

```bash
# Select different image
python main.py --index 42

# Change local epochs (1=easy, 10=hard to attack)
python main.py --local_epochs 3

# Adjust DLG iterations (more=better quality, slower)
python main.py --dlg_iterations 500

# Change learning rate (lower=easier to attack)
python main.py --lr 0.001

# Use custom image
python main.py --image /path/to/image.png
```

## 🎯 Key Files

| File | Purpose |
|------|---------|
| `main.py` | Basic DLG attack |
| `dlg_advanced.py` | Advanced features (TV, label inference) |
| `compare_algorithms.py` | FedAvg vs SCAFFOLD vs FedProx |
| `README.md` | Full documentation |
| `USAGE_EXAMPLES.md` | Practical examples |
| `PROJECT_OVERVIEW.md` | Project summary |

## 📊 Expected Outputs

| File | Description |
|------|-------------|
| `ground_truth.png` | Original image |
| `initial_dummy.png` | Random start |
| `dlg_reconstruction_progress.png` | 12 snapshots |
| `dlg_loss_curve.png` | Convergence |
| `dlg_final_comparison.png` | Result |

## 🔢 Parameter Guidelines

### Local Epochs
- `1` → Easy attack, good reconstruction ✅
- `2-3` → Moderate difficulty
- `5+` → Hard attack, poor reconstruction
- `10+` → Very hard, privacy protected

### Learning Rate
- `0.001` → Very easy to attack
- `0.01` → Easy (default) ✅
- `0.1` → Moderate
- `0.5+` → Hard

### DLG Iterations
- `100` → Quick test
- `300` → Good balance (default) ✅
- `500` → Better quality
- `1000+` → Best quality

## 📈 Quality Metrics

| Metric | Good | Moderate | Poor |
|--------|------|----------|------|
| **PSNR (dB)** | > 25 | 15-25 | < 15 |
| **MSE** | < 0.01 | 0.01-0.1 | > 0.1 |
| **Correlation** | > 0.95 | 0.8-0.95 | < 0.8 |

## 🎓 Common Use Cases

### Maximum Quality
```bash
python dlg_advanced.py \
    --local_epochs 1 \
    --lr 0.001 \
    --dlg_iterations 1000 \
    --use_tv --tv_weight 0.0001
```

### Realistic FL Setting
```bash
python main.py \
    --local_epochs 5 \
    --lr 0.01 \
    --dlg_iterations 500
```

### Label Inference
```bash
python dlg_advanced.py \
    --infer_label \
    --dlg_iterations 500
```

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Poor quality | Use `--local_epochs 1` |
| Slow | Use `--dlg_iterations 100` |
| Out of memory | Set `CUDA_VISIBLE_DEVICES=""` |
| Module error | Run `pip install -r requirements.txt` |
| Not converging | Try `--dlg_lr 0.5` or `--optimizer Adam` |

## 💻 Device Selection

```bash
# Auto (default) - uses CUDA/MPS if available
python main.py

# Force CPU
CUDA_VISIBLE_DEVICES="" python main.py

# Check available device
python -c "import torch; print(torch.cuda.is_available())"
```

## 📚 Documentation Quick Links

- **Getting Started**: README.md (section 1-3)
- **Algorithm Details**: README.md (Algorithm Details)
- **Usage Examples**: USAGE_EXAMPLES.md
- **Full Overview**: PROJECT_OVERVIEW.md
- **Parameters**: `python main.py --help`

## 🔍 Key Code Locations

### SCAFFOLD Training
- File: `main.py`
- Lines: ~84-108 (local training with control variates)

### DLG Attack
- File: `main.py`
- Lines: ~131-149 (gradient matching optimization)

### Gradient Computation
- File: `main.py`
- Lines: ~87-93 (target gradient extraction)

## 🎯 Success Checklist

- [ ] Installed dependencies
- [ ] Ran basic attack successfully
- [ ] Generated 5 output images
- [ ] PSNR > 20 dB (for 1 epoch)
- [ ] Image is recognizable
- [ ] Understood key parameters
- [ ] Tried different configurations

## ⚡ One-Liners

```bash
# Quick test (100 iterations)
python main.py --dlg_iterations 100

# Best quality (may be slow)
python main.py --local_epochs 1 --lr 0.001 --dlg_iterations 1000

# Hard attack scenario
python main.py --local_epochs 10 --lr 0.1

# Save to custom directory (advanced)
python dlg_advanced.py --output_dir ./my_results

# Compare all algorithms
python compare_algorithms.py --dlg_iterations 300
```

## 📞 Get Help

```bash
# View all options
python main.py --help
python dlg_advanced.py --help

# Check versions
python -c "import torch; print(torch.__version__)"

# Test imports
python -c "from models.vision import LeNet; from utils import label_to_onehot; print('OK')"
```

## 🎓 Learning Order

1. **Beginner**: Run `python main.py` → examine outputs
2. **Intermediate**: Try different `--index` and `--local_epochs`
3. **Advanced**: Use `dlg_advanced.py` with various options
4. **Expert**: Run `compare_algorithms.py` and modify code

## 💡 Pro Tips

✅ Always start with `--local_epochs 1` for best results
✅ Use GPU when available (10x faster)
✅ Try indices 25, 42, 77, 100 for variety
✅ Save outputs before running again (they overwrite)
✅ Use `dlg_advanced.py` for organized outputs
✅ Read README.md Algorithm Details section
✅ Experiment with TV regularization
✅ Compare results across algorithms

## 📊 Typical Results (1 epoch, lr=0.01)

```
Iteration   0: Loss = 12.345678
Iteration 100: Loss = 0.234567
Iteration 200: Loss = 0.012345
Iteration 300: Loss = 0.000523

Final Metrics:
  MSE: 0.000523
  PSNR: 32.81 dB
  Correlation: 0.9847

Status: ✅ Excellent reconstruction
```

## 🎯 Research Questions to Explore

1. How does batch size affect attack success?
2. What's the minimum number of iterations needed?
3. How do different optimizers compare?
4. Can we infer labels accurately?
5. What learning rate threshold makes attack impractical?
6. How does TV weight affect quality?
7. Are ResNets easier/harder to attack than LeNet?
8. What's the effect of gradient clipping?

---

**Remember**: This is for research/educational purposes only!

**More Info**: See README.md, USAGE_EXAMPLES.md, PROJECT_OVERVIEW.md

**Quick Start**: `python main.py`

**Get Help**: `python main.py --help`

