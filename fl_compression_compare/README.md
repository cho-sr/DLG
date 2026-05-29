# Federated Learning Compression Comparison

This folder contains a standalone PyTorch experiment for comparing convergence
under non-IID federated learning when client updates are compressed with:

- top-k sparsification by update magnitude
- symmetric uniform quantization

The default experiment uses CIFAR-10 with a Dirichlet non-IID client split.
Results are saved as CSV, JSON, and PNG convergence plots.

## Quick start

```bash
python fl_compression_compare/run_experiment.py
```

Outputs are written to:

```text
fl_compression_compare/outputs/
```

## Useful examples

Run a small smoke test:

```bash
python fl_compression_compare/run_experiment.py \
  --dataset fake \
  --rounds 2 \
  --train_samples 1000 \
  --test_samples 300 \
  --keep_ratios 1.0 0.1 \
  --quant_bits 32 8
```

Run a stronger non-IID comparison:

```bash
python fl_compression_compare/run_experiment.py \
  --alpha 0.1 \
  --rounds 50 \
  --keep_ratios 1.0 0.2 0.05 0.01 \
  --quant_bits 32 8 4 2
```

Run on CIFAR-10:

```bash
python fl_compression_compare/run_experiment.py \
  --dataset cifar10 \
  --rounds 30 \
  --lr 0.02 \
  --train_samples 20000 \
  --test_samples 5000
```

Run on MNIST:

```bash
python fl_compression_compare/run_experiment.py \
  --dataset mnist \
  --rounds 30 \
  --train_samples 12000 \
  --test_samples 2000
```

## Main arguments

- `--alpha`: Dirichlet concentration for non-IID split. Smaller values create
  more label-skewed clients.
- `--keep_ratios`: fraction of model-update entries uploaded after top-k
  sparsification. `1.0` means dense upload.
- `--quant_bits`: quantization bit-width. `32` means no quantization.
- `--frac`: fraction of clients sampled per communication round.
- `--local_epochs`: local SGD epochs per selected client.

## Output files

- `history.csv`: per-round train/test metrics and communication estimates.
- `summary.json`: run configuration and final metrics for each condition.
- `client_label_distribution.json`: non-IID label histogram per client.
- `convergence_accuracy.png`: test accuracy over communication rounds.
- `convergence_loss.png`: test loss over communication rounds.
- `accuracy_vs_upload.png`: accuracy against cumulative upload cost.
