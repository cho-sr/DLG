# FedSGD + DLG Simulation

This is a clean, single-process PyTorch simulation of:

- FedSGD with multiple clients
- function-call-only server/client interaction
- DLG reconstruction from one victim client's leaked gradient

## Files

- `model.py`: LeNet-style model for CIFAR-10
- `data.py`: CIFAR-10 loading and client partitioning
- `fedsgd_sim.py`: one-round FedSGD simulation
- `dlg_attack.py`: DLG with L-BFGS and pure L2 gradient matching
- `main.py`: runnable end-to-end script

## Run

If CIFAR-10 is not already present locally:

```bash
python3 fedsgd_dlg_sim/main.py --download
```

If CIFAR-10 is already present at the default data path:

```bash
python3 fedsgd_dlg_sim/main.py
```

Useful optional flags:

```bash
python3 fedsgd_dlg_sim/main.py \
  --download \
  --num-clients 2 \
  --victim-client-id 0 \
  --client-batch-size 1 \
  --server-lr 0.1 \
  --dlg-iters 300 \
  --dlg-lr 1.0 \
  --seed 0
```

## Outputs

The script saves these under `fedsgd_dlg_sim/outputs/` by default:

- `ground_truth.png`
- `reconstructed.png`
- `comparison.png`
- `dlg_loss_history.json`
- `dlg_loss_history.pt`
