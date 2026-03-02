import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from models.vision import LeNet, weights_init
from utils import cross_entropy_for_onehot, label_to_onehot


DEFAULT_ATTACK_ROUNDS = {5, 10, 15, 20, 25, 30}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def denormalize_for_viz(x: torch.Tensor, mean, std) -> torch.Tensor:
    if mean is None or std is None:
        return x
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return x * std_t + mean_t


def extract_client_gradient(model, x, y, device, num_classes=10):
    """Extract a single-step true gradient on the client side."""
    net = copy.deepcopy(model).to(device)
    net.eval()

    x = x.to(device)
    y = y.to(device)
    y_onehot = label_to_onehot(y, num_classes=num_classes)

    pred = net(x)
    loss = cross_entropy_for_onehot(pred, y_onehot)
    dy_dx = torch.autograd.grad(loss, net.parameters())
    return [g.detach().cpu().clone() for g in dy_dx]


def save_server_gradient_record(
    save_dir: Path,
    round_idx: int,
    client_id: int,
    gradients: List[torch.Tensor],
    model_state: Dict[str, torch.Tensor],
    num_classes: int = 10,
    input_shape: tuple = (1, 3, 32, 32),
    gt_data: Optional[torch.Tensor] = None,
    gt_label: Optional[int] = None,
) -> Path:
    """
    Save one server-side gradient record for later DLG attack.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"round_{round_idx:03d}_client_{client_id}.pt"

    payload = {
        "round": int(round_idx),
        "client_id": int(client_id),
        "num_classes": int(num_classes),
        "input_shape": tuple(input_shape),
        "gradients": [g.detach().cpu().clone() for g in gradients],
        "model_state": {k: v.detach().cpu().clone() for k, v in model_state.items()},
    }

    if gt_data is not None:
        payload["gt_data"] = gt_data.detach().cpu().clone()
    if gt_label is not None:
        payload["gt_label"] = int(gt_label)

    torch.save(payload, out_path)
    return out_path


def dlg_attack(target_gradients, model_state, gt_shape, num_classes, iters, device):
    net = LeNet().to(device)
    if model_state is not None:
        net.load_state_dict(model_state, strict=True)
    else:
        net.apply(weights_init)
    net.eval()

    criterion = cross_entropy_for_onehot
    dummy_data = torch.randn(gt_shape, device=device, requires_grad=True)
    dummy_label = torch.randn((gt_shape[0], num_classes), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [dummy_data, dummy_label],
    )

    loss_history = []
    for it in range(iters):

        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            soft_label = F.softmax(dummy_label, dim=-1)
            loss = criterion(pred, soft_label)
            dummy_dy_dx = torch.autograd.grad(loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, target_gradients):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

        if it % 10 == 0:
            with torch.enable_grad():
                pred = net(dummy_data)
                soft_label = F.softmax(dummy_label, dim=-1)
                loss = criterion(pred, soft_label)
                dummy_dy_dx = torch.autograd.grad(loss, net.parameters(), create_graph=False)
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, target_gradients):
                    grad_diff += ((gx - gy) ** 2).sum()
            loss_history.append((it, float(grad_diff.item())))
            print(f"  DLG iter {it:3d} | grad loss {float(grad_diff.item()):.6f}")

    with torch.no_grad():
        pred_class = int(torch.argmax(net(dummy_data), dim=1).item())
        dummy_class = int(torch.argmax(F.softmax(dummy_label, dim=-1), dim=1).item())
    return dummy_data.detach().cpu(), pred_class, dummy_class, loss_history


def run_attack_for_rounds(
    records_dir: Path,
    output_dir: Path,
    iters: int,
    rounds: Optional[List[int]] = None,
    clients: Optional[List[int]] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    attack_targets = []
    target_rounds = set(rounds) if rounds else set(DEFAULT_ATTACK_ROUNDS)

    for record_path in sorted(records_dir.glob("round_*_client_*.pt")):
        payload = torch.load(record_path, map_location="cpu", weights_only=False)
        rnd = int(payload["round"])
        cid = int(payload["client_id"])

        if rnd not in target_rounds:
            continue
        if clients is not None and cid not in clients:
            continue
        attack_targets.append((record_path, payload))

    if not attack_targets:
        print("No matching gradient records found for target rounds.")
        return

    device = get_device()
    summary = []
    print(f"Running DLG on device: {device}")
    print(f"Target rounds: {sorted(target_rounds)}")

    for record_path, payload in attack_targets:
        rnd = int(payload["round"])
        cid = int(payload["client_id"])
        num_classes = int(payload.get("num_classes", 10))
        input_shape = tuple(payload.get("input_shape", (1, 3, 32, 32)))
        norm_mean = payload.get("norm_mean")
        norm_std = payload.get("norm_std")
        gradients = [g.to(device) for g in payload["gradients"]]
        model_state = {k: v.to(device) for k, v in payload["model_state"].items()}

        print(f"\n[Attack] round={rnd}, client={cid}, file={record_path.name}")
        recon, pred_class, dummy_class, loss_history = dlg_attack(
            target_gradients=gradients,
            model_state=model_state,
            gt_shape=input_shape,
            num_classes=num_classes,
            iters=iters,
            device=device,
        )

        recon_file = output_dir / f"recon_round_{rnd:03d}_client_{cid}.png"
        recon_viz = denormalize_for_viz(recon, norm_mean, norm_std).clamp(0, 1)
        save_image(recon_viz, recon_file.as_posix())

        gt_saved = None
        if "gt_data" in payload:
            gt = payload["gt_data"]
            gt_file = output_dir / f"gt_round_{rnd:03d}_client_{cid}.png"
            gt_viz = denormalize_for_viz(gt, norm_mean, norm_std).clamp(0, 1)
            save_image(gt_viz, gt_file.as_posix())
            gt_saved = gt_file.name

        item = {
            "round": rnd,
            "client_id": cid,
            "source_record": record_path.name,
            "recon_file": recon_file.name,
            "gt_file": gt_saved,
            "pred_class": pred_class,
            "dummy_class": dummy_class,
            "gt_label": payload.get("gt_label"),
            "loss_history": loss_history,
        }
        summary.append(item)

    summary_path = output_dir / "dlg_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")


def parse_int_list(s: str) -> List[int]:
    if not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    if not s.strip():
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser("FL DLG demo (fixed rounds)")
    p.add_argument("--records_dir", type=str, default="sparsification/dlg_records")
    p.add_argument("--output_dir", type=str, default="sparsification/dlg_outputs")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument(
        "--pt_files",
        type=str,
        default="",
        help="Comma-separated .pt files to attack directly. If set, rounds filter is ignored.",
    )
    p.add_argument(
        "--rounds",
        type=str,
        default="5,10,15,20,25,30",
        help="Comma-separated rounds to attack. Example: 5,10,15",
    )
    p.add_argument(
        "--clients",
        type=str,
        default="",
        help="Comma-separated client ids to attack. Empty means all.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rounds = parse_int_list(args.rounds)
    clients = parse_int_list(args.clients)
    pt_files = parse_str_list(args.pt_files)

    if pt_files:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        device = get_device()
        summary = []

        print(f"Running DLG on device: {device}")
        print(f"Direct .pt mode ({len(pt_files)} files)")

        for raw_path in pt_files:
            record_path = Path(raw_path)
            if not record_path.is_absolute():
                record_path = Path.cwd() / record_path
            if not record_path.exists():
                print(f"Skip missing file: {record_path}")
                continue

            payload = torch.load(record_path, map_location="cpu", weights_only=False)
            rnd = int(payload["round"])
            cid = int(payload["client_id"])
            num_classes = int(payload.get("num_classes", 10))
            input_shape = tuple(payload.get("input_shape", (1, 3, 32, 32)))
            norm_mean = payload.get("norm_mean")
            norm_std = payload.get("norm_std")
            gradients = [g.to(device) for g in payload["gradients"]]
            model_state = {k: v.to(device) for k, v in payload["model_state"].items()}

            print(f"\n[Attack] round={rnd}, client={cid}, file={record_path.name}")
            recon, pred_class, dummy_class, loss_history = dlg_attack(
                target_gradients=gradients,
                model_state=model_state,
                gt_shape=input_shape,
                num_classes=num_classes,
                iters=args.iters,
                device=device,
            )

            recon_file = output_dir / f"recon_round_{rnd:03d}_client_{cid}.png"
            recon_viz = denormalize_for_viz(recon, norm_mean, norm_std).clamp(0, 1)
            save_image(recon_viz, recon_file.as_posix())

            gt_saved = None
            if "gt_data" in payload:
                gt = payload["gt_data"]
                gt_file = output_dir / f"gt_round_{rnd:03d}_client_{cid}.png"
                gt_viz = denormalize_for_viz(gt, norm_mean, norm_std).clamp(0, 1)
                save_image(gt_viz, gt_file.as_posix())
                gt_saved = gt_file.name

            item = {
                "round": rnd,
                "client_id": cid,
                "source_record": record_path.name,
                "recon_file": recon_file.name,
                "gt_file": gt_saved,
                "pred_class": pred_class,
                "dummy_class": dummy_class,
                "gt_label": payload.get("gt_label"),
                "loss_history": loss_history,
            }
            summary.append(item)

        summary_path = output_dir / "dlg_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved summary: {summary_path}")
        return

    run_attack_for_rounds(
        records_dir=Path(args.records_dir),
        output_dir=Path(args.output_dir),
        iters=args.iters,
        rounds=rounds if rounds else None,
        clients=clients if clients else None,
    )


if __name__ == "__main__":
    main()
