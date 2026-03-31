# -*- coding: utf-8 -*-
import argparse

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from utils import label_to_onehot, cross_entropy_for_onehot
from models.vision import LeNet, weights_init


FRONT_RATIO = 0.1
BACK_RATIO = 0.9
LAYER_ORDER = ["body.0", "body.2", "body.4", "fc.0"]
FRONT_LAYERS = {"body.0", "body.2"}
BACK_LAYERS = {"body.4", "fc.0"}


def layer_name_from_param_name(param_name):
    parts = param_name.split(".")
    if len(parts) < 2:
        raise ValueError(f"Unsupported parameter name: {param_name}")

    layer_name = ".".join(parts[:2])
    if layer_name not in FRONT_LAYERS | BACK_LAYERS:
        raise ValueError(f"Unexpected LeNet layer: {param_name}")
    return layer_name


def retention_ratio_for_layer(layer_name):
    if layer_name in FRONT_LAYERS:
        return FRONT_RATIO
    if layer_name in BACK_LAYERS:
        return BACK_RATIO
    raise ValueError(f"Unknown layer name: {layer_name}")


def sparsify_gradients_layerwise_topk(gradients, param_names):
    if len(gradients) != len(param_names):
        raise ValueError("gradients and param_names must have the same length.")

    grouped = {layer_name: [] for layer_name in LAYER_ORDER}
    for idx, (param_name, grad_tensor) in enumerate(zip(param_names, gradients)):
        grouped[layer_name_from_param_name(param_name)].append((idx, grad_tensor))

    sparse_grads = [None] * len(gradients)
    layer_stats = []
    total_kept = 0
    total_params = 0

    for layer_name in LAYER_ORDER:
        entries = grouped[layer_name]
        flat = torch.cat([grad_tensor.reshape(-1) for _, grad_tensor in entries])
        total = flat.numel()
        ratio = retention_ratio_for_layer(layer_name)
        kept = total if ratio >= 1.0 else max(1, int(total * ratio))

        if kept >= total:
            sparse_flat = flat.clone()
        else:
            _, topk_idx = torch.topk(flat.abs(), kept, largest=True, sorted=False)
            mask = torch.zeros_like(flat)
            mask[topk_idx] = 1.0
            sparse_flat = flat * mask

        offset = 0
        for idx, grad_tensor in entries:
            n = grad_tensor.numel()
            sparse_grads[idx] = sparse_flat[offset : offset + n].view_as(grad_tensor)
            offset += n

        total_kept += kept
        total_params += total
        layer_stats.append(
            {
                "layer": layer_name,
                "kept": kept,
                "total": total,
                "retention_ratio": kept / float(total),
            }
        )

    stats = {
        "kept": total_kept,
        "total": total_params,
        "retention_ratio": total_kept / float(total_params),
        "layer_stats": layer_stats,
    }
    return sparse_grads, stats


def main():
    print(torch.__version__, torchvision.__version__)

    parser = argparse.ArgumentParser(description="Deep Leakage from Gradients with layer-wise sparsification.")
    parser.add_argument("--index", type=int, default="25", help="the index for leaking images on CIFAR.")
    parser.add_argument("--image", type=str, default="", help="the path to customized image.")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print("Running on %s" % device)

    dst = datasets.CIFAR10("~/.torch", download=True)
    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()

    img_index = args.index
    gt_data = tp(dst[img_index][0]).to(device)

    if len(args.image) > 1:
        gt_data = Image.open(args.image)
        gt_data = tp(gt_data).to(device)

    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1,)
    gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

    plt.imshow(tt(gt_data[0].cpu()))

    net = LeNet().to(device)
    param_names = [name for name, _ in net.named_parameters()]

    torch.manual_seed(1234)

    net.apply(weights_init)
    criterion = cross_entropy_for_onehot

    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = [grad_tensor.detach().clone() for grad_tensor in dy_dx]
    original_dy_dx, sparse_stats = sparsify_gradients_layerwise_topk(original_dy_dx, param_names)

    print("[Layer-wise sparsification]")
    for layer_stat in sparse_stats["layer_stats"]:
        print(
            f"  {layer_stat['layer']} kept "
            f"{layer_stat['kept']}/{layer_stat['total']} "
            f"({layer_stat['retention_ratio'] * 100:.2f}%)"
        )
    print(
        "  total kept "
        f"{sparse_stats['kept']}/{sparse_stats['total']} "
        f"({sparse_stats['retention_ratio'] * 100:.2f}%)"
    )

    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    plt.imshow(tt(dummy_data[0].cpu()))

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    history = []
    for iters in range(300):
        def closure():
            optimizer.zero_grad()

            dummy_pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        if iters % 10 == 0:
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
            history.append(tt(dummy_data[0].cpu()))

    plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * 10))
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
