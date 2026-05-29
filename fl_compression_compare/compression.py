from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import torch


TensorDict = OrderedDict[str, torch.Tensor]


def clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> TensorDict:
    cloned: TensorDict = OrderedDict()
    for key, tensor in state_dict.items():
        cloned[key] = tensor.detach().cpu().clone()
    return cloned


def subtract_state_dicts(
    local_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
) -> TensorDict:
    delta: TensorDict = OrderedDict()
    for key, local_tensor in local_state.items():
        if not local_tensor.is_floating_point():
            continue
        delta[key] = local_tensor.detach().cpu() - global_state[key].detach().cpu()
    return delta


def add_delta_to_model(model: torch.nn.Module, delta: Mapping[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    updated_state = OrderedDict()

    for key, tensor in model_state.items():
        if key in delta:
            updated_state[key] = tensor + delta[key].to(tensor.device, dtype=tensor.dtype)
        else:
            updated_state[key] = tensor

    model.load_state_dict(updated_state, strict=True)


def aggregate_deltas(
    client_deltas: Iterable[Mapping[str, torch.Tensor]],
    client_sizes: Iterable[int],
) -> TensorDict:
    client_deltas = list(client_deltas)
    client_sizes = list(client_sizes)
    if not client_deltas:
        raise ValueError("client_deltas must contain at least one update.")

    total_size = float(sum(client_sizes))
    if total_size <= 0:
        raise ValueError("client_sizes must sum to a positive value.")

    aggregated: TensorDict = OrderedDict()
    for key in client_deltas[0].keys():
        weighted = None
        for delta, size in zip(client_deltas, client_sizes):
            term = delta[key].float() * (float(size) / total_size)
            weighted = term if weighted is None else weighted + term
        aggregated[key] = weighted
    return aggregated


def _flatten_tensor_dict(delta: Mapping[str, torch.Tensor]):
    keys: List[str] = []
    shapes: List[torch.Size] = []
    dtypes: List[torch.dtype] = []
    sizes: List[int] = []
    flat_parts: List[torch.Tensor] = []

    for key, tensor in delta.items():
        tensor_cpu = tensor.detach().cpu()
        keys.append(key)
        shapes.append(tensor_cpu.shape)
        dtypes.append(tensor_cpu.dtype)
        sizes.append(tensor_cpu.numel())
        flat_parts.append(tensor_cpu.reshape(-1).float())

    if flat_parts:
        flat = torch.cat(flat_parts, dim=0)
    else:
        flat = torch.empty(0, dtype=torch.float32)

    metadata = {
        "keys": keys,
        "shapes": shapes,
        "dtypes": dtypes,
        "sizes": sizes,
    }
    return flat, metadata


def _unflatten_tensor_dict(flat: torch.Tensor, metadata: Mapping[str, List]) -> TensorDict:
    restored: TensorDict = OrderedDict()
    offset = 0

    for key, shape, dtype, size in zip(
        metadata["keys"],
        metadata["shapes"],
        metadata["dtypes"],
        metadata["sizes"],
    ):
        chunk = flat[offset : offset + size].view(shape).to(dtype=dtype)
        restored[key] = chunk
        offset += size

    return restored


def quantize_symmetric(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:
        return tensor.clone()
    if bits < 1:
        raise ValueError("bits must be >= 1.")
    if tensor.numel() == 0:
        return tensor.clone()

    max_abs = tensor.abs().max()
    if float(max_abs.item()) == 0.0:
        return tensor.clone()

    if bits == 1:
        return torch.sign(tensor) * max_abs

    levels = (2 ** (bits - 1)) - 1
    normalized = torch.clamp(tensor / max_abs, min=-1.0, max=1.0)
    quantized = torch.round(normalized * levels) / float(levels)
    return quantized * max_abs


def sparsify_topk(flat: torch.Tensor, keep_ratio: float) -> Tuple[torch.Tensor, int]:
    if keep_ratio < 0.0 or keep_ratio > 1.0:
        raise ValueError("keep_ratio must satisfy 0 <= keep_ratio <= 1.")
    if flat.numel() == 0:
        return flat.clone(), 0

    total = flat.numel()
    if keep_ratio == 0.0:
        return torch.zeros_like(flat), 0

    kept = total if keep_ratio >= 1.0 else max(1, int(total * keep_ratio))
    if kept >= total:
        return flat.clone(), total

    _, topk_idx = torch.topk(flat.abs(), kept, largest=True, sorted=False)
    sparse = torch.zeros_like(flat)
    sparse[topk_idx] = flat[topk_idx]
    return sparse, kept


def compress_delta(
    delta: Mapping[str, torch.Tensor],
    keep_ratio: float,
    quant_bits: int,
) -> Tuple[TensorDict, Dict[str, float]]:
    flat, metadata = _flatten_tensor_dict(delta)
    sparse_flat, kept = sparsify_topk(flat, keep_ratio)
    compressed_flat = quantize_symmetric(sparse_flat, quant_bits)
    compressed_delta = _unflatten_tensor_dict(compressed_flat, metadata)

    total = int(flat.numel())
    index_bits = int(math.ceil(math.log2(max(2, total))))
    bits_per_value = int(quant_bits if quant_bits < 32 else 32)
    if kept >= total:
        upload_bits = kept * bits_per_value
    else:
        upload_bits = kept * (bits_per_value + index_bits)

    dense_upload_bits = total * 32
    quantization_mse = 0.0
    compression_mse = 0.0
    if total > 0:
        quantization_mse = float(torch.mean((compressed_flat - sparse_flat) ** 2).item())
        compression_mse = float(torch.mean((compressed_flat - flat) ** 2).item())

    stats = {
        "num_params": float(total),
        "kept_params": float(kept),
        "retention_ratio": float(kept / total) if total else 0.0,
        "quantization_bits": float(quant_bits),
        "upload_bits": float(upload_bits),
        "dense_upload_bits": float(dense_upload_bits),
        "relative_upload": float(upload_bits / dense_upload_bits) if dense_upload_bits else 0.0,
        "quantization_mse": quantization_mse,
        "compression_mse": compression_mse,
        "nonzero_params": float(torch.count_nonzero(compressed_flat).item()),
    }
    return compressed_delta, stats


def mean_stats(stats_rows: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    rows = list(stats_rows)
    if not rows:
        return {}

    result: Dict[str, float] = {}
    for key in rows[0].keys():
        result[key] = float(sum(float(row[key]) for row in rows) / len(rows))
    return result
