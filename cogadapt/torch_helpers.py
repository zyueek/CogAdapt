"""Selected experimental functions; see provenance/code_excerpts.json.

This review module omits model loading, datasets, and experiment runners.
"""

from __future__ import annotations
from typing import Any
import torch
import torch.nn.functional as functional


def weighted_causal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor,
    task_weights: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous().to(logits.device)
    shifted_weights = token_weights[:, 1:].contiguous().to(logits.device)
    losses = functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shifted_labels)
    mask = shifted_labels.ne(-100)
    per_example = (losses * shifted_weights * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return (per_example * task_weights.to(logits.device)).mean()


def _mask_trainable(
    parameters_by_block: dict[int, list[torch.nn.Parameter]], active_blocks: set[int]
) -> None:
    for block, parameters in parameters_by_block.items():
        enabled = block in active_blocks
        for parameter in parameters:
            parameter.requires_grad = enabled


def _activate_blocks(
    active: set[int],
    routers: dict[int, torch.nn.Parameter],
    base_gates: dict[int, torch.Tensor],
    trained_gates: dict[int, torch.Tensor],
    lora_by_block: dict[int, list[Any]],
) -> None:
    for block, modules in lora_by_block.items():
        scale = 1.0 if block in active else 0.0
        for module in modules:
            for adapter in module.scaling:
                module.set_scale(adapter, scale)
    for block, parameter in routers.items():
        source = (
            trained_gates[block]
            if block in active and block in trained_gates
            else base_gates[block]
        )
        parameter.data.copy_(source.to(parameter.device, parameter.dtype))


def linear_cka(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """Exact float32 linear CKA formula used by the reference decoder script."""
    if features_x.ndim != 2 or features_y.ndim != 2:
        raise ValueError("features must be 2D (n_examples x dim)")
    if features_x.shape[0] != features_y.shape[0]:
        raise ValueError("CKA feature matrices have different example counts")
    x = features_x.float()
    y = features_y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    k_xy = x.t().mm(y)
    k_xx = x.t().mm(x)
    k_yy = y.t().mm(y)
    dot_xy = torch.norm(k_xy).pow(2)
    denominator = (torch.norm(k_xx) * torch.norm(k_yy)).clamp(1e-12)
    return float((dot_xy / denominator).item())
