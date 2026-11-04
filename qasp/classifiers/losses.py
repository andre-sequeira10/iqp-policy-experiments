"""Differentiable objectives for discrete quantum classifiers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


_ALIASES = {
    "mse": "mean_squared_error",
    "mean_squared_error": "mean_squared_error",
    "hinge": "hinge",
    "hinge_loss": "hinge",
    "binary_cross_entropy": "binary_cross_entropy",
    "binary_ce": "binary_cross_entropy",
    "bce": "binary_cross_entropy",
    "categorical_cross_entropy": "categorical_cross_entropy",
    "cross_entropy": "categorical_cross_entropy",
    "cce": "categorical_cross_entropy",
    "nll": "categorical_cross_entropy",
}


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    """Apply none, mean or sum reduction to loss values; reject unknown reduction names."""
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def _class_indices(targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Convert floating one-hot/soft targets with final dimension n_classes by argmax; otherwise cast targets to integer labels."""
    targets = torch.as_tensor(targets)
    if targets.ndim > 0 and targets.shape[-1] == n_classes and targets.is_floating_point():
        targets = targets.argmax(dim=-1)
    return targets.to(torch.long)


def _multiclass_outputs(outputs: torch.Tensor, targets: torch.Tensor) -> bool:
    """Detect multiclass output layout from batch rank or a scalar target paired with multiple scores."""
    if outputs.ndim >= 2:
        return outputs.shape[-1] > 1
    return targets.ndim == 0 and outputs.numel() > 1


def mean_squared_error(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE with one-hot multiclass and ``{-1, +1}`` binary targets."""
    outputs = torch.as_tensor(outputs)
    targets = torch.as_tensor(targets, device=outputs.device)
    if _multiclass_outputs(outputs, targets):
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(0)
        indices = _class_indices(targets, outputs.shape[-1]).reshape(-1)
        encoded = F.one_hot(indices, num_classes=outputs.shape[-1]).to(outputs.dtype)
        values = (outputs - encoded).square().mean(dim=-1)
    else:
        scores = outputs.squeeze(-1)
        encoded = 2.0 * targets.to(outputs.dtype) - 1.0
        values = (scores - encoded).square()
    return _reduce(values, reduction)


def hinge_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary hinge or multiclass Crammer-Singer hinge loss."""
    scores = torch.as_tensor(scores)
    targets = torch.as_tensor(targets, device=scores.device)
    if _multiclass_outputs(scores, targets):
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        indices = _class_indices(targets, scores.shape[-1]).reshape(-1)
        true_scores = scores.gather(1, indices[:, None]).squeeze(1)
        mask = F.one_hot(indices, num_classes=scores.shape[-1]).bool()
        other_scores = scores.masked_fill(mask, -torch.inf).max(dim=-1).values
        values = torch.relu(float(margin) + other_scores - true_scores)
    else:
        binary_scores = scores.squeeze(-1)
        signed_targets = 2.0 * targets.to(scores.dtype) - 1.0
        values = torch.relu(float(margin) - signed_targets * binary_scores)
    return _reduce(values, reduction)


def binary_cross_entropy(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary cross entropy for positive-class or two-class probabilities."""
    probabilities = torch.as_tensor(probabilities)
    targets = torch.as_tensor(targets, device=probabilities.device)
    if probabilities.ndim >= 2 and probabilities.shape[-1] == 2:
        positive = probabilities[..., 1]
    elif probabilities.ndim == 1 and targets.ndim == 0 and probabilities.numel() == 2:
        positive = probabilities[1]
    else:
        positive = probabilities.squeeze(-1)
    if eps is None:
        eps = torch.finfo(positive.dtype).eps
    positive = positive.clamp(min=eps, max=1.0 - eps)
    values = -(
        targets.to(positive.dtype) * torch.log(positive)
        + (1.0 - targets.to(positive.dtype)) * torch.log1p(-positive)
    )
    return _reduce(values, reduction)


def categorical_cross_entropy(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Categorical cross entropy for normalized class probabilities."""
    probabilities = torch.as_tensor(probabilities)
    targets = torch.as_tensor(targets, device=probabilities.device)
    if probabilities.ndim == 1:
        probabilities = probabilities.unsqueeze(0)
    if probabilities.shape[-1] < 2:
        raise ValueError("categorical_cross_entropy requires at least two classes.")
    indices = _class_indices(targets, probabilities.shape[-1]).reshape(-1)
    if probabilities.shape[0] != indices.numel():
        raise ValueError("The batch dimensions of probabilities and targets do not match.")
    if eps is None:
        eps = torch.finfo(probabilities.dtype).eps
    selected = probabilities.gather(1, indices[:, None]).squeeze(1).clamp_min(eps)
    return _reduce(-torch.log(selected), reduction)


def classification_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    objective: str,
    *,
    reduction: str = "mean",
    margin: float = 1.0,
) -> torch.Tensor:
    """Dispatch to a named classifier objective."""
    if objective not in _ALIASES:
        supported = ", ".join(sorted(_ALIASES))
        raise ValueError(
            f"Unknown objective '{objective}'. Supported values: {supported}."
        )
    canonical = _ALIASES[objective]
    if canonical == "mean_squared_error":
        return mean_squared_error(outputs, targets, reduction=reduction)
    if canonical == "hinge":
        return hinge_loss(outputs, targets, margin=margin, reduction=reduction)
    if canonical == "binary_cross_entropy":
        return binary_cross_entropy(outputs, targets, reduction=reduction)
    return categorical_cross_entropy(outputs, targets, reduction=reduction)


class ClassifierLoss(nn.Module):
    """PyTorch module wrapper around :func:`classification_loss`."""

    def __init__(
        self,
        objective: str,
        *,
        reduction: str = "mean",
        margin: float = 1.0,
    ) -> None:
        """Configure a stateless loss module using a registered objective alias.

        ``reduction`` selects none, mean or sum when the loss is evaluated; ``margin``
        is used by hinge objectives. Unknown objectives raise ValueError immediately.
        This module owns no trainable parameters."""
        super().__init__()
        if objective not in _ALIASES:
            supported = ", ".join(sorted(_ALIASES))
            raise ValueError(
                f"Unknown objective '{objective}'. Supported values: {supported}."
            )
        self.objective = _ALIASES[objective]
        self.reduction = reduction
        self.margin = float(margin)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Evaluate classification_loss on outputs and targets with the configured objective, reduction and hinge margin."""
        return classification_loss(
            outputs,
            targets,
            self.objective,
            reduction=self.reduction,
            margin=self.margin,
        )


__all__ = [
    "ClassifierLoss",
    "binary_cross_entropy",
    "categorical_cross_entropy",
    "classification_loss",
    "hinge_loss",
    "mean_squared_error",
]
