"""Dependency-light evaluation metrics for QASP classifiers."""

from __future__ import annotations

import torch

from .losses import categorical_cross_entropy


def class_predictions(outputs: torch.Tensor, *, threshold: float = 0.0) -> torch.Tensor:
    """Convert binary scores or class-valued outputs to integer predictions."""
    outputs = torch.as_tensor(outputs)
    if outputs.ndim >= 2 and outputs.shape[-1] > 1:
        return outputs.argmax(dim=-1).to(torch.long)
    if outputs.ndim == 1 and outputs.numel() > 1:
        return (outputs > threshold).to(torch.long)
    return (outputs.squeeze(-1) > threshold).to(torch.long)


def confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    n_classes: int | None = None,
) -> torch.Tensor:
    """Count predictions with true classes in rows and predicted classes in columns.

    Flatten equally sized, nonempty integer-label inputs. Return an int64 tensor
    of shape (n_classes, n_classes); infer n_classes from the largest label when
    omitted. Labels must lie in [0, n_classes), with at least two classes. Inputs
    must reside on compatible devices. Invalid sizes or labels raise ValueError."""
    targets = torch.as_tensor(targets).reshape(-1).to(torch.long)
    predictions = torch.as_tensor(predictions).reshape(-1).to(torch.long)
    if targets.numel() != predictions.numel():
        raise ValueError("targets and predictions must contain the same number of items.")
    if targets.numel() == 0:
        raise ValueError("targets and predictions must not be empty.")
    if n_classes is None:
        n_classes = int(torch.cat((targets, predictions)).max().item()) + 1
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2.")
    if torch.any(targets < 0) or torch.any(targets >= n_classes):
        raise ValueError("targets contain a class outside [0, n_classes).")
    if torch.any(predictions < 0) or torch.any(predictions >= n_classes):
        raise ValueError("predictions contain a class outside [0, n_classes).")
    encoded = targets * n_classes + predictions
    return torch.bincount(encoded, minlength=n_classes * n_classes).reshape(
        n_classes, n_classes
    )


def accuracy_score(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """Return the float64 mean of elementwise label matches after flattening.

    Targets and predictions must contain the same number of items; unequal sizes
    raise ValueError. Empty inputs yield NaN through the underlying mean."""
    targets = torch.as_tensor(targets).reshape(-1)
    predictions = torch.as_tensor(predictions).reshape(-1)
    if targets.numel() != predictions.numel():
        raise ValueError("targets and predictions must contain the same number of items.")
    return (targets == predictions).to(torch.float64).mean()


def _per_class_statistics(matrix: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return per-class precision, recall, F1 and row support from a count matrix; undefined ratios contribute zero."""
    matrix = matrix.to(torch.float64)
    true_positive = matrix.diag()
    false_positive = matrix.sum(dim=0) - true_positive
    false_negative = matrix.sum(dim=1) - true_positive
    support = matrix.sum(dim=1)
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float64).eps)
    return precision, recall, f1, support


def _average(values: torch.Tensor, support: torch.Tensor, average: str | None) -> torch.Tensor:
    """Reduce class values by macro mean or support-weighted mean; None/none retains the vector."""
    if average is None or average == "none":
        return values
    if average == "macro":
        return values.mean()
    if average == "weighted":
        return (values * support).sum() / support.sum().clamp_min(1.0)
    raise ValueError("average must be 'macro', 'weighted', or None.")


def precision_score(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    n_classes: int | None = None,
    average: str | None = "macro",
) -> torch.Tensor:
    """Compute per-class precision with zero for classes never predicted.

    Targets/predictions are integer labels. ``average`` is macro (default),
    weighted by target support, or None/none for a class vector. ``n_classes``
    fixes matrix size or is inferred as in confusion_matrix. Returns a tensor."""
    matrix = confusion_matrix(targets, predictions, n_classes)
    precision, _, _, support = _per_class_statistics(matrix)
    return _average(precision, support, average)


def recall_score(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    n_classes: int | None = None,
    average: str | None = "macro",
) -> torch.Tensor:
    """Compute per-class recall with zero for classes absent from targets.

    Accept integer labels and optional n_classes. Return a float64 scalar for
    macro/weighted averaging or a class vector for average=None/none. Macro
    averaging includes all configured classes, including zero-support classes."""
    matrix = confusion_matrix(targets, predictions, n_classes)
    _, recall, _, support = _per_class_statistics(matrix)
    return _average(recall, support, average)


def f1_score(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    n_classes: int | None = None,
    average: str | None = "macro",
) -> torch.Tensor:
    """Return the harmonic mean of class precision and recall.

    Integer targets/predictions use confusion_matrix validation. average may be
    macro, target-support weighted, or None/none. Undefined class F1 is zero;
    return a float64 scalar or vector according to the averaging mode."""
    matrix = confusion_matrix(targets, predictions, n_classes)
    _, _, f1, support = _per_class_statistics(matrix)
    return _average(f1, support, average)


def balanced_accuracy_score(
    targets: torch.Tensor, predictions: torch.Tensor, *, n_classes: int | None = None
) -> torch.Tensor:
    """Return macro recall over all inferred or explicitly configured classes, including zero-support classes."""
    return recall_score(
        targets, predictions, n_classes=n_classes, average="macro"
    )


def top_k_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, *, k: int = 1
) -> torch.Tensor:
    """Return the fraction of targets among the k largest output scores.

    Outputs have shape (batch, classes), or (classes,) for one sample; targets
    contain one integer class per row. Scores need not be probabilities. Reject
    k outside [1, classes]. Returns a float64 scalar tensor."""
    outputs = torch.as_tensor(outputs)
    targets = torch.as_tensor(targets, device=outputs.device).reshape(-1).to(torch.long)
    if outputs.ndim == 1:
        outputs = outputs.unsqueeze(0)
    if not 1 <= k <= outputs.shape[-1]:
        raise ValueError(f"k must be between 1 and {outputs.shape[-1]}.")
    top_k = outputs.topk(k, dim=-1).indices
    return (top_k == targets[:, None]).any(dim=-1).to(torch.float64).mean()


def log_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return mean categorical cross-entropy from class probabilities and target labels; see categorical_cross_entropy for target conversion and clipping."""
    return categorical_cross_entropy(probabilities, targets)


def brier_score(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the mean sum of squared errors against one-hot class labels.

    Probabilities have shape (batch, classes) or (classes,) for one sample.
    Targets are integer class indices. This is the multiclass, class-summed
    Brier score; inputs are not normalized or clipped by this function."""
    probabilities = torch.as_tensor(probabilities)
    targets = torch.as_tensor(targets, device=probabilities.device).to(torch.long)
    if probabilities.ndim == 1:
        probabilities = probabilities.unsqueeze(0)
    encoded = torch.nn.functional.one_hot(
        targets.reshape(-1), num_classes=probabilities.shape[-1]
    ).to(probabilities.dtype)
    return (probabilities - encoded).square().sum(dim=-1).mean()


def _binary_auc_inputs(
    targets: torch.Tensor,
    scores: torch.Tensor,
    positive_label: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate binary labels/scores, select the positive-class column when needed, and return binary targets with float64 scores."""
    scores = torch.as_tensor(scores)
    targets = torch.as_tensor(targets, device=scores.device).reshape(-1)
    if scores.ndim == 2:
        if scores.shape[1] == 1:
            scores = scores[:, 0]
        elif scores.shape[1] != 2:
            raise ValueError(
                "Binary AUC scores must be one-dimensional or have two class columns."
            )
        elif positive_label not in {0, 1}:
            raise ValueError(
                "positive_label must be 0 or 1 when scores contain two class columns."
            )
        else:
            scores = scores[:, positive_label]
    else:
        scores = scores.reshape(-1)
    if targets.numel() != scores.numel():
        raise ValueError("targets and scores must contain the same number of items.")
    if targets.numel() == 0:
        raise ValueError("targets and scores must not be empty.")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must contain only finite values.")
    return (targets == positive_label).to(torch.long), scores.to(torch.float64)


def _binary_threshold_counts(
    binary_targets: torch.Tensor, scores: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cumulative true/false positives at distinct descending score thresholds; tied scores share a threshold."""
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = binary_targets[order].to(torch.float64)
    true_positives = sorted_targets.cumsum(dim=0)
    false_positives = (1.0 - sorted_targets).cumsum(dim=0)

    # Evaluate once at the end of every tied-score group. This gives ties the
    # same threshold and matches sklearn's rank/threshold interpretation.
    group_end = torch.ones_like(sorted_scores, dtype=torch.bool)
    group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    indices = torch.nonzero(group_end, as_tuple=False).squeeze(1)
    return true_positives[indices], false_positives[indices]


def _binary_roc_auc(
    targets: torch.Tensor, scores: torch.Tensor, positive_label: int = 1
) -> torch.Tensor:
    """Integrate the binary ROC curve by trapezoids; return NaN when either target class is absent."""
    binary_targets, scores = _binary_auc_inputs(targets, scores, positive_label)
    positives = binary_targets.sum().to(torch.float64)
    negatives = (binary_targets.numel() - binary_targets.sum()).to(torch.float64)
    if positives == 0 or negatives == 0:
        return scores.new_tensor(float("nan"))

    true_positives, false_positives = _binary_threshold_counts(
        binary_targets, scores
    )
    true_positive_rate = torch.cat(
        (scores.new_zeros(1), true_positives / positives)
    )
    false_positive_rate = torch.cat(
        (scores.new_zeros(1), false_positives / negatives)
    )
    widths = false_positive_rate[1:] - false_positive_rate[:-1]
    heights = 0.5 * (true_positive_rate[1:] + true_positive_rate[:-1])
    return (widths * heights).sum()


def _binary_average_precision(
    targets: torch.Tensor, scores: torch.Tensor, positive_label: int = 1
) -> torch.Tensor:
    """Integrate binary precision over recall increments; return NaN when no positive targets exist."""
    binary_targets, scores = _binary_auc_inputs(targets, scores, positive_label)
    positives = binary_targets.sum().to(torch.float64)
    if positives == 0:
        return scores.new_tensor(float("nan"))

    true_positives, false_positives = _binary_threshold_counts(
        binary_targets, scores
    )
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / positives
    previous_recall = torch.cat((scores.new_zeros(1), recall[:-1]))
    # Step-wise integration is average precision, the standard PR-area summary
    # for imbalanced classification and sklearn's average_precision_score.
    return ((recall - previous_recall) * precision).sum()


def _multiclass_auc(
    targets: torch.Tensor,
    scores: torch.Tensor,
    metric,
    average: str | None,
) -> torch.Tensor:
    """Apply a binary area metric one versus rest; support micro, macro, support-weighted or per-class outputs and omit undefined classes from macro/weighted means."""
    scores = torch.as_tensor(scores)
    if scores.ndim != 2 or scores.shape[1] < 3:
        raise ValueError("Multiclass AUC scores must have shape [n_samples, n_classes].")
    targets = torch.as_tensor(targets, device=scores.device)
    n_classes = scores.shape[1]
    if targets.ndim == 2:
        if targets.shape != scores.shape:
            raise ValueError("One-hot targets must have the same shape as scores.")
        one_hot = targets.to(torch.long)
        class_indices = targets.argmax(dim=-1)
    else:
        class_indices = targets.reshape(-1).to(torch.long)
        if class_indices.numel() != scores.shape[0]:
            raise ValueError("targets and scores must have the same batch size.")
        if torch.any(class_indices < 0) or torch.any(class_indices >= n_classes):
            raise ValueError("targets contain a class outside the score columns.")
        one_hot = torch.nn.functional.one_hot(
            class_indices, num_classes=n_classes
        )

    if average == "micro":
        return metric(one_hot.reshape(-1), scores.reshape(-1), positive_label=1)

    values = torch.stack(
        [
            metric(one_hot[:, class_index], scores[:, class_index], positive_label=1)
            for class_index in range(n_classes)
        ]
    )
    if average is None or average == "none":
        return values
    valid = ~torch.isnan(values)
    if not valid.any():
        return values.new_tensor(float("nan"))
    if average == "macro":
        return values[valid].mean()
    if average == "weighted":
        support = torch.bincount(class_indices, minlength=n_classes).to(values.dtype)
        weights = support[valid]
        return (values[valid] * weights).sum() / weights.sum().clamp_min(1.0)
    raise ValueError("average must be 'macro', 'weighted', 'micro', or None.")


def roc_auc_score(
    targets: torch.Tensor,
    scores: torch.Tensor,
    *,
    average: str | None = "macro",
    positive_label: int = 1,
) -> torch.Tensor:
    """Compute binary or multiclass one-vs-rest ROC AUC.

    Binary ``scores`` may contain one positive-class score per sample or two
    class-probability columns. Multiclass scores must contain one column per
    class and support ``macro``, ``weighted``, ``micro``, or per-class output.
    Undefined class-wise ROC areas are represented by ``nan`` and excluded from
    macro/weighted averages.
    """
    scores = torch.as_tensor(scores)
    if scores.ndim == 2 and scores.shape[1] > 2:
        return _multiclass_auc(targets, scores, _binary_roc_auc, average)
    return _binary_roc_auc(targets, scores, positive_label)


def pr_auc_score(
    targets: torch.Tensor,
    scores: torch.Tensor,
    *,
    average: str | None = "macro",
    positive_label: int = 1,
) -> torch.Tensor:
    """Compute binary or multiclass one-vs-rest PR AUC.

    QASP uses step-wise average precision for PR AUC because it does not apply
    optimistic linear interpolation between operating points. This is the same
    definition used by sklearn's ``average_precision_score`` and is especially
    useful when the positive class is rare.
    """
    scores = torch.as_tensor(scores)
    if scores.ndim == 2 and scores.shape[1] > 2:
        return _multiclass_auc(
            targets, scores, _binary_average_precision, average
        )
    return _binary_average_precision(targets, scores, positive_label)


def average_precision_score(
    targets: torch.Tensor,
    scores: torch.Tensor,
    *,
    average: str | None = "macro",
    positive_label: int = 1,
) -> torch.Tensor:
    """Alias for :func:`pr_auc_score` using sklearn terminology."""
    return pr_auc_score(
        targets,
        scores,
        average=average,
        positive_label=positive_label,
    )


def classification_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    n_classes: int | None = None,
    probabilities: torch.Tensor | None = None,
    auc_average: str | None = "macro",
) -> dict[str, torch.Tensor]:
    """Return aggregate/per-class metrics and an optional AUC evaluation.

    Supplying class ``probabilities`` adds ``roc_auc`` and ``pr_auc``. Binary
    AUC uses class 1 as positive; multiclass AUC uses one-vs-rest averaging.
    """
    matrix = confusion_matrix(targets, predictions, n_classes)
    precision, recall, f1, support = _per_class_statistics(matrix)
    result = {
        "accuracy": accuracy_score(targets, predictions),
        "balanced_accuracy": recall.mean(),
        "precision_macro": precision.mean(),
        "recall_macro": recall.mean(),
        "f1_macro": f1.mean(),
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "support": support.to(torch.long),
        "confusion_matrix": matrix,
    }
    if probabilities is not None:
        result["roc_auc"] = roc_auc_score(
            targets, probabilities, average=auc_average
        )
        result["pr_auc"] = pr_auc_score(
            targets, probabilities, average=auc_average
        )
    return result


__all__ = [
    "accuracy_score",
    "average_precision_score",
    "balanced_accuracy_score",
    "brier_score",
    "class_predictions",
    "classification_metrics",
    "confusion_matrix",
    "f1_score",
    "log_loss",
    "precision_score",
    "pr_auc_score",
    "recall_score",
    "roc_auc_score",
    "top_k_accuracy",
]
