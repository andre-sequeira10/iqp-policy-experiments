"""Probability losses and Gaussian MMD as a mixture of Pauli-Z expectations.

Bandwidths are standard deviations: k(x,y)=exp(-||x-y||²/(2 sigma²)).
See Recio-Armengol et al., arXiv:2503.02934, Sec. 4.2. All differentiable
operations preserve Torch or PennyLane NumPy/Autograd graphs.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pennylane as qml


def positive_integer(value, name):
    """Validate a positive integral dimension without silently rounding."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def binary_matrix(data, n_qubits=None, name="data"):
    """Return validated nonempty binary rows, without integer basis conversion."""
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    array = np.asarray(data)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must have nonempty shape (rows, n_qubits).")
    if n_qubits is not None and array.shape[1] != n_qubits:
        raise ValueError(f"{name} must have {n_qubits} columns.")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary values 0 and 1.")
    return array.astype(np.int64, copy=True)


def basis_indices(data):
    """Convert binary rows to big-endian indices for small probability readouts."""
    data = binary_matrix(data)
    if data.shape[1] > 62:
        raise ValueError("Basis indices require at most 62 bits; use MMD on bit rows.")
    return data @ (1 << np.arange(data.shape[1] - 1, -1, -1))


def nll_loss(probs, batch_idx):
    """Mean negative log likelihood of integer basis indices (epsilon=1e-12)."""
    return -qml.math.mean(qml.math.log(probs[batch_idx] + 1e-12))


def kl_divergence_loss(probs, target_probs):
    """Smoothed KL(target || model); the second argument is a distribution."""
    target_probs = qml.math.convert_like(target_probs, probs)
    target_probs = qml.math.cast_like(target_probs, probs)
    return qml.math.sum(
        target_probs
        * (qml.math.log(target_probs + 1e-12) - qml.math.log(probs + 1e-12))
    )


def mmd_loss(
    model_moments,
    target_moments,
    weights=None,
    *,
    model_moments_2=None,
    target_correction=None,
):
    """Squared moment discrepancy, optionally using independent noisy estimates.

    ``weights=None`` means a uniform mean. For finite-shot model expectations,
    pass an independently evaluated ``model_moments_2`` to avoid squaring shot
    noise. ``target_correction`` subtracts the target sample-mean variance for
    the data U-statistic. No clipping: unbiased estimates can be negative.
    """

    def like(value):
        """Match a constant to the differentiable moment array."""
        return qml.math.cast_like(
            qml.math.convert_like(value, model_moments), model_moments
        )

    target = like(target_moments)
    other = model_moments if model_moments_2 is None else model_moments_2
    terms = (model_moments - target) * (other - target)
    if target_correction is not None:
        terms = terms - like(target_correction)
    return (
        qml.math.mean(terms) if weights is None else qml.math.sum(like(weights) * terms)
    )


class PauliMMD:
    """Gaussian-kernel MMD² over bitstrings without a full probability vector.

    ``mode='sampled'`` draws n_observables independent masks from a uniform
    mixture over bandwidths. Each bit is Bernoulli((1-exp(-1/(2*sigma²)))/2).
    ``mode='exact'`` enumerates all masks, exponentially, up to max_exact_qubits.
    The zero mask is retained with its zero discrepancy and original weight.
    ``estimator='empirical'`` compares to the empirical data distribution;
    ``'unbiased'`` removes the diagonal data contribution (requires >=2 rows).
    """

    def __init__(
        self,
        n_qubits,
        *,
        bandwidths=(1.0,),
        n_observables=128,
        mode="sampled",
        estimator="empirical",
        seed=0,
        max_exact_qubits=16,
    ):
        """Validate kernel settings and initialize an isolated random generator."""
        self.n_qubits = positive_integer(n_qubits, "n_qubits")
        self.n_observables = positive_integer(n_observables, "n_observables")
        self.bandwidths = np.atleast_1d(np.asarray(bandwidths, dtype=float))
        if (
            self.bandwidths.ndim != 1
            or not self.bandwidths.size
            or not np.all(np.isfinite(self.bandwidths) & (self.bandwidths > 0))
        ):
            raise ValueError("bandwidths must be finite positive standard deviations.")
        if mode not in {"sampled", "exact"}:
            raise ValueError("MMD mode must be 'sampled' or 'exact'.")
        if estimator not in {"empirical", "unbiased"}:
            raise ValueError("MMD estimator must be 'empirical' or 'unbiased'.")
        self.mode, self.estimator = mode, estimator
        self.rng = np.random.default_rng(seed)
        self.max_exact_qubits = positive_integer(max_exact_qubits, "max_exact_qubits")
        if mode == "exact" and n_qubits > self.max_exact_qubits:
            raise ValueError("Exact MMD enumerates 2**n masks; use mode='sampled'.")
        self.masks, self.weights = self.draw_masks()

    def draw_masks(self):
        """Return (binary masks, normalized weights), preserving identity mass."""
        p = -np.expm1(-0.5 / self.bandwidths**2) / 2
        if self.mode == "exact":
            masks = (
                np.arange(1 << self.n_qubits)[:, None]
                >> np.arange(self.n_qubits - 1, -1, -1)
            ) & 1
            weight = masks.sum(axis=1)
            weights = np.mean(
                p[:, None] ** weight * (1 - p[:, None]) ** (self.n_qubits - weight),
                axis=0,
            )
        else:
            probabilities = p[self.rng.integers(len(p), size=self.n_observables)]
            masks = (
                self.rng.random((self.n_observables, self.n_qubits))
                < probabilities[:, None]
            ).astype(int)
            weights = np.full(self.n_observables, 1 / self.n_observables)
        return masks, weights

    def resample(self):
        """Replace sampled masks once per optimizer step; exact masks stay fixed."""
        if self.mode == "sampled":
            self.masks, self.weights = self.draw_masks()
        return self.masks

    def target_statistics(self, data, masks=None):
        """Compute empirical parities and optional U-statistic correction."""
        masks = (
            self.masks
            if masks is None
            else binary_matrix(masks, self.n_qubits, "masks")
        )
        data = binary_matrix(data, self.n_qubits)
        means = (1 - 2 * ((data @ masks.T) % 2)).mean(axis=0)
        correction = None
        if self.estimator == "unbiased":
            if len(data) < 2:
                raise ValueError("Unbiased MMD requires at least two data rows.")
            correction = (1 - means**2) / (len(data) - 1)
        return means, correction

    def __call__(self, model_moments, data, *, model_moments_2=None):
        """Evaluate MMD² on the current masks and supplied binary data."""
        means, correction = self.target_statistics(data)
        return mmd_loss(
            model_moments,
            means,
            self.weights,
            model_moments_2=model_moments_2,
            target_correction=correction,
        )


__all__ = ["PauliMMD", "kl_divergence_loss", "mmd_loss", "nll_loss"]
