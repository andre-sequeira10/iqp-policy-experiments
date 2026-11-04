"""Centered kernel-target alignment and optimization of quantum kernel weights.

Kernel Target Alignment (KTA) for trainable quantum kernels.

Given a *trainable* quantum kernel (one built with an ``ansatz`` so that its
encoding ``U(x; theta)`` carries variational angles) this module optimises
``theta`` to maximise the **centered** alignment between the kernel Gram matrix
and the label-induced target matrix.

Centered kernel-target alignment (Cortes, Mohri & Rostamizadeh, JMLR 2012):

    A(K, T) = <Kc, Tc>_F / ( ||Kc||_F * ||Tc||_F ),

where ``Kc = H K H`` and ``Tc = H T H`` with the centering matrix
``H = I - (1/n) 11^T`` and the ideal target ``T_ij = 1[y_i == y_j]`` (which
covers both binary and multi-class problems).  Alignment lies in ``[-1, 1]`` and
larger is better, so we minimise ``-A`` with a Torch optimizer.

Typical use
-----------
>>> from qasp.qkernels import create_kernel
>>> from qasp.qkernels.kta import KernelTargetAlignment, target_alignment
>>> cfg = {"kernel_type": "fidelity", "feature_map": "AngleEmbedding",
...        "feature_map_config": {"rotations": "Z", "imprimitive_pattern": "ring"},
...        "ansatz": "AngleAnsatz",
...        "ansatz_config": {"n_layers": 2, "rotations": "ZY",
...                          "include_imprimitive": True, "imprimitive": "CZ",
...                          "imprimitive_pattern": "ring", "reuploading": True},
...        "init": "random_-pi_pi"}
>>> kernel = create_kernel(cfg, n_qubits=4)
>>> print("before:", target_alignment(kernel.gram(X), y))
>>> kta = KernelTargetAlignment(kernel)
>>> history = kta.fit(X, y, steps=40, lr=0.05)
>>> print("after :", target_alignment(kernel.gram(X), y))"""

from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# target matrices and (centered) alignment -- numpy, for reporting
# ---------------------------------------------------------------------------
def target_matrix(y) -> np.ndarray:
    """Ideal target Gram ``T_ij = 1[y_i == y_j]`` (binary and multi-class)."""
    y = np.asarray(y).reshape(-1)
    return (y[:, None] == y[None, :]).astype(float)


def _center_np(M):
    """Double-center a square NumPy Gram matrix with H=I-11.T/n."""
    n = M.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ M @ H


def target_alignment(K, y, centered: bool = True) -> float:
    """Centered kernel-target alignment between Gram matrix ``K`` and the target
    induced by labels ``y`` (returns a python float)."""
    K = np.asarray(K, dtype=float)
    T = target_matrix(y)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square Gram matrix.")
    if K.shape != T.shape:
        raise ValueError("The number of labels must match the size of K.")
    if centered:
        K = _center_np(K)
        T = _center_np(T)
    num = float((K * T).sum())
    den = float(np.linalg.norm(K) * np.linalg.norm(T))
    return num / den if den > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# torch (differentiable) alignment
# ---------------------------------------------------------------------------
def _center_torch(M):
    """Double-center a square Torch Gram matrix while preserving dtype, device and autograd."""
    n = M.shape[0]
    H = torch.eye(n, dtype=M.dtype, device=M.device)
    H -= torch.ones(n, n, dtype=M.dtype, device=M.device) / n
    return H @ M @ H


def alignment_torch(K, T, centered: bool = True):
    """Return differentiable normalized Frobenius alignment of square matrices.

    K and target matrix T must have compatible shape/device. centered=True
    subtracts row/column means via double centering. The norm product is clamped
    to 1e-12, so degenerate zero-norm inputs return zero instead of dividing by
    zero. Gradients remain connected to K and T."""
    if centered:
        K = _center_torch(K)
        T = _center_torch(T)
    num = (K * T).sum()
    den = torch.norm(K) * torch.norm(T)
    return num / den.clamp_min(1e-12)


# ---------------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------------
class KernelTargetAlignment:
    """Optimise a trainable kernel's angles to maximise centered target
    alignment.

    Args:
        kernel: a trainable quantum kernel (built with an ``ansatz``).  Must
            expose :meth:`parameters` and a ``gram(X, return_tensor=True)`` that
            returns a differentiable ``torch.Tensor``.
        centered (bool): use centered alignment (recommended).
    """

    def __init__(self, kernel, centered: bool = True):
        """Attach a kernel with trainable parameters and select centered or uncentered alignment; raise ValueError for an encoding-only kernel."""
        self.kernel = kernel
        self.centered = centered
        params = kernel.parameters()
        if not params:
            raise ValueError(
                "Kernel has no trainable parameters. Build it with an "
                "'ansatz' (and 'ansatz_config') to enable target alignment."
            )

    # ------------------------------------------------------------------
    def alignment(self, X, y) -> float:
        """Current (centered) alignment on ``(X, y)`` as a python float."""
        return target_alignment(self.kernel.gram(X), y, centered=self.centered)

    def fit(
        self,
        X,
        y,
        steps: int = 50,
        lr: float = 0.05,
        optimizer=None,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        """Train the kernel angles.

        Args:
            X, y: training inputs / labels.
            steps (int): optimisation steps.
            lr (float): learning rate for the default Adam optimizer.
            optimizer: optional pre-built Torch optimizer over
                ``kernel.parameters()``.
            batch_size (int, optional): if set, a random subset of rows is used
                each step (useful to keep the O(n^2) fidelity Gram cheap).
            seed (int, optional): RNG seed for mini-batch sampling.
            verbose (bool): print progress.

        Returns:
            dict with ``"alignment"`` (per-step) and ``"loss"`` histories.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        n = len(X)
        if n != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if n < 2:
            raise ValueError("Kernel target alignment needs at least two samples.")
        if steps < 1:
            raise ValueError("steps must be at least 1.")
        if batch_size is not None and not 2 <= batch_size <= n:
            raise ValueError("batch_size must be between 2 and len(X).")
        params = self.kernel.parameters()
        opt = optimizer or torch.optim.Adam(params, lr=lr)
        rng = np.random.default_rng(seed)

        history = {"alignment": [], "loss": []}
        for step in range(steps):
            if batch_size is not None and batch_size < n:
                idx = rng.choice(n, size=batch_size, replace=False)
            else:
                idx = np.arange(n)
            Xb, yb = X[idx], y[idx]
            opt.zero_grad()
            K = self.kernel.gram(Xb, return_tensor=True)
            T = torch.as_tensor(
                (yb[:, None] == yb[None, :]).astype(float),
                dtype=torch.float64,
                device=K.device,
            )
            A = alignment_torch(K.double(), T, centered=self.centered)
            loss = -A
            loss.backward()
            opt.step()

            history["alignment"].append(float(A.detach()))
            history["loss"].append(float(loss.detach()))
            if verbose:
                print(f"step {step + 1:3d}/{steps}  alignment={float(A):.4f}",
                      flush=True)
        return history


def align_kernel(kernel, X, y, steps: int = 50, lr: float = 0.05,
                 batch_size: Optional[int] = None, centered: bool = True,
                 seed: Optional[int] = None, verbose: bool = True):
    """Convenience one-shot: train ``kernel`` on ``(X, y)`` and return the
    fitted kernel together with the optimisation history."""
    kta = KernelTargetAlignment(kernel, centered=centered)
    history = kta.fit(X, y, steps=steps, lr=lr, batch_size=batch_size,
                      seed=seed, verbose=verbose)
    return kernel, history
