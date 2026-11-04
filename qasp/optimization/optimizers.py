"""Optimizer adapters used by the optimization module.

SciPy consumes NumPy arrays, PennyLane optimizers consume Autograd arrays,
and PyTorch optimizers consume Torch tensors. Keeping those families explicit
prevents silent conversions between incompatible differentiation frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import OptimizeResult, minimize


SCIPY_OPTIMIZERS = (
    "COBYLA",
    "Nelder-Mead",
    "Powell",
    "BFGS",
    "L-BFGS-B",
    "SLSQP",
)

PENNYLANE_OPTIMIZERS = (
    "GradientDescent",
    "Adam",
    "Adagrad",
    "Momentum",
    "NesterovMomentum",
    "RMSProp",
    "QNG",
    "SPSA",
    "QNSPSA",
    "ShotAdaptive",
)

TORCH_OPTIMIZERS = (
    "Adam",
    "AdamW",
    "SGD",
    "RMSprop",
    "Adagrad",
    "Adadelta",
    "ASGD",
    "LBFGS",
)

# Backwards-compatible aggregate used for optimizer discovery.
AVAILABLE_OPTIMIZERS = tuple(
    dict.fromkeys(SCIPY_OPTIMIZERS + PENNYLANE_OPTIMIZERS + TORCH_OPTIMIZERS)
)


def normalize_backend(backend: str) -> str:
    """Return the canonical optimization backend name."""
    aliases = {
        "numpy": "numpy",
        "autograd": "numpy",
        "pennylane": "numpy",
        "torch": "torch",
        "pytorch": "torch",
    }
    try:
        return aliases[str(backend).lower()]
    except KeyError as exc:
        raise ValueError("backend must be 'numpy' or 'torch'.") from exc


def _lookup(name: str, choices: tuple[str, ...], family: str) -> str:
    """Resolve an optimizer name case-insensitively or raise ValueError listing the supported family."""
    lookup = {candidate.lower(): candidate for candidate in choices}
    try:
        return lookup[str(name).lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown {family} optimizer {name!r}; choose from {choices}."
        ) from exc


def is_pennylane_optimizer(optimizer) -> bool:
    """Whether a string names one of PennyLane's Autograd optimizers."""
    return isinstance(optimizer, str) and optimizer.lower() in {
        name.lower() for name in PENNYLANE_OPTIMIZERS
    }


@dataclass(frozen=True)
class ClassicalOptimizer:
    """Reusable SciPy optimizer compatible with QASP optimization solvers."""

    method: str = "COBYLA"
    options: dict | None = None

    def __call__(self, objective, initial_parameters, *, jac=None, **overrides):
        """Minimize a scalar objective with scipy.optimize.minimize; merge option overrides, convert initial parameters to NumPy and optionally pass a Jacobian."""
        options = dict(self.options or {})
        options.update(overrides)
        return minimize(
            objective,
            np.asarray(initial_parameters, dtype=float),
            method=self.method,
            jac=jac,
            options=options,
        )


def create_optimizer(name: str = "COBYLA", **options) -> ClassicalOptimizer:
    """Create a reusable SciPy optimizer by method name.

    PennyLane and PyTorch optimizers own framework-specific parameter state,
    so they are selected by name in ``solver.solve(...)`` instead.
    """
    method = _lookup(name, SCIPY_OPTIMIZERS, "SciPy")
    return ClassicalOptimizer(method=method, options=options)


def minimize_objective(
    objective: Callable[[np.ndarray], float],
    initial_parameters,
    *,
    optimizer: str | Callable = "COBYLA",
    optimizer_options: dict | None = None,
    jac: Callable[[np.ndarray], np.ndarray] | None = None,
):
    """Minimize an objective with SciPy or a user supplied optimizer callable."""
    x0 = np.asarray(initial_parameters, dtype=float).reshape(-1)
    options = dict(optimizer_options or {})
    if callable(optimizer):
        if jac is None:
            result = optimizer(objective, x0.copy(), **options)
        else:
            result = optimizer(objective, x0.copy(), jac=jac, **options)
        if hasattr(result, "x"):
            return result
        parameters = np.asarray(result, dtype=float).reshape(-1)
        return OptimizeResult(
            x=parameters,
            fun=float(objective(parameters)),
            success=True,
            message="Custom optimizer returned parameters.",
        )
    method = _lookup(str(optimizer), SCIPY_OPTIMIZERS, "SciPy")
    return minimize(
        objective,
        x0,
        method=method,
        jac=jac,
        options=options,
    )


_PENNYLANE_CLASSES = {
    "gradientdescent": qml.GradientDescentOptimizer,
    "adam": qml.AdamOptimizer,
    "adagrad": qml.AdagradOptimizer,
    "momentum": qml.MomentumOptimizer,
    "nesterovmomentum": qml.NesterovMomentumOptimizer,
    "rmsprop": qml.RMSPropOptimizer,
    "qng": qml.QNGOptimizer,
    "spsa": qml.SPSAOptimizer,
    "qnspsa": qml.QNSPSAOptimizer,
    "shotadaptive": qml.ShotAdaptiveOptimizer,
}

_QNODE_ONLY_OPTIMIZERS = {"qng", "qnspsa", "shotadaptive"}


def minimize_pennylane(
    objective,
    initial_parameters,
    *,
    optimizer: str = "Adam",
    optimizer_options: dict | None = None,
    qnode=None,
):
    """Minimize with a PennyLane optimizer on the Autograd interface.

    ``QNG``, ``QNSPSA``, and ``ShotAdaptive`` require a single scalar QNode.
    QAOA supplies one; composite PCE losses do not support those optimizers.
    """
    canonical = _lookup(optimizer, PENNYLANE_OPTIMIZERS, "PennyLane")
    key = canonical.lower()
    options = dict(optimizer_options or {})
    steps = int(
        options.pop(
            "max_steps",
            options.pop("steps", options.pop("maxiter", 100)),
        )
    )
    if steps < 1:
        raise ValueError("The number of optimizer steps must be positive.")
    tolerance = options.pop("tolerance", options.pop("tol", None))
    if "lr" in options and "stepsize" not in options:
        options["stepsize"] = options.pop("lr")
    if key == "spsa":
        options.setdefault("maxiter", steps)
    if key == "shotadaptive":
        options.setdefault("min_shots", 10)
    if key in _QNODE_ONLY_OPTIMIZERS and qnode is None:
        raise ValueError(
            f"PennyLane {canonical} requires a single scalar QNode objective; "
            "it is supported by QAOA but not by the composite PCE loss."
        )
    if key == "shotadaptive":
        shots = getattr(getattr(qnode.device, "shots", None), "total_shots", None)
        if shots is None:
            raise ValueError(
                "PennyLane ShotAdaptive requires a QAOA device with finite shots."
            )

    optimizer_instance = _PENNYLANE_CLASSES[key](**options)
    target = qnode if key in _QNODE_ONLY_OPTIMIZERS else objective
    parameters = pnp.array(
        np.asarray(initial_parameters, dtype=float).reshape(-1),
        requires_grad=True,
    )
    history = []
    converged = False
    for _ in range(steps):
        parameters, previous_cost = optimizer_instance.step_and_cost(target, parameters)
        history.append(float(previous_cost))
        if (
            tolerance is not None
            and len(history) > 1
            and abs(history[-1] - history[-2]) <= float(tolerance)
        ):
            converged = True
            break

    values = np.asarray(parameters, dtype=float).reshape(-1)
    final_cost = float(objective(parameters))
    return OptimizeResult(
        x=values,
        fun=final_cost,
        nit=len(history),
        success=True,
        message=(
            "PennyLane optimizer converged."
            if converged
            else "PennyLane optimizer completed the requested steps."
        ),
        history=history,
        optimizer=canonical,
        backend="numpy",
    )


def minimize_torch(
    objective,
    initial_parameters,
    *,
    optimizer: str | type = "Adam",
    optimizer_options: dict | None = None,
):
    """Minimize a Torch-interface scalar objective with ``torch.optim``."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - project dependency
        raise ImportError("The torch backend requires PyTorch.") from exc

    classes = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "asgd": torch.optim.ASGD,
        "lbfgs": torch.optim.LBFGS,
    }
    if isinstance(optimizer, str):
        canonical = _lookup(optimizer, TORCH_OPTIMIZERS, "PyTorch")
        optimizer_class = classes[canonical.lower()]
    elif isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
        canonical = optimizer.__name__
        optimizer_class = optimizer
    else:
        raise TypeError(
            "For backend='torch', optimizer must be a supported name or a "
            "torch.optim.Optimizer class."
        )

    options = dict(optimizer_options or {})
    steps = int(
        options.pop(
            "max_steps",
            options.pop("steps", options.pop("maxiter", 100)),
        )
    )
    if steps < 1:
        raise ValueError("The number of optimizer steps must be positive.")
    tolerance = options.pop("tolerance", options.pop("tol", None))
    options.setdefault("lr", 0.05)

    parameters = torch.nn.Parameter(
        torch.as_tensor(initial_parameters, dtype=torch.float64).reshape(-1).clone()
    )
    optimizer_instance = optimizer_class([parameters], **options)
    history = []
    converged = False

    def closure():
        """Zero gradients, evaluate the differentiable scalar objective and backpropagate for optimizers that require a closure."""
        optimizer_instance.zero_grad(set_to_none=True)
        loss = objective(parameters)
        loss.backward()
        return loss

    for _ in range(steps):
        if isinstance(optimizer_instance, torch.optim.LBFGS):
            optimizer_instance.step(closure)
        else:
            closure()
            optimizer_instance.step()
        with torch.no_grad():
            current = float(objective(parameters).detach().cpu())
        history.append(current)
        if (
            tolerance is not None
            and len(history) > 1
            and abs(history[-1] - history[-2]) <= float(tolerance)
        ):
            converged = True
            break

    values = parameters.detach().cpu().numpy().astype(float, copy=True)
    return OptimizeResult(
        x=values,
        fun=history[-1],
        nit=len(history),
        success=True,
        message=(
            "PyTorch optimizer converged."
            if converged
            else "PyTorch optimizer completed the requested steps."
        ),
        history=history,
        optimizer=canonical,
        backend="torch",
        optimizer_instance=optimizer_instance,
    )


__all__ = [
    "AVAILABLE_OPTIMIZERS",
    "ClassicalOptimizer",
    "PENNYLANE_OPTIMIZERS",
    "SCIPY_OPTIMIZERS",
    "TORCH_OPTIMIZERS",
    "create_optimizer",
    "is_pennylane_optimizer",
    "minimize_objective",
    "minimize_pennylane",
    "minimize_torch",
    "normalize_backend",
]
