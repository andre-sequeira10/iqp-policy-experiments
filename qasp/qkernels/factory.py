"""Dictionary factories and circuit drawing for fidelity/projected kernels.

High-level factory for quantum kernels, mirroring ``qasp.rl.create_model``.
A user describes the kernel with a configuration dictionary and a single call
to :func:`create_kernel` returns a ready-to-use kernel object.

Example
-------
>>> config = {
...     "kernel_type": "fidelity",            # or "projected"
...     "feature_map": "AngleEmbedding",
...     "feature_map_config": {"rotations": "Z",
...                             "imprimitive_pattern": "ring"},
...     "reps": 2,
...     "device": "default.qubit",
... }
>>> kernel = create_kernel(config, n_qubits=4)
>>> K = kernel.gram(X_train)"""

from copy import deepcopy

import numpy as np

from .quantum_kernels import (
    FidelityKernel,
    ProjectedKernel,
)

# overlap (two-input) kernels are drawn with both x and y
_OVERLAP_KERNELS = (FidelityKernel,)


def _tikz_circuit_kwargs(kernel, x, y=None, *, overlap=False):
    """Build the arguments used to record a kernel's underlying circuit."""
    if x is None:
        x = np.random.random(kernel.nfeat)
    if overlap and y is None:
        y = np.random.random(kernel.nfeat)

    if kernel.trainable:
        state = kernel.qlayer.state_dict()
        if overlap:
            import torch

            inputs = torch.cat((kernel._x_tensor(x), kernel._x_tensor(y)))
        else:
            inputs = kernel._x_tensor(x)

        circuit_kwargs = {
            "inputs": inputs,
            "params": state["params"],
        }
        if kernel.fm_input_scaling:
            circuit_kwargs["input_scaling_params"] = state[
                "input_scaling_params"
            ]
        return circuit_kwargs

    if overlap:
        inputs = np.stack(
            (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        )
    else:
        inputs = np.asarray(x, dtype=float)

    circuit_kwargs = {"inputs": inputs}
    if kernel.fm_input_scaling:
        circuit_kwargs["input_scaling_params"] = kernel._fixed_isp
    return circuit_kwargs


_KERNELS = {
    "fidelity": FidelityKernel,
    "projected": ProjectedKernel,
    "projected_quantum_kernel": ProjectedKernel,
}


def create_kernel(config: dict, n_qubits: int):
    """Build a quantum kernel from a configuration dictionary.

    Args:
        config (dict): must contain ``kernel_type`` (one of ``"fidelity"`` or
            ``"projected"``) and ``feature_map`` plus
            ``feature_map_config``.  Additional keys are forwarded to the kernel
            constructor:

            - common: ``reps``, ``input_scaling_params``, ``device``,
              ``diff_method``, ``shots``, ``mpi``, ``device_kwargs`` (forwarded to
              ``CircuitBuilder.compile`` like the RL / VQC models).
            - trainable encoding: ``ansatz``, ``ansatz_config``, ``init``
              (turns the kernel into a trainable circuit usable with
              kernel-target alignment).
            - fidelity: optional ``projectors`` with labels such as
              ``"P 0 2"``. Omitting them gives global fidelity; multiple
              labels produce their averaged localized fidelity.
            - projected: ``observables`` (Pauli strings), ``weights``, ``gamma``.
        n_qubits (int): number of qubits.

    Returns:
        A kernel instance exposing ``gram`` / ``__call__`` and ``draw``.
    """
    cfg = deepcopy(config)
    kernel_type = cfg.pop("kernel_type", "fidelity")
    if kernel_type not in _KERNELS:
        raise ValueError(
            f"Unknown kernel_type '{kernel_type}'. "
            f"Choose from {sorted(_KERNELS)}."
        )

    # resolve callables (allow n_qubits-dependent values, like create_model)
    fm_cfg = dict(cfg.get("feature_map_config", {}) or {})
    if callable(fm_cfg.get("features")):
        fm_cfg["features"] = fm_cfg["features"](n_qubits)
    fm_cfg.setdefault("n_qubits", n_qubits)
    cfg["feature_map_config"] = fm_cfg

    # resolve ansatz config callables (e.g. n_layers as a function of n_qubits)
    if cfg.get("ansatz") is not None:
        ans_cfg = dict(cfg.get("ansatz_config", {}) or {})
        if callable(ans_cfg.get("n_layers")):
            ans_cfg["n_layers"] = ans_cfg["n_layers"](n_qubits)
        ans_cfg.setdefault("n_qubits", n_qubits)
        cfg["ansatz_config"] = ans_cfg

    cfg["n_qubits"] = n_qubits

    KernelClass = _KERNELS[kernel_type]
    return KernelClass(**cfg)


def draw_kernel(
    kernel=None,
    config=None,
    n_qubits=None,
    x=None,
    y=None,
    style: str = "pennylane",
    decimals: int = 2,
    output: str = "matplotlib",
    tikz_options=None,
):
    """Draw the circuit underlying a kernel.

    Either pass an existing ``kernel`` object or a ``config`` + ``n_qubits``
    pair that will be used to build one on the fly.

    Set ``output="tikz"`` to return Quantikz-compatible LaTeX source.
    ``tikz_options`` accepts the circuit exporter options, including
    ``compress``, ``standalone`` and ``gate_colors``.
    """
    if kernel is None:
        if config is None or n_qubits is None:
            raise ValueError(
                "Provide either a kernel object or (config, n_qubits)."
            )
        kernel = create_kernel(config, n_qubits)

    normalized_output = output.lower()
    if normalized_output in {"tikz", "tex", "latex"}:
        from qasp.circuits import circuit_to_tikz

        overlap = isinstance(kernel, _OVERLAP_KERNELS)
        circuit_kwargs = _tikz_circuit_kwargs(
            kernel,
            x,
            y,
            overlap=overlap,
        )
        options = dict(tikz_options or {})
        options.setdefault("decimals", decimals)
        if options.get("compress"):
            options.setdefault("mirror_layers", overlap)
        return circuit_to_tikz(
            kernel.builder,
            circuit_kwargs=circuit_kwargs,
            **options,
        )

    if normalized_output not in {"matplotlib", "mpl"}:
        raise ValueError(
            "output must be one of 'matplotlib', 'mpl', 'tikz', 'tex', "
            "or 'latex'."
        )

    if isinstance(kernel, _OVERLAP_KERNELS):
        return kernel.draw(x=x, y=y, style=style, decimals=decimals)
    return kernel.draw(x=x, style=style, decimals=decimals)
