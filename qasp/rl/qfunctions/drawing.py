"""Draw the quantum circuit of a PQDQN using QASP's existing exporters."""

from __future__ import annotations

import pennylane as qml
import torch

from .models import PQDQN


def draw_qfunction(
    model: PQDQN | None = None,
    *,
    config: dict | None = None,
    n_qubits: int | None = None,
    n_actions: int | None = None,
    inputs=None,
    output: str = "matplotlib",
    decimals: int = 2,
    style: str = "pennylane",
    tikz_options: dict | None = None,
):
    """Draw a PQDQN as Matplotlib, text, or Quantikz/TikZ source.

    Pass an existing model or its configuration dictionary. Dimensions may be
    supplied in the dictionary or as keyword overrides. Omitted inputs default
    to a zero feature vector; supplied inputs must describe one observation and
    receive the model's input activation, just as in ``forward``.

    TikZ options (e.g. ``standalone``, ``symbolic_parameters``, ``wire_labels``)
    pass through to ``circuit_to_tikz``, just as for policies and kernels.
    ``compress=True`` expands the first complete layer; for data reuploading
    this includes its feature map. ``standalone=False`` produces a paper
    snippet, and ``gate_colors=False`` selects monochrome. The drawing shows the quantum gates and
    Pauli measurements; the classical coefficient matrix and linear readout are
    not circuit operations and are not included. Existing models are not rebuilt
    or modified. TikZ output is source, not a compiled PDF.
    """
    output = output.lower()
    if output not in {"matplotlib", "mpl", "text", "ascii", "tikz", "tex", "latex"}:
        raise ValueError(
            "output must be 'matplotlib', 'text', or 'tikz' (or an alias)."
        )
    if model is None:
        if config is None:
            raise ValueError("Provide a PQDQN model or config.")
        model = PQDQN.from_config(config, n_qubits=n_qubits, n_actions=n_actions)

    state = model.qlayer.state_dict()
    params = state["params"]
    circuit_kwargs = {"params": params}
    feature_map = model.circuit_builder.ansatz.feature_map
    if feature_map is not None:
        n_features = getattr(feature_map, "features", None) or model.n_qubits
        if inputs is None:
            inputs = torch.zeros(
                int(n_features), dtype=params.dtype, device=params.device
            )
        inputs = torch.as_tensor(
            inputs, dtype=params.dtype, device=params.device
        ).detach()
        if inputs.ndim != 1 or inputs.numel() != n_features:
            raise ValueError(
                f"Drawing requires one observation with {n_features} features."
            )
        if model.input_activation == "tanh":
            inputs = torch.tanh(inputs)
        circuit_kwargs["inputs"] = inputs
        if getattr(feature_map, "input_scaling", False):
            circuit_kwargs["input_scaling_params"] = state["input_scaling_params"]

    if output in {"tikz", "tex", "latex"}:
        from qasp.circuits import circuit_to_tikz

        options = dict(tikz_options or {})
        options.setdefault("decimals", decimals)
        return circuit_to_tikz(
            model.circuit_builder, circuit_kwargs=circuit_kwargs, **options
        )
    if output in {"text", "ascii"}:
        return qml.draw(model.circuit, decimals=decimals)(**circuit_kwargs)
    return qml.draw_mpl(model.circuit, decimals=decimals, style=style)(**circuit_kwargs)


__all__ = ["draw_qfunction"]
