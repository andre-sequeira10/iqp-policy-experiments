"""Paper-ready circuit drawing for PQCBM without executing a quantum device."""

from copy import copy

import numpy as np
import pennylane as qml

from .factory import create_qcbm


def draw_qcbm(
    model=None,
    *,
    config=None,
    n_qubits=None,
    inputs=None,
    output="matplotlib",
    decimals=2,
    style="pennylane",
    tikz_options=None,
):
    """Draw the Born sampling circuit as Matplotlib, text, or Quantikz source.

    Supply a model or (config,n_qubits). Drawings show computational-basis output
    measurements; MMD's classical moment aggregation is not a circuit gate.
    ``tikz_options`` passes through to circuit_to_tikz: standalone, compress,
    symbolic_parameters, wire_labels, gate_colors and all existing options.
    The model and its training measurement are unchanged. No simulator runs.
    """
    output = output.lower()
    if output not in {"matplotlib", "mpl", "text", "ascii", "tikz", "tex", "latex"}:
        raise ValueError("output must be matplotlib, text, or tikz (or an alias).")
    if model is None:
        if config is None:
            raise ValueError("Provide a model or config.")
        config = dict(config)
        if inputs is not None:
            config["inputs"] = inputs
        elif config.get("inputs") is None:
            builder = config.get("circuit_builder", config.get("circuit"))
            feature_map = getattr(getattr(builder, "ansatz", None), "feature_map", None)
            if config.get("feature_map") is not None or feature_map is not None:
                width = (
                    n_qubits
                    or config.get("n_qubits")
                    or getattr(builder, "n_qubits", None)
                )
                features = (config.get("feature_map_config") or {}).get(
                    "features", getattr(feature_map, "features", None) or width
                )
                if callable(features):
                    features = features(width)
                config["inputs"] = np.zeros(features)
        model = create_qcbm(config, n_qubits=n_qubits)
    kwargs = model._kwargs(inputs=inputs)
    if model.circuit_builder is not None:
        builder = copy(model.circuit_builder)
        builder.set_measurement("probs")
        function = builder.build()
        target = builder
    else:
        function = model.circuit.func
        target = function
    if output in {"tikz", "tex", "latex"}:
        from qasp.circuits import circuit_to_tikz

        options = dict(tikz_options or {})
        options.setdefault("decimals", decimals)
        return circuit_to_tikz(target, circuit_kwargs=kwargs, **options)
    if output in {"text", "ascii"}:
        return qml.draw(function, decimals=decimals)(**kwargs)
    return qml.draw_mpl(function, decimals=decimals, style=style)(**kwargs)


__all__ = ["draw_qcbm"]
