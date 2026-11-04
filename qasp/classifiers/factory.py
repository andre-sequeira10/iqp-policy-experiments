"""Configuration-driven construction for QASP quantum classifiers."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pennylane as qml
import torch

from qasp.circuits import AngleAnsatz, IQPAnsatz
from qasp.feature_maps import AngleEmbedding, NeuralEmbedding

from .models import PQCClassifier


_ANSATZES = {
    "AngleAnsatz": AngleAnsatz,
    "angle": AngleAnsatz,
    "IQPAnsatz": IQPAnsatz,
    "iqp": IQPAnsatz,
}
_FEATURE_MAPS = {
    "AngleEmbedding": AngleEmbedding,
    "angle": AngleEmbedding,
    "NeuralEmbedding": NeuralEmbedding,
    "neural": NeuralEmbedding,
}


def _resolve(component, registry, kind):
    """Resolve a registered component name or pass through an explicitly supplied class."""
    if not isinstance(component, str):
        return component
    if component not in registry:
        raise ValueError(
            f"Unknown {kind} '{component}'. Choose from {sorted(registry)}."
        )
    return registry[component]


def create_classifier(config: dict, n_qubits: int, n_classes: int) -> PQCClassifier:
    """Build a classifier using QASP's standard configuration notation.

    ``device``, ``mpi`` (or legacy ``enable_mpi``), and ``device_kwargs`` are
    forwarded to the model and its circuit builder.
    """
    cfg = deepcopy(config)
    classifier_type = cfg.pop("classifier_type", "pqc")
    if classifier_type not in {"pqc", "PQCClassifier", "quantum"}:
        raise ValueError("classifier_type must be 'pqc' or 'PQCClassifier'.")

    if cfg.get("circuit_builder") is None:
        if "ansatz" not in cfg or "feature_map" not in cfg:
            raise ValueError(
                "config requires ansatz and feature_map when circuit_builder is absent."
            )
        cfg["ansatz"] = _resolve(cfg["ansatz"], _ANSATZES, "ansatz")
        cfg["feature_map"] = _resolve(
            cfg["feature_map"], _FEATURE_MAPS, "feature map"
        )

        ansatz_config = dict(cfg.get("ansatz_config", {}) or {})
        if callable(ansatz_config.get("n_layers")):
            ansatz_config["n_layers"] = ansatz_config["n_layers"](n_qubits)
        ansatz_config.setdefault("n_qubits", n_qubits)
        cfg["ansatz_config"] = ansatz_config

        feature_map_config = dict(cfg.get("feature_map_config", {}) or {})
        if callable(feature_map_config.get("features")):
            feature_map_config["features"] = feature_map_config["features"](n_qubits)
        feature_map_config.setdefault("features", n_qubits)
        feature_map_config.setdefault("n_qubits", n_qubits)
        cfg["feature_map_config"] = feature_map_config

    cfg["n_qubits"] = n_qubits
    cfg["n_classes"] = n_classes
    return PQCClassifier(**cfg)


def _drawing_kwargs(classifier: PQCClassifier, inputs=None):
    """Build concrete circuit arguments from a trained classifier."""
    state = classifier.qlayer.state_dict()
    kwargs = {"params": state["params"]}
    ansatz = classifier.circuit_builder.ansatz
    feature_map = getattr(ansatz, "feature_map", None)
    if feature_map is not None:
        n_features = getattr(feature_map, "features", None) or classifier.n_qubits
        if inputs is None:
            inputs = np.zeros(int(n_features), dtype=float)
        kwargs["inputs"] = torch.as_tensor(
            inputs, dtype=state["params"].dtype
        )
        if getattr(feature_map, "input_scaling", False):
            kwargs["input_scaling_params"] = state["input_scaling_params"]
    return kwargs


def draw_classifier(
    classifier: PQCClassifier | None = None,
    *,
    config: dict | None = None,
    n_qubits: int | None = None,
    n_classes: int | None = None,
    inputs=None,
    style: str = "pennylane",
    decimals: int = 2,
    output: str = "matplotlib",
    tikz_options: dict | None = None,
):
    """Draw a classifier circuit as Matplotlib, text, or Quantikz/TikZ.

    Pass an existing classifier, or ``config`` together with ``n_qubits`` and
    ``n_classes``. TikZ options are forwarded to QASP's circuit exporter;
    useful paper settings include ``standalone=True``,
    ``symbolic_parameters=True``, ``compress=True``, and custom
    ``wire_labels`` / ``gate_colors``.
    """
    if classifier is None:
        if config is None or n_qubits is None or n_classes is None:
            raise ValueError(
                "Provide classifier or (config, n_qubits, n_classes)."
            )
        classifier = create_classifier(config, n_qubits, n_classes)

    circuit_kwargs = _drawing_kwargs(classifier, inputs)
    normalized_output = output.lower()
    if normalized_output in {"tikz", "tex", "latex"}:
        from qasp.circuits import circuit_to_tikz

        options = dict(tikz_options or {})
        options.setdefault("decimals", decimals)
        return circuit_to_tikz(
            classifier.circuit_builder,
            circuit_kwargs=circuit_kwargs,
            **options,
        )
    if normalized_output in {"text", "ascii"}:
        return qml.draw(classifier.circuit, decimals=decimals)(**circuit_kwargs)
    if normalized_output not in {"matplotlib", "mpl"}:
        raise ValueError(
            "output must be one of 'matplotlib', 'mpl', 'text', 'ascii', "
            "'tikz', 'tex', or 'latex'."
        )
    return qml.draw_mpl(
        classifier.circuit, decimals=decimals, style=style
    )(**circuit_kwargs)


__all__ = ["create_classifier", "draw_classifier"]
