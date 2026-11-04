"""Circuit drawing helpers for QASP optimization algorithms."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pennylane as qml

from .pce import PauliCorrelationEncoding
from .qaoa import LRQAOA, QAOA, WarmStartQAOA
from .recursive import RecursiveQAOA
from .results import OptimizationResult


def _result_parameters(parameters):
    """Extract parameter arrays from an OptimizationResult; pass other inputs through."""
    if isinstance(parameters, OptimizationResult):
        return parameters.parameters
    return parameters


def _qaoa_parameters(model: QAOA, parameters):
    """Resolve draw-time angles, defaulting to an LR ramp or zeros, and validate through the model initializer."""
    parameters = _result_parameters(parameters)
    if parameters is None:
        if isinstance(model, LRQAOA):
            parameters = model.ramp_parameters.reshape(-1)
        else:
            parameters = model.initial_parameters("zeros")
    return model.initial_parameters(parameters)


def _pce_parameters(model: PauliCorrelationEncoding, parameters):
    """Resolve PCE drawing angles from a result/array or zeros and reshape to the model parameter tensor."""
    parameters = _result_parameters(parameters)
    if parameters is None:
        parameters = model.initial_parameters("zeros")
    values = np.asarray(parameters, dtype=float)
    if values.size != model.n_parameters:
        raise ValueError(
            f"Expected {model.n_parameters} PCE parameters, got {values.size}."
        )
    return values.reshape(model.parameter_shape)


def _numeric_label(value, decimals):
    """Format a scalar gate label to the requested precision, falling back to a symbolic theta for nonnumeric values."""
    try:
        return f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return r"\theta"


def _qaoa_parameter_formatter(model: QAOA, decimals: int):
    """Create paper-friendly gamma/beta labels from QAOA operation IDs."""

    def formatter(value, operation, parameter_index, global_index):
        """Translate tagged QAOA operations into layer/term-aware gamma, beta or warm-start labels for Quantikz."""
        del parameter_index
        operation_id = str(getattr(operation, "id", "") or "")
        parts = operation_id.split(":")
        if len(parts) >= 3 and parts[:2] == ["qasp", "warm_init"]:
            return rf"\vartheta_{{{int(parts[2]) + 1}}}"
        if len(parts) >= 5 and parts[:2] == ["qasp", "warm_basis"]:
            return rf"\vartheta_{{{int(parts[4]) + 1}}}"
        if len(parts) >= 4 and parts[0] == "qasp" and parts[1] in {"cost", "mixer"}:
            kind = parts[1]
            layer = int(parts[2]) + 1
            term = int(parts[3]) + 1
            symbol = "gamma" if kind == "cost" else "beta"
            coefficient = "c" if kind == "cost" else "m"
            if model.parametrization == "standard":
                angle = rf"\{symbol}_{{{layer}}}"
            else:
                angle = rf"\{symbol}_{{{layer},{term}}}"
            return rf"2{coefficient}_{{{term}}}{angle}"
        return rf"\theta_{{{global_index + 1}}}" if operation_id else _numeric_label(value, decimals)

    return formatter


def _qaoa_compression_options(model: QAOA):
    """Infer initial-state and per-layer gate counts for compression; require explicit offsets for callable initial-state preparation."""
    if model.warm_start is not None:
        initial_count = model.n_qubits
    elif callable(model.initial_state):
        raise ValueError(
            "Automatic QAOA compression cannot infer a callable initial state; "
            "provide first_layer_start in tikz_options."
        )
    elif isinstance(model.initial_state, str):
        initial_count = (
            model.n_qubits
            if model.initial_state.lower() in {"plus", "+"}
            else 0
        )
    else:
        initial_count = 1  # StatePrep
    mixer_count = (
        3 * model.n_qubits if model._warm_mixer else len(model.mixer_terms)
    )
    return {
        "n_layers": model.n_layers,
        "first_layer_start": initial_count,
        "first_layer_operation_count": len(model.cost_terms) + mixer_count,
    }


def draw_qaoa(
    model: QAOA | None = None,
    *,
    problem=None,
    config: dict | None = None,
    parameters=None,
    measurement: str = "probabilities",
    output: str = "matplotlib",
    style: str = "pennylane",
    decimals: int = 2,
    tikz_options: dict | None = None,
):
    """Draw a QAOA-family circuit as Matplotlib, text, or Quantikz/TikZ.

    Pass an existing standard, multi-angle, warm-start, or linear-ramp QAOA
    model.  Alternatively, pass ``problem`` and constructor ``config``.
    ``parameters`` may be a flat array or an :class:`OptimizationResult`.
    """
    if model is None:
        if problem is None:
            raise ValueError("Provide a QAOA model or a QUBO problem.")
        model = QAOA(problem, **dict(config or {}))
    if not isinstance(model, QAOA):
        raise TypeError("model must be a QAOA, WarmStartQAOA, or LRQAOA instance.")

    values = _qaoa_parameters(model, parameters)
    circuit = model.build_circuit(measurement=measurement)
    normalized_output = output.lower()
    if normalized_output in {"text", "ascii"}:
        return qml.draw(circuit, decimals=decimals)(values)
    if normalized_output in {"tikz", "tex", "latex"}:
        from qasp.circuits import circuit_to_tikz

        options = dict(tikz_options or {})
        options.setdefault("decimals", decimals)
        options.setdefault("symbolic_parameters", True)
        if (
            options.get("symbolic_parameters")
            and "parameter_labels" not in options
            and "parameter_formatter" not in options
        ):
            options["parameter_formatter"] = _qaoa_parameter_formatter(
                model, decimals
            )
        if options.get("compress"):
            for name, value in _qaoa_compression_options(model).items():
                options.setdefault(name, value)
        return circuit_to_tikz(circuit, values, **options)
    if normalized_output not in {"matplotlib", "mpl"}:
        raise ValueError(
            "output must be 'matplotlib', 'mpl', 'text', 'ascii', 'tikz', "
            "'tex', or 'latex'."
        )
    return qml.draw_mpl(circuit, decimals=decimals, style=style)(values)


def draw_pce(
    model: PauliCorrelationEncoding | None = None,
    *,
    problem=None,
    config: dict | None = None,
    parameters=None,
    output: str = "matplotlib",
    style: str = "pennylane",
    decimals: int = 2,
    tikz_options: dict | None = None,
):
    """Draw a PCE QASP ansatz and its Pauli-correlation measurements."""
    if model is None:
        if problem is None:
            raise ValueError("Provide a PCE model or a QUBO problem.")
        model = PauliCorrelationEncoding(problem, **dict(config or {}))
    if not isinstance(model, PauliCorrelationEncoding):
        raise TypeError("model must be a PauliCorrelationEncoding instance.")

    values = _pce_parameters(model, parameters)
    circuit_kwargs = {"params": values}
    normalized_output = output.lower()
    if normalized_output in {"text", "ascii"}:
        return qml.draw(model.qnode, decimals=decimals)(**circuit_kwargs)
    if normalized_output in {"tikz", "tex", "latex"}:
        from qasp.circuits import circuit_to_tikz

        options = dict(tikz_options or {})
        options.setdefault("decimals", decimals)
        options.setdefault("symbolic_parameters", True)
        return circuit_to_tikz(
            model.builder,
            circuit_kwargs=circuit_kwargs,
            **options,
        )
    if normalized_output not in {"matplotlib", "mpl"}:
        raise ValueError(
            "output must be 'matplotlib', 'mpl', 'text', 'ascii', 'tikz', "
            "'tex', or 'latex'."
        )
    return qml.draw_mpl(
        model.qnode, decimals=decimals, style=style
    )(**circuit_kwargs)


def draw_recursive_qaoa(
    model: RecursiveQAOA,
    *,
    step: int = 0,
    parameters=None,
    **drawing_options,
):
    """Draw the QAOA circuit used at one completed RQAOA reduction step."""
    if not isinstance(model, RecursiveQAOA):
        raise TypeError("model must be a RecursiveQAOA instance.")
    step_solvers = getattr(model, "step_solvers", None)
    step_results = getattr(model, "step_results", None)
    if not step_solvers:
        raise RuntimeError("Call RecursiveQAOA.solve() before drawing a step.")
    try:
        solver = step_solvers[step]
        if parameters is None:
            parameters = step_results[step]
    except IndexError as exc:
        raise IndexError(
            f"RQAOA step {step} is unavailable; completed {len(step_solvers)} steps."
        ) from exc
    return draw_qaoa(
        solver,
        parameters=parameters,
        **drawing_options,
    )


def draw_optimization(
    model=None,
    *,
    algorithm: str = "qaoa",
    problem=None,
    config: dict | None = None,
    parameters=None,
    step: int = 0,
    **drawing_options,
):
    """Draw any supported optimization model with one high-level function."""
    if model is None:
        if problem is None:
            raise ValueError("Provide an optimization model or a QUBO problem.")
        cfg = deepcopy(config or {})
        normalized = algorithm.lower().replace("-", "_")
        if normalized in {"qaoa", "standard_qaoa"}:
            model = QAOA(problem, **cfg)
        elif normalized in {"multi_angle", "multi_angle_qaoa"}:
            cfg["parametrization"] = "multi_angle"
            model = QAOA(problem, **cfg)
        elif normalized in {"warm_start", "warm_start_qaoa"}:
            model = WarmStartQAOA(problem, **cfg)
        elif normalized in {"lr_qaoa", "lrqaoa", "linear_ramp"}:
            model = LRQAOA(problem, **cfg)
        elif normalized in {"pce", "pauli_correlation_encoding"}:
            model = PauliCorrelationEncoding(problem, **cfg)
        else:
            raise ValueError(
                "algorithm must be qaoa, multi_angle_qaoa, warm_start_qaoa, "
                "lr_qaoa, or pce."
            )

    if isinstance(model, RecursiveQAOA):
        return draw_recursive_qaoa(
            model,
            step=step,
            parameters=parameters,
            **drawing_options,
        )
    if isinstance(model, PauliCorrelationEncoding):
        return draw_pce(
            model,
            parameters=parameters,
            **drawing_options,
        )
    if isinstance(model, QAOA):
        return draw_qaoa(
            model,
            parameters=parameters,
            **drawing_options,
        )
    raise TypeError("Unsupported optimization model type.")


__all__ = [
    "draw_optimization",
    "draw_pce",
    "draw_qaoa",
    "draw_recursive_qaoa",
]
