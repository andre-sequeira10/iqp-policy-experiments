"""Finite-sample cost-concentration diagnostics for QASP models.

A single-model call diagnoses one constructed model.  A width-sweep call uses
a user-provided factory so every qubit count is represented by the exact QASP
classifier, kernel, or policy configuration intended for the experiment.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch


def _initialization_label(spec):
    """Keep result metadata readable for names, specifications and callables."""
    if spec is None or isinstance(spec, str):
        return spec
    return getattr(spec, "__name__", repr(spec))


def _generator(device: torch.device, seed: int | None) -> torch.Generator:
    """Create a device-local Torch generator with an explicit seed or entropy-based seed."""
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed))
    return generator


def _sample_tensor(
    tensor: torch.Tensor,
    initialization: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return a new tensor using the shared initializer and a local RNG."""
    from qasp.initialization import initialize_tensor

    return initialize_tensor(torch.empty_like(tensor), initialization, generator=generator)


def _model_type(model: Any, requested: str) -> str:
    """Resolve a requested model family or infer classifier, kernel, policy or generic by protocol."""
    normalized = requested.lower()
    if normalized != "auto":
        if normalized not in {"classifier", "kernel", "policy", "generic"}:
            raise ValueError(
                "model_type must be 'auto', 'classifier', 'kernel', 'policy', "
                "or 'generic'."
            )
        return normalized
    if hasattr(model, "predict_proba") and hasattr(model, "n_classes"):
        return "classifier"
    if hasattr(model, "gram") and hasattr(model, "nfeat"):
        return "kernel"
    if hasattr(model, "action_probs") and hasattr(model, "n_actions"):
        return "policy"
    return "generic"


def _named_parameters(model: Any, parameter_scope: str | Iterable):
    """Select named trainable tensors from the quantum layer/all model, or accept an explicit iterable."""
    if not isinstance(parameter_scope, str):
        raw = list(parameter_scope)
        return [(f"parameter_{index}", value) for index, value in enumerate(raw)]

    scope = parameter_scope.lower()
    if scope not in {"quantum", "all"}:
        raise ValueError("parameter_scope must be 'quantum', 'all', or an iterable.")
    owner = getattr(model, "qlayer", None) if scope == "quantum" else model
    if owner is None:
        return []
    if hasattr(owner, "named_parameters"):
        named = list(owner.named_parameters())
    elif hasattr(owner, "parameters"):
        named = [
            (f"parameter_{index}", value)
            for index, value in enumerate(owner.parameters())
        ]
    else:
        named = []

    result = []
    seen = set()
    for name, value in named:
        if not isinstance(value, torch.Tensor) or not value.requires_grad:
            continue
        if id(value) in seen:
            continue
        seen.add(id(value))
        result.append((name, value))
    return result


def _feature_count(model: Any) -> int | None:
    """Infer embedded feature count from model metadata; return None when it cannot be determined."""
    value = getattr(model, "nfeat", None)
    if value is not None:
        return int(value)
    config = getattr(model, "feature_map_config", None)
    if isinstance(config, dict) and config.get("features") is not None:
        value = config["features"]
        if not callable(value):
            return int(value)
    builder = getattr(model, "circuit_builder", None)
    ansatz = getattr(builder, "ansatz", None)
    feature_map = getattr(ansatz, "feature_map", None)
    value = getattr(feature_map, "features", None)
    if value is not None:
        return int(value)
    return None


def _as_model_tensor(value: Any, reference: torch.Tensor) -> torch.Tensor:
    """Convert a value to the dtype and device of a reference parameter tensor."""
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _random_state(
    model: Any,
    kind: str,
    initialization: str,
    generator: torch.Generator,
    reference: torch.Tensor,
    state_shape: int | tuple[int, ...] | None,
):
    """Sample one embedded state, or a pair for kernels, using explicit/inferred feature dimensions."""
    if state_shape is None:
        features = _feature_count(model)
        if features is None:
            raise ValueError(
                "Could not infer the embedded-state size. Pass state, "
                "state_sampler, or state_shape."
            )
        shape = (features,)
    elif isinstance(state_shape, int):
        shape = (state_shape,)
    else:
        shape = tuple(state_shape)
    template = torch.empty(shape, dtype=reference.dtype, device=reference.device)

    def one():
        """Sample a state tensor using the selected initialization distribution and local generator."""
        return _sample_tensor(template, initialization, generator)

    if kind == "kernel":
        return one(), one()
    return one()


def _select_output(values: Any, output_index: int, label: str) -> torch.Tensor:
    """Select one final-axis output and average leading dimensions to obtain a scalar cost."""
    tensor = torch.as_tensor(values)
    if tensor.ndim == 0:
        if output_index not in {0, -1}:
            raise IndexError(f"{label} is scalar; output_index must be 0.")
        return tensor
    if not -tensor.shape[-1] <= output_index < tensor.shape[-1]:
        raise IndexError(
            f"output_index={output_index} is outside the {tensor.shape[-1]} "
            f"available {label} outputs."
        )
    return tensor[..., output_index].mean()


def _z_expectations_from_probabilities(
    probabilities: Any, n_qubits: int
) -> torch.Tensor:
    """Recover local Z expectations from big-endian basis probabilities."""
    values = torch.as_tensor(probabilities)
    expected_size = 1 << int(n_qubits)
    if values.ndim == 0 or values.shape[-1] != expected_size:
        raise ValueError(
            "Probability output cannot be converted to local Z expectations: "
            f"expected a final dimension of {expected_size}, got "
            f"{tuple(values.shape)}."
        )
    basis = torch.arange(expected_size, device=values.device)
    signs = []
    for position in range(n_qubits):
        shift = n_qubits - 1 - position
        bit = (basis >> shift) & 1
        signs.append(1 - 2 * bit)
    z_table = torch.stack(signs, dim=-1).to(values.dtype)
    return values @ z_table


def _kernel_expectations(model: Any, state: Any) -> torch.Tensor:
    """Return raw projector/Pauli expectations used by a QASP kernel."""
    is_fidelity = hasattr(model, "projector_labels")
    if isinstance(state, (tuple, list)) and len(state) == 2:
        x, y = state
        if is_fidelity:
            return torch.as_tensor(model._overlap_out(x, y)).reshape(-1)
        return torch.cat(
            (
                torch.as_tensor(model._state_out(x)).reshape(-1),
                torch.as_tensor(model._state_out(y)).reshape(-1),
            )
        )

    data = np.asarray(torch.as_tensor(state).detach().cpu(), dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        raise ValueError(
            "Kernel expectation extraction requires state=(x, y) or a "
            "two-dimensional input dataset."
        )
    if is_fidelity:
        pairs = [
            torch.as_tensor(model._overlap_out(data[i], data[j])).reshape(-1)
            for i in range(len(data))
            for j in range(i + 1, len(data))
        ]
        if not pairs:
            pairs = [
                torch.as_tensor(model._overlap_out(data[0], data[0])).reshape(-1)
            ]
        return torch.cat(pairs)
    return torch.cat(
        [torch.as_tensor(model._state_out(sample)).reshape(-1) for sample in data]
    )


def _default_expectations(model: Any, state: Any, kind: str) -> torch.Tensor:
    """Extract pre-decoder quantum expectations for a supported QASP model."""
    if kind == "classifier":
        reference = next(iter(_named_parameters(model, "all")))[1]
        inputs = _as_model_tensor(state, reference)
        raw = model._quantum_output(inputs)
        if getattr(model, "readout", None) == "probabilities":
            raw = _z_expectations_from_probabilities(
                raw, len(model.measurement_qubits)
            )
        return torch.as_tensor(raw).reshape(-1)

    if kind == "policy":
        reference = next(iter(_named_parameters(model, "all")))[1]
        inputs = _as_model_tensor(state, reference)
        if getattr(model, "policy_type", None) == "softmax":
            if getattr(model, "device", None) == "lightning.pp":
                raw = model._get_pp_output(inputs)
            else:
                raw = model.qlayer(inputs)
            return torch.as_tensor(raw).reshape(-1)

        if hasattr(model, "_circuit_output"):
            raw = model._circuit_output(inputs)
        else:
            # Compatibility with the original PQCBorn class, whose public
            # action-probability path applies tanh before circuit execution.
            if model.__class__.__name__ == "PQCBorn":
                inputs = torch.tanh(inputs)
            raw = model.qlayer(inputs)
        if getattr(model, "training_method", None) == "probs":
            raw = _z_expectations_from_probabilities(
                raw, len(model.measurement_qubits)
            )
        return torch.as_tensor(raw).reshape(-1)

    if kind == "kernel":
        return _kernel_expectations(model, state).reshape(-1)

    raise ValueError(
        "A generic model requires expectation_function(model, state) to define "
        "the raw quantum expectation output."
    )


def _default_cost(
    model: Any,
    state: Any,
    kind: str,
    output_index: int,
    targets: Any,
    objective: str | None,
) -> tuple[torch.Tensor, str]:
    """Choose the family-specific scalar diagnostic; generic models require an explicit cost callback."""
    if kind == "classifier":
        reference = next(iter(_named_parameters(model, "all")))[1]
        inputs = _as_model_tensor(state, reference)
        if targets is not None:
            if inputs.ndim == 1:
                inputs = inputs.unsqueeze(0)
            labels = torch.as_tensor(targets, device=inputs.device)
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)
            value = model.compute_loss(inputs, labels, objective=objective)
            return torch.as_tensor(value), objective or model.default_objective
        probabilities = model.predict_proba(inputs)
        return (
            _select_output(probabilities, output_index, "class-probability"),
            f"class_probability[{output_index}]",
        )

    if kind == "policy":
        reference = next(iter(_named_parameters(model, "all")))[1]
        inputs = _as_model_tensor(state, reference)
        if getattr(model, "policy_type", None) == "softmax":
            probabilities = torch.softmax(
                model.forward(inputs, temperature=None), dim=-1
            )
        else:
            probabilities = model.action_probs(inputs, temperature=None)
        return (
            _select_output(probabilities, output_index, "action-probability"),
            f"action_probability[{output_index}]",
        )

    if kind == "kernel":
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            raise ValueError(
                "The default kernel diagnostic needs state=(x, y). When state "
                "is omitted, QASP samples both inputs automatically."
            )
        x, y = state
        value = model.gram(
            np.asarray(torch.as_tensor(x).detach().cpu()).reshape(1, -1),
            np.asarray(torch.as_tensor(y).detach().cpu()).reshape(1, -1),
            return_tensor=True,
        )[0, 0]
        return torch.as_tensor(value), "kernel_similarity"

    raise ValueError(
        "A generic model requires cost_function(model, state) to define the "
        "scalar diagnostic quantity."
    )


def _diagnose_single_model(
    model: Any,
    n_samples: int = 2000,
    initialization: str | Callable[[torch.Tensor], Any] = "random_-pi_pi",
    *,
    state: Any = None,
    input_scaling_init=None,
    state_initialization="normal_0_1",
    state_sampler: Callable[[int, torch.Generator], Any] | None = None,
    state_shape: int | tuple[int, ...] | None = None,
    cost_function: Callable[[Any, Any], Any] | None = None,
    expectation_function: Callable[[Any, Any], Any] | None = None,
    targets: Any = None,
    objective: str | None = None,
    output_index: int = 0,
    parameter_scope: str | Iterable = "quantum",
    model_type: str = "auto",
    seed: int | None = None,
    ddof: int = 1,
    return_samples: bool = False,
) -> dict[str, Any]:
    r"""Estimate cost concentration across random circuit initializations.

    For sampled parameters :math:`\theta_s`, the reported statistic is

    .. math::

       \widehat{\mathrm{Var}}[C]
       = \frac{1}{N-\mathrm{ddof}}\sum_s (C_s-\bar C)^2.

    A fixed ``state`` estimates parameter-only variance conditional on that
    input.  If ``state`` is omitted, a fresh embedded state is sampled for
    every parameter initialization, matching QASP's historical
    ``policy_variance`` behavior but estimating joint state/parameter variance.
    Kernel defaults use a random pair ``(x, y)``.

    Without ``cost_function``, classifiers and policies use the selected class
    or action probability and kernels use ``k(x, y)``.  Passing classifier
    ``targets`` instead evaluates ``model.compute_loss``.  A custom callable
    receives ``(model, sampled_state)`` and must return one scalar.

    The same initialization samples also produce raw quantum expectation
    outputs before class/action decoding and loss post-processing. Observable
    circuits use their measured values directly. For Born-probability circuits,
    local Pauli-Z expectations are reconstructed exactly from the basis
    distribution. ``expectation_function(model, sampled_state)`` can override
    this extraction and may return a scalar or vector.

    initialization accepts shared initializer objects, legacy strings or tensor
    callables. input_scaling_init optionally overrides the input-scale group;
    built-in distributions use the diagnostic's local seeded RNG.

    ``parameter_scope="quantum"`` resamples only the model's ``qlayer``;
    ``"all"`` also includes trainable classical post-processing parameters.
    The model's original parameters and training/evaluation mode are restored.

    Notes:
        Cost concentration is a useful finite-sample warning signal, but the
        standard barren-plateau criterion is exponential decay of gradient
        variance with system size. Compare like-for-like model sizes, costs,
        inputs, and initialization distributions before drawing that stronger
        conclusion.
    """
    if not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("n_samples must be an integer of at least 2.")
    if not isinstance(ddof, int) or ddof < 0 or ddof >= n_samples:
        raise ValueError("ddof must satisfy 0 <= ddof < n_samples.")
    if state is not None and state_sampler is not None:
        raise ValueError("Pass either state or state_sampler, not both.")
    if cost_function is not None and (targets is not None or objective is not None):
        raise ValueError("targets/objective cannot be combined with cost_function.")

    kind = _model_type(model, model_type)
    if targets is not None and kind != "classifier":
        raise ValueError(
            "targets/objective are built-in only for classifiers; use "
            "cost_function for a supervised kernel or policy objective."
        )
    if objective is not None and targets is None:
        raise ValueError("objective requires classifier targets.")
    named_parameters = _named_parameters(model, parameter_scope)
    if not named_parameters:
        raise ValueError(
            "No trainable parameters were found in the selected parameter_scope. "
            "Trainable QASP kernels must be constructed with an ansatz."
        )
    parameters = [value for _, value in named_parameters]
    originals = [value.detach().clone() for value in parameters]
    reference = parameters[0]
    generators: dict[str, torch.Generator] = {}

    def generator_for(device: torch.device):
        """Cache one generator per device, with reproducible seed offsets across devices."""
        key = str(device)
        if key not in generators:
            offset = len(generators)
            generators[key] = _generator(
                device, None if seed is None else int(seed) + offset
            )
        return generators[key]

    state_generator = generator_for(reference.device)
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()

    costs = []
    expectation_samples = []
    running_variance = []
    running_expectation_variance = []
    running_expectation_value_variance = []
    running_mean = 0.0
    running_m2 = 0.0
    expectation_value_mean = 0.0
    expectation_value_m2 = 0.0
    expectation_mean = None
    expectation_m2 = None
    cost_name = getattr(cost_function, "__name__", "custom_cost")
    state_mode = "fixed" if state is not None else (
        "sampler" if state_sampler is not None else "random"
    )

    try:
        with torch.no_grad():
            for sample_index in range(n_samples):
                for name, parameter in named_parameters:
                    spec = (input_scaling_init
                            if name.rsplit(".", 1)[-1] == "input_scaling_params"
                            and input_scaling_init is not None else initialization)
                    from qasp.initialization import initialize_tensor

                    initialize_tensor(parameter, spec, generator=generator_for(parameter.device))

                if state_sampler is not None:
                    sampled_state = state_sampler(sample_index, state_generator)
                elif state is not None:
                    sampled_state = state
                else:
                    sampled_state = _random_state(
                        model,
                        kind,
                        state_initialization,
                        state_generator,
                        reference,
                        state_shape,
                    )

                if cost_function is None:
                    value, cost_name = _default_cost(
                        model,
                        sampled_state,
                        kind,
                        output_index,
                        targets,
                        objective,
                    )
                else:
                    value = torch.as_tensor(cost_function(model, sampled_state))
                if value.numel() != 1:
                    raise ValueError(
                        "cost_function must return exactly one scalar per sample; "
                        f"received shape {tuple(value.shape)}."
                    )
                cost = float(value.detach().cpu().item())
                if not math.isfinite(cost):
                    raise ValueError(
                        f"The diagnostic cost is non-finite at sample {sample_index}."
                    )
                costs.append(cost)

                if expectation_function is None:
                    expectation_value = _default_expectations(
                        model, sampled_state, kind
                    )
                else:
                    expectation_value = torch.as_tensor(
                        expectation_function(model, sampled_state)
                    ).reshape(-1)
                expectation_array = np.asarray(
                    expectation_value.detach().cpu(), dtype=float
                ).reshape(-1)
                if expectation_array.size == 0:
                    raise ValueError(
                        "expectation_function must return at least one value."
                    )
                if not np.isfinite(expectation_array).all():
                    raise ValueError(
                        "The quantum expectation output is non-finite at sample "
                        f"{sample_index}."
                    )
                if expectation_mean is None:
                    expectation_mean = np.zeros_like(expectation_array)
                    expectation_m2 = np.zeros_like(expectation_array)
                elif expectation_array.shape != expectation_mean.shape:
                    raise ValueError(
                        "The quantum expectation output shape changed between "
                        f"samples: expected {expectation_mean.shape}, received "
                        f"{expectation_array.shape}."
                    )
                expectation_samples.append(expectation_array.copy())
                expectation_scalar = float(expectation_array.mean())

                count = sample_index + 1
                delta = cost - running_mean
                running_mean += delta / count
                running_m2 += delta * (cost - running_mean)
                expectation_delta = expectation_array - expectation_mean
                expectation_mean += expectation_delta / count
                expectation_m2 += expectation_delta * (
                    expectation_array - expectation_mean
                )
                expectation_value_delta = (
                    expectation_scalar - expectation_value_mean
                )
                expectation_value_mean += expectation_value_delta / count
                expectation_value_m2 += expectation_value_delta * (
                    expectation_scalar - expectation_value_mean
                )
                denominator = count - ddof
                running_variance.append(
                    running_m2 / denominator if denominator > 0 else float("nan")
                )
                if denominator > 0:
                    running_expectation_variance.append(
                        (expectation_m2 / denominator).copy()
                    )
                else:
                    running_expectation_variance.append(
                        np.full_like(expectation_mean, np.nan)
                    )
                running_expectation_value_variance.append(
                    expectation_value_m2 / denominator
                    if denominator > 0
                    else float("nan")
                )
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals):
                parameter.copy_(original)
        if was_training is not None and hasattr(model, "train"):
            model.train(bool(was_training))

    variance = running_m2 / (n_samples - ddof)
    expectation_variance = expectation_m2 / (n_samples - ddof)
    expectation_value_variance = expectation_value_m2 / (n_samples - ddof)
    standard_deviation = math.sqrt(max(variance, 0.0))
    result = {
        "model_type": kind,
        "cost": cost_name,
        "n_samples": n_samples,
        "initialization": _initialization_label(initialization),
        "input_scaling_init": _initialization_label(input_scaling_init),
        "parameter_scope": (
            parameter_scope if isinstance(parameter_scope, str) else "custom"
        ),
        "parameter_names": tuple(name for name, _ in named_parameters),
        "n_parameters": sum(parameter.numel() for parameter in parameters),
        "state_mode": state_mode,
        "state_initialization": (
            _initialization_label(state_initialization) if state_mode == "random" else None
        ),
        "seed": seed,
        "ddof": ddof,
        "mean": running_mean,
        "variance": variance,
        "cost_variance": variance,
        "standard_deviation": standard_deviation,
        "standard_error_of_mean": standard_deviation / math.sqrt(n_samples),
        "sample_counts": np.arange(1, n_samples + 1),
        "variance_by_sample": np.asarray(running_variance, dtype=float),
        "cost_variance_by_sample": np.asarray(running_variance, dtype=float),
        "expectation_mean": expectation_mean.copy(),
        "expectation_variance": expectation_variance.copy(),
        "expectation_standard_deviation": np.sqrt(
            np.maximum(expectation_variance, 0.0)
        ),
        "expectation_value_mean": expectation_value_mean,
        "expectation_value_variance": expectation_value_variance,
        "expectation_value_standard_deviation": math.sqrt(
            max(expectation_value_variance, 0.0)
        ),
        "expectation_variance_by_sample": np.stack(
            running_expectation_variance
        ),
        "expectation_value_variance_by_sample": np.asarray(
            running_expectation_value_variance, dtype=float
        ),
    }
    if return_samples:
        result["samples"] = np.asarray(costs, dtype=float)
        result["cost_samples"] = np.asarray(costs, dtype=float)
        result["expectation_samples"] = np.stack(expectation_samples)
        result["expectation_value_samples"] = np.asarray(
            expectation_samples, dtype=float
        ).mean(axis=1)
    return result


def barren_plateau_diagnoser(
    model: Any = None,
    n_samples: int = 2000,
    initialization: str | Callable[[torch.Tensor], Any] = "random_-pi_pi",
    *,
    qubit_list: Iterable[int] | None = None,
    model_factory: Callable[[int], Any] | None = None,
    state_factory: Callable[[int], Any] | None = None,
    state: Any = None,
    input_scaling_init=None,
    state_initialization="normal_0_1",
    state_sampler: Callable[[int, torch.Generator], Any] | None = None,
    state_shape: int | tuple[int, ...] | None = None,
    cost_function: Callable[[Any, Any], Any] | None = None,
    expectation_function: Callable[[Any, Any], Any] | None = None,
    targets: Any = None,
    objective: str | None = None,
    output_index: int = 0,
    parameter_scope: str | Iterable = "quantum",
    model_type: str = "auto",
    seed: int | None = None,
    ddof: int = 1,
    return_samples: bool = False,
) -> dict[str, Any]:
    r"""Estimate concentration for one model or across a qubit-width sweep.

    With ``qubit_list=None``, ``model`` is a constructed QASP classifier,
    trainable kernel, or policy and the return value is the single-model
    diagnostic.

    For a barren-plateau scaling experiment, pass ``qubit_list`` and a
    ``model_factory(n_qubits)`` callable.  QASP constructs one model per width
    and performs ``n_samples`` independent parameter initializations at every
    width.  The returned ``cost_variance`` and
    ``expectation_value_variance`` arrays are aligned with ``qubit_list``.
    Componentwise raw-output variances can have different lengths at different
    widths and are therefore returned in
    ``expectation_variance_by_qubit``.  Full single-width results are retained
    in ``results_by_qubit``.

    A fixed ``state`` is reused across widths.  When input dimensionality grows
    with circuit width, ``state_factory(n_qubits)`` supplies one fixed input per
    width, while leaving ``state=None`` samples fresh inputs of each model's
    inferred size.

    The default ``ddof=1`` gives the unbiased estimator

    .. math::

       \widehat{\operatorname{Var}}[C_n]
       = \frac{1}{N-1}\sum_{s=1}^N(C_{n,s}-\bar C_n)^2

    independently for every qubit count :math:`n`.  A small value at one width
    is only a concentration warning; barren-plateau evidence requires studying
    the scaling of a controlled cost (and conventionally its gradient) across
    widths.
    """
    if qubit_list is None:
        if model_factory is not None:
            raise ValueError("model_factory requires qubit_list.")
        if state_factory is not None:
            raise ValueError("state_factory requires qubit_list.")
        if model is None:
            raise ValueError("Pass a constructed model for a single-model diagnosis.")
        return _diagnose_single_model(
            model,
            n_samples=n_samples,
            initialization=initialization,
            input_scaling_init=input_scaling_init,
            state=state,
            state_initialization=state_initialization,
            state_sampler=state_sampler,
            state_shape=state_shape,
            cost_function=cost_function,
            expectation_function=expectation_function,
            targets=targets,
            objective=objective,
            output_index=output_index,
            parameter_scope=parameter_scope,
            model_type=model_type,
            seed=seed,
            ddof=ddof,
            return_samples=return_samples,
        )

    widths = list(qubit_list)
    if not widths:
        raise ValueError("qubit_list must contain at least one qubit count.")
    if any(isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in widths):
        raise ValueError("Every entry in qubit_list must be a positive integer.")
    if len(set(widths)) != len(widths):
        raise ValueError("qubit_list must not contain duplicate qubit counts.")
    if model is not None:
        raise ValueError(
            "A qubit sweep constructs a different model at every width; pass "
            "model_factory instead of model."
        )
    if not callable(model_factory):
        raise ValueError(
            "A qubit sweep requires model_factory(n_qubits) to construct each model."
        )
    if not isinstance(parameter_scope, str):
        raise ValueError(
            "A qubit sweep requires parameter_scope='quantum' or 'all'; custom "
            "parameter iterables belong to one constructed model only."
        )
    if state_factory is not None and not callable(state_factory):
        raise ValueError("state_factory must be callable.")
    if state_factory is not None and (state is not None or state_sampler is not None):
        raise ValueError(
            "state_factory cannot be combined with state or state_sampler."
        )

    results_by_qubit = {}
    for sweep_index, n_qubits in enumerate(widths):
        width_model = model_factory(n_qubits)
        if width_model is None:
            raise ValueError(f"model_factory returned None for {n_qubits} qubits.")
        width_seed = None if seed is None else int(seed) + sweep_index
        width_state = state_factory(n_qubits) if state_factory is not None else state
        width_result = _diagnose_single_model(
            width_model,
            n_samples=n_samples,
            initialization=initialization,
            input_scaling_init=input_scaling_init,
            state=width_state,
            state_initialization=state_initialization,
            state_sampler=state_sampler,
            state_shape=state_shape,
            cost_function=cost_function,
            expectation_function=expectation_function,
            targets=targets,
            objective=objective,
            output_index=output_index,
            parameter_scope=parameter_scope,
            model_type=model_type,
            seed=width_seed,
            ddof=ddof,
            return_samples=return_samples,
        )
        width_result["n_qubits"] = n_qubits
        results_by_qubit[n_qubits] = width_result

    def aligned(name):
        """Collect one numeric diagnostic field in the requested qubit-width order."""
        return np.asarray([results_by_qubit[n][name] for n in widths], dtype=float)

    cost_variance = aligned("cost_variance")
    expectation_value_variance = aligned("expectation_value_variance")
    result = {
        "mode": "qubit_sweep",
        "qubit_list": np.asarray(widths, dtype=int),
        "n_samples": n_samples,
        "n_samples_per_qubit": n_samples,
        "initialization": results_by_qubit[widths[0]]["initialization"],
        "input_scaling_init": results_by_qubit[widths[0]]["input_scaling_init"],
        "parameter_scope": results_by_qubit[widths[0]]["parameter_scope"],
        "state_mode": results_by_qubit[widths[0]]["state_mode"],
        "state_initialization": results_by_qubit[widths[0]][
            "state_initialization"
        ],
        "seed": seed,
        "ddof": ddof,
        "variance": cost_variance,
        "cost_variance": cost_variance,
        "cost_variances": cost_variance,
        "mean": aligned("mean"),
        "cost_mean": aligned("mean"),
        "standard_deviation": aligned("standard_deviation"),
        "standard_error_of_mean": aligned("standard_error_of_mean"),
        "expectation_value_mean": aligned("expectation_value_mean"),
        "expectation_value_variance": expectation_value_variance,
        "expectation_value_variances": expectation_value_variance,
        "expectation_value_standard_deviation": aligned(
            "expectation_value_standard_deviation"
        ),
        "expectation_variance_by_qubit": {
            n: results_by_qubit[n]["expectation_variance"].copy() for n in widths
        },
        "results_by_qubit": results_by_qubit,
    }
    if return_samples:
        result["cost_samples_by_qubit"] = {
            n: results_by_qubit[n]["cost_samples"].copy() for n in widths
        }
        result["expectation_samples_by_qubit"] = {
            n: results_by_qubit[n]["expectation_samples"].copy() for n in widths
        }
        result["expectation_value_samples_by_qubit"] = {
            n: results_by_qubit[n]["expectation_value_samples"].copy()
            for n in widths
        }
    return result


__all__ = ["barren_plateau_diagnoser"]
