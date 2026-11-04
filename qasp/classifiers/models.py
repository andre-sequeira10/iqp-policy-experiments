"""Parameterized-quantum-circuit classifiers built from QASP primitives."""

from __future__ import annotations

import itertools
import math
from copy import deepcopy
from numbers import Real
from typing import Any

import pennylane as qml
import torch
from torch import nn

from qasp.circuits.constructor import CircuitBuilder

from .decoders import ClassProbabilityDecoder, balanced_majority_blocks
from .losses import classification_loss


class PQCClassifier(nn.Module):
    """A discrete classifier backed by a QASP parameterized quantum circuit.

    Two readout families are supported:

    ``readout="probabilities"``
        Measure Born probabilities and aggregate basis states into classes with
        the contiguous, parity, or majority decoder.

    ``readout="expectation"``
        For binary classification, measure one or more observables and use
        their arithmetic mean as the signed classifier score. For multiclass
        classification, a flat list supplies one observable score per class;
        alternatively, a nested list supplies one nonempty observable group
        per class and each class score is that group's arithmetic mean.
        :meth:`predict_proba` applies sigmoid/softmax post-processing.

    A preconfigured :class:`~qasp.circuits.CircuitBuilder` can be passed, or the
    same ``ansatz`` / ``feature_map`` class-and-config notation used by QASP's
    policy models can construct the circuit internally.
    """

    _READOUT_ALIASES = {
        "probabilities": "probabilities",
        "probability": "probabilities",
        "probs": "probabilities",
        "born": "probabilities",
        "expectation": "expectation",
        "expectations": "expectation",
        "expval": "expectation",
        "scores": "expectation",
    }

    def __init__(
        self,
        circuit_builder: CircuitBuilder | None = None,
        *,
        ansatz=None,
        ansatz_config: dict[str, Any] | None = None,
        feature_map=None,
        feature_map_config: dict[str, Any] | None = None,
        n_qubits: int = 1,
        n_classes: int = 2,
        readout: str = "probabilities",
        decoder: str | None = None,
        measurement_qubits=None,
        majority_blocks=None,
        strict_majority_blocks: bool = False,
        observables=None,
        observable=None,
        init="normal_0_1",
        input_scaling_init=None,
        device: str = "default.qubit",
        diff_method: str = "best",
        shots: int | None = None,
        output_scaling: bool | float | None = None,
        temperature: float = 1.0,
        enable_mpi: bool | None = None,
        mpi: bool | None = None,
        device_kwargs: dict | None = None,
        pp_truncation: int | None = None,
        pp_topM: int | None = 20000,
        pp_tau: float = 1e-10,
        pp_prune_every: int = 5,
        pp_prune: bool = True,
        cutting=None,
    ) -> None:
        """Construct a Torch classifier from an independent copy of a builder or component configs.

        ``n_qubits`` and ``n_classes`` define circuit width and output labels.
        Supply ``circuit_builder`` or ansatz/feature_map classes and their configs;
        registered names are accepted by the factory. init accepts a shared
        initializer object, legacy name or tensor callable. input_scaling_init
        independently initializes feature-map scales (defaults to init).
        Probability readout aggregates
        basis probabilities using decoder (default contiguous). Expectation readout
        uses observable(s); binary groups are averaged to one score, while multiclass
        requires one observable/group per class. Decoder is then disallowed.

        ``measurement_qubits`` controls the measured physical wires. Majority
        blocks are resolved against those wires. ``output_scaling=True`` registers
        trainable output scales; a number fixes the scale. Temperature must be
        positive. Device, shots, diff_method and device_kwargs configure execution;
        MPI options are reconciled before device creation.

        The internal lightning.pp route supports analytic expectations only, with
        Torch differentiation. pp_truncation limits Pauli weight; pp_topM, pp_tau,
        pp_prune_every and pp_prune control coefficient pruning. These approximate
        the output/gradient, not just runtime. See the class methods for output
        probabilities and objective-dependent losses."""
        self.cutting = cutting
        super().__init__()
        if circuit_builder is None and ansatz is None:
            raise ValueError("Either circuit_builder or ansatz must be provided.")
        if int(n_qubits) < 1:
            raise ValueError("n_qubits must be positive.")
        if int(n_classes) < 2:
            raise ValueError("n_classes must be at least 2.")
        if readout not in self._READOUT_ALIASES:
            supported = ", ".join(sorted(self._READOUT_ALIASES))
            raise ValueError(f"Unknown readout '{readout}'. Supported values: {supported}.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.n_qubits = int(n_qubits)
        self.n_classes = int(n_classes)
        self.readout = self._READOUT_ALIASES[readout]
        if self.readout == "probabilities":
            self.decoder_name = "contiguous" if decoder is None else decoder
        else:
            if decoder is not None:
                raise ValueError(
                    "decoder applies only to probability readout; expectation "
                    "readout uses observable scores directly."
                )
            self.decoder_name = None
        self.init = init
        self.input_scaling_init = input_scaling_init
        self.device_name = device
        self.diff_method = diff_method
        self.shots = shots
        self.temperature = float(temperature)
        from qasp.devices import resolve_mpi

        self.mpi = self.enable_mpi = resolve_mpi(mpi, enable_mpi)
        self.device_kwargs = device_kwargs
        self.pp_truncation = pp_truncation
        self.pp_topM = pp_topM
        self.pp_tau = pp_tau
        self.pp_prune_every = pp_prune_every
        self.pp_prune = bool(pp_prune)
        self.pp_engine = None
        self.pp_tape = None
        if self.device_name == "lightning.pp" and self.cutting is None:
            if self.readout != "expectation":
                raise ValueError(
                    "device='lightning.pp' supports only expectation readout; "
                    "Born-probability readout requires a probability-capable simulator."
                )
            if self.shots is not None:
                raise ValueError(
                    "device='lightning.pp' is an analytic Pauli-propagation "
                    "backend and requires shots=None."
                )
            if self.enable_mpi:
                raise ValueError(
                    "device='lightning.pp' does not support enable_mpi=True."
                )
            if self.pp_truncation is not None and int(self.pp_truncation) < 1:
                raise ValueError("pp_truncation must be a positive integer or None.")
            if self.pp_topM is not None and int(self.pp_topM) < 1:
                raise ValueError("pp_topM must be a positive integer or None.")
            if float(self.pp_tau) < 0:
                raise ValueError("pp_tau must be nonnegative.")
            if int(self.pp_prune_every) < 0:
                raise ValueError("pp_prune_every must be nonnegative.")
        self.requested_majority_blocks = majority_blocks
        self.strict_majority_blocks = bool(strict_majority_blocks)
        if observable is not None and observables is not None:
            raise ValueError("Pass either observable or observables, not both.")
        configured_observables = observables if observables is not None else observable
        if configured_observables is None:
            self.observables = None
        elif isinstance(configured_observables, (str, qml.operation.Operator)):
            self.observables = (configured_observables,)
        else:
            self.observables = tuple(configured_observables)
        self.observable_groups = None

        self.ansatz = ansatz
        self.ansatz_config = None if ansatz_config is None else dict(ansatz_config)
        self.feature_map = feature_map
        self.feature_map_config = (
            None if feature_map_config is None else dict(feature_map_config)
        )

        if circuit_builder is None:
            if feature_map is None or feature_map_config is None:
                raise ValueError(
                    "feature_map and feature_map_config are required when "
                    "circuit_builder is not provided."
                )
            fm_config = dict(feature_map_config)
            fm_config.setdefault("n_qubits", self.n_qubits)
            fm = feature_map(**fm_config)
            ans_config = dict(ansatz_config or {})
            ans_config.setdefault("n_qubits", self.n_qubits)
            circuit_ansatz = ansatz(**ans_config, feature_map=fm)
            self.circuit_builder = CircuitBuilder(n_qubits=self.n_qubits)
            self.circuit_builder.set_ansatz(circuit_ansatz)
        else:
            # Readout configuration mutates a builder, so protect a caller's
            # reusable circuit definition from classifier-specific measurements.
            self.circuit_builder = deepcopy(circuit_builder)
            if self.circuit_builder.n_qubits != self.n_qubits:
                raise ValueError(
                    "n_qubits does not match circuit_builder.n_qubits: "
                    f"{self.n_qubits} != {self.circuit_builder.n_qubits}."
                )

        self.measurement_qubits = self._resolve_measurement_qubits(
            measurement_qubits
        )
        self.majority_blocks = tuple()
        self.unused_majority_qubits = tuple()
        self.probability_decoder = None
        self.n_outputs = self.n_classes
        self.n_measurements = self.n_classes

        if self.readout == "probabilities":
            self._configure_probability_readout()
            default_objective = (
                "binary_cross_entropy"
                if self.n_classes == 2
                else "categorical_cross_entropy"
            )
        else:
            self._configure_expectation_readout()
            default_objective = "hinge"
        self.default_objective = default_objective

        self.circuit = self.circuit_builder.build()
        compiled = self.circuit_builder.compile(
            device=self.device_name,
            diff_method=self.diff_method,
            shots=self.shots,
            mpi=self.enable_mpi,
            device_kwargs=self.device_kwargs,
            cutting=self.cutting,
        )
        if self.device_name == "lightning.pp" and self.cutting is None:
            self.qnode, self.pp_tape = compiled
            # Import only when requested so ordinary classifier workflows do
            # not eagerly load the reinforcement-learning dependency stack.
            from qasp.rl.differentiable_pp_sim import DifferentiablePauliProp

            self.pp_engine = DifferentiablePauliProp(
                topM=self.pp_topM,
                tau=self.pp_tau,
                prune_every=self.pp_prune_every,
            )
        else:
            self.qnode = compiled
        self.qlayer = self.circuit_builder.create_torchlayer(
            initialization=self.init, input_scaling_init=self.input_scaling_init
        )

        self.output_scaling = output_scaling
        if output_scaling is True:
            self.output_scale = nn.Parameter(torch.ones(self.n_outputs))
        elif isinstance(output_scaling, Real) and not isinstance(output_scaling, bool):
            self.register_buffer(
                "output_scale", torch.full((self.n_outputs,), float(output_scaling))
            )
        else:
            self.output_scale = None

    def _resolve_measurement_qubits(self, measurement_qubits) -> list[int]:
        """Choose default measured wires or validate an explicit unique, in-range sequence."""
        decoder = ClassProbabilityDecoder._ALIASES.get(self.decoder_name)
        if self.readout == "probabilities" and decoder is None:
            supported = ", ".join(sorted(ClassProbabilityDecoder._ALIASES))
            raise ValueError(
                f"Unknown decoder '{self.decoder_name}'. Supported values: {supported}."
            )

        if measurement_qubits is None:
            if self.readout == "probabilities" and decoder == "contiguous":
                n_measured = math.ceil(math.log2(self.n_classes))
                wires = list(range(n_measured))
            else:
                wires = list(range(self.n_qubits))
        else:
            wires = [int(wire) for wire in measurement_qubits]

        if not wires:
            raise ValueError("measurement_qubits must be non-empty.")
        if len(set(wires)) != len(wires):
            raise ValueError("measurement_qubits must not contain duplicates.")
        if any(wire < 0 or wire >= self.n_qubits for wire in wires):
            raise ValueError(
                f"measurement_qubits must lie in [0, {self.n_qubits})."
            )
        return wires

    def _configure_probability_readout(self) -> None:
        """Install the probability measurement and decoder matrix for the configured classes."""
        n_measured = len(self.measurement_qubits)
        if self.n_classes > (1 << n_measured):
            raise ValueError(
                f"n_classes={self.n_classes} requires at least "
                f"ceil(log2(n_classes))={math.ceil(math.log2(self.n_classes))} "
                f"measured qubits; received {n_measured}."
            )
        normalized = ClassProbabilityDecoder._ALIASES[self.decoder_name]
        position_blocks = None
        if normalized == "majority":
            (
                self.majority_blocks,
                position_blocks,
                self.unused_majority_qubits,
            ) = balanced_majority_blocks(
                self.measurement_qubits,
                self.n_classes,
                self.requested_majority_blocks,
                strict=self.strict_majority_blocks,
            )

        self.probability_decoder = ClassProbabilityDecoder(
            decoder=self.decoder_name,
            n_classes=self.n_classes,
            n_measured_qubits=n_measured,
            majority_blocks=position_blocks,
        )
        self.circuit_builder.set_measurement(
            "probs", wires=self.measurement_qubits
        )

    def _default_observables(self, n_outputs: int):
        """Generate enough nonidentity Z products in increasing support size for class scores."""
        labels = []
        for size in range(1, len(self.measurement_qubits) + 1):
            for combination in itertools.combinations(self.measurement_qubits, size):
                labels.append(
                    ("Z" * size) + " " + " ".join(map(str, combination))
                )
                if len(labels) == n_outputs:
                    return labels
        if n_outputs > len(labels):
            raise ValueError(
                f"The {len(self.measurement_qubits)} measured qubits provide only "
                f"{len(labels)} distinct non-identity Z observables, fewer than the "
                f"{n_outputs} requested classifier outputs. Provide observables explicitly."
            )
        return labels[:n_outputs]

    def _configure_expectation_readout(self) -> None:
        """Validate observable groups, configure expectations and store their mean-aggregation matrix."""
        self.n_outputs = 1 if self.n_classes == 2 else self.n_classes
        if self.observables is None:
            self.observables = tuple(self._default_observables(self.n_outputs))
        if not self.observables:
            raise ValueError("Expectation readout requires at least one observable.")

        def is_observable(value) -> bool:
            """Recognize a Pauli specification string or an existing PennyLane operator."""
            return isinstance(value, (str, qml.operation.Operator))

        contains_groups = any(not is_observable(value) for value in self.observables)
        if contains_groups:
            if any(is_observable(value) for value in self.observables):
                raise ValueError(
                    "Do not mix individual observables and observable groups. "
                    "Use a flat list or one nested group per classifier output."
                )
            expected_groups = 1 if self.n_classes == 2 else self.n_classes
            if len(self.observables) != expected_groups:
                raise ValueError(
                    f"Expectation readout requires {expected_groups} observable "
                    f"groups for n_classes={self.n_classes}; received "
                    f"{len(self.observables)}."
                )
            groups = []
            for index, configured_group in enumerate(self.observables):
                try:
                    group = tuple(configured_group)
                except TypeError as error:
                    raise TypeError(
                        f"Observable group {index} must be an iterable of "
                        "observable strings or PennyLane operators."
                    ) from error
                if not group:
                    raise ValueError(f"Observable group {index} must be nonempty.")
                if any(not is_observable(value) for value in group):
                    raise TypeError(
                        f"Observable group {index} contains an unsupported value. "
                        "Use observable strings or PennyLane operators."
                    )
                groups.append(group)
            self.observable_groups = tuple(groups)
            self.observables = tuple(
                observable
                for group in self.observable_groups
                for observable in group
            )
        else:
            if self.n_classes > 2 and len(self.observables) != self.n_outputs:
                raise ValueError(
                    f"Expectation readout requires {self.n_outputs} observables "
                    f"(one per class) or {self.n_outputs} nested observable groups "
                    f"for n_classes={self.n_classes}; received "
                    f"{len(self.observables)} observables."
                )
            if self.n_classes == 2:
                self.observable_groups = (tuple(self.observables),)
            else:
                self.observable_groups = tuple(
                    (observable,) for observable in self.observables
                )

        self.n_measurements = len(self.observables)
        self.circuit_builder.set_measurement(
            "expectation",
            wires=self.measurement_qubits,
            observables=self.observables,
        )

    def _standardize_expectations(self, values: torch.Tensor) -> torch.Tensor:
        """Move PennyLane's measurement axis to the conventional final axis."""
        if self.n_measurements == 1:
            if values.ndim == 0:
                return values.reshape(1)
            if values.shape[-1] == 1:
                return values
            if values.shape[0] == 1:
                values = values.movedim(0, -1)
                if values.ndim >= 2 and values.shape[-2] == 1:
                    values = values.squeeze(-2)
                return values
            return values.unsqueeze(-1)

        if values.ndim == 1 and values.shape[0] == self.n_measurements:
            return values
        if values.shape[-1] == self.n_measurements:
            return values
        if values.shape[0] == self.n_measurements:
            values = values.movedim(0, -1)
            if values.ndim >= 2 and values.shape[-2] == 1:
                values = values.squeeze(-2)
            return values
        raise ValueError(
            "Could not identify the expectation-measurement axis in output shape "
            f"{tuple(values.shape)}; expected {self.n_measurements} measurements."
        )

    def expectation_values(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return every flattened observable expectation before aggregation."""
        if self.readout != "expectation":
            raise RuntimeError(
                "expectation_values is available only for expectation readout."
            )
        return self._standardize_expectations(self._quantum_output(inputs))

    def _pp_single_output(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate one example with differentiable Pauli propagation."""
        if self.pp_engine is None or self.pp_tape is None:
            raise RuntimeError(
                "The differentiable Pauli-propagation backend is not initialized."
            )

        weights = dict(self.qlayer.named_parameters())
        if "params" not in weights:
            raise RuntimeError(
                "Could not find `params` in the classifier quantum layer. "
                f"Available parameters: {list(weights)}"
            )

        tape_kwargs = {"params": weights["params"]}
        feature_map = getattr(self.circuit_builder.ansatz, "feature_map", None)
        if feature_map is not None:
            tape_kwargs["inputs"] = inputs
            if getattr(feature_map, "input_scaling", False):
                tape_kwargs["input_scaling_params"] = weights.get(
                    "input_scaling_params"
                )

        tape = self.pp_tape(**tape_kwargs)
        return self.pp_engine.execute_from_tape(
            tape,
            k=self.pp_truncation,
            prune=self.pp_prune,
        )

    def _quantum_output(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute the configured exact or Pauli-propagation backend."""
        if self.device_name != "lightning.pp" or self.cutting is not None:
            return self.qlayer(inputs)
        if inputs.ndim == 1:
            return self._pp_single_output(inputs)
        if inputs.ndim == 2:
            return torch.stack(
                [self._pp_single_output(sample) for sample in inputs], dim=0
            )
        raise ValueError(
            "Classifier inputs must be a feature vector or a two-dimensional "
            "batch of feature vectors for device='lightning.pp'."
        )

    def _raw_output(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute the quantum layer and aggregate expectation groups before score scaling."""
        values = self._quantum_output(inputs)
        if self.readout == "probabilities":
            return self.probability_decoder(values)
        values = self._standardize_expectations(values)
        if self.n_classes == 2:
            return values.mean(dim=-1, keepdim=True)
        class_scores = []
        start = 0
        for group in self.observable_groups:
            stop = start + len(group)
            class_scores.append(values[..., start:stop].mean(dim=-1))
            start = stop
        return torch.stack(class_scores, dim=-1)

    def forward(
        self, inputs: torch.Tensor, *, temperature: float | None = None
    ) -> torch.Tensor:
        """Return class probabilities or raw expectation-value scores."""
        outputs = self._raw_output(inputs)
        if self.output_scale is not None and self.readout == "expectation":
            outputs = outputs * self.output_scale

        if self.readout == "probabilities":
            outputs = outputs.clamp_min(0.0)
            eps = torch.finfo(outputs.dtype).eps
            outputs = outputs / outputs.sum(dim=-1, keepdim=True).clamp_min(eps)
            selected_temperature = self.temperature if temperature is None else temperature
            if selected_temperature <= 0:
                raise ValueError("temperature must be positive.")
            if self.output_scale is not None or selected_temperature != 1.0:
                scale = 1.0 if self.output_scale is None else self.output_scale
                outputs = torch.softmax(
                    scale * torch.log(outputs.clamp_min(eps)) / selected_temperature,
                    dim=-1,
                )
        return outputs

    def decision_function(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return raw classifier scores (probabilities for Born readout)."""
        return self.forward(inputs)

    def predict_proba(
        self, inputs: torch.Tensor, *, temperature: float | None = None
    ) -> torch.Tensor:
        """Return a normalized probability for every discrete class."""
        selected_temperature = self.temperature if temperature is None else temperature
        if selected_temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.readout == "probabilities":
            return self.forward(inputs, temperature=selected_temperature)

        scores = self.forward(inputs)
        if self.n_classes == 2:
            positive = torch.sigmoid(scores.squeeze(-1) / selected_temperature)
            return torch.stack((1.0 - positive, positive), dim=-1)
        return torch.softmax(scores / selected_temperature, dim=-1)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the maximum-probability class index."""
        if self.readout == "expectation" and self.n_classes == 2:
            return (self.forward(inputs).squeeze(-1) > 0.0).to(torch.long)
        return self.predict_proba(inputs).argmax(dim=-1)

    def draw(self, inputs=None, **kwargs):
        """Draw this classifier via :func:`qasp.classifiers.draw_classifier`."""
        from .factory import draw_classifier

        return draw_classifier(classifier=self, inputs=inputs, **kwargs)

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        objective: str | None = None,
        **loss_kwargs,
    ) -> torch.Tensor:
        """Evaluate a named QASP classifier objective on a batch."""
        objective = self.default_objective if objective is None else objective
        normalized = objective.lower()
        if normalized in {
            "binary_cross_entropy",
            "binary_ce",
            "bce",
            "categorical_cross_entropy",
            "cross_entropy",
            "cce",
            "nll",
        }:
            outputs = self.predict_proba(inputs)
        else:
            outputs = self.forward(inputs)
        return classification_loss(outputs, targets, objective, **loss_kwargs)

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Return the quantum layer's parameter tensors."""
        return self.qlayer.qnode_weights

    def set_weights(self, *, params=None, input_scaling_params=None) -> None:
        """Delegate strict parameter loading to the circuit builder; include input scales if enabled."""
        self.circuit_builder.set_weights(
            params=params, input_scaling_params=input_scaling_params
        )


QuantumClassifier = PQCClassifier


__all__ = ["PQCClassifier", "QuantumClassifier"]
