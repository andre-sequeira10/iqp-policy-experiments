"""Parametrized quantum deep Q-networks built from QASP circuits."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import nn

from qasp.circuits import AngleAnsatz, CircuitBuilder, IQPAnsatz
from qasp.feature_maps import AngleEmbedding, NeuralEmbedding

from .observables import resolve_observables


class PQDQN(nn.Module):
    """Quantum Q(s, a), with an optional affine classical readout.

    ``observables`` accepts a shared list of Pauli strings, one list per action
    (summed with unit coefficients), or ``'parity-like'`` / ``'majority-modulo'``.
    Named readouts expose the non-identity Pauli-Z terms of QASP's decoder
    decompositions individually. ``preset_actions`` selects the decoder size,
    independently of the output ``n_actions``. ``fc_bias`` defaults to True for
    shared and preset features, False for nested per-action sums.
    ``fc_layer=True`` learns an unrestricted map from these features to
    Q-values. With ``False``, the feature count must equal ``n_actions`` and raw
    expectation values are returned (bounded, usually unsuitable for DQN).

    Inputs have shape ``(..., features)``; outputs have shape ``(..., n_actions)``.
    Sequential circuit evaluation supports every existing feature map. Set
    ``batch_mode='broadcast'`` only for broadcast-compatible circuits, e.g.
    AngleEmbedding with string rotations. No softmax or probability clipping.
    """

    def __init__(
        self,
        *,
        n_qubits,
        n_actions,
        observables,
        circuit_builder=None,
        ansatz=None,
        ansatz_config=None,
        feature_map=None,
        feature_map_config=None,
        fc_layer=True,
        fc_bias=None,
        init="normal_0_1",
        input_scaling_init=None,
        device="default.qubit",
        diff_method="best",
        shots=None,
        measurement_qubits=None,
        majority_blocks=None,
        preset_actions=None,
        preset_mode="features",
        input_activation="tanh",
        batch_mode="sequential",
        mpi=False,
        device_kwargs=None,
        pp_truncation=None,
        pp_topM=20000,
        pp_tau=1e-10,
        pp_prune_every=5,
        pp_prune=True,
        cutting=None,
    ):
        """Build a differentiable observable-based Q-function with n_actions outputs.

        Supply n_qubits, n_actions >= 2 and observables, plus either a builder
        (copied) or ansatz/feature_map names/classes and their dictionaries.
        Dictionary n_layers/features may be callables of n_qubits.

        Readout:
            A flat Pauli-string list gives shared independent features. Nested
            lists require one group per action and sum within groups. Named parity
            or majority presets expose independent nonidentity Pauli terms by
            default. preset_actions sets the decomposition size independently of
            the number of Q-values; measurement_qubits/majority_blocks refine it.
            preset_mode='projectors' retains historical fixed reconstruction only.
            fc_layer=True applies an affine map to n_actions. fc_bias=None enables
            bias for shared lists/presets and disables it for nested groups.
            Without the head, feature count must equal n_actions.

        Execution:
            input_activation is tanh or None. batch_mode is sequential or broadcast;
            NeuralEmbedding and explicit-word AngleEmbedding require sequential
            execution on statevector backends. init and input_scaling_init select
            independent ansatz and feature-map initializers.
            device/diff_method/shots/mpi/device_kwargs configure PennyLane.
            lightning.pp uses analytic Torch autograd, no MPI or shots. Its
            pp_truncation limits Pauli weight; pp_topM/pp_tau/pp_prune_every and
            pp_prune control coefficient pruning. PP always evaluates scalar tapes
            per sample. Truncation and pruning change both predictions and gradients.

        Invalid dimensions, layouts and backend combinations raise ValueError.
        The model returns unconstrained Q-values, not action probabilities."""
        self.cutting = cutting
        super().__init__()
        if not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError("n_qubits must be a positive integer.")
        if not isinstance(n_actions, int) or n_actions < 2:
            raise ValueError("n_actions must be an integer >= 2.")
        if batch_mode not in {"sequential", "broadcast"}:
            raise ValueError("batch_mode must be sequential or broadcast.")
        if input_activation not in {None, "tanh"}:
            raise ValueError("input_activation must be None or 'tanh'.")
        if not isinstance(fc_layer, bool):
            raise ValueError("fc_layer must be a boolean.")
        if fc_bias is not None and not isinstance(fc_bias, bool):
            raise ValueError("fc_bias must be a boolean or None.")
        if device == "lightning.pp" and self.cutting is None:
            if shots is not None:
                raise ValueError("lightning.pp is analytic and requires shots=None.")
            if mpi:
                raise ValueError("lightning.pp does not support mpi=True.")
            if diff_method not in {"best", "backprop"}:
                raise ValueError(
                    "lightning.pp uses Torch autograd; use diff_method='backprop' or 'best'."
                )
            for name, value in [("pp_truncation", pp_truncation), ("pp_topM", pp_topM)]:
                if value is not None and (type(value) is not int or value < 1):
                    raise ValueError(f"{name} must be a positive integer or None.")
            if type(pp_prune_every) is not int or pp_prune_every < 0:
                raise ValueError("pp_prune_every must be a nonnegative integer.")
            if (
                not isinstance(pp_tau, (int, float))
                or not math.isfinite(pp_tau)
                or pp_tau < 0
            ):
                raise ValueError("pp_tau must be finite and nonnegative.")
            if not isinstance(pp_prune, bool):
                raise ValueError("pp_prune must be a boolean.")
        self.device_name = device
        self.pp_truncation, self.pp_prune = pp_truncation, pp_prune
        self.pp_engine, self.pp_tape = None, None
        self.n_qubits, self.n_actions = n_qubits, n_actions
        self.input_activation, self.batch_mode = input_activation, batch_mode
        strings, operators, coefficients = resolve_observables(
            observables,
            n_qubits,
            n_actions,
            measurement_qubits,
            majority_blocks,
            preset_actions=preset_actions,
            preset_mode=preset_mode,
        )
        self.preset_actions = (
            (n_actions if preset_actions is None else preset_actions)
            if isinstance(observables, str)
            else None
        )
        self.preset_mode = preset_mode if isinstance(observables, str) else None
        if fc_bias is None:
            fc_bias = isinstance(observables, str) or all(
                isinstance(s, str) for s in observables
            )
        self.observable_strings = tuple(strings)
        self.register_buffer(
            "readout_coefficients",
            torch.as_tensor(coefficients, dtype=torch.get_default_dtype()),
        )
        if not fc_layer and len(coefficients) != n_actions:
            raise ValueError(
                "Without fc_layer, there must be exactly n_actions readout features."
            )

        if circuit_builder is None:

            def resolve(value, registry, kind):
                """Look up a registered component name or validate an explicitly supplied callable."""
                if isinstance(value, str):
                    if value not in registry:
                        raise ValueError(f"Unknown {kind}: {value!r}.")
                    return registry[value]
                if not callable(value):
                    raise ValueError(f"Provide a {kind} class or registered name.")
                return value

            ansatz = resolve(
                ansatz, {"AngleAnsatz": AngleAnsatz, "IQPAnsatz": IQPAnsatz}, "ansatz"
            )
            feature_map = resolve(
                feature_map,
                {"AngleEmbedding": AngleEmbedding, "NeuralEmbedding": NeuralEmbedding},
                "feature_map",
            )
            ac, fc = deepcopy(ansatz_config or {}), deepcopy(feature_map_config or {})
            for config in (ac, fc):
                if config.get("n_qubits", n_qubits) != n_qubits:
                    raise ValueError("Component n_qubits must match the model.")
                config["n_qubits"] = n_qubits
            if callable(ac.get("n_layers")):
                ac["n_layers"] = ac["n_layers"](n_qubits)
            if callable(fc.get("features")):
                fc["features"] = fc["features"](n_qubits)
            fc.setdefault("features", n_qubits)
            self.fm = feature_map(**fc)
            circuit_builder = CircuitBuilder(n_qubits=n_qubits)
            circuit_builder.set_ansatz(ansatz(**ac, feature_map=self.fm))
        else:
            circuit_builder = deepcopy(circuit_builder)
            if circuit_builder.n_qubits != n_qubits:
                raise ValueError("CircuitBuilder n_qubits must match the model.")
            if circuit_builder.ansatz is None:
                raise ValueError("CircuitBuilder must have an ansatz.")
            self.fm = circuit_builder.ansatz.feature_map
        if (
            device != "lightning.pp"
            and batch_mode == "broadcast"
            and (
                isinstance(self.fm, NeuralEmbedding)
                or (
                    isinstance(self.fm, AngleEmbedding)
                    and not isinstance(self.fm.rotations, str)
                )
            )
        ):
            raise ValueError("This feature map requires batch_mode='sequential'.")
        self.circuit_builder = circuit_builder
        self.circuit_builder.set_measurement("expectation", observables=operators)
        self.circuit_builder.build()
        compiled = self.circuit_builder.compile(
            device=device,
            diff_method=diff_method,
            shots=shots,
            mpi=mpi,
            device_kwargs=device_kwargs,
            cutting=self.cutting,
        )
        if device == "lightning.pp" and self.cutting is None:
            from qasp.rl.differentiable_pp_sim import DifferentiablePauliProp

            self.circuit, self.pp_tape = compiled
            self.pp_engine = DifferentiablePauliProp(
                topM=pp_topM,
                tau=pp_tau,
                prune_every=pp_prune_every,
            )
        else:
            self.circuit = compiled
        self.qlayer = self.circuit_builder.create_torchlayer(initialization=init, input_scaling_init=input_scaling_init)
        self.fc_layer = (
            nn.Linear(len(coefficients), n_actions, bias=fc_bias) if fc_layer else None
        )

    @classmethod
    def from_config(cls, config, *, n_qubits=None, n_actions=None):
        """Build from a copied policy-style dictionary; never mutate caller data."""
        config = deepcopy(config)
        if n_qubits is not None:
            config["n_qubits"] = n_qubits
        if n_actions is not None:
            config["n_actions"] = n_actions
        if callable(config.get("n_actions")):
            config["n_actions"] = config["n_actions"](config["n_qubits"])
        return cls(**config)

    def _pp_single_output(self, inputs):
        # Live Parameters (not state_dict tensors) preserve the autograd graph
        # through rotations, input scaling, and neural embeddings.
        """Trace one input with live parameters and return differentiable PP expectations."""
        weights = dict(self.qlayer.named_parameters())
        kwargs = {"params": weights["params"]}
        if self.fm is not None:
            kwargs["inputs"] = inputs
            if getattr(self.fm, "input_scaling", False):
                kwargs["input_scaling_params"] = weights["input_scaling_params"]
        return self.pp_engine.execute_from_tape(
            self.pp_tape(**kwargs),
            k=self.pp_truncation,
            prune=self.pp_prune,
        ).reshape(-1)

    def quantum_features(self, inputs):
        """Return readout features with shape (..., n_readout_features).

        inputs must have a nonempty final feature axis and match the embedding's
        feature dimension. Conversion uses the quantum parameters' dtype/device;
        optional tanh precedes encoding. Leading batch axes are preserved, including
        empty batches. Raw Pauli expectations are combined by readout_coefficients;
        no affine head or softmax is applied here."""
        parameter = next(self.qlayer.parameters())
        x = torch.as_tensor(inputs, dtype=parameter.dtype, device=parameter.device)
        if x.ndim < 1 or x.shape[-1] == 0:
            raise ValueError("inputs must have shape (..., features).")
        if self.input_activation == "tanh":
            x = torch.tanh(x)
        flat = x.reshape(-1, x.shape[-1])
        if len(flat) == 0:
            return x.new_empty((*x.shape[:-1], self.readout_coefficients.shape[0]))
        if self.pp_engine is not None:
            # The PP engine consumes scalar-angle tapes; both batching modes
            # evaluate samples individually while retaining all leading axes.
            raw = torch.stack([self._pp_single_output(row) for row in flat])
        elif self.batch_mode == "broadcast":
            # CircuitBuilder returns a tuple of measurements; TorchLayer stacks
            # it as (measurements, batch, 1), including singleton dimensions.
            raw = self.qlayer(flat).reshape(len(self.observable_strings), len(flat)).T
        else:
            raw = torch.stack([self.qlayer(row).reshape(-1) for row in flat])
        features = raw @ self.readout_coefficients.T
        return features.reshape(*x.shape[:-1], self.readout_coefficients.shape[0])

    def forward(self, inputs):
        """Return Q-values of shape (..., n_actions) for inputs shaped (..., features).

        Computes quantum_features and then the optional affine head. The output
        is differentiable and unnormalized; greedy action selection belongs to the
        trainer or caller."""
        features = self.quantum_features(inputs)
        return self.fc_layer(features) if self.fc_layer is not None else features

    def q_values(self, inputs):
        """Alias for forward, returning unconstrained action values with the same batch shape."""
        return self(inputs)

    def draw(self, inputs=None, **kwargs):
        """Draw this circuit; use ``output='tikz'`` to return Quantikz source.

        See :func:`qasp.rl.qfunctions.draw_qfunction` for drawing options.
        """
        from .drawing import draw_qfunction

        return draw_qfunction(self, inputs=inputs, **kwargs)
