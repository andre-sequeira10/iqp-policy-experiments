"""Parameterized quantum circuit Born machines using QASP's circuit backends."""

from __future__ import annotations

import warnings
from copy import deepcopy

import numpy as np
import pennylane as qml
import torch

from qasp.circuits import AngleAnsatz, CircuitBuilder, IQPAnsatz
from qasp.devices import create_qnode, create_device, resolve_mpi
from qasp.feature_maps import AngleEmbedding, NeuralEmbedding

from .losses import PauliMMD, basis_indices, binary_matrix, nll_loss, positive_integer


def _resolve(value, registry, name):
    """Resolve a public template name or accept an explicit template class."""
    if isinstance(value, str):
        if value not in registry:
            raise ValueError(
                f"Unknown {name} {value!r}; choose from {sorted(registry)}."
            )
        return registry[value]
    if not callable(value):
        raise TypeError(f"{name} must be a registered name or template class.")
    return value


def _numpy(value):
    """Detach a parameter for export, diagnostics or initialization."""
    return (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )


class PQCBM:
    """Train a Born distribution with NLL or expectation-based Gaussian MMD².

    Construct from ``ansatz`` / ``ansatz_config`` and optional ``feature_map`` /
    ``feature_map_config``, a CircuitBuilder, or an existing probability QNode.
    Data are binary rows (N,n_qubits), with wire zero the most significant bit.
    Feature maps encode one fixed conditioning vector ``inputs``; training data
    are output bitstrings, not the feature-map inputs. Input scaling is trainable.

    Torch is the default interface; ``interface='autograd'`` supports PennyLane
    optimizers and the legacy natural-gradient path. ``train`` (alias ``fit``)
    returns (params, pre-update losses). This object is not a torch.nn.Module.
    ``parameters()`` supplies the live tensors for external Torch optimizers.

    ``loss='mmd'`` uses PauliMMD and never needs a probability vector. The
    ``lightning.pp`` backend uses differentiable Pauli propagation and requires
    Torch, analytic execution, and no MPI. Statevector devices forward MPI and
    plugin options through CircuitBuilder; a compatible installed plugin is
    required. There is no silent serial fallback. See docs/genlearn.md.
    """

    def __init__(
        self,
        circuit=None,
        init_params=None,
        data=None,
        *,
        n_qubits=None,
        circuit_builder=None,
        ansatz="AngleAnsatz",
        ansatz_config=None,
        feature_map=None,
        feature_map_config=None,
        inputs=None,
        input_scaling_params=None,
        loss="nll",
        mmd_config=None,
        optimizer=None,
        lr=0.1,
        batch_size=None,
        seed=0,
        interface="torch",
        init="random_0_2pi",
        input_scaling_init=None,
        device="default.qubit",
        diff_method="best",
        shots=None,
        mpi=None,
        enable_mpi=None,
        device_kwargs=None,
        natural_grad=False,
        commuting=False,
        fisher="quantum",
        lambda_reg=1e-2,
        pp_truncation=None,
        pp_topM=20000,
        pp_tau=1e-10,
        pp_prune_every=5,
        pp_prune=True,
        max_probability_qubits=20,
        cutting=None,
    ):
        """Configure a model; device allocation occurs but circuits do not execute.

        ``init`` accepts qasp.initialization objects, legacy strings or Torch-style
        callables. ``input_scaling_init`` separately initializes input scales
        (ones by default). Explicit init_params/input_scaling_params override
        their respective distribution. seed controls built-in initialization, data batches and
        default MMD masks independently of global RNG state. Set identical seeds,
        data and configs on every MPI rank. Plugin shot RNG uses device_kwargs.
        ``mmd_config`` accepts PauliMMD arguments except n_qubits. Natural gradient
        supports analytic Autograd models without trainable input scales.
        """
        self.cutting = cutting
        if circuit is not None and circuit_builder is not None:
            raise ValueError("Provide circuit or circuit_builder, not both.")
        circuit = circuit if circuit is not None else circuit_builder
        if n_qubits is None:
            n_qubits = getattr(circuit, "n_qubits", None)
            if n_qubits is None and isinstance(circuit, qml.QNode):
                n_qubits = len(circuit.device.wires)
            if n_qubits is None and data is not None:
                n_qubits = binary_matrix(data).shape[1]
        self.n_qubits = positive_integer(n_qubits, "n_qubits")
        if interface not in {"torch", "autograd"}:
            raise ValueError("interface must be 'torch' or 'autograd'.")
        if loss not in {"nll", "mmd"}:
            raise ValueError("loss must be 'nll' or 'mmd'.")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive.")
        if batch_size is not None:
            batch_size = positive_integer(batch_size, "batch_size")
        self.mpi = resolve_mpi(mpi, enable_mpi)
        if self.mpi and seed is None:
            raise ValueError("MPI training requires an explicit shared seed.")
        self.seed, self.rng = seed, np.random.default_rng(seed)
        self.interface, self.loss_name = interface, loss
        self.lr, self.batch_size = lr, batch_size
        self.device_name, self.diff_method, self.shots = device, diff_method, shots
        self.device_kwargs = dict(device_kwargs or {})
        self.max_probability_qubits = positive_integer(
            max_probability_qubits, "max_probability_qubits"
        )
        self.data = None if data is None else binary_matrix(data, self.n_qubits)
        self.inputs = None if inputs is None else np.asarray(inputs, dtype=float)
        self.pp_engine = None
        self.pp_truncation, self.pp_prune = pp_truncation, pp_prune
        if device == "lightning.pp" and self.cutting is None:
            if interface != "torch" or loss != "mmd":
                raise ValueError(
                    "lightning.pp requires interface='torch' and loss='mmd'."
                )
            if shots is not None or self.mpi:
                raise ValueError("lightning.pp requires shots=None and mpi=False.")
            if diff_method not in {"best", "backprop"}:
                raise ValueError(
                    "lightning.pp uses backprop/best through Torch autograd."
                )
            for key, value in (("pp_truncation", pp_truncation), ("pp_topM", pp_topM)):
                if value is not None:
                    positive_integer(value, key)
            if type(pp_prune_every) is not int or pp_prune_every < 0:
                raise ValueError("pp_prune_every must be a nonnegative integer.")
            if not np.isfinite(pp_tau) or pp_tau < 0 or type(pp_prune) is not bool:
                raise ValueError(
                    "pp_tau must be finite and nonnegative; pp_prune must be boolean."
                )
            from qasp.rl.differentiable_pp_sim import DifferentiablePauliProp

            self.pp_engine = DifferentiablePauliProp(
                topM=pp_topM, tau=pp_tau, prune_every=pp_prune_every
            )
        self.mmd = None
        if loss == "mmd":
            options = dict(mmd_config or {})
            options.setdefault("seed", seed)
            if (
                self.mpi
                and options["seed"] is None
                and options.get("mode", "sampled") == "sampled"
            ):
                raise ValueError(
                    "MPI sampled MMD requires an explicit shared MMD seed."
                )
            self.mmd = PauliMMD(self.n_qubits, **options)
        elif mmd_config:
            raise ValueError("mmd_config requires loss='mmd'.")
        if loss == "nll":
            self._check_probability_size()

        self.circuit_builder = None
        self.fm = None
        self._source_node = circuit if isinstance(circuit, qml.QNode) else None
        if self._source_node is not None:
            if init_params is None:
                raise ValueError("An existing QNode requires init_params.")
            if self.pp_engine is not None:
                raise ValueError(
                    "For lightning.pp, pass a CircuitBuilder or ansatz configuration."
                )
            if device_kwargs:
                raise ValueError("device_kwargs cannot configure an existing QNode.")
            create_device(circuit.device, wires=circuit.device.wires, mpi=self.mpi)
            self.device = circuit.device
            if getattr(self.device.shots, "has_partitioned_shots", False):
                raise ValueError(
                    "PQCBM requires a single shot count, not a shot vector."
                )
            actual_shots = self.device.shots.total_shots
            if shots is not None and shots != actual_shots:
                raise ValueError("shots cannot reconfigure an existing QNode.")
            self.shots = actual_shots
            self.diff_method = circuit.diff_method
            self.circuit = circuit
            self.wires = list(self.device.wires)[: self.n_qubits]
            if self.cutting is not None:
                self.device = create_device(circuit.device, wires=self.wires, cutting=self.cutting)
                self.circuit = create_qnode(circuit.func, self.device, interface=interface, diff_method=self.diff_method)
            param_shape, scale_shape = np.shape(init_params), None
        else:
            if circuit is None:
                ac = deepcopy(ansatz_config or {})
                if ac.get("n_qubits", self.n_qubits) != self.n_qubits:
                    raise ValueError("ansatz n_qubits must match the model.")
                ac["n_qubits"] = self.n_qubits
                if callable(ac.get("n_layers")):
                    ac["n_layers"] = ac["n_layers"](self.n_qubits)
                if feature_map is not None:
                    fc = deepcopy(feature_map_config or {})
                    if fc.get("n_qubits", self.n_qubits) != self.n_qubits:
                        raise ValueError("feature_map n_qubits must match the model.")
                    fc["n_qubits"] = self.n_qubits
                    fc.setdefault("features", self.n_qubits)
                    if callable(fc.get("features")):
                        fc["features"] = fc["features"](self.n_qubits)
                    self.fm = _resolve(
                        feature_map,
                        {
                            "AngleEmbedding": AngleEmbedding,
                            "angle": AngleEmbedding,
                            "NeuralEmbedding": NeuralEmbedding,
                            "neural": NeuralEmbedding,
                        },
                        "feature_map",
                    )(**fc)
                elif feature_map_config:
                    raise ValueError("feature_map_config requires feature_map.")
                circuit = CircuitBuilder(self.n_qubits)
                cls = _resolve(
                    ansatz,
                    {
                        "AngleAnsatz": AngleAnsatz,
                        "angle": AngleAnsatz,
                        "IQPAnsatz": IQPAnsatz,
                        "iqp": IQPAnsatz,
                    },
                    "ansatz",
                )
                circuit.set_ansatz(cls(**ac, feature_map=self.fm))
            else:
                circuit = deepcopy(circuit)
                if circuit.n_qubits != self.n_qubits or circuit.ansatz is None:
                    raise ValueError(
                        "CircuitBuilder must have a matching width and an ansatz."
                    )
                self.fm = circuit.ansatz.feature_map
            self.circuit_builder = circuit
            self.wires = list(range(self.n_qubits))
            if self.fm is not None:
                features = getattr(self.fm, "features", None) or self.n_qubits
                if self.inputs is None or self.inputs.shape != (features,):
                    raise ValueError(
                        f"Provide fixed inputs with shape ({features},) for the feature map."
                    )
            elif self.inputs is not None:
                raise ValueError(
                    "inputs requires a feature map or an existing encoded QNode."
                )
            shapes = circuit.ansatz.get_param_shapes()
            param_shape, scale_shape = (
                shapes[-1],
                shapes[0] if len(shapes) > 1 else None,
            )
            circuit.set_measurement(
                "expectation", observables=[qml.Z(0)]
            ) if loss == "mmd" else circuit.set_measurement("probs")
            circuit.build()
            compiled = circuit.compile(
                device=device,
                diff_method=diff_method,
                shots=shots,
                mpi=self.mpi,
                device_kwargs=device_kwargs,
                cutting=self.cutting,
            )
            self.circuit = compiled[0] if self.pp_engine is not None else compiled
            self.device = circuit.device
            if getattr(self.device.shots, "has_partitioned_shots", False):
                raise ValueError(
                    "PQCBM requires a single shot count, not a shot vector."
                )
            self.shots = self.device.shots.total_shots
        self.params = self._parameter(init_params, param_shape, init)
        self.input_scaling_params = None
        if self.fm is not None and self.fm.input_scaling:
            self.input_scaling_params = self._parameter(
                input_scaling_params, scale_shape,
                "ones" if input_scaling_init is None else input_scaling_init
            )
        elif input_scaling_params is not None or input_scaling_init is not None:
            raise ValueError(
                "input_scaling_init/params requires an input-scaling feature map."
            )
        if self._source_node is not None:
            tape = qml.tape.make_qscript(self._source_node.func)(**self._kwargs())
            if len(tape.measurements) != 1 or not isinstance(
                tape.measurements[0], qml.measurements.ProbabilityMP
            ):
                raise ValueError(
                    "An existing QNode must return one computational-basis probability measurement."
                )
            measurement = tape.measurements[0]
            if measurement.obs is not None:
                raise ValueError(
                    "The source QNode must measure probabilities in the computational basis."
                )
            self.wires = list(measurement.wires or self.device.wires)
            if len(self.wires) != self.n_qubits:
                raise ValueError(
                    "n_qubits must match the QNode's probability readout width."
                )
        self._moment_node = create_qnode(
            self._moment_function,
            self.device,
            interface=interface,
            diff_method=self.diff_method if self.pp_engine is None else "best",
        )
        self.natural_grad, self.commuting = natural_grad, commuting
        self.fisher, self.lambda_reg = fisher, lambda_reg
        self.Freg = None
        if natural_grad and self.cutting is not None:
            raise ValueError("Natural-gradient QNode transforms are not yet supported with cutting; use ordinary gradient optimization.")
        if natural_grad:
            if (
                interface != "autograd"
                or self.shots is not None
                or self.input_scaling_params is not None
            ):
                raise ValueError(
                    "Natural gradient requires analytic Autograd execution without trainable input scales."
                )
            if (
                fisher not in {"quantum", "classical"}
                or not np.isfinite(lambda_reg)
                or lambda_reg <= 0
            ):
                raise ValueError(
                    "fisher must be quantum/classical and lambda_reg finite and positive."
                )
            self._check_probability_size()  # Fisher transforms are small-system tools.
            self._fisher_node = create_qnode(
                self._probability_function,
                self.device,
                interface="autograd",
                diff_method=self.diff_method,
            )
            if commuting:
                self.Freg = self._fisher_matrix()
        self.opt = optimizer
        if not natural_grad and interface == "torch" and callable(optimizer):
            self.opt = optimizer(self.parameters())
        if not natural_grad and self.opt is None:
            self.opt = (
                torch.optim.Adam(self.parameters(), lr=lr)
                if interface == "torch"
                else qml.AdamOptimizer(stepsize=lr)
            )
        if not natural_grad and interface == "torch":
            if not isinstance(self.opt, torch.optim.Optimizer):
                raise TypeError(
                    "Torch training requires an optimizer factory, e.g. lambda p: torch.optim.SGD(p, lr=0.1)."
                )
            bound = {id(p) for group in self.opt.param_groups for p in group["params"]}
            if bound != {id(p) for p in self.parameters()}:
                raise ValueError(
                    "The optimizer must own this model's parameters; supply an optimizer factory."
                )
        elif not natural_grad and not hasattr(self.opt, "step_and_cost"):
            raise TypeError(
                "Autograd training requires a PennyLane optimizer with step_and_cost."
            )
        self.loss_history = []

    def _parameter(self, value, shape, initialization):
        """Create one trainable parameter array with a reproducible initializer."""
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if value is None:
            from qasp.initialization import sample_parameters

            value = sample_parameters(shape, initialization, rng=self.rng)
        value = _numpy(value)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(
                f"Initial parameter shape must be {shape} with finite entries; got {value.shape}."
            )
        if self.interface == "torch":
            return torch.nn.Parameter(torch.tensor(value.copy(), dtype=torch.float64))
        return qml.numpy.array(value.copy(), requires_grad=True)

    def parameters(self):
        """Return the live ansatz and optional input-scale tensors in order."""
        return [self.params] + (
            [] if self.input_scaling_params is None else [self.input_scaling_params]
        )

    @property
    def n_parameters(self):
        """Total trainable scalar count, including learned feature-map scales."""
        return sum(int(np.prod(parameter.shape)) for parameter in self.parameters())

    @property
    def indices(self):
        """Lazily convert stored binary data to indices (never used by MMD)."""
        if self.data is None:
            raise ValueError("No training data supplied.")
        return basis_indices(self.data)

    @classmethod
    def from_config(cls, config, *, n_qubits=None, data=None):
        """Construct from a copied dictionary with optional dimension/data overrides."""
        config = deepcopy(config)
        if n_qubits is not None:
            if config.get("n_qubits", n_qubits) != n_qubits:
                raise ValueError("n_qubits conflicts with config.")
            config["n_qubits"] = n_qubits
        if data is not None:
            config["data"] = data
        return cls(**config)

    def _kwargs(self, params=None, scales=None, inputs=None):
        """Build circuit arguments while preserving live differentiable weights."""
        result = {"params": self.params if params is None else params}
        x = self.inputs if inputs is None else inputs
        if x is not None:
            result["inputs"] = qml.math.cast_like(
                qml.math.convert_like(x, result["params"]), result["params"]
            )
        if self.input_scaling_params is not None:
            result["input_scaling_params"] = (
                self.input_scaling_params if scales is None else scales
            )
        return result

    def _queue(self, params, scales=None, inputs=None):
        """Queue the source model gates only, stripping its terminal measurement."""
        kwargs = self._kwargs(params, scales, inputs)
        if self.circuit_builder is not None:
            self.circuit_builder.ansatz(**kwargs)
        else:
            with qml.QueuingManager.stop_recording():
                tape = qml.tape.make_qscript(self._source_node.func)(**kwargs)
            for op in tape.operations:
                qml.apply(op)

    def _moment_function(self, params, masks, scales=None):
        """Queue the ansatz and requested nonidentity Pauli-Z observables."""
        self._queue(params, scales)
        return tuple(
            qml.expval(qml.prod(*(qml.Z(self.wires[i]) for i in np.flatnonzero(mask))))
            for mask in masks
        )

    def _probability_function(self, params, scales=None):
        """Queue a complete computational-basis readout for diagnostics/NLL."""
        self._queue(params, scales)
        return qml.probs(wires=self.wires)

    def moments(self, masks=None, *, params=None, input_scaling_params=None):
        """Return differentiable Z-mask expectations, deduplicating circuit work.

        Identity is exactly one and never sent to PP or a device. Duplicate
        masks preserve their original weights/order after evaluation.
        """
        if masks is None:
            if self.mmd is None:
                raise ValueError("Supply masks or configure loss='mmd'.")
            masks = self.mmd.masks
        masks = binary_matrix(masks, self.n_qubits, "masks")
        p = self.params if params is None else params
        scales = (
            self.input_scaling_params
            if input_scaling_params is None
            else input_scaling_params
        )
        unique, inverse = np.unique(masks, axis=0, return_inverse=True)
        active = unique.any(axis=1)
        values = None
        if active.any():
            if self.pp_engine is not None:
                tape = qml.tape.make_qscript(self._moment_function)(
                    p, unique[active], scales
                )
                values = self.pp_engine.execute_from_tape(
                    tape, k=self.pp_truncation, prune=self.pp_prune
                )
            else:
                values = self._moment_node(p, unique[active], scales)
            values = (
                qml.math.stack(list(values))
                if isinstance(values, (tuple, list))
                else qml.math.reshape(values, (-1,))
            )
        # Keep a graph even if every sampled observable is identity or truncated.
        zero = qml.math.sum(p) * 0
        j, ordered = 0, []
        for present in active:
            ordered.append(values[j] + zero if present else zero + 1)
            j += int(present)
        return qml.math.stack(ordered)[inverse]

    def _check_probability_size(self):
        """Guard accidental exponential readout allocations."""
        if self.n_qubits > self.max_probability_qubits:
            raise ValueError(
                "Full probabilities exceed max_probability_qubits; use expectation MMD or explicitly raise the limit."
            )

    def get_probs(
        self, *, params=None, input_scaling_params=None, device=None, device_kwargs=None
    ):
        """Return 2**n probabilities; PP requires an explicit sampling backend.

        Diagnostic probability readouts use best differentiation, since adjoint
        devices may support expectation gradients but not probability gradients.
        """
        self._check_probability_size()
        if self.pp_engine is not None and device is None:
            raise ValueError(
                "lightning.pp computes expectations only; provide an explicit device for probabilities."
            )
        dev = (
            self.device
            if device is None
            else create_device(
                device,
                wires=self.n_qubits + getattr(self.circuit_builder, "n_ancillas", 0),
                device_kwargs=device_kwargs,
                cutting=self.cutting,
            )
        )
        node = create_qnode(
            self._probability_function,
            dev,
            interface=self.interface,
            diff_method="best",
        )
        return node(
            self.params if params is None else params,
            self.input_scaling_params
            if input_scaling_params is None
            else input_scaling_params,
        )

    def loss(self, data=None, *, params=None, input_scaling_params=None, loss_fn=None):
        """Evaluate the current objective without changing masks or parameters.

        Custom callbacks receive (probabilities, integer_indices). Finite-shot
        MMD uses two independent expectation executions to remove model-square
        bias. Autograd/parameter-shift optimizers see fixed masks within a step.
        """
        data = self.data if data is None else binary_matrix(data, self.n_qubits)
        if data is None:
            raise ValueError("Supply data to construction, loss, or train.")
        p = self.params if params is None else params
        scales = (
            self.input_scaling_params
            if input_scaling_params is None
            else input_scaling_params
        )
        if loss_fn is not None or self.loss_name == "nll":
            if self.pp_engine is not None:
                raise ValueError(
                    "Probability callbacks are unavailable on lightning.pp."
                )
            self._check_probability_size()
            probs = (
                self.circuit(**self._kwargs(p, scales))
                if self.loss_name == "nll"
                else self.get_probs(params=p, input_scaling_params=scales)
            )
            return (loss_fn or nll_loss)(probs, basis_indices(data))
        first = self.moments(params=p, input_scaling_params=scales)
        second = (
            self.moments(params=p, input_scaling_params=scales)
            if self.shots is not None
            else None
        )
        return self.mmd(first, data, model_moments_2=second)

    def _fisher_matrix(self):
        """Compute a regularized full Fisher matrix in flattened parameter order."""
        if self.fisher == "quantum":
            # Hadamard-test auxiliary wires are unnecessary with this full metric.
            transform = qml.gradients.adjoint_metric_tensor(self._fisher_node)
            matrix = 4 * transform(self.params)
        else:
            matrix = qml.gradients.classical_fisher(self._fisher_node)(self.params)
        matrix = qml.math.reshape(matrix, (self.params.size, self.params.size))
        return matrix + self.lambda_reg * np.eye(self.params.size)

    def train(
        self,
        steps,
        loss_fn=None,
        print_every=0,
        *,
        data=None,
        resample_observables=True,
    ):
        """Optimize and return (current ansatz params, pre-update loss history).

        A zero-step call is allowed. Minibatches sample rows without replacement.
        Every MPI rank must enter this method with identical data and settings;
        only rank zero should write results. PP pruning changes gradients.
        """
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a nonnegative integer.")
        if type(print_every) is not int or print_every < 0:
            raise ValueError("print_every must be a nonnegative integer.")
        if data is not None:
            self.data = binary_matrix(data, self.n_qubits)
        if self.data is None:
            raise ValueError("Supply binary data before training.")
        losses = []
        for step in range(steps):
            batch = self.data
            if self.batch_size is not None and self.batch_size < len(batch):
                batch = batch[
                    self.rng.choice(len(batch), self.batch_size, replace=False)
                ]
            if self.mmd is not None and resample_observables:
                self.mmd.resample()

            def objective(p, *scales, _batch=batch):
                """Hold data and masks fixed for all evaluations in one update."""
                return self.loss(
                    _batch,
                    params=p,
                    input_scaling_params=scales[0] if scales else None,
                    loss_fn=loss_fn,
                )

            if self.natural_grad:
                value = objective(self.params)
                gradient = qml.grad(objective)(self.params).reshape(-1)
                if not self.commuting:
                    self.Freg = self._fisher_matrix()
                direction = np.linalg.solve(np.asarray(self.Freg), np.asarray(gradient))
                self.params = qml.numpy.array(
                    self.params - self.lr * direction.reshape(self.params.shape),
                    requires_grad=True,
                )
            elif self.interface == "torch":
                self.opt.zero_grad()
                value = objective(*self.parameters())
                value.backward()
                self.opt.step()
            else:
                updated, value = self.opt.step_and_cost(objective, *self.parameters())
                if self.input_scaling_params is None:
                    self.params = updated
                else:
                    self.params, self.input_scaling_params = updated
            losses.append(float(_numpy(value)))
            if print_every and (step + 1) % print_every == 0:
                print(f"Step {step + 1}/{steps}: loss = {losses[-1]:.6g}")
        self.loss_history.extend(losses)
        return self.params, losses

    fit = train

    def sample(self, shots=1000, *, return_bits=True, device=None, device_kwargs=None):
        """Execute finite-shot sampling; return (shots,n) bits or basis indices.

        For a PP-trained model choose a real sampling device explicitly. Uses
        the trained gates and scales directly, without constructing probabilities.
        Existing devices must have a shot count compatible with PennyLane's
        per-call shot override. MPI sampling remains a collective operation.
        """
        shots = positive_integer(shots, "shots")
        if self.pp_engine is not None and device is None:
            raise ValueError(
                "Choose an explicit sampling device for a lightning.pp model."
            )
        if not return_bits and self.n_qubits > 62:
            raise ValueError("Use return_bits=True for more than 62 qubits.")
        dev = (
            self.device
            if device is None
            else create_device(
                device,
                wires=self.n_qubits + getattr(self.circuit_builder, "n_ancillas", 0),
                shots=shots,
                device_kwargs=device_kwargs,
                cutting=self.cutting,
            )
        )

        def sample_function(params, scales=None):
            """Queue the trained model and computational-basis samples."""
            self._queue(params, scales)
            return qml.sample(wires=self.wires)

        node = create_qnode(sample_function, dev, diff_method=None)
        bits = np.asarray(
            qml.set_shots(node, shots=shots)(self.params, self.input_scaling_params),
            dtype=int,
        ).reshape(shots, self.n_qubits)
        return bits if return_bits else basis_indices(bits)

    def shannon_entropy(self, *, device=None):
        """Return empirical data and full model entropy in nats (small systems)."""
        if self.data is None:
            raise ValueError("No training data supplied.")
        counts = np.unique(self.data, axis=0, return_counts=True)[1]
        target = counts / counts.sum()
        probs = _numpy(self.get_probs(device=device))
        positive = probs[probs > 0]
        return float(-np.sum(target * np.log(target))), float(
            -np.sum(positive * np.log(positive))
        )

    def draw(self, **kwargs):
        """Draw the model using draw_qcbm, including Quantikz export."""
        from .drawing import draw_qcbm

        return draw_qcbm(self, **kwargs)


class BornMachine(PQCBM):
    """Deprecated compatibility wrapper; new code should use PQCBM.

    Retains Autograd and integer-index sample defaults from the original API.
    """

    def __init__(self, *args, **kwargs):
        """Forward legacy constructor arguments with an Autograd default."""
        warnings.warn(
            "BornMachine was renamed to PQCBM; migrate to PQCBM or create_qcbm.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.setdefault("interface", "autograd")
        source = (
            args[0] if args else kwargs.get("circuit", kwargs.get("circuit_builder"))
        )
        if isinstance(source, CircuitBuilder):
            if len(args) > 1:
                args = (args[0], _numpy(args[1]).reshape(-1), *args[2:])
            elif kwargs.get("init_params") is not None:
                kwargs["init_params"] = _numpy(kwargs["init_params"]).reshape(-1)
        super().__init__(*args, **kwargs)

    def sample(self, shots=1000, *, return_bits=False, **kwargs):
        """Retain integer sample output unless return_bits=True is requested."""
        return super().sample(shots, return_bits=return_bits, **kwargs)
