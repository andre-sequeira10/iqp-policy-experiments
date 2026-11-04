"""Simulator-independent fragment execution and QNode/TorchLayer integration."""
import inspect
import numpy as np
import pennylane as qml
from qasp.devices import create_device
from .planning import canonicalize, plan_cuts
from .specs import as_config


class FragmentExecutor:
    """Execute plans on a PennyLane device/name/factory or differentiable PP.

    A factory is called as ``factory(wires=integer, shots=integer_or_None)``. Installed
    plugins retain their native capability constraints. No simulator fallback
    occurs. Instances and factories can provide device-specific setup, including
    MPI. Device instances must have enough wires including record ancillas.
    """
    def __init__(self, device="default.qubit", *, interface="auto", diff_method="best",
                 device_kwargs=None, mpi=False, pp_options=None, max_diff=1, gradient_kwargs=None):
        self.device_spec = device
        self.interface, self.diff_method = interface, diff_method
        self.device_kwargs, self.mpi = dict(device_kwargs or {}), mpi
        seed = self.device_kwargs.get("seed")
        self._seed_sequence = np.random.SeedSequence(int(seed)) if isinstance(seed, (int, np.integer)) else None
        self.pp_options = dict(pp_options or {})
        self.max_diff = max_diff
        self.gradient_kwargs = dict(gradient_kwargs or {})
        self.devices = {}
        if device == "lightning.pp" and mpi:
            raise ValueError("Differentiable Pauli propagation does not support MPI.")
        if device == "lightning.pp" and interface not in {"auto", "torch", None}:
            raise ValueError("Cutting with lightning.pp supports Torch gradients; use interface='torch'.")
        if device == "lightning.pp" and diff_method not in {None, "best", "backprop"}:
            raise ValueError("lightning.pp uses Torch backprop; select diff_method='best' or 'backprop'.")

    def __call__(self, plan):
        if not plan.tapes:
            return []
        if self.device_spec == "lightning.pp":
            if any(t.shots for t in plan.tapes):
                raise ValueError("lightning.pp is analytic; finite fragment shots are unsupported.")
            from qasp.rl.differentiable_pp_sim import DifferentiablePauliProp
            opts = {"topM": None, "tau": 0.0, "prune_every": 0, **self.pp_options}
            k, prune = opts.pop("k", None), opts.pop("prune", False)
            engine = DifferentiablePauliProp(**opts)
            results = []
            for tape in plan.tapes:
                # Lower arbitrary PauliRot factors to the engine's elementary
                # gate set. This retains live Torch tensors and record ancillas.
                batch, _ = qml.transforms.decompose(tape, gate_set={
                    qml.CNOT, qml.CZ, qml.Hadamard, qml.RX, qml.RY, qml.RZ})
                results.append(engine.execute_from_tape(batch[0], k=k, prune=prune))
            return results
        results = [None] * len(plan.tapes)
        batches = {}
        for i, tape in enumerate(plan.tapes):
            batches.setdefault((len(tape.wires), tape.shots), []).append(i)
        for (width, shots), indices in batches.items():
            key = (width, shots)
            if key not in self.devices:
                spec = self.device_spec
                if callable(spec) and not isinstance(spec, qml.devices.Device):
                    if self.device_kwargs:
                        raise ValueError("Configure custom factory options inside the factory, not device_kwargs.")
                    dev = spec(wires=width, shots=shots.total_shots)
                    if self.mpi and not getattr(dev, "_mpi", False):
                        raise ValueError("mpi=True requires the factory to return an MPI-enabled device.")
                else:
                    options = dict(self.device_kwargs)
                    if self._seed_sequence is not None:
                        options["seed"] = int(self._seed_sequence.spawn(1)[0].generate_state(1)[0])
                    dev = create_device(spec, wires=width, shots=shots.total_shots,
                                        device_kwargs=options, mpi=self.mpi)
                if dev.wires is not None and len(dev.wires) < width:
                    raise ValueError(f"Fragment requires {width} device wires, but device has {len(dev.wires)}.")
                self.devices[key] = dev
            dev = self.devices[key]
            tapes = [plan.tapes[i] for i in indices]
            if dev.wires is not None:
                mapping = dict(zip(range(width), list(dev.wires)[:width]))
                tapes = [qml.map_wires(t, mapping)[0][0] for t in tapes]
            if dev.name == "default.mixed" and not shots and getattr(dev, "readout_err", None) is None:
                # PL 0.42's Pauli measurement path takes sqrt(probabilities)
                # before marginalizing. Exact zero probabilities then yield NaN
                # backprop gradients. Hermitian uses Tr(rho O) directly. These
                # matrices are fragment-local, never a full source state.
                tapes = [qml.tape.QuantumScript(
                    t.operations,
                    [qml.expval(qml.Hermitian(
                        qml.matrix(m.obs), wires=m.obs.wires)) for m in t.measurements],
                    shots=t.shots) for t in tapes]
            diff = self.diff_method
            if shots and diff in {"adjoint", "backprop"}:
                raise ValueError(f"diff_method={diff} is incompatible with finite fragment shots; use parameter-shift.")
            try:
                values = qml.execute(tapes, dev, interface=self.interface,
                                     diff_method=diff, max_diff=self.max_diff,
                                     gradient_kwargs=self.gradient_kwargs)
            except Exception as exc:
                raise RuntimeError(f"Cut fragment execution failed on {dev.name}: {exc}") from exc
            for i, value in zip(indices, values):
                results[i] = value
        return results


class CuttingQNode(qml.QNode):
    """A QNode-compatible cutting evaluator, including ordinary TorchLayer use.

    The source device is a metadata-only dynamic-width default.qubit. All actual
    execution is through ``executor``. ``last_plan`` and ``last_report`` expose
    the most recent scalar/broadcast member. Plans are rebuilt to retain current
    coefficient gradients. QNode transforms applied after construction are not
    supported; transform the quantum function/tape before constructing this node.
    """
    def __init__(self, func, *, cutting, device="default.qubit", interface="auto",
                 diff_method="best", shots=None, device_kwargs=None, mpi=False,
                 executor=None, max_diff=1, gradient_kwargs=None):
        self.cutting = as_config(cutting)
        self.fragment_device = device
        if shots is None and isinstance(device, qml.devices.Device):
            shots = device.shots
        super().__init__(func, qml.device("default.qubit", wires=None, shots=shots),
                         interface=interface, diff_method=diff_method, max_diff=max_diff,
                         gradient_kwargs=gradient_kwargs)
        self.device._qasp_cutting = self.cutting
        self.device._qasp_fragment_device = device
        self.device._qasp_fragment_options = {"device_kwargs": device_kwargs, "mpi": mpi}
        self.executor = executor or FragmentExecutor(
            device, interface=interface, diff_method=diff_method,
            device_kwargs=device_kwargs, mpi=mpi, pp_options=self.cutting.pp_options,
            max_diff=max_diff, gradient_kwargs=gradient_kwargs)
        self.last_plan = self.last_report = None
        self.cut_batch_inputs = False
        self._cut_rng = np.random.default_rng(self.cutting.seed)

    def _impl_call(self, *args, **kwargs):
        if len(self.transform_program):
            raise ValueError("Apply transforms before constructing CuttingQNode; post-construction transforms are unsupported.")
        # TorchLayer flattens leading input batch dimensions; execute each row
        # without assuming every template implements native broadcasting.
        bound = inspect.signature(self.func).bind_partial(*args, **{k:v for k,v in kwargs.items() if k != "shots"})
        inputs = bound.arguments.get("inputs")
        if self.cut_batch_inputs and inputs is not None and qml.math.ndim(inputs) > 1:
            values = []
            for row in inputs:
                kw = dict(bound.arguments, inputs=row)
                if "shots" in kwargs:
                    kw["shots"] = kwargs["shots"]
                values.append(self._impl_call(**kw))
            if isinstance(values[0], (tuple, list)):
                return type(values[0])(qml.math.stack([v[i] for v in values]) for i in range(len(values[0])))
            return qml.math.stack(values)
        tape = canonicalize(self.construct(args, kwargs))
        if tape.shots.has_partitioned_shots:
            raise ValueError("Cutting currently requires a scalar shot count, not shot vectors.")
        # Expand native parameter broadcasting, including batched rotation inputs.
        batches, _ = qml.transforms.broadcast_expand(tape)
        outputs = []
        for scalar_tape in batches:
            self.last_plan = plan_cuts(scalar_tape, self.cutting)
            values = self.last_plan.reconstruct(self.executor(self.last_plan))
            self.last_report = self.last_plan.resources()
            self.last_report["shots_per_fragment_tape"] = scalar_tape.shots.total_shots
            # A plugin may split noncommuting observables into further
            # executions. This is a lower bound, not a hardware-shot count.
            self.last_report["fragment_shots_lower_bound"] = (
                len(self.last_plan.tapes) * scalar_tape.shots.total_shots
                if scalar_tape.shots else None)
            for i, (kind, _) in enumerate(self.last_plan.readouts):
                if kind in {"sample", "counts"}:
                    if not scalar_tape.shots:
                        raise ValueError("Sampling a reconstructed distribution requires finite shots.")
                    p = np.asarray(qml.math.toarray(values[i]), dtype=float)
                    if np.min(p) < -1e-8 and self.cutting.probability_policy == "strict":
                        raise ValueError("Finite-shot reconstructed probabilities are negative. Increase shots or "
                                         "explicitly choose probability_policy='project' (biased sampling).")
                    p = np.maximum(p, 0)
                    if not np.isfinite(p).all() or p.sum() <= 0:
                        raise ValueError("Cannot sample an invalid reconstructed probability distribution.")
                    p /= p.sum()
                    draws = self._cut_rng.choice(len(p), size=scalar_tape.shots.total_shots, p=p)
                    n = int(np.log2(len(p)))
                    bits = ((draws[:, None] >> np.arange(n-1, -1, -1)) & 1).astype(int)
                    if kind == "counts":
                        labels, counts = np.unique(["".join(map(str, b)) for b in bits], return_counts=True)
                        values[i] = dict(zip(labels, counts))
                        if scalar_tape.measurements[i].all_outcomes:
                            values[i] = {format(j, f"0{n}b"): values[i].get(format(j, f"0{n}b"), 0)
                                         for j in range(len(p))}
                    else:
                        values[i] = bits
            outputs.append(values)
        values = outputs[0] if tape.batch_size is None else [
            qml.math.stack([o[i] for o in outputs]) for i in range(len(outputs[0]))]
        if isinstance(self._qfunc_output, tuple):
            return tuple(values)
        if isinstance(self._qfunc_output, list):
            return values
        return values[0]


def cut_qnode(func, *, cutting, device=None, **kwargs):
    """Construct a cutting evaluator from a quantum function or an existing QNode.

    When wrapping a QNode, preserve its public execution settings unless
    explicitly overridden. Pass a device name/factory to size devices per
    fragment instead of retaining an existing full-width device instance.
    """
    if isinstance(func, qml.QNode):
        node = func
        if len(node.transform_program):
            raise ValueError("Wrapping a transformed QNode would discard its transforms; transform the quantum function first.")
        func = node.func
        if device is None:
            device = getattr(node, "fragment_device", node.device)
        kwargs.setdefault("interface", node.interface)
        kwargs.setdefault("diff_method", node.diff_method)
        kwargs.setdefault("shots", node.device.shots)
    return CuttingQNode(func, cutting=cutting, device=device if device is not None else "default.qubit", **kwargs)
