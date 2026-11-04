"""Mutable construction of QASP quantum functions, QNodes and Torch layers.

The ansatz owns input encoding; the builder owns measurement and backend
configuration. Building, compiling and allocating trainable Torch parameters
are separate steps. TikZ export traces the circuit without executing it."""

import functools
from functools import partial

import pennylane as qml
import torch
from pennylane import numpy as np, qnode

import os
from qasp.devices import create_device

class CircuitBuilder:
    """Assemble a QASP ansatz, measurement, QNode and optional TorchLayer.

    The builder is mutable. Configure an ansatz (which owns its feature map),
    call ``build()``, then ``compile()`` before ``create_torchlayer()``.
    Building only queues gates; compiling chooses the execution backend.

    Example::

        from qasp.circuits.templates import AngleAnsatz
        builder = CircuitBuilder(n_qubits=4)
        builder.set_ansatz(AngleAnsatz(n_qubits=4))
        builder.set_measurement("probs")
        quantum_function = builder.build()
        qnode = builder.compile(device="default.qubit")

    Use keyword ``params`` when calling an unencoded QNode. For encoded
    circuits also provide ``inputs`` and, if enabled, ``input_scaling_params``."""

    def __init__(
        self,
        n_qubits: int,
        n_ancillas: int = 0,
        dev_kwargs: dict = None,
    ):
        """Initialize an uncompiled builder with computational-basis probabilities.

        Args:
            n_qubits: Number of data wires, numbered from zero.
            n_ancillas: Additional wires used by optional readout constructions.
            dev_kwargs: Copied device options; compile-time options override them.

        No quantum device or trainable parameters are allocated yet."""
        self.n_qubits = n_qubits
        self.dev_kwargs = dict(dev_kwargs or {})
        self.ansatz = None
        self.feature_map = None
        self.measurement = ("probs", list(range(n_qubits)))
        self.obs = None
        self.is_compiled = False
        self.z_gates = None
        self.CNOT_Gates = None
        self.n_ancillas = n_ancillas

    def set_ansatz(self, ansatz_fn):
        """Attach a QASP-compatible ansatz instance without copying it.

        The object must expose ``feature_map`` and be callable with ``params``
        and the applicable input arguments. TorchLayer construction additionally
        requires ``get_param_shapes()``. Its gates use the builder's data wires."""
        self.ansatz = ansatz_fn

    def set_CNOT_gates(self, CNOT_Gates: list):
        """
        CNOT_Gates (list[tuple[int, int]]): list of (control, target) qubit pairs for CNOT gates.
        """
        self.CNOT_Gates = CNOT_Gates
        self.ansatz.CNOT_Gates = CNOT_Gates

    def set_parity_check(self):
        """Assign prefix-parity CNOT pairs to the attached ansatz.

        Each ancilla receives CNOTs from a successively shorter data-wire prefix.
        Requires an attached ansatz; mutates its ``parity_gates`` and returns None."""
        parity_gates = []
        for i in range(self.n_ancillas):
            for j in range(self.n_qubits - i):
                parity_gates.append((j, self.n_qubits + self.n_ancillas - 1 - i))

        self.ansatz.parity_gates = parity_gates

    def set_z_gates(self, z_gates: list):
        """Store a Z-gate mask on the builder; return None.

        This setter does not copy the mask to the ansatz. Templates that consult
        ``ansatz.z_gates`` must be configured there separately."""
        self.z_gates = z_gates

    def set_feature_map(self, feature_map_fn):
        """Store feature-map metadata on this builder; return None.

        Circuit construction reads ``ansatz.feature_map``. Attach the feature map
        to the ansatz when constructing it; this setter alone does not replace it."""
        self.feature_map = feature_map_fn

    def set_measurement(
        self,
        meas_type: str,
        wires=None,
        observables=None,
    ):
        """Configure the terminal measurement; return None.

        Args:
            meas_type: ``probs``, ``state``, ``expectation``, ``variance`` or ``sample``.
            wires: Wires for probabilities/samples; default is every data wire.
                ``build`` sorts these wires. State measurements return the device state.
            observables: Required for expectation/variance. Entries are PennyLane
                operators or Pauli specifications such as ``"Z 0"``, ``"ZZ 0 2"``
                and ``"-X 1"``; word characters correspond to the listed wires.

        Raises ValueError for unsupported measurement names or missing observables,
        and TypeError for unsupported observable entries. Rebuild after changing it."""
        valid = {"probs", "state", "expectation", "variance", "sample"}
        if meas_type not in valid:
            raise ValueError(f"Invalid measurement '{meas_type}'. Choose from {valid}.")

        if wires is None:
            wires = range(self.n_qubits)
        self.measurement = (meas_type, wires)
        if meas_type in {"expectation", "variance"}:
            if observables is None:
                raise ValueError(
                    "observables must be provided for expectation/variance."
                )
            self.obs = []
            for item in observables:
                # Accept already constructed PennyLane observables, including
                # SparseHamiltonian action projectors.
                if isinstance(item, qml.operation.Operator):
                    self.obs.append(item)
                    continue
                if not isinstance(item, str):
                    raise TypeError(
                        "Expectation observables must be PennyLane operators or "
                        f"Pauli-string specifications; received {type(item)}."
                    )

                signal = 1
                parts = item.split()
                pauli_string = parts[0]
                if pauli_string.startswith("-"):
                    signal = -1
                    pauli_string = pauli_string[1:]
                wire_indices = list(map(int, parts[1:]))
                wire_map = {wire: idx for idx, wire in enumerate(wire_indices)}
                #print(f"Parsed observable: signal={signal}, pauli_string='{pauli_string}', wires={wire_indices} , wire_map={wire_map}", flush=True)
                # Use PennyLane's string_to_pauli_word
                pauli_word = qml.pauli.string_to_pauli_word(
                    pauli_string, wire_map=wire_map
                )
                #self.obs.append(qml.Hamiltonian([signal * 1.0], [pauli_word]))
                self.obs.append(signal * pauli_word)
                #print(f"Constructed observable: {self.obs[-1]}", flush=True)

    def build(self):
        """Queue the attached ansatz and return a plain quantum function.

        The callable's signature reflects whether input encoding and trainable
        input scales are present. Probabilities/state/sample use one measurement;
        expectations/variances return a tuple in observable order. Stores ``circuit``
        but does not create a QNode. Raises RuntimeError if no ansatz is attached."""
        if self.ansatz is None:
            raise RuntimeError("Ansatz function not set. Call set_ansatz() first.")

        meas_type, wires = self.measurement
        # @qml.compile
        wires = sorted(wires)  # Ensure wires are in a consistent order
        if self.ansatz.feature_map is not None:
            if self.ansatz.feature_map.input_scaling:

                def circuit_fm_is(inputs=None, input_scaling_params=None, params=None):

                    """Queue scaled input encoding and the ansatz, then return configured measurements."""
                    self.ansatz(
                        inputs=inputs,
                        input_scaling_params=input_scaling_params,
                        params=params,
                    )

                    if meas_type == "probs":
                        return qml.probs(wires=wires)
                    if meas_type == "state":
                        return qml.state()
                    if meas_type == "sample":
                        return qml.sample(wires=wires)
                    if meas_type == "expectation":
                        return tuple(qml.expval(o) for o in self.obs)
                    if meas_type == "variance":
                        return tuple(qml.var(o) for o in self.obs)
                    # unreachable
                    raise RuntimeError("Unknown measurement type")
                    # wrap in QNode

                self.circuit = circuit_fm_is

            else:

                def circuit_fm(inputs=None, params=None):
                    """Queue unscaled input encoding and the ansatz, then return configured measurements."""
                    self.ansatz(inputs=inputs, input_scaling_params=None, params=params)

                    if meas_type == "probs":
                        return qml.probs(wires=wires)
                    if meas_type == "state":
                        return qml.state()
                    if meas_type == "sample":
                        return qml.sample(wires=wires)
                    if meas_type == "expectation":
                        return tuple(qml.expval(o) for o in self.obs)
                    if meas_type == "variance":
                        return tuple(qml.var(o) for o in self.obs)
                    # unreachable
                    raise RuntimeError("Unknown measurement type")
                    # wrap in QNode

                self.circuit = circuit_fm
        else:

            def circuit(params=None):
                """Queue a parameter-only ansatz and return the configured measurements."""
                self.ansatz(params=params)

                if meas_type == "probs":
                    return qml.probs(wires=wires)
                if meas_type == "state":
                    return qml.state()
                if meas_type == "sample":
                    return qml.sample(wires=wires)
                if meas_type == "expectation":
                    return tuple(qml.expval(o) for o in self.obs)
                if meas_type == "variance":
                    return tuple(qml.var(o) for o in self.obs)
                # unreachable
                raise RuntimeError("Unknown measurement type")
                # wrap in QNode

            self.circuit = circuit

        return self.circuit

    def build_adaptive(self, for_torchlayer=False):
        """Build a quantum function whose measurement selector is supplied at call time.

        With a feature map, inputs contain ``(features, selector)``; in TorchLayer
        mode they are concatenated, with trailing entries converted to integer
        wire indices on CPU. Without encoding, pass ``measurement`` separately.
        Selectors are wires for probabilities/samples or operators for expectation/
        variance. Selection is discrete, not differentiable; this is an experimental
        single-sample path. Stores and returns the quantum function."""
        if self.ansatz is None:
            raise RuntimeError("Ansatz function not set. Call set_ansatz() first.")

        self.for_torchlayer = for_torchlayer
        meas_type, wires = self.measurement
        # @qml.compile

        if self.ansatz.feature_map is not None:
            if self.ansatz.feature_map.input_scaling:

                def circuit_fm_is(inputs=None, input_scaling_params=None, params=None):
                    """Unpack features and a discrete selector; queue scaled encoding and selected measurements."""
                    if self.for_torchlayer:
                        inpt = inputs[: self.ansatz.feature_map.features]
                        inpt_ = inputs[self.ansatz.feature_map.features :]
                        if type(inpt_) == torch.Tensor:
                            inpt_measure = inpt_.int().numpy()
                        else:
                            inpt_measure = inpt_
                    else:
                        inpt = inputs[0]
                        inpt_measure = inputs[1]

                    self.ansatz(
                        inputs=inpt,
                        input_scaling_params=input_scaling_params,
                        params=params,
                    )

                    if meas_type == "probs":
                        return qml.probs(wires=inpt_measure)
                    if meas_type == "state":
                        return qml.state()
                    if meas_type == "sample":
                        return qml.sample(wires=inpt_measure)
                    if meas_type == "expectation":
                        return [qml.expval(o) for o in inpt_measure]
                    if meas_type == "variance":
                        return [qml.var(o) for o in inpt_measure]
                    # unreachable
                    raise RuntimeError("Unknown measurement type")
                    # wrap in QNode

                self.circuit = circuit_fm_is

            else:

                def circuit_fm(inputs=None, params=None):
                    """Unpack features and a discrete selector; queue encoding and selected measurements."""
                    if self.for_torchlayer:
                        inpt = inputs[: self.ansatz.feature_map.features]
                        inpt_ = inputs[self.ansatz.feature_map.features :]
                        if type(inpt_) == torch.Tensor:
                            inpt_measure = inpt_.int().numpy()
                        else:
                            inpt_measure = inpt_
                    else:
                        inpt = inputs[0]
                        inpt_measure = inputs[1]

                    self.ansatz(inputs=inpt, input_scaling_params=None, params=params)

                    if meas_type == "probs":
                        return qml.probs(wires=inpt_measure)
                    if meas_type == "state":
                        return qml.state()
                    if meas_type == "sample":
                        return qml.sample(wires=inpt_measure)
                    if meas_type == "expectation":
                        return [qml.expval(o) for o in inpt_measure]
                    if meas_type == "variance":
                        return [qml.var(o) for o in inpt_measure]
                    # unreachable
                    raise RuntimeError("Unknown measurement type")
                    # wrap in QNode

                self.circuit = circuit_fm
        else:
            # no inputs for this model, so receives the adaptive measurement as a new argument "measurement",
            # but it does not work with torch since it interprets as a trainable parameter which is not
            def circuit(params=None, measurement=None):
                """Queue a parameter-only ansatz and the call-time measurement selector."""
                self.ansatz(params=params)

                if meas_type == "probs":
                    return qml.probs(wires=measurement)
                if meas_type == "state":
                    return qml.state()
                if meas_type == "sample":
                    return qml.sample(wires=measurement)
                if meas_type == "expectation":
                    return [qml.expval(o) for o in measurement]
                if meas_type == "variance":
                    return [qml.var(o) for o in measurement]
                # unreachable
                raise RuntimeError("Unknown measurement type")
                # wrap in QNode

            self.circuit = circuit

        return self.circuit

    def build_kernel(self, for_torchlayer=False):
        """Construct an overlap circuit for fidelity-style kernels.

        The circuit applies ``U(x)`` followed by ``U(x')^\u2020`` while sharing
        the same variational parameters. For a TorchLayer, the two examples
        are supplied as one concatenated input; otherwise they are supplied as
        a two-row array.
        """
        if self.ansatz is None:
            raise RuntimeError("Ansatz function not set. Call set_ansatz() first.")

        self.for_torchlayer = for_torchlayer
        meas_type, wires = self.measurement
        wires = sorted(wires)
        feature_map = self.ansatz.feature_map
        n_features = self.n_qubits
        if feature_map is not None and feature_map.features is not None:
            n_features = feature_map.features

        def split_inputs(inputs):
            """Split a concatenated TorchLayer feature pair or unpack a two-item input pair."""
            if for_torchlayer:
                return inputs[:n_features], inputs[n_features : 2 * n_features]
            return inputs[0], inputs[1]

        def measure():
            """Queue the configured terminal measurement for the overlap circuit."""
            if meas_type == "probs":
                return qml.probs(wires=wires)
            if meas_type == "state":
                return qml.state()
            if meas_type == "sample":
                return qml.sample(wires=wires)
            if meas_type == "expectation":
                return tuple(qml.expval(o) for o in self.obs)
            if meas_type == "variance":
                return tuple(qml.var(o) for o in self.obs)
            raise RuntimeError("Unknown measurement type")

        if feature_map is not None and feature_map.input_scaling:

            def circuit(inputs=None, input_scaling_params=None, params=None):
                """Queue U(x) followed by the adjoint of U(y), sharing trainable weights."""
                x, x_prime = split_inputs(inputs)
                self.ansatz(
                    inputs=x,
                    input_scaling_params=input_scaling_params,
                    params=params,
                )
                qml.adjoint(self.ansatz)(
                    inputs=x_prime,
                    input_scaling_params=input_scaling_params,
                    params=params,
                )
                return measure()

        else:

            def circuit(inputs=None, params=None):
                """Queue U(x) followed by the adjoint of U(y), sharing trainable weights."""
                x, x_prime = split_inputs(inputs)
                self.ansatz(inputs=x, input_scaling_params=None, params=params)
                qml.adjoint(self.ansatz)(
                    inputs=x_prime,
                    input_scaling_params=None,
                    params=params,
                )
                return measure()

        self.circuit = circuit
        return self.circuit

    def compile(
        self,
        device="auto",
        diff_method="best",
        method="mps",
        shots=None,
        mpi=False,
        cut_circuit=False,
        dev_wires=10,
        spsa_num_directions=1,
        device_vjp=False,
        batch_obs=True,
        device_kwargs=None,
        cutting=None,
    ):
        """Compile a previously built circuit and store its device and QNode.

        Args:
            device: PennyLane device name/instance, or ``auto``. Automatic choices
                are default.qubit for probabilities, lightning.qubit for samples,
                and lightning.kokkos for expectations. Use an explicit device for
                state/variance measurements. MPI auto-selects lightning.kokkos.
            diff_method: PennyLane differentiation method, e.g. backprop or adjoint.
            method: Tensor-network method for default.tensor (default ``mps``).
            shots: Shot count; None requests analytic execution, except auto sample
                mode defaults to one shot.
            mpi: Request distributed execution through the selected plugin.
            cut_circuit: Experimental fragment execution on lightning.kokkos.
            dev_wires: Number of device wires for circuit-cut fragments.
            spsa_num_directions: Direction count passed to SPSA differentiation.
            device_vjp: Request device VJP support from PennyLane.
            cutting: Optional qasp.cutting.CutConfig or dictionary. Routes gate
                and wire cuts through differentiable fragment reconstruction.
            batch_obs: Default Lightning observable-batching option.
            device_kwargs: Device options overriding constructor options.

        Returns:
            A QNode, or the experimental cutting adapter. For QASP's internal
            ``lightning.pp`` route, returns ``(qnode, tape_factory)``: the QNode
            owns parameters while the caller must execute the tape with the PP
            engine. This name is not an installed PennyLane device plugin.

        Backend/differentiation compatibility is validated by PennyLane and the
        selected plugin. Compilation requires ``build`` to have set ``circuit``."""
        options = dict(self.dev_kwargs)
        options.update(device_kwargs or {})
        if cutting is not None:
            if cut_circuit:
                raise ValueError("Use cutting= or the legacy cut_circuit flag, not both.")
            if device_vjp:
                raise ValueError("Cutting uses fragment Jacobians; device_vjp=True is unsupported.")
            from qasp.cutting import CuttingQNode
            self.qnode = CuttingQNode(
                self.circuit, cutting=cutting,
                device="default.qubit" if device == "auto" else device,
                interface="auto", diff_method=diff_method, shots=shots,
                device_kwargs=options, mpi=mpi,
                gradient_kwargs={"num_directions": spsa_num_directions}
                if diff_method == "spsa" else None,
            )
            self.device, self.diff_method = self.qnode.device, diff_method
            self.is_compiled = True
            return self.qnode
        if mpi and device == "lightning.pp":
            raise ValueError("lightning.pp does not support mpi=True.")
        # Auto selection must not silently select a serial simulator for
        # probability or sample measurements when MPI was requested.
        if mpi and device == "auto":
            device = "lightning.kokkos"
        if isinstance(device, qml.devices.Device):
            self.device = create_device(
                device, wires=self.n_qubits + self.n_ancillas,
                shots=shots, mpi=mpi, device_kwargs=options,
            )
        elif device == "auto":
            if self.measurement[0] == "probs":
                self.device = create_device(
                    "default.qubit", wires=self.n_qubits + self.n_ancillas + 1,
                    shots=shots, device_kwargs=options,
                )
            elif self.measurement[0] == "sample":
                self.device = create_device(
                    "lightning.qubit",
                    wires=self.n_qubits + self.n_ancillas + 1,
                    shots=1 if shots is None else shots, device_kwargs=options,
                )
            elif self.measurement[0] == "expectation":
                print("Using Kokkos device for expectation measurements.", flush=True)
                options.setdefault("batch_obs", batch_obs)
                self.device = create_device(
                    "lightning.kokkos",
                    wires=self.n_qubits + self.n_ancillas + 1,
                    shots=shots, device_kwargs=options,
                )

        elif device == "default.tensor":
            options.setdefault("method", method)
            self.device = create_device(
                "default.tensor",
                wires=self.n_qubits + self.n_ancillas + 1,
                shots=shots, mpi=mpi, device_kwargs=options,
            )
        
        elif device == "lightning.pp":
            # Parameter-owning placeholder only; PP executes the traced tape.
            # Do not require an optional Lightning/Kokkos installation here.
            self.device = create_device(
                "default.qubit", wires=self.n_qubits + self.n_ancillas + 1,
                shots=shots, device_kwargs=options,
            )

        elif device in {"lightning.kokkos", "lightning.gpu", "lightning.qubit"}:
            options.setdefault("batch_obs", batch_obs)
            self.device = create_device(
                device, wires=self.n_qubits + self.n_ancillas,
                mpi=mpi, shots=shots, device_kwargs=options,
            )
        elif device == "default.clifford":      
            options.setdefault("max_workers", None)
            self.device = create_device(
                "default.clifford", wires=self.n_qubits + self.n_ancillas,
                shots=shots, mpi=mpi, device_kwargs=options,
            )
        else:
            self.device = create_device(
                device, wires=self.n_qubits + self.n_ancillas,
                shots=shots, mpi=mpi, device_kwargs=options,
            )
         
        self.diff_method = diff_method


        #n_threads = str(os.cpu_count())

        #os.environ["OMP_NUM_THREADS"] = n_threads
        #os.environ["OMP_NUM_THREADS"] = "1"
        #os.environ["MKL_NUM_THREADS"] = "1"
        #os.environ["OPENBLAS_NUM_THREADS"] = "1"
        if cut_circuit:
            #self.qnode = qml.QNode(
                #self.circuit, device=qml.device(
                 #       "lightning.kokkos", wires=12, mpi=mpi,batch_obs=True
                  #  ), diff_method=self.diff_method
            #)
            #self.qnode = qml.transforms.make_tape(self.circuit)
            #make a class that returns a new object , the qtape and has a method called forward 
            class CutQNode:
                """Experimental adapter that executes cut fragments and reconstructs their result."""
                def __init__(self, circuit, device, diff_method):
                    """Capture the tape factory and allocate a Kokkos device with dev_wires wires."""
                    self.qtape = qml.tape.make_qscript(circuit)
                    self.small_dev = create_device(
                        "lightning.kokkos", wires=dev_wires, shots=shots,
                        mpi=mpi, device_kwargs={"batch_obs": True, **options},
                    )
                    self.diff_method = diff_method

                def forward(self, *args, **kwargs):
                    
                    """Cut a traced tape, execute fragments using adjoint differentiation, and reconstruct."""
                    fragment_tapes, processing_fn = qml.cut_circuit(

                        self.qtape(*args, **kwargs),

                        device_wires=self.small_dev.wires,

                    )

                    fragment_results = qml.execute(fragment_tapes, self.small_dev, diff_method="adjoint", interface="torch")

                    return processing_fn(fragment_results)
            self.qnode = CutQNode(self.circuit, device=self.device, diff_method=self.diff_method)
            return self.qnode
        else:
            if device == "default.clifford":
                self.qnode = qml.QNode(
                    self.circuit, device=self.device, interface="torch" , diff_method=self.diff_method, gradient_kwargs={"num_directions": spsa_num_directions} if self.diff_method == "spsa" else None 
                )
                self.qnode = qml.transforms.clifford_t_decomposition(self.qnode)
                #self.qnode = qml.transforms.decompose(self.qnode, gate_set={"H", "X", "Y" , "Z" , "CNOT"})
                #self.qnode = qml.transforms.decompose(self.qnode , gate_set={qml.Hadamard , qml.PauliX , qml.PauliY , qml.PauliZ , qml.CNOT , qml.CZ})
            else:
                # ``lightning.pp`` is a QASP logical backend. Its QNode is
                # retained only so TorchLayer can own and initialize the
                # circuit parameters; execution and autograd are handled by
                # DifferentiablePauliProp from a traced tape. Asking the
                # placeholder for a device-specific derivative is unnecessary:
                # that QNode is never used for PP evaluation.
                qnode_diff_method = (
                    "best" if device == "lightning.pp" else self.diff_method
                )
                self.qnode = qml.QNode(
                    self.circuit,
                    device=self.device,
                    diff_method=qnode_diff_method,
                    device_vjp=device_vjp,
                    gradient_kwargs={"num_directions": spsa_num_directions}
                    if self.diff_method == "spsa"
                    else None,
                )
        #if shots is not None:
            #self.qnode = qml.set_shots(self.qnode, shots=shots)

        #self.qnode, proc_fn = qml.cut_circuit(self.qnode, device_wires=range(30),auto_cutter=True)

        self.is_compiled = True



        if device == "lightning.pp":
            self.ansatz_tape = qml.transforms.make_tape(self.circuit)
            """
            self.ansatz_tape_compiled = qml.compile(
                self.ansatz_tape,
                basis_set=["CNOT", "RX", "RY", "RZ"],
                num_passes=2
            )
            """

            return self.qnode, self.ansatz_tape  # _compiled
        else:
            return self.qnode

    def to_tikz(self, *args, **kwargs):
        """Return this builder's circuit as Quantikz/TikZ LaTeX source.

        The circuit is traced with ``args`` and circuit keyword arguments. See
        :func:`qasp.circuits.circuit_to_tikz` for formatting options.
        """
        from .tikz import circuit_to_tikz

        return circuit_to_tikz(self, *args, **kwargs)

    def save_tikz(self, path, *args, **kwargs):
        """Save this builder's Quantikz/TikZ source to a ``.tex`` file."""
        from .tikz import save_tikz

        return save_tikz(path, self, *args, **kwargs)

    def set_weights(self, params=None, input_scaling_params=None) -> None:
        """Copy supplied weights into the layer, leaving unspecified groups unchanged.

        Values must have the exact shape of their parameter group and be finite.
        Copies preserve each parameter's dtype, device and optimizer registration.
        """
        import torch

        replacements = {}
        for name, value in (("params", params), ("input_scaling_params", input_scaling_params)):
            if value is None:
                continue
            if name not in self.weight_shapes:
                raise ValueError(f"This circuit has no {name} parameter group.")
            target = getattr(self.qlayer, name)
            value = torch.as_tensor(value, dtype=target.dtype, device=target.device)
            if value.shape != target.shape or not torch.isfinite(value).all():
                raise ValueError(f"{name} must have shape {tuple(target.shape)} and finite values.")
            replacements[name] = value
        with torch.no_grad():
            for name, value in replacements.items():
                getattr(self.qlayer, name).copy_(value)

    def create_torchlayer(
        self, params=None, input_scaling_params=None, initialization=None,
        *, input_scaling_init=None,
    ) -> qml.qnn.TorchLayer:
        """Create a trainable layer with independent ansatz and input-scale initializers.

        ``initialization`` accepts qasp.initialization objects, legacy strings,
        or Torch-style tensor callables. ``input_scaling_init=None`` uses the
        same specification for the input scales. Explicit arrays independently
        override the initializer for their group. Without a specification, the
        historical uniform [0, 2*pi) default is retained. Unknown names raise.
        The legacy ``discrete_0_pi_over_2`` alias still draws 0/pi here; use
        ``Choice((0, pi/2))`` for an unambiguous half-pi distribution.
        """
        import functools
        import torch
        from qasp.initialization import initialize_tensor, resolve_initializer

        shapes = self.ansatz.get_param_shapes()
        scaled = (self.ansatz.feature_map is not None
                  and self.ansatz.feature_map.input_scaling)
        if not scaled and (input_scaling_init is not None or input_scaling_params is not None):
            raise ValueError("input_scaling_init/params requires an input-scaling feature map.")
        self.weight_shapes = ({"input_scaling_params": shapes[0], "params": shapes[1]}
                              if scaled else {"params": shapes[0]})
        self.initialization = initialization
        self.input_scaling_init = input_scaling_init
        specs = {"params": initialization,
                 "input_scaling_params": (initialization if input_scaling_init is None
                                          else input_scaling_init)}
        explicit = {"params": params, "input_scaling_params": input_scaling_params}
        self.init_method = {}
        for name, shape in self.weight_shapes.items():
            value = explicit[name]
            if value is not None:
                expected = (shape,) if isinstance(shape, int) else tuple(shape)
                value = torch.as_tensor(value).detach().clone()
                if tuple(value.shape) != expected or not torch.isfinite(value).all():
                    raise ValueError(f"{name} must have shape {expected} and finite values.")
                self.init_method[name] = lambda tensor, value=value: tensor.copy_(value.to(tensor))
            else:
                spec = resolve_initializer(specs[name], legacy_discrete_pi=True)
                self.init_method[name] = functools.partial(initialize_tensor, spec=spec)

        self.qlayer = qml.qnn.TorchLayer(
            self.qnode, self.weight_shapes, init_method=self.init_method
        )
        if hasattr(self.qnode, "cut_batch_inputs"):
            self.qnode.cut_batch_inputs = True
        return self.qlayer
