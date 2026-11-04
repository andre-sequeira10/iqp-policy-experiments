"""QAOA, warm-start QAOA, multi-angle QAOA, and LR-QAOA."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any
from collections import Counter

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize
from qasp.devices import create_qnode, create_device

from .optimizers import (
    is_pennylane_optimizer,
    minimize_objective,
    minimize_pennylane,
    minimize_torch,
    normalize_backend,
)
from .problem import PauliTerm, QUBO
from .results import OptimizationResult


def _rotation_from_spec(spec, *, default_wire: int | None = None) -> PauliTerm:
    """Parse the same ``'XX 0 1'`` notation used by QASP AngleAnsatz."""
    coefficient = 1.0
    if isinstance(spec, str):
        parts = spec.split()
        pauli = parts[0].upper()
        if len(parts) == 1 and default_wire is not None:
            wires = (default_wire,)
        else:
            wires = tuple(map(int, parts[1:]))
    elif isinstance(spec, Mapping):
        pauli = str(spec.get("pauli", spec.get("rotation", ""))).upper()
        wires_value = spec.get("wires", spec.get("wire", ()))
        if isinstance(wires_value, (int, np.integer)):
            wires = (int(wires_value),)
        else:
            wires = tuple(map(int, wires_value))
        coefficient = float(spec.get("coefficient", spec.get("weight", 1.0)))
    elif isinstance(spec, Sequence) and len(spec) in {2, 3}:
        pauli = str(spec[0]).upper()
        wires_value = spec[1]
        if isinstance(wires_value, (int, np.integer)):
            wires = (int(wires_value),)
        else:
            wires = tuple(map(int, wires_value))
        if len(spec) == 3:
            coefficient = float(spec[2])
    else:
        raise TypeError(f"Unsupported rotation specification: {spec!r}.")
    if not pauli or any(letter not in "IXYZ" for letter in pauli):
        raise ValueError(f"Invalid Pauli word {pauli!r}.")
    if len(pauli) != len(wires):
        raise ValueError(
            f"Pauli word {pauli!r} needs {len(pauli)} wires, received {wires}."
        )
    return PauliTerm(coefficient, pauli, wires)


def _parse_mixer(mixer, n_qubits: int) -> tuple[list[PauliTerm], Any]:
    """Normalize a QASP-style mixer dictionary into Pauli rotations."""
    initial_state = "plus"
    if mixer is None or (isinstance(mixer, str) and mixer.lower() == "x"):
        return [PauliTerm(1.0, "X", (wire,)) for wire in range(n_qubits)], initial_state
    if isinstance(mixer, Mapping):
        initial_state = mixer.get("initial_state", "plus")
        rotations = mixer.get("rotations", mixer.get("terms"))
        if rotations is None:
            raise ValueError("A mixer dictionary requires 'rotations' or 'terms'.")
        if isinstance(rotations, str) and " " not in rotations:
            wires = mixer.get("wires", range(n_qubits))
            # Match AngleAnsatz: a compact string such as "XY" means X and Y
            # rotations broadcast over every selected wire.
            specs = [
                f"{pauli} {int(wire)}"
                for pauli in rotations
                for wire in wires
            ]
        elif isinstance(rotations, (str, Mapping)):
            specs = [rotations]
        else:
            specs = list(rotations)
    else:
        specs = list(mixer)
    terms = [_rotation_from_spec(spec) for spec in specs]
    if not terms:
        raise ValueError("The mixer must contain at least one rotation.")
    if any(min(term.wires) < 0 or max(term.wires) >= n_qubits for term in terms):
        raise ValueError("Mixer wires must lie inside the QUBO register.")
    return terms, initial_state


class QAOA:
    """Solve a QUBO with standard or multi-angle QAOA.

    ``parametrization='standard'`` uses one ``gamma`` and one ``beta`` per
    layer. ``parametrization='multi_angle'`` assigns an independent parameter
    to every non-constant cost gate and every mixer gate in every layer.

    ``mpi=True`` enables distributed simulation on the selected device.
    All ranks must call the solver with identical data and parameters.
    Additional plugin settings are passed through ``device_kwargs``.

    A custom mixer follows QASP's dictionary notation, for example::

        {"rotations": ["X 0", "X 1", "XX 0 1"],
         "initial_state": "plus"}
    """

    _PARAMETRIZATION_ALIASES = {
        "standard": "standard",
        "single": "standard",
        "multi_angle": "multi_angle",
        "multi-angle": "multi_angle",
        "ma": "multi_angle",
    }

    def __init__(
        self,
        problem: QUBO | np.ndarray,
        *,
        n_layers: int = 1,
        parametrization: str = "standard",
        mixer=None,
        warm_start: Sequence[float] | str | bool | None = None,
        warm_start_epsilon: float = 0.25,
        device: str | qml.devices.Device = "default.qubit",
        mpi: bool = False,
        device_kwargs: dict | None = None,
        shots: int | None = None,
        diff_method: str | None = "auto",
        backend: str = "numpy",
        seed: int | None = None,
        mixer_evolution_sign: float = 1.0,
        init="random",
        cutting=None,
    ):
        """Construct QAOA state-preparation, expectation, probability and sample QNodes.

        Args:
            problem: QUBO or square matrix representing offset + x.T @ Q @ x.
            n_layers: Positive number of alternating cost/mixer layers.
            parametrization: Standard (two angles per layer) or multi_angle (one
                angle per nonconstant cost term and per mixer term).
            mixer: Custom rotation dictionary/specification; None uses the X mixer.
            warm_start: None/False, a vector in [0,1], or True/relax/relaxed to run a
                bounded SLSQP relaxation at construction. An implicit mixer is then
                adapted to the clipped relaxed state.
            warm_start_epsilon: Clip relaxed entries to [epsilon,1-epsilon]; <0.5.
            device: PennyLane device name or configured instance.
            mpi: Request collective distributed simulation where supported.
            device_kwargs: Plugin-specific options for device creation.
            shots: Training shots; None requests analytic execution.
            diff_method: Auto selects adjoint analytically and parameter-shift with
                finite shots. Adjoint/backprop cannot be combined with finite shots.
            backend: NumPy/Autograd or Torch optimization interface.
            seed: Seed for built-in parameter initialization, not a shot-noise seed.
            init: Shared initializer object, legacy mode or explicit angle array.
                The default random draws uniformly in [-0.1, 0.1). solve can override it.
            mixer_evolution_sign: Sign multiplying mixer angles (default +1).

        Constructing with a relaxation warm start performs classical optimization;
        ordinary construction does not optimize the QAOA angles. MPI ranks must use
        identical parameters and participate in all collective circuit calls."""
        self.cutting = cutting
        self.problem = problem if isinstance(problem, QUBO) else QUBO(problem)
        self.n_qubits = self.problem.n_variables
        self.n_layers = int(n_layers)
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive.")
        try:
            self.parametrization = self._PARAMETRIZATION_ALIASES[parametrization]
        except KeyError as exc:
            raise ValueError(
                "parametrization must be 'standard' or 'multi_angle'."
            ) from exc
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.init = "random" if init is None else init
        self.backend = normalize_backend(backend)
        self.interface = "torch" if self.backend == "torch" else "autograd"
        self.cost_terms = self.problem.pauli_terms(include_constant=False)
        self.warm_start = self._resolve_warm_start(
            warm_start, epsilon=float(warm_start_epsilon)
        )
        self._warm_mixer = self.warm_start is not None and mixer is None
        self.mixer_terms, self.initial_state = _parse_mixer(mixer, self.n_qubits)
        self.mixer_evolution_sign = float(mixer_evolution_sign)
        self.mpi = bool(mpi or getattr(device, "_mpi", False))
        self.device = create_device(
            device, wires=self.n_qubits, shots=shots, mpi=mpi,
            device_kwargs=device_kwargs, cutting=self.cutting,
        )
        device_shots = getattr(getattr(self.device, "shots", None), "total_shots", None)
        self.training_shots = shots if shots is not None else device_shots
        # Preserve the original public attribute while making its role clear.
        self.shots = self.training_shots
        if diff_method == "auto":
            diff_method = (
                ("best" if self.cutting is not None else "adjoint")
                if self.training_shots is None else "parameter-shift"
            )
        if self.training_shots is not None and diff_method in {"adjoint", "backprop"}:
            raise ValueError(
                f"diff_method={diff_method!r} requires analytic execution; use "
                "'parameter-shift' or a derivative-free optimizer with finite shots."
            )
        self.diff_method = diff_method
        self.cost_hamiltonian = self.problem.hamiltonian()
        self._expectation_qnode = create_qnode(
            self._expectation_circuit,
            self.device,
            interface=self.interface,
            diff_method=self.diff_method,
        )
        self._probability_qnode = create_qnode(
            self._probability_circuit,
            self.device,
            interface=None,
            diff_method=None,
        )
        self._sample_qnode = create_qnode(
            self._sample_circuit,
            self.device,
            interface=None,
            diff_method=None,
        )

    @property
    def n_cost_parameters_per_layer(self) -> int:
        """Number of cost angles in one layer: one for standard, one per cost term for multi-angle."""
        return 1 if self.parametrization == "standard" else len(self.cost_terms)

    @property
    def n_mixer_parameters_per_layer(self) -> int:
        """Number of mixer angles per layer, including one per wire for a multi-angle warm mixer."""
        if self.parametrization == "standard":
            return 1
        return self.n_qubits if self._warm_mixer else len(self.mixer_terms)

    @property
    def parameters_per_layer(self) -> int:
        """Total number of cost and mixer angles in a layer."""
        return self.n_cost_parameters_per_layer + self.n_mixer_parameters_per_layer

    @property
    def n_parameters(self) -> int:
        """Total trainable angle count across all layers."""
        return self.n_layers * self.parameters_per_layer

    @property
    def parameter_shape(self) -> tuple[int, int]:
        """Return (n_layers, parameters_per_layer), with cost angles preceding mixer angles."""
        return self.n_layers, self.parameters_per_layer

    def _resolve_warm_start(self, warm_start, *, epsilon: float):
        """Resolve or classically relax a warm-start vector, validate its size/range and clip it away from deterministic endpoints."""
        if warm_start is None or warm_start is False:
            return None
        if not 0.0 <= epsilon < 0.5:
            raise ValueError("warm_start_epsilon must lie in [0, 0.5).")
        if warm_start is True or (
            isinstance(warm_start, str) and warm_start.lower() in {"relax", "relaxed"}
        ):
            result = minimize(
                lambda values: self.problem.offset
                + values @ self.problem.matrix @ values,
                np.full(self.n_qubits, 0.5),
                method="SLSQP",
                bounds=[(0.0, 1.0)] * self.n_qubits,
            )
            values = np.asarray(result.x, dtype=float)
        else:
            values = np.asarray(warm_start, dtype=float)
        if values.shape != (self.n_qubits,):
            raise ValueError(
                f"warm_start must have shape ({self.n_qubits},), got {values.shape}."
            )
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("warm_start entries must lie in [0, 1].")
        return np.clip(values, epsilon, 1.0 - epsilon)

    def initial_parameters(self, initialization=None) -> np.ndarray:
        """Sample a flat vector using init, an override, a legacy mode or explicit values.

        Built-in initializers use this solver's seeded RNG. None selects the
        constructor's init; the default 'random' retains uniform [-0.1, 0.1).
        'ramp' retains the layer-dependent schedule. Explicit arrays take precedence.
        """
        from qasp.initialization import sample_parameters, Uniform

        initialization = self.init if initialization is None else initialization
        if not isinstance(initialization, str) and not callable(initialization):
            values = np.asarray(initialization, dtype=float)
            if values.size != self.n_parameters or not np.isfinite(values).all():
                raise ValueError(f"Expected {self.n_parameters} finite initial parameters, got {values.size}.")
            return values.reshape(-1).copy()
        name = initialization.lower() if isinstance(initialization, str) else None
        if name == "ramp":
            gamma = np.linspace(0.05, 0.5, self.n_layers)
            beta = np.linspace(0.5, 0.05, self.n_layers)
            if self.parametrization == "standard":
                return np.column_stack((gamma, beta)).reshape(-1)
            layers = []
            for layer in range(self.n_layers):
                layers.append(
                    np.concatenate(
                        (
                            np.full(len(self.cost_terms), gamma[layer]),
                            np.full(self.n_mixer_parameters_per_layer, beta[layer]),
                        )
                    )
                )
            return np.asarray(layers).reshape(-1)
        if name == "random":
            initialization = Uniform(-0.1, 0.1)
        return sample_parameters((self.n_parameters,), initialization, rng=self.rng)

    def _parameter_matrix(self, parameters):
        """Validate total parameter count and reshape into the layer/angle matrix without detaching its differentiation interface."""
        if qml.math.size(parameters) != self.n_parameters:
            raise ValueError(
                f"Expected {self.n_parameters} parameters, "
                f"got {qml.math.size(parameters)}."
            )
        return qml.math.reshape(parameters, self.parameter_shape)

    def _prepare_initial_state(self):
        """Queue the custom, warm-start, all-plus, all-zero or explicit normalized state preparation."""
        if callable(self.initial_state):
            self.initial_state()
            return
        if self.warm_start is not None:
            angles = 2.0 * np.arcsin(np.sqrt(self.warm_start))
            for wire, angle in enumerate(angles):
                qml.RY(angle, wires=wire, id=f"qasp:warm_init:{wire}")
            return
        if isinstance(self.initial_state, str):
            state = self.initial_state.lower()
            if state in {"zero", "zeros", "computational"}:
                return
            if state in {"plus", "+"}:
                for wire in range(self.n_qubits):
                    qml.Hadamard(wires=wire)
                return
        state = np.asarray(self.initial_state)
        if state.shape == (2**self.n_qubits,):
            qml.StatePrep(state, wires=range(self.n_qubits), normalize=True)
            return
        raise ValueError("Unsupported mixer initial_state specification.")

    @staticmethod
    def _apply_pauli_evolution(term: PauliTerm, angle, *, gate_id=None):
        """Queue exp(-i * angle * coefficient * PauliWord) using a PauliRot with twice that angle."""
        qml.PauliRot(
            2.0 * angle * term.coefficient,
            term.pauli,
            wires=term.wires,
            id=gate_id,
        )

    def _apply_warm_mixer(self, beta_values, *, layer: int):
        """Queue exact local evolutions about the warm-start axes, with mixer_evolution_sign applied to beta."""
        angles = 2.0 * np.arcsin(np.sqrt(self.warm_start))
        if qml.math.ndim(beta_values) == 0:
            beta_values = [beta_values] * self.n_qubits
        for wire, (theta, beta) in enumerate(zip(angles, beta_values)):
            # Exact evolution under -sin(theta)X-cos(theta)Z, whose ground
            # state is the warm-start state prepared above.
            qml.RY(
                -theta,
                wires=wire,
                id=f"qasp:warm_basis:pre:{layer}:{wire}",
            )
            qml.RZ(
                -2.0 * self.mixer_evolution_sign * beta,
                wires=wire,
                id=f"qasp:mixer:{layer}:{wire}",
            )
            qml.RY(
                theta,
                wires=wire,
                id=f"qasp:warm_basis:post:{layer}:{wire}",
            )

    def _apply_layers(self, parameters):
        """Queue initial-state preparation and every cost/mixer layer using standard or multi-angle parameter ordering."""
        matrix = self._parameter_matrix(parameters)
        self._prepare_initial_state()
        for layer, layer_parameters in enumerate(matrix):
            if self.parametrization == "standard":
                gamma, beta = layer_parameters
                for term_index, term in enumerate(self.cost_terms):
                    self._apply_pauli_evolution(
                        term,
                        gamma,
                        gate_id=f"qasp:cost:{layer}:{term_index}",
                    )
                if self._warm_mixer:
                    self._apply_warm_mixer(beta, layer=layer)
                else:
                    for term_index, term in enumerate(self.mixer_terms):
                        self._apply_pauli_evolution(
                            term,
                            self.mixer_evolution_sign * beta,
                            gate_id=f"qasp:mixer:{layer}:{term_index}",
                        )
                continue

            split = len(self.cost_terms)
            gammas = layer_parameters[:split]
            betas = layer_parameters[split:]
            for term_index, (term, gamma) in enumerate(zip(self.cost_terms, gammas)):
                self._apply_pauli_evolution(
                    term,
                    gamma,
                    gate_id=f"qasp:cost:{layer}:{term_index}",
                )
            if self._warm_mixer:
                self._apply_warm_mixer(betas, layer=layer)
            else:
                for term_index, (term, beta) in enumerate(
                    zip(self.mixer_terms, betas)
                ):
                    self._apply_pauli_evolution(
                        term,
                        self.mixer_evolution_sign * beta,
                        gate_id=f"qasp:mixer:{layer}:{term_index}",
                    )

    def _expectation_circuit(self, parameters):
        """Queue QAOA and measure the QUBO-derived cost Hamiltonian expectation."""
        self._apply_layers(parameters)
        return qml.expval(self.cost_hamiltonian)

    def _probability_circuit(self, parameters):
        """Queue QAOA and return full-register computational-basis probabilities."""
        self._apply_layers(parameters)
        return qml.probs(wires=range(self.n_qubits))

    def _sample_circuit(self, parameters):
        """Queue QAOA and sample full-register bit strings; requires a finite-shot execution."""
        self._apply_layers(parameters)
        return qml.sample(wires=range(self.n_qubits))

    def expectation_tensor(self, parameters):
        """Return the differentiable Hamiltonian expectation value."""
        return self._expectation_qnode(parameters)

    def expectation(self, parameters) -> float:
        """Evaluate the training Hamiltonian expectation value.

        This uses ``training_shots``: analytic when it is ``None`` and a
        finite-shot estimator otherwise.
        """
        if self.backend == "torch":
            import torch

            values = torch.as_tensor(parameters, dtype=torch.float64)
            return float(self.expectation_tensor(values).detach().cpu())
        values = pnp.array(np.asarray(parameters, dtype=float), requires_grad=False)
        return float(self.expectation_tensor(values))

    def gradient(self, parameters) -> np.ndarray:
        """Differentiate the Hamiltonian expectation with the configured method."""
        if self.diff_method is None:
            raise RuntimeError("This QAOA instance was configured without gradients.")
        if self.backend == "torch":
            import torch

            values = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
            self.expectation_tensor(values).backward()
            return values.grad.detach().cpu().numpy().astype(float, copy=True)
        values = pnp.array(np.asarray(parameters, dtype=float), requires_grad=True)
        gradient = qml.grad(self.expectation_tensor)(values)
        return np.asarray(gradient, dtype=float)

    def probabilities(self, parameters, *, shots: int | None = None) -> np.ndarray:
        """Return exact or finite-shot basis-state probabilities.

        Passing ``shots=None`` explicitly performs analytic evaluation,
        independently of the shot count used during training.  For a sampled
        distribution with counts, use :meth:`sample_solutions` or
        :meth:`solution_counts`.
        """
        qnode = self._probability_qnode.update_shots(shots)
        return np.asarray(
            qnode(np.asarray(parameters, dtype=float)), dtype=float
        )

    def sample_solutions(self, parameters, *, shots: int = 1000) -> np.ndarray:
        """Draw computational-basis solution bitstrings after training."""
        shots = int(shots)
        if shots < 1:
            raise ValueError("shots must be positive.")
        qnode = self._sample_qnode.update_shots(shots)
        samples = np.asarray(qnode(np.asarray(parameters, dtype=float)), dtype=int)
        return samples.reshape(shots, self.n_qubits)

    def _sample_statistics(self, parameters, *, shots: int):
        """Return a dense empirical probability vector and sparse bit-string count dictionary from the requested shots."""
        samples = self.sample_solutions(parameters, shots=shots)
        powers = 1 << np.arange(self.n_qubits - 1, -1, -1)
        indices = samples @ powers
        frequencies = np.bincount(indices, minlength=2**self.n_qubits)
        probabilities = frequencies.astype(float) / int(shots)
        counts = {
            format(index, f"0{self.n_qubits}b"): int(count)
            for index, count in enumerate(frequencies)
            if count
        }
        return probabilities, counts

    def estimate_probabilities(self, parameters, *, shots: int = 1000) -> np.ndarray:
        """Estimate the full solution distribution from finite-shot samples."""
        probabilities, _ = self._sample_statistics(parameters, shots=int(shots))
        return probabilities

    def solution_counts(self, parameters, *, shots: int = 1000) -> dict[str, int]:
        """Return observed solution bitstrings and their finite-shot counts."""
        samples = self.sample_solutions(parameters, shots=shots)
        return dict(Counter("".join(map(str, row)) for row in samples))

    def correlations(self, parameters):
        """Measure Z and ZZ expectations without a full probability vector."""
        pairs = [(i, j) for i in range(self.n_qubits) for j in range(i + 1, self.n_qubits)]

        def circuit(values):
            """Measure all one-body Z expectations followed by the selected two-body Z correlations."""
            self._apply_layers(values)
            return tuple(
                [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
                + [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in pairs]
            )

        node = create_qnode(circuit, self.device, interface=None, diff_method=None)
        values = np.asarray(node(parameters), dtype=float)
        two_body = np.eye(self.n_qubits)
        for value, (i, j) in zip(values[self.n_qubits:], pairs):
            two_body[i, j] = two_body[j, i] = value
        return values[:self.n_qubits], two_body

    def probability_of(
        self,
        bitstring: str,
        parameters,
        *,
        shots: int | None = None,
    ) -> float:
        """Return the exact or shot-estimated probability of one solution."""
        if len(bitstring) != self.n_qubits or any(bit not in "01" for bit in bitstring):
            raise ValueError(f"bitstring must contain {self.n_qubits} binary digits.")
        if shots is None:
            probabilities = self.probabilities(parameters, shots=None)
        else:
            counts = self.solution_counts(parameters, shots=shots)
            return counts.get(bitstring, 0) / int(shots)
        return float(probabilities[int(bitstring, 2)])

    def build_circuit(self, measurement: str = "expectation"):
        """Return the compiled PennyLane QNode for inspection or drawing."""
        if measurement in {"expectation", "expval", "energy"}:
            return self._expectation_qnode
        if measurement in {"probabilities", "probs"}:
            return self._probability_qnode
        raise ValueError("measurement must be 'expectation' or 'probabilities'.")

    def draw(self, parameters=None, **drawing_options):
        """Draw this QAOA circuit as Matplotlib, text, or Quantikz/TikZ."""
        from .drawing import draw_qaoa

        return draw_qaoa(self, parameters=parameters, **drawing_options)

    def _result(
        self,
        parameters,
        *,
        algorithm: str,
        history=None,
        raw_result=None,
        result_shots: int | None = None,
        include_probabilities: bool = True,
    ):
        """Assemble decoded mode, QUBO energy, expectation objective and solver metadata; optionally use finite-shot counts instead of a dense probability vector."""
        parameters = np.asarray(parameters, dtype=float).reshape(-1)
        if not include_probabilities:
            if result_shots is None:
                raise ValueError("include_probabilities=False requires result_shots.")
            result_shots = int(result_shots)
            counts = self.solution_counts(parameters, shots=result_shots)
            probabilities = None
            bitstring = min(counts, key=lambda bits: (-counts[bits], bits))
        elif result_shots is None:
            probabilities = self.probabilities(parameters, shots=None)
            counts = None
        else:
            result_shots = int(result_shots)
            probabilities, counts = self._sample_statistics(
                parameters, shots=result_shots
            )
        if probabilities is not None:
            basis_index = int(np.argmax(probabilities))
            bitstring = format(basis_index, f"0{self.n_qubits}b")
        objective = self.expectation(parameters)
        return OptimizationResult(
            algorithm=algorithm,
            parameters=parameters,
            bitstring=bitstring,
            energy=self.problem.energy(bitstring),
            objective=objective,
            probabilities=probabilities,
            counts=counts,
            shots=result_shots,
            history=list(history or []),
            success=bool(getattr(raw_result, "success", True)),
            message=str(getattr(raw_result, "message", "")),
            metadata={
                "n_layers": self.n_layers,
                "parametrization": self.parametrization,
                "training_shots": self.training_shots,
                "diff_method": self.diff_method,
                "backend": self.backend,
                "mpi": self.mpi,
                "optimizer": getattr(raw_result, "optimizer", None),
                "warm_start": None
                if self.warm_start is None
                else self.warm_start.copy(),
            },
            raw_result=raw_result,
        )

    def solve(
        self,
        initial_parameters=None,
        *,
        optimizer: str | Callable | type | None = None,
        optimizer_options: dict | None = None,
        use_gradients: bool = False,
        result_shots: int | None = None,
        include_probabilities: bool = True,
    ) -> OptimizationResult:
        """Optimize QAOA and return an analytic or sampled solution.

        The NumPy backend accepts SciPy optimizers and PennyLane optimizers
        such as ``Adam``, ``QNG``, ``SPSA``, and ``QNSPSA``. The Torch backend
        accepts ``torch.optim`` names such as ``Adam``, ``AdamW``, ``SGD``,
        ``RMSprop``, and ``LBFGS``. ``result_shots`` is independent of the
        training shot count. With ``include_probabilities=False``, supply
        ``result_shots`` to return sparse counts without an exponential-size
        probability array. The selected solution is the most frequent sample.
        """
        if not include_probabilities and result_shots is None:
            raise ValueError("include_probabilities=False requires result_shots.")
        x0 = self.initial_parameters(initial_parameters)
        optimizer = optimizer or ("Adam" if self.backend == "torch" else "COBYLA")
        options = {"maxiter": 100}
        options.update(optimizer_options or {})
        if self.backend == "torch":
            raw_result = minimize_torch(
                self.expectation_tensor,
                x0,
                optimizer=optimizer,
                optimizer_options=options,
            )
            history = list(raw_result.history)
        elif is_pennylane_optimizer(optimizer):
            raw_result = minimize_pennylane(
                self.expectation_tensor,
                x0,
                optimizer=optimizer,
                optimizer_options=options,
                qnode=self._expectation_qnode,
            )
            history = list(raw_result.history)
        else:
            history = []

            def objective(values):
                """Evaluate the expected cost and append each objective call to optimization history."""
                value = self.expectation(values)
                history.append(value)
                return value

            raw_result = minimize_objective(
                objective,
                x0,
                optimizer=optimizer,
                optimizer_options=options,
                jac=self.gradient if use_gradients else None,
            )
        algorithm = "warm_start_qaoa" if self.warm_start is not None else "qaoa"
        if self.parametrization == "multi_angle":
            algorithm = f"multi_angle_{algorithm}"
        return self._result(
            raw_result.x,
            algorithm=algorithm,
            history=history,
            raw_result=raw_result,
            result_shots=result_shots,
            include_probabilities=include_probabilities,
        )


class WarmStartQAOA(QAOA):
    """Convenience wrapper that always enables warm-start QAOA."""

    def __init__(self, problem, *, warm_start: Sequence[float] | str | bool = True, **kwargs):
        """Initialize QAOA with warm_start=True by default; accept a supplied relaxed vector and forward other options to QAOA."""
        super().__init__(problem, warm_start=warm_start, **kwargs)


class LRQAOA(QAOA):
    """Deterministic linear-ramp QAOA without a classical optimizer.

    Layer ``i`` uses ``gamma_i=(i+1)*delta_gamma/p`` and
    ``beta_i=(1-i/p)*delta_beta``.  Defaults follow the commonly used
    universal schedule ``delta_gamma=0.6`` and ``delta_beta=0.3``.
    """

    def __init__(
        self,
        problem,
        *,
        n_layers: int = 1,
        delta_gamma: float = 0.6,
        delta_beta: float = 0.3,
        **kwargs,
    ):
        """Construct a standard-parameter QAOA with a fixed linear angle ramp.

        n_layers sets depth; delta_gamma and delta_beta scale the increasing cost
        and decreasing mixer schedules. Multi-angle parametrization is rejected.
        The default mixer sign is -1, corresponding to exp(+i beta H_B).
        Remaining keyword options are forwarded to QAOA."""
        if "init" in kwargs:
            raise ValueError("LRQAOA uses a fixed ramp; configure delta_gamma/delta_beta instead of init.")
        if kwargs.get("parametrization", "standard") != "standard":
            raise ValueError("LRQAOA uses the standard two-angle parametrization.")
        kwargs["parametrization"] = "standard"
        # The LR-QAOA definition uses exp(+i beta H_B).
        kwargs.setdefault("mixer_evolution_sign", -1.0)
        super().__init__(problem, n_layers=n_layers, **kwargs)
        self.delta_gamma = float(delta_gamma)
        self.delta_beta = float(delta_beta)

    @property
    def ramp_parameters(self) -> np.ndarray:
        """Return the (n_layers, 2) gamma/beta schedule: gamma increases from delta_gamma/p to delta_gamma; beta decreases from delta_beta to delta_beta/p."""
        layers = np.arange(self.n_layers, dtype=float)
        gammas = (layers + 1.0) * self.delta_gamma / self.n_layers
        betas = (1.0 - layers / self.n_layers) * self.delta_beta
        return np.column_stack((gammas, betas))

    def solve(self, *, result_shots: int | None = None, include_probabilities: bool = True) -> OptimizationResult:
        """Execute the fixed ramp and return the most probable state."""
        result = self._result(
            self.ramp_parameters.reshape(-1),
            algorithm="lr_qaoa",
            history=[],
            result_shots=result_shots,
            include_probabilities=include_probabilities,
        )
        result.metadata.update(
            {
                "delta_gamma": self.delta_gamma,
                "delta_beta": self.delta_beta,
                "ramp_parameters": self.ramp_parameters.copy(),
            }
        )
        return result


LinearRampQAOA = LRQAOA


__all__ = [
    "LRQAOA",
    "LinearRampQAOA",
    "QAOA",
    "WarmStartQAOA",
]
