"""Recursive QAOA for reducing and exactly finishing QUBO problems."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .problem import QUBO
from .qaoa import QAOA
from .results import RecursiveQAOAResult


@dataclass
class _ReducedIsing:
    """Mutable reduced Ising objective indexed by original variable labels, with an accumulated offset."""
    active: list[int]
    offset: float
    linear: dict[int, float]
    quadratic: dict[tuple[int, int], float]

    @staticmethod
    def _add(mapping, key, value):
        """Accumulate a coefficient in place and remove terms numerically close to zero."""
        mapping[key] = mapping.get(key, 0.0) + float(value)
        if np.isclose(mapping[key], 0.0, atol=1e-12):
            mapping.pop(key, None)

    def as_qubo(self) -> QUBO:
        """Return a QUBO in active-variable order, retaining the accumulated energy offset."""
        position = {label: index for index, label in enumerate(self.active)}
        linear = {position[label]: value for label, value in self.linear.items()}
        quadratic = {
            tuple(sorted((position[i], position[j]))): value
            for (i, j), value in self.quadratic.items()
        }
        return QUBO.from_ising(
            len(self.active), linear, quadratic, offset=self.offset
        )

    def fix(self, label: int, sign: int):
        """Substitute spin[label]=sign in place, update offset/neighbor fields, and remove the fixed variable."""
        self.offset += self.linear.pop(label, 0.0) * sign
        for pair, value in list(self.quadratic.items()):
            if label not in pair:
                continue
            other = pair[1] if pair[0] == label else pair[0]
            self._add(self.linear, other, sign * value)
            self.quadratic.pop(pair)
        self.active.remove(label)

    def relate(self, removed: int, kept: int, sign: int):
        """Substitute ``Z_removed = sign * Z_kept``."""
        self._add(self.linear, kept, sign * self.linear.pop(removed, 0.0))
        for pair, value in list(self.quadratic.items()):
            if removed not in pair:
                continue
            self.quadratic.pop(pair)
            other = pair[1] if pair[0] == removed else pair[0]
            if other == kept:
                self.offset += sign * value
            else:
                replacement = tuple(sorted((kept, other)))
                self._add(self.quadratic, replacement, sign * value)
        self.active.remove(removed)

    def exact_ground_state(self) -> tuple[dict[int, int], str, float]:
        """Enumerate remaining spins and return the minimizing original-label assignment, active-order bit string and energy; exponential in active count."""
        best_spins = None
        best_value = np.inf
        for spin_tuple in product((1, -1), repeat=len(self.active)):
            spins = dict(zip(self.active, spin_tuple))
            value = self.offset
            value += sum(coefficient * spins[i] for i, coefficient in self.linear.items())
            value += sum(
                coefficient * spins[i] * spins[j]
                for (i, j), coefficient in self.quadratic.items()
            )
            if value < best_value:
                best_spins, best_value = spins, value
        bits = "".join("0" if best_spins[label] == 1 else "1" for label in self.active)
        return best_spins, bits, float(best_value)


class RecursiveQAOA:
    """Recursively eliminate variables using QAOA Pauli correlations.

    At each step the largest-magnitude expectation among Hamiltonian terms
    ``<Z_i>`` and ``<Z_i Z_j>`` is rounded to a sign and substituted into the
    Ising model.  Once ``cutoff`` variables remain, the reduced problem is
    solved exactly and all eliminations are reconstructed in reverse order.
    """

    def __init__(
        self,
        problem: QUBO | np.ndarray,
        *,
        cutoff: int = 2,
        n_layers: int = 1,
        qaoa_options: dict | None = None,
        optimizer: str = "COBYLA",
        optimizer_options: dict | None = None,
        device=None,
        mpi: bool | None = None,
        device_kwargs: dict | None = None,
        init=None,
        cutting=None,
    ):
        """Configure correlation-based QAOA reduction followed by exact enumeration.

        problem is a QUBO or matrix. cutoff is the positive number of variables left
        for brute-force solution; n_layers configures each intermediate QAOA. The
        qaoa_options dictionary configures these solvers, with explicit device/mpi/
        device_kwargs overriding it. optimizer and optimizer_options apply per
        reduction step (default COBYLA, maxiter=100). step_solvers and step_results
        record the later solve operation; construction alone does not reduce the QUBO.
        init sets each subproblem's initializer and overrides qaoa_options['init']."""
        self.problem = problem if isinstance(problem, QUBO) else QUBO(problem)
        self.cutoff = int(cutoff)
        if self.cutoff < 1:
            raise ValueError("cutoff must be positive.")
        self.n_layers = int(n_layers)
        self.qaoa_options = dict(qaoa_options or {})
        if cutting is not None:
            self.qaoa_options["cutting"] = cutting
        if init is not None:
            self.qaoa_options["init"] = init
        if device is not None:
            self.qaoa_options["device"] = device
        if mpi is not None:
            self.qaoa_options["mpi"] = mpi
        if device_kwargs is not None:
            self.qaoa_options["device_kwargs"] = dict(device_kwargs)
        self.optimizer = optimizer
        self.optimizer_options = {"maxiter": 100}
        self.optimizer_options.update(optimizer_options or {})
        self.step_solvers = []
        self.step_results = []

    @staticmethod
    def _correlations(probabilities, n_qubits):
        """Return one- and two-body Z expectations from a dense basis-probability vector in most-significant-bit wire order."""
        basis = np.arange(2**n_qubits, dtype=np.uint64)
        bit_positions = np.arange(n_qubits - 1, -1, -1, dtype=np.uint64)
        bits = ((basis[:, None] >> bit_positions) & 1).astype(float)
        spins = 1.0 - 2.0 * bits
        one_body = probabilities @ spins
        two_body = np.einsum("s,si,sj->ij", probabilities, spins, spins)
        return one_body, two_body

    @staticmethod
    def _strongest_rule(reduced, one_body, two_body):
        """Choose the largest-magnitude supported single/pair correlation and its sign; choose a deterministic positive single-spin rule when no terms remain."""
        position = {label: index for index, label in enumerate(reduced.active)}
        candidates = []
        for label in reduced.linear:
            value = float(one_body[position[label]])
            candidates.append((abs(value), "single", (label,), value))
        for pair in reduced.quadratic:
            i, j = pair
            value = float(two_body[position[i], position[j]])
            candidates.append((abs(value), "pair", pair, value))
        if not candidates:
            label = reduced.active[-1]
            return "single", (label,), 1.0
        _, kind, labels, value = max(candidates, key=lambda item: item[0])
        return kind, labels, 1.0 if value >= 0.0 else -1.0

    def solve(self, *, result_shots=None, include_probabilities=True) -> RecursiveQAOAResult:
        """Eliminate variables with optional counts-only intermediate results.

        Counts-only execution uses direct Z/ZZ measurements for elimination,
        with the QAOA device's training shot count. MPI communicator size is
        fixed, so the cutoff register must still fit that communicator.
        """
        if not include_probabilities and result_shots is None:
            raise ValueError("include_probabilities=False requires result_shots.")
        self.step_solvers = []
        self.step_results = []
        offset, linear, quadratic = self.problem.to_ising()
        reduced = _ReducedIsing(
            active=list(range(self.problem.n_variables)),
            offset=offset,
            linear=dict(linear),
            quadratic=dict(quadratic),
        )
        rules = []
        intermediate_results = []

        while len(reduced.active) > self.cutoff:
            reduced_qubo = reduced.as_qubo()
            options = dict(self.qaoa_options)
            options.setdefault("seed", len(intermediate_results))
            qaoa = QAOA(reduced_qubo, n_layers=self.n_layers, **options)
            result = qaoa.solve(
                optimizer=self.optimizer,
                optimizer_options=self.optimizer_options,
                result_shots=result_shots,
                include_probabilities=include_probabilities,
            )
            intermediate_results.append(result)
            self.step_solvers.append(qaoa)
            self.step_results.append(result)
            if result.probabilities is None:
                one_body, two_body = qaoa.correlations(result.parameters)
            else:
                one_body, two_body = self._correlations(
                    result.probabilities, len(reduced.active)
                )
            kind, labels, sign = self._strongest_rule(
                reduced, one_body, two_body
            )
            sign = int(sign)
            if kind == "single":
                label = labels[0]
                rules.append({"singlet": (label,), "bias": sign})
                reduced.fix(label, sign)
            else:
                kept, removed = labels
                rules.append(
                    {"pair": (kept, removed), "correlation": sign}
                )
                reduced.relate(removed, kept, sign)

        spins, cutoff_bitstring, cutoff_energy = reduced.exact_ground_state()
        for rule in reversed(rules):
            if "singlet" in rule:
                spins[rule["singlet"][0]] = int(rule["bias"])
            else:
                kept, removed = rule["pair"]
                spins[removed] = int(rule["correlation"]) * spins[kept]
        bitstring = "".join(
            "0" if spins[index] == 1 else "1"
            for index in range(self.problem.n_variables)
        )
        return RecursiveQAOAResult(
            bitstring=bitstring,
            energy=self.problem.energy(bitstring),
            elimination_rules=rules,
            intermediate_results=intermediate_results,
            cutoff_bitstring=cutoff_bitstring,
            metadata={
                "cutoff": self.cutoff,
                "cutoff_energy": cutoff_energy,
                "n_steps": len(rules),
            },
        )

    def draw(self, *, step: int = 0, parameters=None, **drawing_options):
        """Draw the constituent QAOA circuit from one completed recursion step."""
        from .drawing import draw_recursive_qaoa

        return draw_recursive_qaoa(
            self,
            step=step,
            parameters=parameters,
            **drawing_options,
        )


RQAOA = RecursiveQAOA


__all__ = ["RQAOA", "RecursiveQAOA"]
