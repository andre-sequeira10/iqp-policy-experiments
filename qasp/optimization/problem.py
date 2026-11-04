"""QUBO problem definitions and QUBO-to-Ising conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np
import pennylane as qml


@dataclass(frozen=True)
class PauliTerm:
    """One weighted Pauli word used in an optimization Hamiltonian."""

    coefficient: float
    pauli: str
    wires: tuple[int, ...]


class QUBO:
    """A quadratic unconstrained binary optimization problem.

    The objective is ``offset + x.T @ matrix @ x`` for binary variables
    ``x[i] in {0, 1}``.  Non-symmetric matrices are accepted; their two
    off-diagonal entries are combined into one quadratic coefficient.
    """

    def __init__(self, matrix, offset: float = 0.0):
        """Copy a finite square coefficient matrix and constant energy offset.

        The objective is ``offset + x.T @ matrix @ x`` for binary x. Off-diagonal
        entries are not implicitly halved: a pair coefficient is Q[i,j]+Q[j,i].
        Require at least one variable; invalid shape/nonfinite coefficients raise
        ValueError. Subsequent changes to the caller's array do not change this QUBO."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be a square two-dimensional array.")
        if matrix.shape[0] < 1:
            raise ValueError("A QUBO must contain at least one variable.")
        if not np.isfinite(matrix).all() or not np.isfinite(offset):
            raise ValueError("QUBO coefficients and offset must be finite.")
        self.matrix = matrix.copy()
        self.offset = float(offset)

    @classmethod
    def from_terms(
        cls,
        n_variables: int,
        *,
        linear: Mapping[int, float] | Sequence[float] | None = None,
        quadratic: Mapping[tuple[int, int], float] | None = None,
        offset: float = 0.0,
    ) -> "QUBO":
        """Build a QUBO from explicit linear and quadratic coefficients."""
        n_variables = int(n_variables)
        if n_variables < 1:
            raise ValueError("n_variables must be positive.")
        matrix = np.zeros((n_variables, n_variables), dtype=float)
        if linear is not None:
            items = linear.items() if isinstance(linear, Mapping) else enumerate(linear)
            for index, value in items:
                index = int(index)
                if index < 0 or index >= n_variables:
                    raise ValueError(f"Linear index {index} is outside the QUBO.")
                matrix[index, index] += float(value)
        if quadratic is not None:
            for pair, value in quadratic.items():
                if len(pair) != 2:
                    raise ValueError("Quadratic keys must be pairs of indices.")
                i, j = map(int, pair)
                if i == j:
                    matrix[i, i] += float(value)
                    continue
                if min(i, j) < 0 or max(i, j) >= n_variables:
                    raise ValueError(f"Quadratic index {(i, j)} is outside the QUBO.")
                matrix[min(i, j), max(i, j)] += float(value)
        return cls(matrix, offset=offset)

    @classmethod
    def from_ising(
        cls,
        n_variables: int,
        linear: Mapping[int, float],
        quadratic: Mapping[tuple[int, int], float],
        *,
        offset: float = 0.0,
    ) -> "QUBO":
        """Create a QUBO from ``offset + sum h_i Z_i + sum J_ij Z_i Z_j``."""
        h = {int(i): float(value) for i, value in linear.items()}
        couplings = {
            tuple(sorted(map(int, pair))): float(value)
            for pair, value in quadratic.items()
        }
        qubo_offset = float(offset) + sum(h.values()) + sum(couplings.values())
        qubo_linear = {
            i: -2.0 * h.get(i, 0.0)
            - 2.0
            * sum(value for pair, value in couplings.items() if i in pair)
            for i in range(int(n_variables))
        }
        qubo_quadratic = {pair: 4.0 * value for pair, value in couplings.items()}
        return cls.from_terms(
            n_variables,
            linear=qubo_linear,
            quadratic=qubo_quadratic,
            offset=qubo_offset,
        )

    @classmethod
    def maxcut(cls, n_variables: int, edges: Iterable[Sequence[float]]) -> "QUBO":
        """Return the minimization QUBO for weighted MaxCut.

        Each edge is ``(i, j)`` or ``(i, j, weight)``.  The returned objective
        is the negative cut value, so minimizing it maximizes the cut.
        """
        linear: dict[int, float] = {}
        quadratic: dict[tuple[int, int], float] = {}
        for edge in edges:
            if len(edge) == 2:
                i, j = edge
                weight = 1.0
            elif len(edge) == 3:
                i, j, weight = edge
            else:
                raise ValueError("MaxCut edges must be (i, j) or (i, j, weight).")
            i, j, weight = int(i), int(j), float(weight)
            if i == j:
                raise ValueError("MaxCut edges cannot be self-loops.")
            pair = tuple(sorted((i, j)))
            linear[i] = linear.get(i, 0.0) - weight
            linear[j] = linear.get(j, 0.0) - weight
            quadratic[pair] = quadratic.get(pair, 0.0) + 2.0 * weight
        return cls.from_terms(
            n_variables, linear=linear, quadratic=quadratic
        )

    @property
    def n_variables(self) -> int:
        """Number of binary variables, equal to the coefficient matrix dimension."""
        return int(self.matrix.shape[0])

    @property
    def linear(self) -> dict[int, float]:
        """Return non-negligible diagonal coefficients as an index-to-weight dictionary."""
        return {
            i: float(self.matrix[i, i])
            for i in range(self.n_variables)
            if not np.isclose(self.matrix[i, i], 0.0)
        }

    @property
    def quadratic(self) -> dict[tuple[int, int], float]:
        """Return non-negligible unordered pair coefficients Q[i,j]+Q[j,i], keyed by i<j."""
        terms = {}
        for i in range(self.n_variables):
            for j in range(i + 1, self.n_variables):
                value = float(self.matrix[i, j] + self.matrix[j, i])
                if not np.isclose(value, 0.0):
                    terms[(i, j)] = value
        return terms

    def energy(self, bitstring: str | Sequence[int]) -> float:
        """Evaluate the QUBO objective on a binary assignment."""
        if isinstance(bitstring, str):
            if any(bit not in "01" for bit in bitstring):
                raise ValueError("A bitstring may contain only '0' and '1'.")
            bits = np.fromiter((int(bit) for bit in bitstring), dtype=float)
        else:
            bits = np.asarray(bitstring, dtype=float)
        if bits.shape != (self.n_variables,):
            raise ValueError(
                f"Expected {self.n_variables} bits, received shape {bits.shape}."
            )
        if not np.isin(bits, (0.0, 1.0)).all():
            raise ValueError("QUBO assignments must be binary.")
        return float(self.offset + bits @ self.matrix @ bits)

    def to_ising(
        self,
    ) -> tuple[float, dict[int, float], dict[tuple[int, int], float]]:
        """Return ``(offset, h, J)`` under the convention ``x=(1-Z)/2``."""
        linear = self.linear
        quadratic = self.quadratic
        offset = self.offset + 0.5 * sum(linear.values())
        offset += 0.25 * sum(quadratic.values())
        h = {
            i: -0.5 * linear.get(i, 0.0)
            - 0.25
            * sum(value for pair, value in quadratic.items() if i in pair)
            for i in range(self.n_variables)
        }
        h = {i: value for i, value in h.items() if not np.isclose(value, 0.0)}
        couplings = {
            pair: 0.25 * value
            for pair, value in quadratic.items()
            if not np.isclose(value, 0.0)
        }
        return float(offset), h, couplings

    def pauli_terms(self, *, include_constant: bool = False) -> list[PauliTerm]:
        """Return the nonzero Pauli terms in the Ising Hamiltonian."""
        offset, h, couplings = self.to_ising()
        terms = []
        if include_constant and not np.isclose(offset, 0.0):
            terms.append(PauliTerm(offset, "I", (0,)))
        terms.extend(PauliTerm(value, "Z", (i,)) for i, value in h.items())
        terms.extend(
            PauliTerm(value, "ZZ", pair) for pair, value in couplings.items()
        )
        return terms

    def hamiltonian(self):
        """Return the exact PennyLane cost Hamiltonian, including its offset."""
        terms = self.pauli_terms(include_constant=True)
        if not terms:
            return qml.Hamiltonian([0.0], [qml.Identity(0)])
        observables = []
        coefficients = []
        pauli_map = {"I": qml.Identity, "X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
        for term in terms:
            factors = [pauli_map[p](wire) for p, wire in zip(term.pauli, term.wires)]
            observable = factors[0]
            for factor in factors[1:]:
                observable = observable @ factor
            coefficients.append(term.coefficient)
            observables.append(observable)
        return qml.Hamiltonian(coefficients, observables)

    def brute_force(self) -> tuple[str, float]:
        """Return a minimum-energy assignment and its objective value."""
        best_bits = None
        best_energy = np.inf
        for bits in product((0, 1), repeat=self.n_variables):
            energy = self.energy(bits)
            if energy < best_energy:
                best_bits, best_energy = bits, energy
        return "".join(map(str, best_bits)), float(best_energy)

    def __repr__(self) -> str:
        """Return a concise description containing the variable count and constant offset."""
        return f"QUBO(n_variables={self.n_variables}, offset={self.offset})"


__all__ = ["PauliTerm", "QUBO"]
