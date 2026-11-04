"""Result containers shared by QASP optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OptimizationResult:
    """Quantum optimization output separating a candidate from its training objective.

    bitstring is the selected/decoded binary candidate; energy evaluates its
    QUBO cost. objective records the optimized expectation or PCE surrogate.
    parameters contains trained/ramp angles. probabilities and counts are
    optional exact/shot-derived output distributions; shots identifies result
    sampling. history holds optimizer objectives, while success/message report
    optimizer status, not proof of global optimality. metadata stores
    algorithm-specific interpretation and raw_result retains the optimizer object."""

    algorithm: str
    parameters: np.ndarray
    bitstring: str
    energy: float
    objective: float
    probabilities: np.ndarray | None = None
    counts: dict[str, int] | None = None
    shots: int | None = None
    history: list[float] = field(default_factory=list)
    success: bool = True
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_result: Any = None

    @property
    def most_probable_bitstring(self) -> str:
        """Return the stored decoded bitstring (compatibility property).

        For QAOA this is the selected distribution mode. For PCE or recursive
        results it is the algorithm's decoded solution and need not be a sampled mode."""
        return self.bitstring

    @property
    def solution_probability(self) -> float | None:
        """Probability assigned to :attr:`bitstring`, exact or shot-estimated."""
        if self.probabilities is None:
            if self.counts is not None and self.shots:
                return self.counts.get(self.bitstring, 0) / self.shots
            return None
        return float(self.probabilities[int(self.bitstring, 2)])


@dataclass
class RecursiveQAOAResult:
    """Reconstructed solution and provenance of recursive variable elimination.

    bitstring and energy refer to the original QUBO. elimination_rules and
    intermediate_results describe each reduction; cutoff_bitstring is the
    exact solution of the final reduced problem. Exactness of that small
    subproblem does not certify global optimality of the reconstructed result."""

    bitstring: str
    energy: float
    elimination_rules: list[dict[str, Any]]
    intermediate_results: list[OptimizationResult]
    cutoff_bitstring: str
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["OptimizationResult", "RecursiveQAOAResult"]
