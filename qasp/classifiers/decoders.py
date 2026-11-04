"""Class-probability decoders for Born-rule quantum classifiers.

The decoders use the same basis-state partitions as QASP's reinforcement-
learning policies, but expose classification terminology and a standalone API.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class ClassProbabilityDecoder(nn.Module):
    """Aggregate computational-basis probabilities into class probabilities.

    PennyLane orders probabilities as ``|00...0>, |00...1>, ..., |11...1>``.
    ``majority_blocks`` therefore contains positions in the ordered measured
    bit string, rather than physical wire labels.
    """

    _ALIASES = {
        "contiguous": "contiguous",
        "contiguous-like": "contiguous",
        "parity": "parity",
        "parity-like": "parity",
        "majority": "majority",
        "majority-modulo": "majority",
        "majority_modulo": "majority",
    }

    def __init__(
        self,
        decoder: str,
        n_classes: int,
        n_measured_qubits: int,
        majority_blocks: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Precompute the assignment of measured basis states to class indices.

        Args:
            decoder: Contiguous, parity or majority decoder name/registered alias.
            n_classes: At least two and at most 2**n_measured_qubits classes.
            n_measured_qubits: Number of positions in each measured bit string.
            majority_blocks: Disjoint nonempty lists of measured-bit positions,
                required for majority; exactly ceil(log2(n_classes)) blocks.

        Bits are ordered most significant first. Majority uses strict majority
        (ties map to zero); resulting codes are reduced modulo n_classes. Assignment
        indices are registered Torch buffers and follow the module's device."""
        super().__init__()
        if decoder not in self._ALIASES:
            supported = ", ".join(sorted(self._ALIASES))
            raise ValueError(
                f"Unknown decoder '{decoder}'. Supported values: {supported}."
            )

        self.decoder = self._ALIASES[decoder]
        self.n_classes = int(n_classes)
        self.n_measured_qubits = int(n_measured_qubits)
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2.")
        if self.n_measured_qubits < 1:
            raise ValueError("At least one measured qubit is required.")

        n_states = 1 << self.n_measured_qubits
        if self.n_classes > n_states:
            raise ValueError(
                f"n_classes={self.n_classes} exceeds the {n_states} measured "
                "computational-basis states."
            )

        self.n_code_bits = math.ceil(math.log2(self.n_classes))
        self.majority_blocks = (
            None
            if majority_blocks is None
            else tuple(tuple(int(position) for position in block) for block in majority_blocks)
        )
        action_index = self._build_class_index()
        basis_weights = 1 << torch.arange(
            self.n_measured_qubits - 1, -1, -1, dtype=torch.long
        )
        self.register_buffer("class_index", action_index, persistent=True)
        self.register_buffer(
            "basis_state_counts",
            torch.bincount(action_index, minlength=self.n_classes),
            persistent=False,
        )
        self.register_buffer("basis_weights", basis_weights, persistent=False)

    def _basis_bits(self) -> torch.Tensor:
        """Enumerate measured basis states as a (2**n_measured_qubits, n_measured_qubits) integer tensor, most significant bit first."""
        states = torch.arange(1 << self.n_measured_qubits, dtype=torch.long)
        shifts = torch.arange(
            self.n_measured_qubits - 1, -1, -1, dtype=torch.long
        )
        return torch.bitwise_and(
            torch.bitwise_right_shift(states[:, None], shifts[None, :]), 1
        )

    def _build_class_index(self) -> torch.Tensor:
        """Construct the basis-state-to-class vector; validate majority blocks and reduce parity/majority codes modulo n_classes."""
        n_states = 1 << self.n_measured_qubits
        if self.decoder == "contiguous":
            base, remainder = divmod(n_states, self.n_classes)
            sizes = torch.full((self.n_classes,), base, dtype=torch.long)
            sizes[:remainder] += 1
            return torch.repeat_interleave(
                torch.arange(self.n_classes, dtype=torch.long), sizes
            )

        bits = self._basis_bits()
        if self.decoder == "parity":
            parity_bits = torch.stack(
                [
                    bits[:, : self.n_measured_qubits - offset]
                    .sum(dim=1)
                    .remainder(2)
                    for offset in range(self.n_code_bits)
                ],
                dim=1,
            )
            code_weights = 1 << torch.arange(
                self.n_code_bits - 1, -1, -1, dtype=torch.long
            )
            return (parity_bits * code_weights).sum(dim=1).remainder(self.n_classes)

        blocks = self.majority_blocks
        if blocks is None:
            raise ValueError("majority_blocks are required by the majority decoder.")
        if len(blocks) != self.n_code_bits:
            raise ValueError(
                "The majority decoder requires exactly ceil(log2(n_classes)) "
                f"={self.n_code_bits} blocks; received {len(blocks)}."
            )

        flattened = [position for block in blocks for position in block]
        if any(position < 0 or position >= self.n_measured_qubits for position in flattened):
            raise ValueError(
                "majority_blocks contain positions outside the measured bit string."
            )
        if len(flattened) != len(set(flattened)):
            raise ValueError("majority_blocks must be disjoint.")
        if any(not block for block in blocks):
            raise ValueError("Every majority block must be non-empty.")

        majority_bits = torch.stack(
            [
                (bits[:, list(block)].sum(dim=1) > (len(block) // 2)).to(torch.long)
                for block in blocks
            ],
            dim=1,
        )
        code_weights = 1 << torch.arange(
            self.n_code_bits - 1, -1, -1, dtype=torch.long
        )
        return (majority_bits * code_weights).sum(dim=1).remainder(self.n_classes)

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Return class probabilities and preserve every leading batch axis."""
        if probabilities.ndim < 1:
            raise ValueError("probabilities must have at least one dimension.")
        n_states = self.class_index.numel()
        if probabilities.shape[-1] != n_states:
            raise ValueError(
                f"Expected {n_states} basis probabilities on the final axis, "
                f"received {probabilities.shape[-1]}."
            )

        leading_shape = probabilities.shape[:-1]
        flattened = probabilities.reshape(-1, n_states)
        decoded = flattened.new_zeros((flattened.shape[0], self.n_classes))
        indices = self.class_index.unsqueeze(0).expand(flattened.shape[0], -1)
        decoded.scatter_add_(1, indices, flattened)
        return decoded.reshape(*leading_shape, self.n_classes)

    def decode_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Decode one measured bit string or an arbitrary batch of bit strings."""
        samples = torch.as_tensor(samples, device=self.class_index.device)
        if samples.ndim < 1 or samples.shape[-1] != self.n_measured_qubits:
            raise ValueError(
                f"Expected samples with final dimension {self.n_measured_qubits}; "
                f"received shape {tuple(samples.shape)}."
            )
        if not torch.all((samples == 0) | (samples == 1)):
            raise ValueError("samples must contain only binary values 0 and 1.")
        basis_index = (samples.to(torch.long) * self.basis_weights).sum(dim=-1)
        return self.class_index[basis_index]

    @property
    def reference_class_prior(self) -> torch.Tensor:
        """Class distribution induced by a uniform basis-state distribution."""
        return self.basis_state_counts / self.basis_state_counts.sum()


def balanced_majority_blocks(
    measurement_qubits: Sequence[int],
    n_classes: int,
    majority_blocks: Sequence[Sequence[int]] | None = None,
    *,
    strict: bool = False,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Resolve physical majority blocks and their measured-string positions.

    Returns ``(wire_blocks, position_blocks, unused_wires)``. When blocks are
    omitted, QASP uses maximally covering, nearly balanced odd blocks and leaves
    at most one measured wire unused. Explicit even blocks are accepted and ties
    decode to zero, matching the reinforcement-learning majority decoder.
    """
    wires = tuple(int(wire) for wire in measurement_qubits)
    if len(set(wires)) != len(wires):
        raise ValueError("measurement_qubits must not contain duplicates.")
    n_blocks = math.ceil(math.log2(int(n_classes)))
    if len(wires) < n_blocks:
        raise ValueError(
            f"Need at least ceil(log2(n_classes))={n_blocks} measured qubits; "
            f"received {len(wires)}."
        )

    if majority_blocks is None:
        used = len(wires) if (len(wires) - n_blocks) % 2 == 0 else len(wires) - 1
        pairs, extra = divmod((used - n_blocks) // 2, n_blocks)
        sizes = [1 + 2 * pairs + (2 if index < extra else 0) for index in range(n_blocks)]
        blocks = []
        start = 0
        for size in sizes:
            blocks.append(wires[start : start + size])
            start += size
        unused = wires[start:]
    else:
        blocks = [tuple(int(wire) for wire in block) for block in majority_blocks]
        if len(blocks) != n_blocks:
            raise ValueError(
                f"majority_blocks must contain {n_blocks} blocks; received {len(blocks)}."
            )
        flattened = [wire for block in blocks for wire in block]
        unknown = [wire for wire in flattened if wire not in wires]
        if unknown:
            raise ValueError(
                f"majority_blocks contain wires not in measurement_qubits: {unknown}."
            )
        if len(flattened) != len(set(flattened)):
            raise ValueError("majority_blocks must be disjoint.")
        unused = tuple(wire for wire in wires if wire not in set(flattened))

    sizes = [len(block) for block in blocks]
    if any(size == 0 for size in sizes):
        raise ValueError("Every majority block must be non-empty.")
    if strict and (
        len(set(sizes)) != 1
        or sizes[0] < 3
        or sizes[0] % 2 == 0
        or unused
    ):
        raise ValueError(
            "strict=True requires equal odd block sizes >= 3 and no unused wires."
        )

    wire_to_position = {wire: position for position, wire in enumerate(wires)}
    position_blocks = tuple(
        tuple(wire_to_position[wire] for wire in block) for block in blocks
    )
    return tuple(blocks), position_blocks, tuple(unused)


ProbabilityDecoder = ClassProbabilityDecoder


__all__ = [
    "ClassProbabilityDecoder",
    "ProbabilityDecoder",
    "balanced_majority_blocks",
]
