# torchlane/rl/differentiable_pauli_propagator.py
"""Packed-Pauli Heisenberg propagation with Torch-autograd coefficients.

The engine traces supported gates backward from expectation observables and
contracts with the all-zero input state. k limits Pauli weight, while tau/topM
prune coefficient magnitudes. Pruning masks use detached values: derivatives
flow through surviving coefficients only, not through selection decisions.
Truncation/pruning change outputs and gradients and carry no general error
bound. This is QASP's internal lightning.pp route, not a PennyLane plugin."""
from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import torch
import pennylane as qml
from pennylane.pauli import PauliSentence


# ============================================================
# Differentiable Pauli propagation
# ============================================================
# - Packed integer key for Pauli words:
#       key = x_mask | (z_mask << n)
#
# - Coefficients are torch scalar tensors, so PyTorch can backpropagate.
#
# - Supports hard pruning:
#       k:    Pauli-weight truncation
#       tau:  coefficient threshold
#       topM: keep top-M largest coefficient magnitudes
#
#   tau/topM decisions are made with detached coefficient magnitudes:
#
#       mask = mask(coeff.detach())
#       kept_coeff = coeff
#
#   Therefore:
#       - kept Pauli paths receive gradients;
#       - dropped paths receive zero gradients;
#       - the mask itself is treated as fixed during backprop.
#
# Selection is held fixed during backpropagation; no straight-through mask gradient is used.
# ============================================================


# Local Pauli code from (xbit, zbit):
# code = xbit | (zbit << 1)
#
#   0: I  (0,0)
#   1: X  (1,0)
#   2: Z  (0,1)
#   3: Y  (1,1)
CODE_I, CODE_X, CODE_Z, CODE_Y = 0, 1, 2, 3


def _pack(x_mask: int, z_mask: int, n: int) -> int:
    """Encode n-wire X/Z bit masks as x_mask | (z_mask << n)."""
    return x_mask | (z_mask << n)


def _unpack(key: int, n: int) -> Tuple[int, int]:
    """Recover (x_mask, z_mask) from an n-wire packed Pauli key."""
    x_mask = key & ((1 << n) - 1)
    z_mask = key >> n
    return x_mask, z_mask


def _weight_from_masks(x_mask: int, z_mask: int) -> int:
    """Count nonidentity factors, counting Y (both masks set) only once."""
    return (x_mask | z_mask).bit_count()


def _get_code(x_mask: int, z_mask: int, wire: int) -> int:
    """Read one wire as I=0, X=1, Z=2 or Y=3 from packed masks."""
    xb = (x_mask >> wire) & 1
    zb = (z_mask >> wire) & 1
    return xb | (zb << 1)


def _set_code(x_mask: int, z_mask: int, wire: int, code: int) -> Tuple[int, int]:
    """Replace one wire's I/X/Z/Y code and return the updated X/Z masks."""
    xb = code & 1
    zb = (code >> 1) & 1

    bit = 1 << wire

    x_mask &= ~bit
    z_mask &= ~bit

    if xb:
        x_mask |= bit
    if zb:
        z_mask |= bit

    return x_mask, z_mask


def _mul1_build() -> List[List[Tuple[int, complex]]]:
    """
    Single-qubit Pauli multiplication in CODE ordering (I, X, Z, Y).

    Returns:
        (result_code, phase)

    where phase is one of:
        1, -1, 1j, -1j
    """
    mul = [[(CODE_I, 1)] * 4 for _ in range(4)]

    for a in range(4):
        for b in range(4):
            if a == CODE_I:
                mul[a][b] = (b, 1)
                continue

            if b == CODE_I:
                mul[a][b] = (a, 1)
                continue

            if a == b:
                mul[a][b] = (CODE_I, 1)
                continue

            # Standard Pauli algebra:
            #
            # X Y =  i Z
            # Y X = -i Z
            # Y Z =  i X
            # Z Y = -i X
            # Z X =  i Y
            # X Z = -i Y
            if a == CODE_X and b == CODE_Y:
                mul[a][b] = (CODE_Z, 1j)
            elif a == CODE_Y and b == CODE_X:
                mul[a][b] = (CODE_Z, -1j)
            elif a == CODE_Y and b == CODE_Z:
                mul[a][b] = (CODE_X, 1j)
            elif a == CODE_Z and b == CODE_Y:
                mul[a][b] = (CODE_X, -1j)
            elif a == CODE_Z and b == CODE_X:
                mul[a][b] = (CODE_Y, 1j)
            elif a == CODE_X and b == CODE_Z:
                mul[a][b] = (CODE_Y, -1j)
            else:
                raise RuntimeError(f"Missing Pauli multiplication case for {(a, b)}")

    return mul


_MUL1 = _mul1_build()


def _anticommutes_1q(axis_code: int, local_code: int) -> bool:
    """Test whether a local Pauli anticommutes with a nonidentity rotation axis."""
    return (local_code != CODE_I) and (local_code != axis_code)


def _parse_pauli_word_to_axis_code(pw: str) -> int:
    """Map X, Y or Z to its packed code; reject other one-qubit words."""
    if pw == "X":
        return CODE_X
    if pw == "Y":
        return CODE_Y
    if pw == "Z":
        return CODE_Z

    raise ValueError(f"Unsupported 1-qubit pauli_word: {pw}")


def _parse_2q_gate_to_axis_code(pw2: str) -> int:
    """Map homogeneous XX/YY/ZZ rotation words to an axis code; reject other pairs."""
    if pw2 == "XX":
        return CODE_X
    if pw2 == "YY":
        return CODE_Y
    if pw2 == "ZZ":
        return CODE_Z

    raise ValueError(f"Unsupported 2-qubit pauli_word: {pw2}")


def _as_torch_scalar(x, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Keep torch tensors attached to the graph.

    If x is already a torch.Tensor, return it converted only if necessary.
    If x is a Python/PennyLane scalar, make a constant torch scalar.
    """
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)

    return torch.tensor(x, dtype=dtype, device=device)


def _code_to_xz_phase(code: int) -> Tuple[int, int, complex]:
    """
    Decompose a single-qubit Pauli into X^x Z^z plus a phase.

    Convention:
        Y = i X Z

    because:
        X Z = -i Y  =>  Y = i X Z
    """
    if code == CODE_I:
        return 0, 0, 1
    if code == CODE_X:
        return 1, 0, 1
    if code == CODE_Z:
        return 0, 1, 1
    if code == CODE_Y:
        return 1, 1, 1j

    raise ValueError(f"Unknown Pauli code: {code}")


def _mul_pair(
    current_a: int,
    current_b: int,
    current_phase: complex,
    next_a: int,
    next_b: int,
    next_phase: complex = 1,
) -> Tuple[int, int, complex]:
    """
    Multiply a two-qubit Pauli pair by another pair:
        (current_a ⊗ current_b) * (next_a ⊗ next_b)

    keeping track of local Pauli multiplication phases.
    """
    ra, pha = _MUL1[current_a][next_a]
    rb, phb = _MUL1[current_b][next_b]

    return ra, rb, current_phase * next_phase * pha * phb


def _build_cnot_pair_table() -> Dict[Tuple[int, int], Tuple[int, int, complex]]:
    """
    Exact local two-qubit CNOT conjugation table, including phase.

    CNOT control=a, target=b.

    Generator images:
        X_a -> X_a X_b
        Z_a -> Z_a
        X_b -> X_b
        Z_b -> Z_a Z_b
    """
    table = {}

    for code_a in range(4):
        for code_b in range(4):
            xa, za, pha_in_a = _code_to_xz_phase(code_a)
            xb, zb, pha_in_b = _code_to_xz_phase(code_b)

            out_a = CODE_I
            out_b = CODE_I
            phase = pha_in_a * pha_in_b

            # X_a contribution: X_a X_b
            if xa:
                out_a, out_b, phase = _mul_pair(
                    out_a, out_b, phase, CODE_X, CODE_X
                )

            # Z_a contribution: Z_a
            if za:
                out_a, out_b, phase = _mul_pair(
                    out_a, out_b, phase, CODE_Z, CODE_I
                )

            # X_b contribution: X_b
            if xb:
                out_a, out_b, phase = _mul_pair(
                    out_a, out_b, phase, CODE_I, CODE_X
                )

            # Z_b contribution: Z_a Z_b
            if zb:
                out_a, out_b, phase = _mul_pair(
                    out_a, out_b, phase, CODE_Z, CODE_Z
                )

            table[(code_a, code_b)] = (out_a, out_b, phase)

    return table


def _build_cz_pair_table() -> Dict[Tuple[int, int], Tuple[int, int, complex]]:
    """
    Exact local two-qubit CZ conjugation table, including phase.

    CZ on wires a,b.

    Generator images:
        X_a -> X_a Z_b
        Z_a -> Z_a
        X_b -> Z_a X_b
        Z_b -> Z_b
    """
    table = {}

    for code_a in range(4):
        for code_b in range(4):
            xa, za, pha_in_a = _code_to_xz_phase(code_a)
            xb, zb, pha_in_b = _code_to_xz_phase(code_b)

            out_a = CODE_I
            out_b = CODE_I
            phase = pha_in_a * pha_in_b

            # X_a contribution: X_a Z_b
            if xa:
                out_a, out_b, phase = _mul_pair(
                    out_a, out_b, phase, CODE_X, CODE_Z
                )

            # Z_a contribution: Z_a
            if za:
                out_a, out_b, phase = _mul_pair(
                    out_a, out_b, phase, CODE_Z, CODE_I
                )

            # X_b contribution: Z_a X_b
            if xb:
                out_a, out_b, phase = _mul_pair(
                    out_a, out_b, phase, CODE_Z, CODE_X
                )

            # Z_b contribution: Z_b
            if zb:
                out_a, out_b, phase = _mul_pair(
                    out_a, out_b, phase, CODE_I, CODE_Z
                )

            table[(code_a, code_b)] = (out_a, out_b, phase)

    return table


_CNOT_PAIR_TABLE = _build_cnot_pair_table()
_CZ_PAIR_TABLE = _build_cz_pair_table()


class DifferentiablePauliProp:
    """
    Differentiable Pauli propagation using Torch scalar coefficients.

    This supports PyTorch backpropagation through the surviving Pauli paths.

    Parameters
    ----------
    topM:
        If not None, keep only the topM largest-magnitude coefficients after pruning.

    tau:
        Drop terms whose detached coefficient magnitude is below tau.

    prune_every:
        Apply tau/topM pruning every `prune_every` gates.
        If 0 or None, only final pruning is applied when prune=True.

    Notes
    -----
    - k truncation is structural and non-parametric.
    - tau/topM pruning uses detached magnitudes for the mask.
    - Gradients flow through kept coefficients only.
    """

    def __init__(
        self,
        *,
        topM: Optional[int] = 20000,
        tau: float = 1e-10,
        prune_every: int = 5,
    ):
        """Configure coefficient pruning; execution supplies the circuit width.

        At pruning points, tau removes magnitudes below the threshold and topM
        keeps at most that many largest terms (None disables this limit).
        prune_every sets the gate interval; zero means final pruning only when
        pruning is enabled. Pauli-weight cutoff k is a separate execute argument.
        No circuit or trainable parameter is stored at initialization."""
        self.topM = int(topM) if topM is not None else None
        self.tau = float(tau) if tau is not None else 0.0
        self.prune_every = int(prune_every) if prune_every is not None else 0

        self._n: Optional[int] = None
        self._xmask: Optional[int] = None
        self._dtype: Optional[torch.dtype] = None
        self._device: Optional[torch.device] = None

    # --------------------------------------------------------
    # Basic torch scalar helpers
    # --------------------------------------------------------
    def _zero(self) -> torch.Tensor:
        """Allocate a scalar zero with the current execution dtype/device."""
        assert self._dtype is not None and self._device is not None
        return torch.zeros((), dtype=self._dtype, device=self._device)

    def _one(self) -> torch.Tensor:
        """Allocate a scalar one with the current execution dtype/device."""
        assert self._dtype is not None and self._device is not None
        return torch.ones((), dtype=self._dtype, device=self._device)

    @staticmethod
    def _infer_torch_context(tape: qml.tape.QuantumTape) -> Tuple[torch.dtype, torch.device]:
        """
        Infer dtype/device from the first torch tensor found in the tape.
        """
        for op in tape.operations:
            for data in op.data:
                if isinstance(data, torch.Tensor):
                    return data.dtype, data.device

        return torch.float32, torch.device("cpu")

    # --------------------------------------------------------
    # Differentiable-through-kept-terms pruning
    # --------------------------------------------------------
    def _prune(self, H: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Select coefficients by detached magnitude and preserve surviving graphs.

        No derivative is assigned to threshold/top-M selection itself; this is
        not a straight-through estimator for the discrete mask. Dropped paths
        contribute neither output nor gradient. Return a new packed-term dictionary
        unless no pruning is needed."""
        if not H:
            return H

        # tau pruning
        if self.tau is not None and self.tau > 0.0:
            tau = self.tau

            H = {
                key: coeff
                for key, coeff in H.items()
                if torch.abs(coeff.detach()) >= tau
            }

            if not H:
                return H

        # topM pruning
        if self.topM is not None and len(H) > self.topM:
            items = list(H.items())

            scores = torch.stack(
                [torch.abs(coeff.detach()) for _, coeff in items]
            )

            keep_idx = torch.topk(
                scores,
                k=self.topM,
                largest=True,
            ).indices.tolist()

            keep = set(keep_idx)

            H = {
                key: coeff
                for i, (key, coeff) in enumerate(items)
                if i in keep
            }

        return H

    # --------------------------------------------------------
    # Measurement init + final expectation
    # --------------------------------------------------------
    def sentence_to_packed(
        self,
        pauli_sentence: PauliSentence,
        n: int,
    ) -> Dict[int, torch.Tensor]:
        """
        Convert PennyLane PauliSentence to packed-key dictionary.

        Measurement coefficients are usually real constants.
        """
        H: Dict[int, torch.Tensor] = {}

        for pw, coeff in pauli_sentence.items():
            x_mask = 0
            z_mask = 0

            for wire, letter in pw.items():
                w = int(wire)

                if letter == "I":
                    continue
                elif letter == "X":
                    x_mask |= (1 << w)
                elif letter == "Z":
                    z_mask |= (1 << w)
                elif letter == "Y":
                    x_mask |= (1 << w)
                    z_mask |= (1 << w)
                else:
                    raise ValueError(f"Unknown Pauli letter: {letter}")

            key = _pack(x_mask, z_mask, n)

            coeff_complex = complex(coeff)

            if abs(coeff_complex.imag) > 1e-12:
                raise ValueError(
                    f"Complex measurement coefficient {coeff_complex} is not supported "
                    "by this real-valued differentiable engine."
                )

            c = torch.tensor(
                float(coeff_complex.real),
                dtype=self._dtype,
                device=self._device,
            )

            H[key] = H.get(key, self._zero()) + c

        return H

    def expval_zero_state(self, H: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        <0|P|0> is nonzero iff the Pauli word has no X/Y factors.

        In bitmask language:
            x_mask == 0
        """
        assert self._xmask is not None

        xmask = self._xmask
        total = self._zero()

        for key, coeff in H.items():
            if (key & xmask) == 0:
                total = total + coeff

        return total

    # --------------------------------------------------------
    # Clifford gates
    # --------------------------------------------------------
    def apply_cnot(
        self,
        control: int,
        target: int,
        H: Dict[int, torch.Tensor],
        *,
        k: Optional[int],
    ) -> Dict[int, torch.Tensor]:
        """
        Exact CNOT conjugation using a local pair table, including signs.
        """
        assert self._n is not None and self._xmask is not None

        n = self._n
        xmask = self._xmask

        out: Dict[int, torch.Tensor] = {}

        for key, coeff in H.items():
            x = key & xmask
            z = key >> n

            code_c = _get_code(x, z, control)
            code_t = _get_code(x, z, target)

            new_c, new_t, phase = _CNOT_PAIR_TABLE[(code_c, code_t)]

            nx, nz = x, z
            nx, nz = _set_code(nx, nz, control, new_c)
            nx, nz = _set_code(nx, nz, target, new_t)

            if k is not None and _weight_from_masks(nx, nz) > k:
                continue

            new_key = _pack(nx, nz, n)

            phase_real = complex(phase).real
            phase_imag = complex(phase).imag

            if abs(phase_imag) > 1e-12:
                raise RuntimeError(f"Unexpected complex phase in CNOT table: {phase}")

            out[new_key] = out.get(new_key, self._zero()) + phase_real * coeff

        return out

    def apply_cz(
        self,
        a: int,
        b: int,
        H: Dict[int, torch.Tensor],
        *,
        k: Optional[int],
    ) -> Dict[int, torch.Tensor]:
        """
        Exact CZ conjugation using a local pair table, including signs.
        """
        assert self._n is not None and self._xmask is not None

        n = self._n
        xmask = self._xmask

        out: Dict[int, torch.Tensor] = {}

        for key, coeff in H.items():
            x = key & xmask
            z = key >> n

            code_a = _get_code(x, z, a)
            code_b = _get_code(x, z, b)

            new_a, new_b, phase = _CZ_PAIR_TABLE[(code_a, code_b)]

            nx, nz = x, z
            nx, nz = _set_code(nx, nz, a, new_a)
            nx, nz = _set_code(nx, nz, b, new_b)

            if k is not None and _weight_from_masks(nx, nz) > k:
                continue

            new_key = _pack(nx, nz, n)

            phase_real = complex(phase).real
            phase_imag = complex(phase).imag

            if abs(phase_imag) > 1e-12:
                raise RuntimeError(f"Unexpected complex phase in CZ table: {phase}")

            out[new_key] = out.get(new_key, self._zero()) + phase_real * coeff

        return out

    def apply_hadamard(
        self,
        wire: int,
        H: Dict[int, torch.Tensor],
        *,
        k: Optional[int],
    ) -> Dict[int, torch.Tensor]:
        """
        Hadamard conjugation:

            H X H = Z
            H Z H = X
            H Y H = -Y
            H I H = I
        """
        assert self._n is not None and self._xmask is not None

        n = self._n
        xmask = self._xmask

        out: Dict[int, torch.Tensor] = {}
        bit = 1 << wire

        for key, coeff in H.items():
            x = key & xmask
            z = key >> n

            xb = 1 if (x & bit) else 0
            zb = 1 if (z & bit) else 0

            # Y -> -Y
            if xb and zb:
                coeff = -coeff

            # swap X/Z bits on this wire
            if xb != zb:
                x ^= bit
                z ^= bit

            if k is not None and _weight_from_masks(x, z) > k:
                continue

            new_key = _pack(x, z, n)
            out[new_key] = out.get(new_key, self._zero()) + coeff

        return out

    # --------------------------------------------------------
    # Pauli rotations
    # --------------------------------------------------------
    def apply_1q_rot(
        self,
        axis_code: int,
        wire: int,
        theta,
        H: Dict[int, torch.Tensor],
        *,
        k: Optional[int],
    ) -> Dict[int, torch.Tensor]:
        """
        U = exp(-i theta/2 * P).

        Heisenberg update:

        If [P,Q]=0:
            Q -> Q

        If {P,Q}=0:
            Q -> cos(theta) Q + sin(theta) (i P Q)
        """
        assert self._n is not None and self._xmask is not None

        n = self._n
        xmask = self._xmask

        theta = _as_torch_scalar(theta, dtype=self._dtype, device=self._device)

        c = torch.cos(theta)
        s = torch.sin(theta)

        out: Dict[int, torch.Tensor] = {}

        for key, coeff in H.items():
            x = key & xmask
            z = key >> n

            local = _get_code(x, z, wire)

            # commuting case
            if local == CODE_I or local == axis_code:
                out[key] = out.get(key, self._zero()) + coeff
                continue

            # cos branch
            out[key] = out.get(key, self._zero()) + c * coeff

            # sin branch
            nx, nz = x, z

            if axis_code == CODE_X:
                # Z -> +Y
                # Y -> -Z
                if local == CODE_Z:
                    new_code, sign = CODE_Y, +1.0
                else:
                    new_code, sign = CODE_Z, -1.0

            elif axis_code == CODE_Y:
                # X -> +Z
                # Z -> -X
                if local == CODE_X:
                    new_code, sign = CODE_Z, +1.0
                else:
                    new_code, sign = CODE_X, -1.0

            elif axis_code == CODE_Z:
                # X -> -Y
                # Y -> +X
                if local == CODE_X:
                    new_code, sign = CODE_Y, -1.0
                else:
                    new_code, sign = CODE_X, +1.0

            else:
                raise ValueError(f"Unknown axis_code: {axis_code}")

            nx, nz = _set_code(nx, nz, wire, new_code)

            if k is not None and _weight_from_masks(nx, nz) > k:
                continue

            new_key = _pack(nx, nz, n)
            out[new_key] = out.get(new_key, self._zero()) + (sign * s) * coeff

        return out

    def apply_2q_rot(
        self,
        axis_code: int,
        a: int,
        b: int,
        theta,
        H: Dict[int, torch.Tensor],
        *,
        k: Optional[int],
    ) -> Dict[int, torch.Tensor]:
        """
        U = exp(-i theta/2 * A_a A_b).

        If the Pauli word anticommutes with the generator:
            Q -> cos(theta) Q + sin(theta) (i P Q)

        Otherwise:
            Q -> Q
        """
        assert self._n is not None and self._xmask is not None

        n = self._n
        xmask = self._xmask

        theta = _as_torch_scalar(theta, dtype=self._dtype, device=self._device)

        c = torch.cos(theta)
        s = torch.sin(theta)

        out: Dict[int, torch.Tensor] = {}

        for key, coeff in H.items():
            x = key & xmask
            z = key >> n

            qa = _get_code(x, z, a)
            qb = _get_code(x, z, b)

            anti_a = _anticommutes_1q(axis_code, qa)
            anti_b = _anticommutes_1q(axis_code, qb)

            # commutes if parity of local anticommutes is even
            if not (anti_a ^ anti_b):
                out[key] = out.get(key, self._zero()) + coeff
                continue

            # cos branch
            out[key] = out.get(key, self._zero()) + c * coeff

            # sin branch
            ra, pha = _MUL1[axis_code][qa]
            rb, phb = _MUL1[axis_code][qb]

            total_phase = pha * phb
            sin_sign = 1j * total_phase

            if abs(sin_sign.imag) > 1e-9:
                raise RuntimeError(
                    f"Expected real sin_sign, got {sin_sign} "
                    f"(total_phase={total_phase})"
                )

            sin_sign = float(sin_sign.real)

            nx, nz = x, z
            nx, nz = _set_code(nx, nz, a, ra)
            nx, nz = _set_code(nx, nz, b, rb)

            if k is not None and _weight_from_masks(nx, nz) > k:
                continue

            new_key = _pack(nx, nz, n)
            out[new_key] = out.get(new_key, self._zero()) + (sin_sign * s) * coeff

        return out

    # --------------------------------------------------------
    # Tape compilation
    # --------------------------------------------------------
    def _compile_ops(self, tape: qml.tape.QuantumTape) -> Tuple[int, List[Tuple]]:
        """
        Convert tape.operations into compact reversed op tuples.

        Tags:
            0: CNOT       -> (0, control, target)
            1: 1q rot     -> (1, axis_code, wire, theta)
            2: 2q rot     -> (2, axis_code, wire_a, wire_b, theta)
            3: CZ         -> (3, wire_a, wire_b)
            4: Hadamard   -> (4, wire)
        """
        max_wire = -1

        for op in tape.operations:
            for w in op.wires:
                max_wire = max(max_wire, int(w))

        for meas in tape.measurements:
            if hasattr(meas, "wires"):
                for w in meas.wires:
                    max_wire = max(max_wire, int(w))

        if max_wire < 0:
            raise RuntimeError("Could not infer number of wires from tape.")

        n = max_wire + 1

        compiled: List[Tuple] = []

        for op in reversed(tape.operations):
            if isinstance(op, qml.CNOT):
                compiled.append((0, int(op.wires[0]), int(op.wires[1])))
                continue

            if isinstance(op, qml.CZ):
                compiled.append((3, int(op.wires[0]), int(op.wires[1])))
                continue

            if isinstance(op, qml.Hadamard):
                compiled.append((4, int(op.wires[0])))
                continue

            if isinstance(op, (qml.RX, qml.RY, qml.RZ)):
                axis_code = _parse_pauli_word_to_axis_code(op.name[-1])
                compiled.append((1, axis_code, int(op.wires[0]), op.data[0]))
                continue

            if isinstance(op, qml.PauliRot):
                theta = op.data[0]
                pw = op.hyperparameters.get("pauli_word", None)
                wires = [int(w) for w in op.wires]

                if pw is None:
                    raise RuntimeError("PauliRot missing pauli_word hyperparameter.")

                if len(pw) == 1:
                    axis_code = _parse_pauli_word_to_axis_code(pw)

                    if len(wires) != 1:
                        raise NotImplementedError(
                            f"Unsupported PauliRot {pw} on wires {wires}"
                        )

                    compiled.append((1, axis_code, wires[0], theta))
                    continue

                if len(pw) == 2:
                    if pw not in ("XX", "YY", "ZZ") or len(wires) != 2:
                        raise NotImplementedError(
                            f"Unsupported PauliRot {pw} on wires {wires}"
                        )

                    axis_code = _parse_2q_gate_to_axis_code(pw)
                    compiled.append((2, axis_code, wires[0], wires[1], theta))
                    continue

                raise NotImplementedError(
                    f"PauliRot with pauli_word length {len(pw)} is not supported: {pw}"
                )

            raise NotImplementedError(f"Gate not supported in differentiable PP: {op}")

        return n, compiled

    # --------------------------------------------------------
    # Public execution method
    # --------------------------------------------------------
    def execute_from_tape(
        self,
        tape: qml.tape.QuantumTape,
        *,
        k: Optional[int] = None,
        prune: bool = True,
    ) -> torch.Tensor:
        """
        Differentiable Pauli propagation evaluation.

        Parameters
        ----------
        tape:
            PennyLane QuantumTape with supported gates and Pauli expval measurements.

        k:
            Maximum Pauli weight. If None, no weight truncation is applied.

        prune:
            If True, apply tau/topM pruning using detached masks.

        Returns
        -------
        torch.Tensor
            Shape: (n_measurements,)
        """
        # QASP circuits intentionally retain high-level PennyLane templates
        # such as AngleEmbedding on their traced tape. Expand those templates
        # to the native gate set understood by the propagation rules while
        # preserving Torch tensors and their autograd graph.
        batch, _ = qml.transforms.decompose(
            tape,
            gate_set={
                qml.CNOT,
                qml.CZ,
                qml.Hadamard,
                qml.RX,
                qml.RY,
                qml.RZ,
                qml.PauliRot,
            },
        )
        tape = batch[0]

        dtype, device = self._infer_torch_context(tape)

        self._dtype = dtype
        self._device = device

        n, ops = self._compile_ops(tape)

        self._n = n
        self._xmask = (1 << n) - 1

        expvals: List[torch.Tensor] = []

        for meas in tape.measurements:
            if not hasattr(meas.obs, "pauli_rep"):
                raise RuntimeError(
                    "Differentiable Pauli propagation only supports expectation "
                    "measurements with Pauli observables, i.e. meas.obs.pauli_rep must exist."
                )

            H = self.sentence_to_packed(meas.obs.pauli_rep, n)

            gate_count = 0

            for item in ops:
                gate_count += 1
                tag = item[0]

                if tag == 0:
                    _, c, t = item
                    H = self.apply_cnot(c, t, H, k=k)

                elif tag == 3:
                    _, a, b = item
                    H = self.apply_cz(a, b, H, k=k)

                elif tag == 4:
                    _, w = item
                    H = self.apply_hadamard(w, H, k=k)

                elif tag == 1:
                    _, axis_code, w, theta = item
                    H = self.apply_1q_rot(axis_code, w, theta, H, k=k)

                elif tag == 2:
                    _, axis_code, a, b, theta = item
                    H = self.apply_2q_rot(axis_code, a, b, theta, H, k=k)

                else:
                    raise RuntimeError(f"Unknown compiled op tag: {tag}")

                if prune and self.prune_every and (gate_count % self.prune_every == 0):
                    H = self._prune(H)

                    if not H:
                        break

            if prune:
                H = self._prune(H)

            expvals.append(self.expval_zero_state(H))

        return torch.stack(expvals)
