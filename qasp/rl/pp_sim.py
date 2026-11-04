# torchlane/rl/pauli_propagator.py
"""Packed-Pauli Heisenberg propagation with Python complex coefficients.

FastTruncatedPauliProp is a non-autograd expectation engine for the all-zero
input state. Structural Pauli-weight truncation and magnitude/term-count
pruning approximate propagation. Use DifferentiablePauliProp for Torch training;
this implementation converts coefficients to Python complex numbers."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, List, Any

import torch
import pennylane as qml
from pennylane.pauli import PauliSentence

# ============================================================
# Fast Pauli propagation (Rudolph-style engineering choices)
# ============================================================
# - Use *packed* integer key for Pauli words: key = x_mask | (z_mask << n)
# - Coefficients are plain python complex (fast)
# - Merge-as-you-go (dict accumulate), then truncate *after merging*
# - Avoid CNOT/CZ/H lookup tables where possible: use symplectic/local updates
# - Pre-parse tape ops once into a compact list of op-tuples (no isinstance in hot loop)
#
# Supported gates (as they appear on tape):
#   - qml.CNOT
#   - qml.CZ
#   - qml.Hadamard
#   - qml.RX, qml.RY, qml.RZ
#   - qml.PauliRot with pauli_word in {"X","Y","Z","XX","YY","ZZ"}
#
# Measurements:
#   - expval with Pauli observables (meas.obs.pauli_rep exists)
#
# Truncation knobs (evaluation-time speed control):
#   - k: max Pauli weight (via execute_from_tape(k=...))
#   - tau: drop terms with |coeff| < tau
#   - topM: keep only topM largest-magnitude terms
#   - prune_every: apply (tau/topM) pruning every N gates
#
# NOTE: This version focuses on FAST EVALUATION (not differentiable).
# ============================================================

# Local Pauli code from (xbit, zbit):
# code = xbit | (zbit<<1)
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
    Single-qubit Pauli multiplication in our CODE ordering (I,X,Z,Y):
      returns (result_code, phase) where phase in {1, -1, 1j, -1j}.
    """
    mul = [[(0, 1)] * 4 for _ in range(4)]

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
            # X*Y = iZ, Y*X = -iZ
            # Y*Z = iX, Z*Y = -iX
            # Z*X = iY, X*Z = -iY
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
                raise RuntimeError(f"Missing Pauli mul case for codes {(a, b)}")

    return mul


_MUL1 = _mul1_build()


def _anticommutes_1q(axis_code: int, local_code: int) -> bool:
    # I commutes with everything; equal Paulis commute; others anticommute.
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

    raise ValueError(f"Unsupported 1q pauli_word {pw}")


def _parse_2q_gate_to_axis_code(pw2: str) -> int:
    # pw2 in {"XX","YY","ZZ"}
    """Map homogeneous XX/YY/ZZ rotation words to an axis code; reject other pairs."""
    if pw2 == "XX":
        return CODE_X
    if pw2 == "YY":
        return CODE_Y
    if pw2 == "ZZ":
        return CODE_Z

    raise ValueError(f"Unsupported 2q pauli_word {pw2}")


class FastTruncatedPauliProp:
    """
    Fast truncated Pauli propagation for expectation values on |0...0>.

    This is intentionally non-differentiable: it uses python floats/complex,
    aggressive truncation, and packed integer keys for speed.
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

        # set at execute time
        self._n: Optional[int] = None
        self._xmask: Optional[int] = None

    # ----------------------------
    # Truncation (after merging)
    # ----------------------------
    def _prune(self, H: Dict[int, complex]) -> Dict[int, complex]:
        """Return surviving coefficient terms after magnitude thresholding and optional top-M selection."""
        if not H:
            return H

        if self.tau and self.tau > 0.0:
            tau = self.tau
            H = {key: coeff for key, coeff in H.items() if abs(coeff) >= tau}

            if not H:
                return H

        if self.topM is not None and len(H) > self.topM:
            topM = self.topM
            items = sorted(H.items(), key=lambda kv: abs(kv[1]), reverse=True)[:topM]
            H = dict(items)

        return H

    # ----------------------------
    # Measurement init + expval
    # ----------------------------
    @staticmethod
    def sentence_to_packed(pauli_sentence: PauliSentence, n: int) -> Dict[int, complex]:
        """Convert a PennyLane PauliSentence on integer wires to packed keys and Python complex coefficients."""
        H: Dict[int, complex] = {}

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
                    raise ValueError(f"Unknown Pauli letter {letter}")

            key = _pack(x_mask, z_mask, n)
            H[key] = H.get(key, 0.0 + 0.0j) + complex(coeff)

        return H

    def expval_zero_state(self, H: Dict[int, complex]) -> float:
        # <0|P|0> nonzero iff there are no X/Y factors => x_mask == 0
        """Sum real coefficients of I/Z-only terms using the current execution width."""
        assert self._xmask is not None

        xmask = self._xmask
        total = 0.0

        for key, coeff in H.items():
            if (key & xmask) == 0:
                total += float(coeff.real)

        return total

    # ----------------------------
    # Clifford gates
    # ----------------------------
    def apply_cnot(
        self,
        control: int,
        target: int,
        H: Dict[int, complex],
        *,
        k: Optional[int],
    ) -> Dict[int, complex]:
        """
        Clifford: conjugation by CNOT has no branching.

        Symplectic update:
            x_t ^= x_c
            z_c ^= z_t

        This corresponds to:
            X_c -> X_c X_t
            Z_t -> Z_c Z_t
            Z_c -> Z_c
            X_t -> X_t
        """
        assert self._n is not None and self._xmask is not None

        n = self._n
        xmask = self._xmask

        out: Dict[int, complex] = {}

        bitc = 1 << control
        bitt = 1 << target

        for key, coeff in H.items():
            x = key & xmask
            z = key >> n

            xc = 1 if (x & bitc) else 0
            xt = 1 if (x & bitt) else 0
            zc = 1 if (z & bitc) else 0
            zt = 1 if (z & bitt) else 0

            xt_new = xt ^ xc
            zc_new = zc ^ zt

            if xt_new != xt:
                x ^= bitt
            if zc_new != zc:
                z ^= bitc

            if k is not None and _weight_from_masks(x, z) > k:
                continue

            new_key = _pack(x, z, n)
            out[new_key] = out.get(new_key, 0.0 + 0.0j) + coeff

        return out

    def apply_cz(
        self,
        a: int,
        b: int,
        H: Dict[int, complex],
        *,
        k: Optional[int],
    ) -> Dict[int, complex]:
        """
        Clifford: conjugation by CZ has no branching.

        Heisenberg update:
            X_a -> X_a Z_b
            X_b -> Z_a X_b
            Z_a -> Z_a
            Z_b -> Z_b

        Symplectic update:
            z_b ^= x_a
            z_a ^= x_b
        """
        assert self._n is not None and self._xmask is not None

        n = self._n
        xmask = self._xmask

        out: Dict[int, complex] = {}

        bita = 1 << a
        bitb = 1 << b

        for key, coeff in H.items():
            x = key & xmask
            z = key >> n

            xa = 1 if (x & bita) else 0
            xb = 1 if (x & bitb) else 0

            if xa:
                z ^= bitb
            if xb:
                z ^= bita

            if k is not None and _weight_from_masks(x, z) > k:
                continue

            new_key = _pack(x, z, n)
            out[new_key] = out.get(new_key, 0.0 + 0.0j) + coeff

        return out

    def apply_hadamard(
        self,
        wire: int,
        H: Dict[int, complex],
        *,
        k: Optional[int],
    ) -> Dict[int, complex]:
        """
        Clifford: conjugation by Hadamard has no branching.

        Heisenberg update:
            H X H = Z
            H Z H = X
            H Y H = -Y
            H I H = I

        In bit form, on the selected wire:
            swap x_wire and z_wire
            multiply coefficient by -1 if local Pauli is Y.
        """
        assert self._n is not None and self._xmask is not None

        n = self._n
        xmask = self._xmask

        out: Dict[int, complex] = {}
        bit = 1 << wire

        for key, coeff in H.items():
            x = key & xmask
            z = key >> n

            xb = 1 if (x & bit) else 0
            zb = 1 if (z & bit) else 0

            # Y has x=z=1 and gets a minus sign under H.
            if xb and zb:
                coeff = -coeff

            # swap x and z bits on this wire
            if xb != zb:
                x ^= bit
                z ^= bit

            if k is not None and _weight_from_masks(x, z) > k:
                continue

            new_key = _pack(x, z, n)
            out[new_key] = out.get(new_key, 0.0 + 0.0j) + coeff

        return out

    # ----------------------------
    # Pauli rotations
    # ----------------------------
    def apply_1q_rot(
        self,
        axis_code: int,
        wire: int,
        theta: float,
        H: Dict[int, complex],
        *,
        k: Optional[int],
    ) -> Dict[int, complex]:
        """
        U = exp(-i theta/2 * P), P in {X,Y,Z} on 'wire'.

        If [P,Q]=0:
            Q -> Q

        If {P,Q}=0:
            Q -> cos(theta) Q + sin(theta) (i P Q)

        The sin branch is another Pauli on the same wire with a ± sign.
        """
        assert self._n is not None and self._xmask is not None

        n = self._n
        xmask = self._xmask

        c = math.cos(theta)
        s = math.sin(theta)

        out: Dict[int, complex] = {}

        for key, coeff in H.items():
            x = key & xmask
            z = key >> n

            local = _get_code(x, z, wire)

            if local == CODE_I or local == axis_code:
                out[key] = out.get(key, 0.0 + 0.0j) + coeff
                continue

            # cos branch: unchanged Pauli word
            out[key] = out.get(key, 0.0 + 0.0j) + (c * coeff)

            # sin branch: update wire's Pauli
            nx, nz = x, z

            if axis_code == CODE_X:
                # Z -> +Y ; Y -> -Z
                if local == CODE_Z:
                    new_code, sign = CODE_Y, +1.0
                else:  # local == Y
                    new_code, sign = CODE_Z, -1.0

            elif axis_code == CODE_Y:
                # X -> +Z ; Z -> -X
                if local == CODE_X:
                    new_code, sign = CODE_Z, +1.0
                else:  # local == Z
                    new_code, sign = CODE_X, -1.0

            elif axis_code == CODE_Z:
                # X -> -Y ; Y -> +X
                if local == CODE_X:
                    new_code, sign = CODE_Y, -1.0
                else:  # local == Y
                    new_code, sign = CODE_X, +1.0

            else:
                raise ValueError(f"Unknown axis_code {axis_code}")

            nx, nz = _set_code(nx, nz, wire, new_code)

            if k is not None and _weight_from_masks(nx, nz) > k:
                continue

            new_key = _pack(nx, nz, n)
            out[new_key] = out.get(new_key, 0.0 + 0.0j) + ((sign * s) * coeff)

        return out

    def apply_2q_rot(
        self,
        axis_code: int,
        a: int,
        b: int,
        theta: float,
        H: Dict[int, complex],
        *,
        k: Optional[int],
    ) -> Dict[int, complex]:
        """
        Two-qubit Pauli rotation with generator P = A_a ⊗ A_b where A in {X,Y,Z}.

        If [P,Q]=0:
            Q -> Q

        If {P,Q}=0:
            Q -> cos(theta) Q + sin(theta) (i P Q)

        We compute new local Paulis via single-qubit multiplication:
            (A * Qa, phase_a), (A * Qb, phase_b)
            total_phase = phase_a * phase_b
            sin_sign = i * total_phase

        In the anti-commuting case, sin_sign should be ±1 real.
        """
        assert self._n is not None and self._xmask is not None

        n = self._n
        xmask = self._xmask

        c = math.cos(theta)
        s = math.sin(theta)

        out: Dict[int, complex] = {}

        for key, coeff in H.items():
            x = key & xmask
            z = key >> n

            qa = _get_code(x, z, a)
            qb = _get_code(x, z, b)

            anti_a = _anticommutes_1q(axis_code, qa)
            anti_b = _anticommutes_1q(axis_code, qb)

            # commutes if parity of anticommutes is even
            if not (anti_a ^ anti_b):
                out[key] = out.get(key, 0.0 + 0.0j) + coeff
                continue

            # cos branch: unchanged
            out[key] = out.get(key, 0.0 + 0.0j) + (c * coeff)

            # sin branch
            ra, pha = _MUL1[axis_code][qa]
            rb, phb = _MUL1[axis_code][qb]
            total_phase = pha * phb

            # In XOR case, total_phase should be ± i; i * total_phase becomes ± 1 real.
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
            out[new_key] = out.get(new_key, 0.0 + 0.0j) + ((sin_sign * s) * coeff)

        return out

    # ----------------------------
    # Tape execution
    # ----------------------------
    def _compile_ops(self, tape: qml.tape.QuantumTape) -> Tuple[int, List[Tuple]]:
        """
        Convert tape.operations into a compact list of tuples in reverse order.

        Each tuple begins with an int tag:
          0: CNOT     -> (0, c, t)
          1: 1q rot   -> (1, axis_code, w, theta_float)
          2: 2q rot   -> (2, axis_code, a, b, theta_float)
          3: CZ       -> (3, a, b)
          4: H        -> (4, w)
        """
        # infer n_qubits from operation and measurement wires
        max_wire = -1

        for op in tape.operations:
            for w in op.wires:
                max_wire = max(max_wire, int(w))

        for meas in tape.measurements:
            if hasattr(meas, "wires"):
                for w in meas.wires:
                    max_wire = max(max_wire, int(w))

        n = max_wire + 1

        compiled: List[Tuple] = []

        for op in reversed(tape.operations):
            if isinstance(op, qml.CNOT):
                c = int(op.wires[0])
                t = int(op.wires[1])
                compiled.append((0, c, t))
                continue

            if isinstance(op, qml.CZ):
                a = int(op.wires[0])
                b = int(op.wires[1])
                compiled.append((3, a, b))
                continue

            if isinstance(op, qml.Hadamard):
                w = int(op.wires[0])
                compiled.append((4, w))
                continue

            if isinstance(op, (qml.RX, qml.RY, qml.RZ)):
                axis_letter = op.name[-1]  # "X","Y","Z"
                axis_code = _parse_pauli_word_to_axis_code(axis_letter)
                w = int(op.wires[0])
                theta = float(op.data[0])
                compiled.append((1, axis_code, w, theta))
                continue

            if isinstance(op, qml.PauliRot):
                theta = float(op.data[0])
                pw = op.hyperparameters.get("pauli_word", None)
                wires = [int(w) for w in op.wires]

                if pw is None:
                    raise RuntimeError("PauliRot missing pauli_word hyperparameter")

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
                    f"PauliRot with pauli_word length {len(pw)} not supported: {pw}"
                )

            raise NotImplementedError(f"Gate not supported in PP: {op}")

        return n, compiled

    def execute_from_tape(
        self,
        tape: qml.tape.QuantumTape,
        *,
        k: Optional[int] = None,
        prune: bool = True,
    ) -> torch.Tensor:
        """
        Fast PP evaluation of expvals on a tape.

        Args:
          tape:
            PennyLane tape with supported operations and expval measurements.

          k:
            Max Pauli weight. This is the Rudolph-style truncation by Pauli weight.
            If None, no weight truncation is applied.

          prune:
            If True, apply coefficient/topM pruning. If False, no coefficient/topM
            pruning is applied, but k-truncation still applies.

        Returns:
          torch.Tensor float32 with shape (n_measurements,)
        """
        n, ops = self._compile_ops(tape)

        self._n = n
        self._xmask = (1 << n) - 1

        expvals: List[float] = []

        for meas in tape.measurements:
            if not hasattr(meas.obs, "pauli_rep"):
                raise RuntimeError(
                    "Pauli propagation only supports expectation measurements with Pauli observables "
                    "(meas.obs.pauli_rep must exist)."
                )
            print(f"[DEBUG] meas.obs.pauli_rep = {meas.obs.pauli_rep}", flush=True)
            H = self.sentence_to_packed(meas.obs.pauli_rep, n)
            print(f"[DEBUG] initial H = {H}", flush=True)
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
                    raise RuntimeError(f"Unknown op tag {tag}")

                if prune and self.prune_every and (gate_count % self.prune_every == 0):
                    H = self._prune(H)

                    if not H:
                        break

            if prune:
                H = self._prune(H)

            expvals.append(self.expval_zero_state(H))

        return torch.tensor(expvals, dtype=torch.float32)


