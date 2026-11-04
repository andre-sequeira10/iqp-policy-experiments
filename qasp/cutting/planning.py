"""Tape partitioning, explicit experiment plans, and linear reconstruction."""
from dataclasses import dataclass, field
from itertools import product
import math
import numpy as np
import pennylane as qml
from .specs import as_config
from .decompositions import recipe, weights, gate_terms, instrument, RecordWire, supported


@dataclass(frozen=True)
class Segment:
    """Internal wire segment after a time-like cut."""
    wire: object
    number: int


@dataclass
class CutPlan:
    """Executable fragment tapes and differentiable contraction instructions.

    Plans bind parameters to the current autodiff graph. Rebuild on each model
    call; a planner may cache discrete selections, but not tensors from old calls.
    ``terms`` entries are (moment index, coefficient, fragment result references).
    ``readouts`` linearly combine moments, or form a variance from two moments.
    """
    tapes: tuple
    terms: tuple
    readouts: tuple
    n_moments: int
    groups: tuple
    coefficients: tuple
    settings: int
    config: object
    source_operations: tuple
    zero: object = 0.0
    metadata: dict = field(default_factory=dict)

    def resources(self):
        """Return bound-parameter costs; this diagnostic detaches scalar numbers."""
        gamma = math.prod(sum(abs(float(qml.math.toarray(c))) for c in row)
                          for row in self.coefficients)
        widths = [len(t.wires) for t in self.tapes]
        return {"cuts": len(self.coefficients), "settings": self.settings,
                "gamma": gamma, "sampling_overhead": gamma**2,
                "fragment_tapes": len(self.tapes),
                "fragment_groups": len(self.groups),
                "maximum_fragment_wires": max(widths, default=0),
                "fragment_widths": widths,
                "record_ancillas_included": True,
                "estimator": "enumeration", **self.metadata}

    def reconstruct(self, results):
        """Contract independent local expectations without detaching gradients."""
        if len(results) != len(self.tapes):
            raise ValueError("Result count does not match the cut plan.")
        flat = [((r,) if len(t.measurements) == 1 and qml.math.ndim(r) == 0 else r)
                for t, r in zip(self.tapes, results)]
        moments = [self.zero for _ in range(self.n_moments)]
        for index, coeff, refs in self.terms:
            term = coeff
            for ti, mi in refs:
                term = term * flat[ti][mi]
            moments[index] = moments[index] + term
        out = []
        for kind, rows in self.readouts:
            values = [sum((c * moments[i] for i, c in row), self.zero) for row in rows]
            if kind == "variance":
                out.append(values[1] - values[0]**2)
            elif kind in {"probabilities", "sample", "counts"}:
                # In-place-on-the-list Walsh-Hadamard butterfly (no in-place
                # tensor writes). O(n 2**n) arithmetic and O(2**n) storage.
                stride = 1
                while stride < len(values):
                    for start in range(0, len(values), 2 * stride):
                        for offset in range(stride):
                            a, b = start + offset, start + offset + stride
                            x, y = values[a], values[b]
                            values[a], values[b] = x + y, x - y
                    stride *= 2
                out.append(qml.math.stack(values) / len(values))
            else:
                out.append(values[0])
        return out


def canonicalize(tape):
    """Expand encoding templates; indices in CutConfig refer to this tape.

    Named gate recipes and ordinary operations retain their identity. The
    returned tape is public so learned/manual planners use the exact same index
    convention as execution. State preparation is never assumed factorized.
    """
    ops = []
    def append(op):
        base = op
        while isinstance(base, qml.ops.op_math.Adjoint):
            base = base.base
        if base.name in {"AngleEmbedding", "IQPEmbedding", "BasicEntanglerLayers",
                         "StronglyEntanglingLayers"}:
            for child in op.decomposition():
                append(child)
        elif op.name == "PauliRot" and "I" in op.hyperparameters["pauli_word"]:
            active = [(w, p) for w, p in zip(op.wires, op.hyperparameters["pauli_word"]) if p != "I"]
            ops.append(qml.PauliRot(op.data[0], "".join(p for _, p in active),
                                   wires=[w for w, _ in active])
                       if active else qml.GlobalPhase(op.data[0] / 2))
        else:
            ops.append(op)
    for op in tape.operations:
        append(op)
    return qml.tape.QuantumScript(ops, tape.measurements, shots=tape.shots)


def candidate_cuts(tape):
    """Inspectable candidate actions for external search or learned planners."""
    tape = canonicalize(tape)
    return tuple({"operation_index": i, "name": op.name,
                  "wires": tuple(op.wires)} for i, op in enumerate(tape.operations)
                 if supported(op))


def _paulis(obs):
    rep = obs.pauli_rep
    if rep is None:
        # Explicit small observable fallback, not a simulation of the full state.
        if len(obs.wires) > 10:
            raise ValueError("Non-Pauli observables wider than 10 wires need explicit Pauli decomposition.")
        rep = qml.pauli_decompose(qml.matrix(obs), wire_order=obs.wires, pauli=True)
    return [(dict(pw), coeff) for pw, coeff in rep.items()]


def _wire_terms(old, new, groups, cut):
    # Identity = 1/2 [I|0> + I|1> + X|+> - X|-> + Y|+i> - Y|-i> + Z|0> - Z|1>].
    source = next(i for i, g in enumerate(groups) if old in g)
    target = next(i for i, g in enumerate(groups) if new in g)
    result = []
    coeffs = []
    for axis, sign, value in [("I", 1, 0), ("I", 1, 1), ("X", 1, 0),
                              ("X", -1, 1), ("Y", 1, 0), ("Y", -1, 1),
                              ("Z", 1, 0), ("Z", -1, 1)]:
        ops, records = {}, {}
        if axis != "I":
            record = RecordWire(cut, 0)
            ops[source] = instrument({old: axis}, record)
            records[source] = [record]
        prep = [qml.RX(np.pi, new)] if value else []
        if axis == "X":
            prep += [qml.Hadamard(new)]
        elif axis == "Y":
            prep += [qml.Hadamard(new), qml.RZ(np.pi / 2, new)]
        ops.setdefault(target, []).extend(prep)
        result.append((ops, records))
        coeffs.append(sign / 2)
    return tuple(coeffs), result


@qml.QueuingManager.stop_recording()
def plan_cuts(tape, config):
    """Build a reusable inspectable plan for one scalar-parameter tape.

    A physical measurement record is a fresh local ancilla. All fragments are
    unitary circuits with terminal Pauli expectations, allowing native simulator
    gradients and QASP's differentiable Pauli propagation. Pauli-rotation cuts
    can involve many wires, but must span exactly two selected fragments.
    """
    config = as_config(config)
    tape = canonicalize(tape)
    if any(isinstance(op, (qml.measurements.MidMeasureMP, qml.ops.op_math.Conditional))
           for op in tape.operations):
        raise ValueError("Source mid-circuit measurements and classical conditionals are unsupported.")
    # Fixed rotation angles follow caller precision, including on default.mixed.
    torch_anchor = next((p for p in tape.get_parameters(trainable_only=False)
                         if qml.math.get_interface(p) == "torch"), None)
    if config.planner is not None:
        config = as_config(config.planner(tape, config))
        if config.planner is not None:
            raise ValueError("Planner must return config.selected(...) with planner=None.")
    if tape.batch_size is not None:
        raise ValueError("Expand broadcast tapes before planning (CuttingQNode does this automatically).")
    if any(i >= len(tape.operations) for i in config.gate_indices):
        raise ValueError("gate_indices contains an out-of-range canonical operation index.")
    if any(c.after >= len(tape.operations) or c.wire not in tape.wires for c in config.wire_cuts):
        raise ValueError("Wire cut position or wire is not present in the tape.")
    if len(set(config.wire_cuts)) != len(config.wire_cuts):
        raise ValueError("Duplicate wire cut.")
    partition = {w: j for j, group in enumerate(config.partitions) for w in group}
    if partition and any(w not in partition for w in tape.wires):
        raise ValueError("partitions must cover every source tape wire.")
    selected = set(config.gate_indices)
    current = {w: w for w in tape.wires}
    segments = dict.fromkeys(tape.wires, 0)
    events, cuts = [], []
    all_wires = list(tape.wires)
    # Event is ('op', operation) or ('cut', cut index).
    def split(wire):
        old = current[wire]
        segments[wire] += 1
        new = Segment(wire, segments[wire])
        current[wire] = new
        all_wires.append(new)
        events.append(("cut", len(cuts)))
        cuts.append(("wire", (old, new)))
    for i, original in enumerate(tape.operations):
        if original.name == "WireCut":
            for wire in original.wires:
                split(wire)
        else:
            op = qml.map_wires(original, current)
            active_wires = original.wires
            if original.name == "PauliRot":
                active_wires = [w for w, p in zip(original.wires, original.hyperparameters["pauli_word"]) if p != "I"]
            crossing = partition and len({partition[w] for w in active_wires}) > 1
            if i in selected or crossing:
                rec = recipe(op)
                events.append(("cut", len(cuts)))
                cuts.append(("gate", rec))
            else:
                events.append(("op", op))
        for wc in config.wire_cuts:
            if wc.after == i:
                split(wc.wire)
    # Connected components of the remaining quantum interaction graph.
    parents = {w: w for w in all_wires}
    def root(w):
        while parents[w] != w:
            w = parents[w]
        return w
    def join(a, b):
        parents[root(b)] = root(a)
    if partition:
        # Group initial wires explicitly. Later wire segments remain free to
        # separate in time; their ordinary interactions determine components.
        for group in config.partitions:
            active = [w for w in group if w in parents]
            for w in active[1:]:
                join(active[0], w)
    for kind, op in events:
        if kind == "op" and len(op.wires):
            for w in op.wires[1:]:
                join(op.wires[0], w)
    components = {}
    for w in all_wires:
        components.setdefault(root(w), []).append(w)
    groups = tuple(tuple(g) for g in components.values())
    group_of = {w: i for i, g in enumerate(groups) for w in g}
    options, coeffs = [], []
    for cut, (kind, data) in enumerate(cuts):
        if kind == "gate":
            for wrapper in data.before + data.after:
                if len({group_of[w] for w in wrapper.wires}) != 1:
                    raise ValueError("A gate recipe wrapper is not local to a fragment.")
            options.append(gate_terms(data, groups, cut))
            coeffs.append(weights(data.angle))
        else:
            cs, choices = _wire_terms(*data, groups, cut)
            options.append(choices)
            coeffs.append(cs)
    n_settings = math.prod(map(len, options))
    if n_settings > config.max_settings:
        raise ValueError(f"Cut plan needs {n_settings} settings, exceeding max_settings={config.max_settings}.")
    # Extract global Pauli moments once, then reuse across all readouts.
    moments, moment_indices, readouts = [], {}, []
    def moment(word):
        key = tuple(sorted(word.items(), key=lambda x: all_wires.index(x[0])))
        if key not in moment_indices:
            moment_indices[key] = len(moments)
            moments.append(dict(key))
        return moment_indices[key]
    def observable(obs):
        return [(moment({current[w]: p for w, p in word.items()}), c)
                for word, c in _paulis(obs)]
    for mp in tape.measurements:
        if isinstance(mp, qml.measurements.ExpectationMP):
            readouts.append(("expectation", [observable(mp.obs)]))
        elif isinstance(mp, qml.measurements.VarianceMP):
            readouts.append(("variance", [observable(mp.obs), observable(qml.prod(mp.obs, mp.obs))]))
        elif isinstance(mp, (qml.measurements.ProbabilityMP, qml.measurements.SampleMP,
                             qml.measurements.CountsMP)):
            if mp.obs is not None:
                raise ValueError("Cut probabilities/samples support computational-basis wires only.")
            wires = list(mp.wires or tape.wires)
            if len(wires) > config.max_probability_wires:
                raise ValueError("Probability reconstruction exceeds max_probability_wires; use local expectations.")
            masks = list(product((0, 1), repeat=len(wires)))
            indices = [moment({current[w]: "Z" for w, bit in zip(wires, mask) if bit}) for mask in masks]
            rows = [[(idx, 1.0)] for idx in indices]
            name = "probabilities" if isinstance(mp, qml.measurements.ProbabilityMP) else (
                "sample" if isinstance(mp, qml.measurements.SampleMP) else "counts")
            readouts.append((name, rows))
        else:
            raise ValueError(f"Cutting does not reconstruct {type(mp).__name__}; request expectations or probabilities.")
    # Each variant groups all needed Pauli expectations into one fragment tape.
    tapes, terms, variants = [], [], {}
    for choice in product(*(range(len(o)) for o in options)):
        local_ops = [[] for _ in groups]
        records = [[] for _ in groups]
        signatures = [[] for _ in groups]
        coeff = 1.0
        for j, setting in enumerate(choice):
            coeff = coeff * coeffs[j][setting]
        for kind, value in events:
            if kind == "op":
                if len(value.wires):
                    local_ops[group_of[value.wires[0]]].append(value)
                elif value.name != "GlobalPhase":
                    raise ValueError(f"Unsupported zero-wire operation {value.name}.")
            else:
                op_map, rec_map = options[value][choice[value]]
                for fid, ops in op_map.items():
                    local_ops[fid].extend(ops)
                    signatures[fid].append((value, choice[value]))
                for fid, rs in rec_map.items():
                    records[fid].extend(rs)
        refs_by_moment = [[] for _ in moments]
        for fid, group in enumerate(groups):
            needed = []
            for mi, word in enumerate(moments):
                local = {w: axis for w, axis in word.items() if w in group}
                local.update(dict.fromkeys(records[fid], "Z"))
                if local:
                    needed.append((mi, local))
            if not needed:
                continue  # Trace preservation => exact identity factor.
            key = (fid, tuple(signatures[fid]))
            if key not in variants:
                wires = list(group) + records[fid]
                mapping = dict(zip(wires, range(len(wires))))
                if config.max_fragment_wires is not None and len(wires) > config.max_fragment_wires:
                    raise ValueError(f"Fragment needs {len(wires)} wires including record ancillas, "
                                     f"exceeding max_fragment_wires={config.max_fragment_wires}.")
                measurements, lookup = [], {}
                for mi, local in needed:
                    label = tuple(local.items())
                    if label not in lookup:
                        lookup[label] = len(measurements)
                        pw = qml.pauli.PauliWord({mapping[w]: axis for w, axis in local.items()})
                        measurements.append(qml.expval(pw.operation()))
                mapped = [qml.map_wires(o, mapping) for o in local_ops[fid]]
                if torch_anchor is not None:
                    import torch
                    mapped = [qml.ops.functions.bind_new_parameters(o, [
                        torch.as_tensor(p, dtype=torch_anchor.dtype, device=torch_anchor.device)
                        if isinstance(p, (float, np.floating)) else p
                        for p in o.data]) for o in mapped]
                # Include idle wires in the tape's wire accounting.
                mapped = [qml.Identity(w) for w in range(len(wires))] + mapped
                variants[key] = (len(tapes), lookup)
                tapes.append(qml.tape.QuantumScript(mapped, measurements, shots=tape.shots))
            ti, lookup = variants[key]
            for mi, local in needed:
                refs_by_moment[mi].append((ti, lookup[tuple(local.items())]))
        terms.extend((mi, coeff, tuple(refs)) for mi, refs in enumerate(refs_by_moment))
    zero = 0.0
    for p in tape.get_parameters(trainable_only=False):
        if qml.math.get_interface(p) in {"torch", "autograd", "jax", "tensorflow"}:
            zero = zero + qml.math.sum(p) * 0
    return CutPlan(tuple(tapes), tuple(terms), tuple(readouts), len(moments),
                   groups, tuple(coeffs), n_settings, config, tuple(tape.operations), zero,
                   {"source_wires": len(tape.wires), "gate_cuts": sum(k == "gate" for k, _ in cuts),
                    "wire_cuts": sum(k == "wire" for k, _ in cuts)})
