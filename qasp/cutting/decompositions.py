"""Six-setting Pauli-channel decompositions with coherent signed instruments.

Conventions follow U(theta)=exp(-i theta P/2). A record ancilla stores a
projective Pauli outcome; including its Z in the terminal observable implements
sum_a a Pi_a rho Pi_a without postselection or loss of branch probability.
"""
from dataclasses import dataclass
import numpy as np
import pennylane as qml


@dataclass(frozen=True)
class RecordWire:
    """Collision-free internal wire label for a local measurement record."""
    cut: int
    side: int


@dataclass
class GateRecipe:
    """A bipartite Pauli rotation with optional local unitary wrappers."""
    word: dict
    angle: object
    before: list
    after: list


_REGISTRY = {}
_BUILTINS = {"CNOT", "CZ", "IsingXX", "IsingYY", "IsingZZ", "PauliRot"}


def register_decomposition(name, factory, *, replace=False):
    """Register an operation-name -> GateRecipe factory (audited by the caller).

    Factories must preserve live parameter tensors and return local wrappers.
    This registry is independent of the cut-selection/planner interface.
    """
    if not callable(factory):
        raise TypeError("A decomposition factory must be callable.")
    if (name in _REGISTRY or name in _BUILTINS) and not replace:
        raise ValueError(f"A decomposition for {name} is already registered.")
    _REGISTRY[name] = factory


def recipe(operation):
    """Resolve supported gates, including adjoints, without detaching angles."""
    inverse = False
    while isinstance(operation, qml.ops.op_math.Adjoint):
        inverse = not inverse
        operation = operation.base
    if operation.name in _REGISTRY:
        if inverse:
            raise ValueError("Custom registry recipes must explicitly handle adjoints.")
        return _REGISTRY[operation.name](operation)
    if operation.name in {"CNOT", "CZ"}:
        a, b = operation.wires
        before = [qml.Hadamard(b)] if operation.name == "CNOT" else []
        after = [qml.RZ(-np.pi / 2, a), qml.RZ(-np.pi / 2, b)]
        if operation.name == "CNOT":
            after.append(qml.Hadamard(b))
        return GateRecipe({a: "Z", b: "Z"}, np.pi / 2, before, after)
    if operation.name == "PauliRot":
        word = dict(zip(operation.wires, operation.hyperparameters["pauli_word"]))
    elif operation.name in {"IsingXX", "IsingYY", "IsingZZ"}:
        word = dict.fromkeys(operation.wires, operation.name[-1])
    else:
        raise ValueError(f"Gate cutting does not support {operation.name}; select CNOT, CZ, "
                         "IsingXX/YY/ZZ or PauliRot, or register a decomposition.")
    word = {w: p for w, p in word.items() if p != "I"}
    if len(word) < 2:
        raise ValueError("A gate cut requires at least two nonidentity Pauli factors.")
    return GateRecipe(word, -operation.data[0] if inverse else operation.data[0], [], [])


def supported(operation):
    """Whether an operation has a registered gate-cut recipe."""
    try:
        recipe(operation)
        return True
    except (ValueError, TypeError):
        return False


def weights(angle):
    """Differentiable six-setting coefficients; keep zero terms for derivatives."""
    c, s = qml.math.cos(angle / 2), qml.math.sin(angle / 2)
    return (c*c, s*s, c*s, -c*s, c*s, -c*s)


def rotation(word, angle):
    """Return a local Pauli rotation, retaining multiqubit Pauli factors."""
    if len(word) == 1:
        wire, axis = next(iter(word.items()))
        return [getattr(qml, "R" + axis)(angle, wires=wire)]
    return [qml.PauliRot(angle, "".join(word.values()), wires=list(word))]


def instrument(word, record):
    """Coherently record a local Pauli parity using a fresh zero-state ancilla."""
    ops, undo = [], []
    for wire, axis in word.items():
        if axis == "X":
            ops.append(qml.Hadamard(wire))
            undo.insert(0, qml.Hadamard(wire))
        elif axis == "Y":
            ops.append(qml.RX(np.pi / 2, wire))
            undo.insert(0, qml.RX(-np.pi / 2, wire))
    ops.extend(qml.CNOT([wire, record]) for wire in word)
    return ops + undo


def gate_terms(rec, groups, cut):
    """Return six local operation dictionaries and record-wire dictionaries."""
    sides = []
    for group in groups:
        local = {w: p for w, p in rec.word.items() if w in group}
        if local:
            sides.append(local)
    if len(sides) != 2:
        raise ValueError("Each cut Pauli rotation must span exactly two fragments. "
                         "Use partitions to group its local factors.")
    settings = [("I", "I"), ("P", "P"), ("M", "+"),
                ("M", "-"), ("+", "M"), ("-", "M")]
    result = []
    for setting in settings:
        ops, records = {}, {}
        for side, (word, kind) in enumerate(zip(sides, setting)):
            fid = next(i for i, group in enumerate(groups) if next(iter(word)) in group)
            local_ops = [o for o in rec.before if set(o.wires).issubset(groups[fid])]
            ancillas = []
            if kind == "P":
                local_ops += rotation(word, np.pi)
            elif kind in {"+", "-"}:
                local_ops += rotation(word, np.pi / 2 if kind == "+" else -np.pi / 2)
            elif kind == "M":
                record = RecordWire(cut, side)
                ancillas.append(record)
                local_ops += instrument(word, record)
            local_ops += [o for o in rec.after if set(o.wires).issubset(groups[fid])]
            ops[fid], records[fid] = local_ops, ancillas
        result.append((ops, records))
    return result
