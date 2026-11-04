"""Export PennyLane circuits as publication-ready Quantikz/TikZ source.

The exporter deliberately returns LaTeX source instead of rendering an image so
that circuit figures keep the fonts, line weights, and vector quality of the
paper or presentation in which they are used.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


_SINGLE_QUBIT_LABELS = {
    "Hadamard": "H",
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
    "Identity": "I",
    "S": "S",
    "T": "T",
    "SX": r"\sqrt{X}",
}

_ROTATION_LABELS = {
    "RX": "R_x",
    "RY": "R_y",
    "RZ": "R_z",
    "CRX": "R_x",
    "CRY": "R_y",
    "CRZ": "R_z",
    "PhaseShift": "P",
    "ControlledPhaseShift": "P",
    "U1": "U_1",
    "U2": "U_2",
    "U3": "U_3",
    "Rot": "R",
}

_MULTI_ROTATION_LABELS = {
    "IsingXX": "XX",
    "IsingYY": "YY",
    "IsingZZ": "ZZ",
    "IsingXY": "XY",
    "MultiRZ": "Z\\cdots Z",
}

_DEFAULT_GATE_COLORS = {
    "single": "DCECF8",       # light blue
    "two": "E4D8F2",          # soft purple
    "multi": "EEE2F7",        # pale violet
    "measurement": "E8E5F5",  # very light lavender
}

_COLOR_NAMES = {
    "single": "qaspSingleGate",
    "two": "qaspTwoQubitGate",
    "multi": "qaspMultiQubitGate",
    "measurement": "qaspMeasurement",
}


def _wire_list(wires: Any) -> list[Any]:
    """Normalize an absent, iterable or scalar wire label to a list."""
    if wires is None:
        return []
    try:
        return list(wires)
    except TypeError:
        return [wires]


def _latex_text(value: Any) -> str:
    """Escape a value that will be placed inside ``\\mathrm{...}``."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _scalar_value(value: Any) -> Any:
    """Return a Python scalar when possible without importing NumPy or Torch."""
    if isinstance(value, (bool, int, float, complex)):
        return value
    try:
        if getattr(value, "numel", lambda: -1)() == 1:
            return value.detach().cpu().item()
    except (AttributeError, RuntimeError, TypeError):
        pass
    try:
        if getattr(value, "size", -1) == 1:
            return value.item()
    except (AttributeError, ValueError, TypeError):
        pass
    return None


def _format_number(value: Any, decimals: int) -> str:
    """Format a scalar as compact LaTeX text; use a bold parameter placeholder for arrays."""
    scalar = _scalar_value(value)
    if scalar is None:
        return r"\boldsymbol{p}"
    if isinstance(scalar, complex):
        real = round(scalar.real, decimals)
        imag = round(scalar.imag, decimals)
        return f"{real:g}{imag:+g}i"
    if isinstance(scalar, bool):
        return str(int(scalar))
    if isinstance(scalar, int):
        return str(scalar)
    rounded = round(float(scalar), decimals)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")


class _ParameterFormatter:
    """Track parameter order and render custom, supplied, symbolic or numeric labels."""
    def __init__(
        self,
        labels: Sequence[str] | None,
        symbolic: bool,
        decimals: int,
        formatter: Callable[[Any, Any, int, int], str] | None,
        feature_map_input_scaling: bool = False,
        feature_map_layer_indices: bool = False,
    ) -> None:
        """Copy label options and initialize independent parameter/embedding counters."""
        self.labels = list(labels) if labels is not None else None
        self.symbolic = symbolic
        self.decimals = decimals
        self.formatter = formatter
        self.feature_map_input_scaling = feature_map_input_scaling
        self.feature_map_layer_indices = feature_map_layer_indices
        self.embedding_index = 0
        self.cursor = 0

    def format(self, value: Any, operation: Any, parameter_index: int) -> str:
        """Render one parameter and advance the global cursor; custom callbacks take precedence."""
        global_index = self.cursor
        self.cursor += 1
        if self.formatter is not None:
            return str(self.formatter(value, operation, parameter_index, global_index))
        if self.labels is not None:
            if global_index >= len(self.labels):
                raise ValueError(
                    "parameter_labels contains fewer entries than the circuit has parameters"
                )
            return str(self.labels[global_index])
        if self.symbolic:
            return rf"\theta_{{{global_index + 1}}}"
        return _format_number(value, self.decimals)

    def validate(self) -> None:
        """Raise ValueError unless supplied labels match the consumed parameter count exactly."""
        if self.labels is not None and self.cursor != len(self.labels):
            raise ValueError(
                "parameter_labels must contain exactly one entry per circuit parameter "
                f"({self.cursor} used, {len(self.labels)} provided)"
            )


def _operation_data(operation: Any) -> list[Any]:
    """Return an operation's parameter data as a list, or an empty list if absent."""
    return list(getattr(operation, "data", ()) or ())


def _base_operation(operation: Any) -> tuple[Any, bool]:
    """Unwrap one PennyLane adjoint and return (base_operation, is_adjoint)."""
    name = str(getattr(operation, "name", operation.__class__.__name__))
    base = getattr(operation, "base", None)
    is_adjoint = name.startswith("Adjoint(")
    if is_adjoint and base is not None:
        return base, True
    return operation, is_adjoint


def _operation_label(operation: Any, parameters: _ParameterFormatter) -> str:
    """Build a LaTeX gate label, consuming scalar parameter labels and marking adjoints."""
    original = operation
    operation, is_adjoint = _base_operation(operation)
    name = str(getattr(operation, "name", operation.__class__.__name__))
    data = _operation_data(original) or _operation_data(operation)
    matrix_operations = {
        "QubitUnitary",
        "ControlledQubitUnitary",
        "DiagonalQubitUnitary",
    }
    # Matrix entries would produce unreadable labels and are represented by U.
    formatted = (
        []
        if name in matrix_operations
        else [
            parameters.format(value, original, index)
            for index, value in enumerate(data)
        ]
    )

    if name in _SINGLE_QUBIT_LABELS:
        label = _SINGLE_QUBIT_LABELS[name]
    elif name in _ROTATION_LABELS:
        suffix = f"({', '.join(formatted)})" if formatted else ""
        label = _ROTATION_LABELS[name] + suffix
    elif name in _MULTI_ROTATION_LABELS:
        suffix = f"({', '.join(formatted)})" if formatted else ""
        label = rf"R_{{{_MULTI_ROTATION_LABELS[name]}}}{suffix}"
    elif name == "PauliRot":
        word = getattr(operation, "hyperparameters", {}).get("pauli_word", "P")
        suffix = f"({', '.join(formatted)})" if formatted else ""
        label = rf"R_{{{_latex_text(word)}}}{suffix}"
    elif name in matrix_operations:
        label = "U"
    elif formatted:
        label = rf"\operatorname{{{_latex_text(name)}}}({', '.join(formatted)})"
    else:
        label = rf"\operatorname{{{_latex_text(name)}}}"

    if is_adjoint:
        label = rf"{label}^{{\dagger}}"
    return label


def _row_positions(operation: Any, row_for_wire: Mapping[Any, int]) -> list[int]:
    """Map an operation's displayed wires to row indices in operation-wire order."""
    return [row_for_wire[wire] for wire in _wire_list(getattr(operation, "wires", ())) if wire in row_for_wire]


def _controlled_parts(
    operation: Any,
) -> tuple[list[Any], list[Any], Any, bool] | None:
    """Return controls, targets, base, and adjoint state when available."""
    operation, is_adjoint = _base_operation(operation)
    name = str(getattr(operation, "name", operation.__class__.__name__))
    wires = _wire_list(getattr(operation, "wires", ()))
    known_controls = {
        "CNOT": 1,
        "CZ": 1,
        "CY": 1,
        "CRX": 1,
        "CRY": 1,
        "CRZ": 1,
        "ControlledPhaseShift": 1,
        "Toffoli": 2,
        "CCZ": 2,
        "CSWAP": 1,
    }
    if name in known_controls:
        count = known_controls[name]
        return (
            wires[:count],
            wires[count:],
            getattr(operation, "base", operation),
            is_adjoint,
        )

    controls = _wire_list(getattr(operation, "control_wires", ()))
    if not controls:
        return None
    base = getattr(operation, "base", operation)
    targets = _wire_list(getattr(base, "wires", ()))
    if not targets:
        targets = [wire for wire in wires if wire not in controls]
    return controls, targets, base, is_adjoint


def _resolve_gate_colors(
    gate_colors: Mapping[str, str | None] | bool | None,
) -> tuple[dict[str, str], list[str]]:
    """Resolve semantic gate colors and any required xcolor definitions."""
    if gate_colors is False:
        return {}, []
    palette = dict(_DEFAULT_GATE_COLORS)
    if gate_colors is not None and gate_colors is not True:
        unknown = set(gate_colors) - set(_DEFAULT_GATE_COLORS)
        if unknown:
            raise ValueError(
                "gate_colors keys must be 'single', 'two', 'multi', or "
                f"'measurement'; received {sorted(unknown)}"
            )
        palette.update(gate_colors)

    resolved: dict[str, str] = {}
    definitions = []
    for role, value in palette.items():
        if value is None:
            continue
        value = str(value).strip()
        if re.fullmatch(r"#?[0-9A-Fa-f]{6}", value):
            name = _COLOR_NAMES[role]
            definitions.append(
                rf"\definecolor{{{name}}}{{HTML}}{{{value.lstrip('#').upper()}}}"
            )
            resolved[role] = name
        else:
            # Existing xcolor names and expressions such as ``blue!12`` work too.
            resolved[role] = value
    return resolved, definitions


def _gate(label: str, color: str | None = None, wires: int | None = None) -> str:
    """Build a Quantikz gate command with optional color and multiwire span."""
    options = []
    if wires is not None:
        options.append(f"wires={wires}")
    if color:
        options.append(f"style={{fill={color}}}")
    option_text = f"[{', '.join(options)}]" if options else ""
    return rf"\gate{option_text}{{{label}}}"


def _meter(label: str = "", color: str | None = None) -> str:
    """Build a Quantikz measurement command with an optional label and fill color."""
    option_text = f"[style={{fill={color}}}]" if color else ""
    return rf"\meter{option_text}{{{label}}}"


def _place_multiwire_gate(
    column: list[str],
    positions: Sequence[int],
    label: str,
    colors: Mapping[str, str],
) -> None:
    """Place a multi-wire gate without implying action on skipped wires."""
    unique_positions = sorted(set(positions))
    first, last = unique_positions[0], unique_positions[-1]
    role = "two" if len(unique_positions) == 2 else "multi"
    color = colors.get(role)
    if len(unique_positions) == last - first + 1:
        column[first] = _gate(label, color, wires=len(unique_positions))
        return

    connections = "".join(rf"\vqw{{{position - first}}}" for position in unique_positions[1:])
    # Repeating the label at every participating endpoint makes the operation
    # visibly symmetric. An empty endpoint box looks like a controlled gate.
    column[first] = _gate(label, color) + connections
    for position in unique_positions[1:]:
        column[position] = _gate(label, color)


def _angle_embedding_column(
    operation: Any,
    row_for_wire: Mapping[Any, int],
    parameters: _ParameterFormatter,
    colors: Mapping[str, str],
    *,
    is_adjoint: bool = False,
) -> list[str]:
    """Expand an AngleEmbedding template into its individual rotations."""
    column = [r"\qw" for _ in row_for_wire]
    wires = _wire_list(getattr(operation, "wires", ()))
    data = _operation_data(operation)
    if not data:
        return column
    try:
        feature_values = list(data[0])
    except TypeError:
        feature_values = [data[0]]

    rotation = getattr(operation, "hyperparameters", {}).get("rotation", "R")
    rotation_name = getattr(rotation, "__name__", str(rotation))
    rotation_label = _ROTATION_LABELS.get(
        rotation_name,
        rf"R_{{{_latex_text(rotation_name)}}}",
    )
    if is_adjoint:
        rotation_label = rf"{rotation_label}^{{\dagger}}"

    parameters.embedding_index += 1
    for index, (wire, value) in enumerate(zip(wires, feature_values), start=1):
        if wire not in row_for_wire:
            continue
        if parameters.symbolic:
            angle = rf"f_{{{index}}}"
            if parameters.feature_map_input_scaling:
                subscript = (
                    f"{parameters.embedding_index},{index}"
                    if parameters.feature_map_layer_indices
                    else str(index)
                )
                angle += rf"\lambda_{{{subscript}}}"
        else:
            angle = _format_number(value, parameters.decimals)
        column[row_for_wire[wire]] = _gate(
            rf"{rotation_label}({angle})",
            colors.get("single"),
        )
    return column


def _operation_column(
    operation: Any,
    wire_order: Sequence[Any],
    row_for_wire: Mapping[Any, int],
    parameters: _ParameterFormatter,
    colors: Mapping[str, str],
) -> list[str]:
    """Render one operation as a wire-aligned column of Quantikz cell strings."""
    column = [r"\qw" for _ in wire_order]
    positions = _row_positions(operation, row_for_wire)
    if not positions:
        return column

    base_operation, is_adjoint = _base_operation(operation)
    base_name = str(
        getattr(base_operation, "name", base_operation.__class__.__name__)
    )
    if base_name == "AngleEmbedding":
        return _angle_embedding_column(
            base_operation,
            row_for_wire,
            parameters,
            colors,
            is_adjoint=is_adjoint,
        )

    name = str(getattr(operation, "name", operation.__class__.__name__))
    if name in {"Barrier", "WireCut"}:
        first, last = min(positions), max(positions)
        column[first] = rf"\barrier[0em]{{{last - first + 1}}}"
        return column
    if name in {"MidMeasureMP", "MeasurementValue"}:
        column[positions[0]] = _meter(color=colors.get("measurement"))
        return column

    controlled = _controlled_parts(operation)
    if controlled is not None:
        controls, targets, base, is_adjoint = controlled
        target_positions = [row_for_wire[wire] for wire in targets if wire in row_for_wire]
        if not target_positions:
            return column
        anchor = target_positions[0]
        control_values = list(getattr(operation, "control_values", ()) or ())
        for index, wire in enumerate(controls):
            if wire in row_for_wire:
                control_row = row_for_wire[wire]
                command = "ctrl"
                if index < len(control_values) and not control_values[index]:
                    command = "octrl"
                column[control_row] = rf"\{command}{{{anchor - control_row}}}"

        base_name = str(getattr(base, "name", base.__class__.__name__))
        if name in {"CNOT", "Toffoli"} or base_name == "PauliX":
            column[anchor] = r"\targ{}"
        elif name in {"CZ", "CCZ"} or base_name == "PauliZ":
            column[anchor] = r"\control{}"
        elif name == "CY" or base_name == "PauliY":
            column[anchor] = _gate("Y", colors.get("two"))
        elif name == "CSWAP" or base_name == "SWAP":
            other = target_positions[-1]
            column[anchor] = rf"\swap{{{other - anchor}}}"
            column[other] = r"\targX{}"
        else:
            label_source = base if base is not operation else operation
            label = _operation_label(label_source, parameters)
            if is_adjoint:
                label = rf"{label}^{{\dagger}}"
            if len(target_positions) == 1:
                column[anchor] = _gate(label, colors.get("two"))
            else:
                _place_multiwire_gate(column, target_positions, label, colors)
        return column

    if name == "SWAP" and len(positions) >= 2:
        first, last = positions[0], positions[-1]
        column[first] = rf"\swap{{{last - first}}}"
        column[last] = r"\targX{}"
        return column

    label = _operation_label(operation, parameters)
    if len(positions) == 1:
        column[positions[0]] = _gate(label, colors.get("single"))
    else:
        _place_multiwire_gate(column, positions, label, colors)
    return column


def _schedule_operations(
    operations: Sequence[Any],
    wire_order: Sequence[Any],
    row_for_wire: Mapping[Any, int],
    parameters: _ParameterFormatter,
    colors: Mapping[str, str],
) -> list[list[str]]:
    """Pack independent operations into columns while preserving dependencies."""
    columns: list[list[str]] = []
    last_column = [-1 for _ in wire_order]
    for operation in operations:
        positions = _row_positions(operation, row_for_wire)
        if not positions:
            continue
        # Reserve the complete vertical span so another gate cannot collide
        # with a control line or a multi-wire gate in the rendered diagram.
        occupied_rows = range(min(positions), max(positions) + 1)
        column_index = max(last_column[row] + 1 for row in occupied_rows)
        while len(columns) <= column_index:
            columns.append([r"\qw" for _ in wire_order])

        rendered = _operation_column(
            operation, wire_order, row_for_wire, parameters, colors
        )
        for row in occupied_rows:
            if rendered[row] != r"\qw":
                columns[column_index][row] = rendered[row]
            last_column[row] = column_index
    return columns


def _abstract_layer_column(
    layer: int,
    wire_order: Sequence[Any],
    colors: Mapping[str, str],
    *,
    adjoint: bool = False,
) -> list[str]:
    """Create one full-register block representing a hidden variational layer."""
    column = [r"\qw" for _ in wire_order]
    role = "two" if len(wire_order) == 2 else "multi"
    label = (
        rf"U^{{\dagger}}(\theta_{{{layer}}})"
        if adjoint
        else rf"U(\theta_{{{layer}}})"
    )
    column[0] = _gate(
        label,
        colors.get(role),
        wires=len(wire_order),
    )
    return column


def _ellipsis_column(wire_order: Sequence[Any]) -> list[str]:
    """Return a wire-aligned omission marker for a compressed circuit drawing."""
    column = [r"\qw" for _ in wire_order]
    column[len(wire_order) // 2] = r"\push{\cdots}"
    return column


def _first_layer_overlay(group: tuple[int, int], n_wires: int) -> str:
    """Draw a dashed enclosure after Quantikz has positioned all matrix cells."""
    group_start, group_steps = group
    # Quantikz column 1 contains the left wire label, so circuit column 0 is 2.
    first_column = group_start + 2
    last_column = first_column + group_steps - 1
    nodes = "".join(
        rf"(\tikzcdmatrixname-{row}-{column})"
        for row in range(1, n_wires + 1)
        for column in range(first_column, last_column + 1)
    )
    return (
        "execute at end picture={"
        rf"\node[fit={nodes},inner sep=3pt] (qaspFirstLayer) {{}};"
        rf"\draw[dashed,rounded corners] "
        r"(qaspFirstLayer.north west) rectangle (qaspFirstLayer.south east);"
        r"\node[anchor=north] at ([yshift=-2pt]qaspFirstLayer.south) "
        r"{$U(\theta_{1})$};"
        "}"
    )


def _compressed_columns(
    operations: Sequence[Any],
    wire_order: Sequence[Any],
    row_for_wire: Mapping[Any, int],
    parameters: _ParameterFormatter,
    colors: Mapping[str, str],
    *,
    n_layers: int,
    first_layer_start: int,
    first_layer_operation_count: int,
    suffix_operation_count: int,
    max_explicit_layers: int,
    mirror_layers: bool = False,
    layer_operation_count: int | None = None,
) -> tuple[list[list[str]], tuple[int, int]]:
    """Render the first layer and replace subsequent layers with U blocks."""
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1 when compress=True")
    if first_layer_operation_count < 1:
        raise ValueError("first_layer_operation_count must be at least 1")
    if first_layer_start < 0 or suffix_operation_count < 0:
        raise ValueError("compression operation counts cannot be negative")
    first_layer_end = first_layer_start + first_layer_operation_count
    suffix_start = len(operations) - suffix_operation_count
    if mirror_layers:
        if (
            layer_operation_count is None
            or layer_operation_count < first_layer_operation_count
        ):
            raise ValueError(
                "mirrored compression needs the total layer operation count"
            )
        forward_end = first_layer_start + layer_operation_count
        inverse_start = forward_end + 2 * suffix_operation_count
        inverse_end = inverse_start + layer_operation_count
        if inverse_end > len(operations):
            raise ValueError(
                "mirrored compression metadata does not match the circuit operations"
            )
        suffix_start = inverse_end
    if first_layer_end > suffix_start:
        raise ValueError("compression metadata does not match the circuit operations")

    prefix = _schedule_operations(
        operations[:first_layer_start], wire_order, row_for_wire, parameters, colors
    )
    first_layer = _schedule_operations(
        operations[first_layer_start:first_layer_end],
        wire_order,
        row_for_wire,
        parameters,
        colors,
    )
    if not first_layer:
        raise ValueError("the first compressed layer produced no drawable gates")

    columns = prefix + first_layer
    group = (len(prefix), len(first_layer))
    if n_layers > 1:
        if n_layers <= max_explicit_layers:
            layer_numbers = list(range(2, n_layers + 1))
            for layer in layer_numbers:
                columns.append(_abstract_layer_column(layer, wire_order, colors))
        else:
            columns.append(_abstract_layer_column(2, wire_order, colors))
            columns.append(_ellipsis_column(wire_order))
            columns.append(_abstract_layer_column(n_layers, wire_order, colors))

    if mirror_layers:
        middle = operations[forward_end:inverse_start]
        if middle:
            columns.extend(
                _schedule_operations(
                    middle,
                    wire_order,
                    row_for_wire,
                    parameters,
                    colors,
                )
            )
        if n_layers <= max_explicit_layers:
            inverse_layer_numbers = list(range(n_layers, 0, -1))
            for layer in inverse_layer_numbers:
                columns.append(
                    _abstract_layer_column(
                        layer, wire_order, colors, adjoint=True
                    )
                )
        else:
            columns.append(
                _abstract_layer_column(
                    n_layers, wire_order, colors, adjoint=True
                )
            )
            columns.append(_ellipsis_column(wire_order))
            columns.append(
                _abstract_layer_column(1, wire_order, colors, adjoint=True)
            )
        if suffix_start < len(operations):
            columns.extend(
                _schedule_operations(
                    operations[suffix_start:],
                    wire_order,
                    row_for_wire,
                    parameters,
                    colors,
                )
            )
    elif suffix_operation_count:
        columns.extend(
            _schedule_operations(
                operations[suffix_start:],
                wire_order,
                row_for_wire,
                parameters,
                colors,
            )
        )
    return columns, group


def _observable_operation_label(observable: Any) -> str:
    """Format a schematic observable label; scalar factors are omitted and sums use H."""
    name = str(getattr(observable, "name", observable.__class__.__name__))
    if name == "Prod":
        operands = list(getattr(observable, "operands", ()) or ())
        if operands:
            labels = [_observable_operation_label(operand) for operand in operands]
            return r"\otimes ".join(labels)
    if name == "SProd":
        base = getattr(observable, "base", None)
        if base is not None:
            return _observable_operation_label(base)
    if name in {"Hamiltonian", "LinearCombination", "SparseHamiltonian", "Sum"}:
        return r"\mathcal{H}"
    short = _SINGLE_QUBIT_LABELS.get(name)
    if short is not None:
        return short
    return rf"\mathrm{{{_latex_text(name)}}}"


def _observable_label(measurement: Any) -> str:
    """Return the schematic LaTeX label of a measurement's observable, or empty text."""
    observable = getattr(measurement, "obs", None)
    if observable is None:
        return ""
    return _observable_operation_label(observable)


def _observable_wire_labels(observable: Any) -> dict[Any, str]:
    """Map each wire in a product observable to its local measurement basis."""
    name = str(getattr(observable, "name", observable.__class__.__name__))
    if name == "Prod":
        labels: dict[Any, str] = {}
        for operand in list(getattr(observable, "operands", ()) or ()):
            labels.update(_observable_wire_labels(operand))
        return labels
    if name == "SProd":
        base = getattr(observable, "base", None)
        if base is not None:
            return _observable_wire_labels(base)
    label = _observable_operation_label(observable)
    return {wire: label for wire in _wire_list(getattr(observable, "wires", ()))}


def _measurement_column(
    measurements: Sequence[Any],
    wire_order: Sequence[Any],
    row_for_wire: Mapping[Any, int],
    colors: Mapping[str, str],
) -> list[str] | None:
    """Render terminal measurement meters, or None when none are present.

    Tensor products put local basis meters on participating wires. Identity
    is a known constant and adds no meter. This schematic does not encode
    classical product reconstruction or a model's affine readout head."""
    if not measurements:
        return None
    column = [r"\qw" for _ in wire_order]
    used = False
    for measurement in measurements:
        class_name = measurement.__class__.__name__
        wires = _wire_list(getattr(measurement, "wires", ()))
        observable = getattr(measurement, "obs", None)
        if not wires and observable is not None:
            wires = _wire_list(getattr(observable, "wires", ()))
        if not wires or class_name in {"StateMP", "DensityMatrixMP"}:
            wires = list(wire_order)

        positions = [row_for_wire[wire] for wire in wires if wire in row_for_wire]
        if not positions:
            continue
        if class_name in {"ExpectationMP", "VarianceMP"}:
            observable_label = _observable_label(measurement)
            if observable_label == "I":
                # Identity is a known constant, not a physical measurement.
                # It must not overwrite a real meter on the same wire.
                continue
            if len(positions) == 1:
                if class_name == "ExpectationMP":
                    label = rf"\langle {observable_label}\rangle"
                else:
                    label = rf"\operatorname{{Var}}({observable_label})"
                column[positions[0]] = _meter(label, colors.get("measurement"))
            else:
                # A tensor-product observable is physically measured on every
                # participating qubit, not only on the first wire.
                local_labels = _observable_wire_labels(observable)
                for wire in wires:
                    if wire not in row_for_wire:
                        continue
                    label = local_labels.get(wire, observable_label)
                    if label == "I":
                        continue
                    if class_name == "VarianceMP":
                        label = rf"\operatorname{{Var}}({label})"
                    column[row_for_wire[wire]] = _meter(
                        label, colors.get("measurement")
                    )
        elif class_name in {"StateMP", "DensityMatrixMP"}:
            column[min(positions)] = _meter(
                r"\mathrm{state}", colors.get("measurement")
            )
        else:
            for position in positions:
                column[position] = _meter(color=colors.get("measurement"))
        used = True
    return column if used else None


def _default_wire_label(wire: Any) -> str:
    """Create a LaTeX label from an integer wire index or escaped arbitrary label."""
    if isinstance(wire, int):
        return rf"q_{{{wire}}}"
    return rf"\mathrm{{{_latex_text(wire)}}}"


def _resolve_wire_labels(
    wire_order: Sequence[Any], wire_labels: Mapping[Any, str] | Sequence[str] | None
) -> list[str]:
    """Resolve defaults, partial mappings or an exact-length sequence of display labels."""
    if wire_labels is None:
        return [_default_wire_label(wire) for wire in wire_order]
    if isinstance(wire_labels, Mapping):
        return [str(wire_labels.get(wire, _default_wire_label(wire))) for wire in wire_order]
    labels = [str(label) for label in wire_labels]
    if len(labels) != len(wire_order):
        raise ValueError("wire_labels must have exactly one label per wire")
    return labels


def tape_to_tikz(
    tape: Any,
    *,
    wire_order: Sequence[Any] | None = None,
    wire_labels: Mapping[Any, str] | Sequence[str] | None = None,
    decimals: int = 2,
    symbolic_parameters: bool = False,
    parameter_labels: Sequence[str] | None = None,
    parameter_formatter: Callable[[Any, Any, int, int], str] | None = None,
    standalone: bool = False,
    row_sep: str = "0.35cm",
    column_sep: str = "0.25cm",
    gate_colors: Mapping[str, str | None] | bool | None = None,
    compress: bool = False,
    n_layers: int | None = None,
    first_layer_start: int | None = None,
    first_layer_operation_count: int | None = None,
    suffix_operation_count: int = 0,
    max_explicit_layers: int = 4,
    mirror_layers: bool = False,
    layer_operation_count: int | None = None,
    feature_map_input_scaling: bool = False,
    feature_map_layer_indices: bool = False,
) -> str:
    """Convert a PennyLane tape/QuantumScript to Quantikz source.

    ``parameter_labels`` are LaTeX expressions in displayed circuit order. A custom
    ``parameter_formatter(value, operation, local_index, global_index)`` can be
    supplied for complete control over parameter notation. ``gate_colors`` accepts
    the semantic keys ``single``, ``two``, ``multi``, and ``measurement`` with HTML
    hex or xcolor values; pass ``False`` for monochrome output. With ``compress=True``,
    the first layer remains expanded and later layers become ``U(theta_l)`` blocks.
    """
    operations = list(getattr(tape, "operations", ()) or ())
    measurements = list(getattr(tape, "measurements", ()) or ())
    if wire_order is None:
        wire_order = _wire_list(getattr(tape, "wires", ()))
    else:
        wire_order = list(wire_order)
    if not wire_order:
        raise ValueError("Cannot draw a circuit with no wires")
    if len(set(wire_order)) != len(wire_order):
        raise ValueError("wire_order cannot contain duplicate wires")

    colors, color_definitions = _resolve_gate_colors(gate_colors)
    row_for_wire = {wire: index for index, wire in enumerate(wire_order)}
    parameters = _ParameterFormatter(
        parameter_labels,
        symbolic_parameters,
        decimals,
        parameter_formatter,
        feature_map_input_scaling=feature_map_input_scaling,
        feature_map_layer_indices=feature_map_layer_indices,
    )
    group: tuple[int, int] | None = None
    if compress:
        if n_layers is None or first_layer_operation_count is None:
            raise ValueError(
                "compress=True needs layer metadata; pass a QASP CircuitBuilder or "
                "provide n_layers and first_layer_operation_count"
            )
        columns, group = _compressed_columns(
            operations,
            wire_order,
            row_for_wire,
            parameters,
            colors,
            n_layers=int(n_layers),
            first_layer_start=int(first_layer_start or 0),
            first_layer_operation_count=int(first_layer_operation_count),
            suffix_operation_count=int(suffix_operation_count),
            max_explicit_layers=int(max_explicit_layers),
            mirror_layers=bool(mirror_layers),
            layer_operation_count=(
                None
                if layer_operation_count is None
                else int(layer_operation_count)
            ),
        )
    else:
        columns = _schedule_operations(
            operations, wire_order, row_for_wire, parameters, colors
        )
    measurement_column = _measurement_column(
        measurements, wire_order, row_for_wire, colors
    )
    if measurement_column is not None:
        columns.append(measurement_column)
    parameters.validate()

    labels = _resolve_wire_labels(wire_order, wire_labels)
    rows = []
    for row, label in enumerate(labels):
        cells = [rf"\lstick{{${label}$}}"]
        cells.extend(column[row] for column in columns)
        cells.append(r"\qw")
        rows.append(" & ".join(cells) + r" \\")

    environment_options = [f"row sep={row_sep}", f"column sep={column_sep}"]
    if group is not None:
        environment_options.insert(0, _first_layer_overlay(group, len(wire_order)))
    environment = [
        rf"\begin{{quantikz}}[{', '.join(environment_options)}]",
        *rows,
        r"\end{quantikz}",
    ]
    if not standalone:
        requirements = [
            "% Requires \\usepackage{xcolor}",
            "% Requires \\usepackage{quantikz}",
            "% Requires \\usetikzlibrary{fit}",
            *color_definitions,
        ]
        return "\n".join(requirements + environment) + "\n"

    document = [
        r"\documentclass[tikz,border=4pt]{standalone}",
        r"\usepackage{xcolor}",
        r"\usepackage{quantikz}",
        r"\usetikzlibrary{fit}",
        *color_definitions,
        r"\begin{document}",
        *environment,
        r"\end{document}",
    ]
    return "\n".join(document) + "\n"


def _circuit_tape(circuit: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Trace a builder, QNode or quantum callable without executing it, or reuse a tape."""
    if hasattr(circuit, "operations") and hasattr(circuit, "measurements"):
        return circuit

    # CircuitBuilder is detected by behaviour to avoid a circular import.
    if hasattr(circuit, "build") and hasattr(circuit, "n_qubits"):
        quantum_function = getattr(circuit, "circuit", None) or circuit.build()
        return _circuit_tape(quantum_function, args, kwargs)

    construct = getattr(circuit, "construct", None)
    if callable(construct):
        constructed = construct(args, kwargs)
        if constructed is not None and hasattr(constructed, "operations"):
            return constructed
        for attribute in ("tape", "qtape", "_tape"):
            candidate = getattr(circuit, attribute, None)
            if candidate is not None and hasattr(candidate, "operations"):
                return candidate

    qscript = getattr(circuit, "qtape", None)
    if callable(qscript):
        return qscript(*args, **kwargs)

    if callable(circuit):
        import pennylane as qml

        make_qscript = getattr(qml.tape, "make_qscript", None)
        if make_qscript is None:
            make_qscript = qml.transforms.make_tape
        return make_qscript(circuit)(*args, **kwargs)
    raise TypeError(
        "circuit must be a CircuitBuilder, PennyLane QNode, quantum function, or tape"
    )


def _builder_compression_metadata(
    builder: Any,
    tape: Any,
    *,
    mirror_layers: bool = False,
) -> dict[str, int]:
    """Infer layer boundaries from a QASP CircuitBuilder and its ansatz."""
    ansatz = getattr(builder, "ansatz", None)
    if ansatz is None:
        raise ValueError("compress=True requires a builder with an ansatz")
    n_layers = int(getattr(ansatz, "n_layers", 0))
    if n_layers < 1:
        raise ValueError("compress=True requires an ansatz with n_layers >= 1")
    if (
        getattr(ansatz, "feature_map", None) is not None
        and getattr(ansatz, "include_feature_map_first", True) is False
        and n_layers > 1
    ):
        raise ValueError(
            "Automatic compression requires include_feature_map_first=True"
        )
    if getattr(ansatz, "z_gates", None) is not None:
        raise ValueError(
            "Automatic compression is not available for inverse/Z-gate circuits"
        )

    ansatz_name = ansatz.__class__.__name__
    block_size = int(getattr(ansatz, "_block_size", 0))
    if ansatz_name == "AngleAnsatz":
        edge_count = len(getattr(ansatz, "imprimitive_edges", ()) or ())
        layer_counts = []
        for layer in range(n_layers):
            count = block_size
            if getattr(ansatz, "include_imprimitive", False) and (
                layer < n_layers - 1
                or getattr(ansatz, "include_final_imprimitive", True)
            ):
                count += edge_count
            layer_counts.append(count)
    elif ansatz_name == "IQPAnsatz":
        layer_counts = [block_size + int(ansatz.n_qubits)] * n_layers
    else:
        raise ValueError(
            "Automatic compression currently supports AngleAnsatz and IQPAnsatz. "
            "For another circuit, provide n_layers, first_layer_start, and "
            "first_layer_operation_count explicitly."
        )

    if not layer_counts or layer_counts[0] < 1:
        raise ValueError("Could not infer a non-empty first ansatz layer")
    suffix_count = 0
    parity_gates = getattr(ansatz, "parity_gates", None)
    cnot_gates = getattr(ansatz, "CNOT_Gates", None)
    if parity_gates is not None:
        suffix_count = len(parity_gates)
    elif cnot_gates is not None:
        suffix_count = len(cnot_gates)

    operation_count = len(list(getattr(tape, "operations", ()) or ()))
    if (
        getattr(ansatz, "reuploading", False)
        and getattr(ansatz, "feature_map", None) is not None
    ):
        if mirror_layers:
            raise ValueError(
                "Automatic mirrored compression does not support data reuploading"
            )
        # A reuploading layer contains the encoding followed by its variational
        # gates. Keep initial Hadamards outside, and retain any decoding suffix.
        prefix_count = (
            int(ansatz.n_qubits)
            if (
                ansatz_name == "IQPAnsatz"
                or getattr(ansatz, "include_hadamards_initial", False)
            )
            else 0
        )
        encoding_total = (
            operation_count - prefix_count - sum(layer_counts) - suffix_count
        )
        if encoding_total < n_layers or encoding_total % n_layers:
            raise ValueError(
                "The traced circuit does not contain one encoding block per layer"
            )
        encoding_count = encoding_total // n_layers
        operations = list(tape.operations)

        def signature(ops):
            """Return parameter-independent operation signatures for repeated-block detection."""
            return [(op.name, tuple(op.wires), str(op.hyperparameters)) for op in ops]

        reference = signature(operations[prefix_count : prefix_count + encoding_count])
        cursor = prefix_count
        for count in layer_counts:
            if signature(operations[cursor : cursor + encoding_count]) != reference:
                raise ValueError(
                    "Data-encoding blocks differ structurally between layers"
                )
            cursor += encoding_count + count
        return {
            "n_layers": n_layers,
            "first_layer_start": prefix_count,
            "first_layer_operation_count": encoding_count + layer_counts[0],
            "layer_operation_count": encoding_total + sum(layer_counts),
            "suffix_operation_count": suffix_count,
        }
    if mirror_layers:
        non_layer_count = operation_count - 2 * sum(layer_counts) - 2 * suffix_count
        if non_layer_count < 0 or non_layer_count % 2:
            raise ValueError(
                "The traced overlap circuit does not match the builder's "
                "mirrored layer metadata"
            )
        first_layer_start = non_layer_count // 2
    else:
        first_layer_start = operation_count - sum(layer_counts) - suffix_count
    if first_layer_start < 0:
        raise ValueError("The traced circuit does not match the builder's layer metadata")
    return {
        "n_layers": n_layers,
        "first_layer_start": first_layer_start,
        "first_layer_operation_count": layer_counts[0],
        "layer_operation_count": sum(layer_counts),
        "suffix_operation_count": suffix_count,
    }


def circuit_to_tikz(circuit: Any, *args: Any, **kwargs: Any) -> str:
    """Trace ``circuit`` with ``args`` and return publication-ready TikZ.

    Export options are keyword-only and match :func:`tape_to_tikz`. Arguments
    intended for the circuit can be supplied in ``circuit_kwargs`` when their
    names overlap with exporter options.
    """
    option_names = {
        "wire_order",
        "wire_labels",
        "decimals",
        "symbolic_parameters",
        "parameter_labels",
        "parameter_formatter",
        "standalone",
        "row_sep",
        "column_sep",
        "gate_colors",
        "compress",
        "n_layers",
        "first_layer_start",
        "first_layer_operation_count",
        "suffix_operation_count",
        "max_explicit_layers",
        "mirror_layers",
        "layer_operation_count",
        "feature_map_input_scaling",
        "feature_map_layer_indices",
    }
    options = {name: kwargs.pop(name) for name in list(kwargs) if name in option_names}
    circuit_kwargs = kwargs.pop("circuit_kwargs", None)
    if circuit_kwargs is not None:
        if kwargs:
            raise TypeError("Use either direct circuit keywords or circuit_kwargs, not both")
        kwargs = dict(circuit_kwargs)

    builder_wire_order = None
    if hasattr(circuit, "n_qubits") and hasattr(circuit, "build"):
        builder_wire_order = list(
            range(int(circuit.n_qubits) + int(getattr(circuit, "n_ancillas", 0)))
        )
        ansatz = getattr(circuit, "ansatz", None)
        feature_map = getattr(ansatz, "feature_map", None)
        options.setdefault(
            "feature_map_input_scaling",
            bool(getattr(feature_map, "input_scaling", False)),
        )
        options.setdefault(
            "feature_map_layer_indices", bool(getattr(ansatz, "reuploading", False))
        )
    tape = _circuit_tape(circuit, args, kwargs)
    if "wire_order" not in options and builder_wire_order:
        options["wire_order"] = builder_wire_order
    if options.get("compress") and hasattr(circuit, "ansatz"):
        inferred = _builder_compression_metadata(
            circuit,
            tape,
            mirror_layers=bool(options.get("mirror_layers", False)),
        )
        for name, value in inferred.items():
            options.setdefault(name, value)
    source = tape_to_tikz(tape, **options)
    if (options.get("compress")
        and getattr(getattr(circuit, "ansatz", None), "reuploading", False)
        and getattr(getattr(circuit, "ansatz", None), "feature_map", None) is not None
    ):
        # The hidden blocks include data encoding as well as trainable gates.
        for layer in range(1, int(options["n_layers"]) + 1):
            source = source.replace(
                rf"U(\theta_{{{layer}}})",
                rf"U_{{{layer}}}(\mathbf{{f}};\Theta_{{{layer}}})",
            )
    return source


def save_tikz(path: str | Path, circuit: Any, *args: Any, **kwargs: Any) -> Path:
    """Write :func:`circuit_to_tikz` output to ``path`` and return that path."""
    destination = Path(path)
    destination.write_text(circuit_to_tikz(circuit, *args, **kwargs), encoding="utf-8")
    return destination


__all__ = ["circuit_to_tikz", "save_tikz", "tape_to_tikz"]
