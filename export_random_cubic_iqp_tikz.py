from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml

from qasp.circuits import CircuitBuilder, IQPAnsatz


N_QUBITS = 6
SAMPLE = 0
RESULTS = Path(
    "./tests/results/iqp_random_cubic_rzzz_variance"
)
OUTPUT = RESULTS / "representative_circuits"


def parse_supports(encoded: str) -> list[tuple[int, ...]]:
    return [tuple(map(int, item.split("-"))) for item in encoded.split(";")]


class ThreeLocalIQPAnsatz:
    """Exact QASP-compatible Z/ZZ/ZZZ ansatz used by the executed notebook."""

    def __init__(self, n_qubits, edges, triples):
        self.n_qubits = int(n_qubits)
        self.n_layers = 1
        self.feature_map = None
        self.reuploading = False
        self.include_single_qubit = True
        self.edges = [tuple(map(int, edge)) for edge in edges]
        self.triples = [tuple(map(int, triple)) for triple in triples]
        self.z_gates = None
        self.parity_gates = None
        self.CNOT_Gates = None
        self._block_size = self.n_qubits + len(self.edges) + len(self.triples)

    def get_weights_shape(self):
        return {"params": self._block_size}

    def __call__(self, inputs=None, input_scaling_params=None, params=None):
        if params.ndim != 1 or len(params) != self._block_size:
            raise ValueError(
                f"params must be 1D of length {self._block_size}; got {params.shape}"
            )
        for wire in range(self.n_qubits):
            qml.Hadamard(wires=wire)
        for wire in range(self.n_qubits):
            qml.RZ(params[wire], wires=wire)
        offset = self.n_qubits
        for edge_index, edge in enumerate(self.edges):
            qml.PauliRot(params[offset + edge_index], "ZZ", wires=list(edge))
        offset += len(self.edges)
        for triple_index, triple in enumerate(self.triples):
            qml.PauliRot(params[offset + triple_index], "ZZZ", wires=list(triple))
        for wire in range(self.n_qubits):
            qml.Hadamard(wires=wire)


def probability_builder(ansatz):
    builder = CircuitBuilder(n_qubits=ansatz.n_qubits)
    builder.set_ansatz(ansatz)
    builder.set_measurement("probs", wires=list(range(ansatz.n_qubits)))
    builder.build()
    return builder


def main():
    circuits = pd.read_csv(RESULTS / "sampled_circuits.csv")
    row = circuits.loc[
        (circuits["n_qubits"] == N_QUBITS) & (circuits["sample"] == SAMPLE)
    ].iloc[0]
    edges = parse_supports(row["edges"])
    triples = parse_supports(row["triples"])

    angle_rng = np.random.default_rng(int(row["angle_component_seed"]))
    base_angles = angle_rng.uniform(-np.pi, np.pi, size=N_QUBITS + len(edges))
    triple_angles = angle_rng.uniform(-np.pi, np.pi, size=N_QUBITS)
    augmented_angles = np.concatenate([base_angles, triple_angles])
    assert np.array_equal(augmented_angles[: len(base_angles)], base_angles)

    base_ansatz = IQPAnsatz(
        n_qubits=N_QUBITS,
        n_layers=1,
        imprimitive_pattern=edges,
        include_single_qubit=True,
    )
    augmented_ansatz = ThreeLocalIQPAnsatz(N_QUBITS, edges, triples)
    base_builder = probability_builder(base_ansatz)
    augmented_builder = probability_builder(augmented_ansatz)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    tikz_options = {
        "standalone": True,
        "symbolic_parameters": True,
        "wire_labels": {wire: rf"q_{{{wire + 1}}}" for wire in range(N_QUBITS)},
        "gate_colors": {
            "single": "7950C5",
            "two": "E07A2D",
            "multi": "00A6A6",
            "measurement": "2E8B57",
        },
        "row_sep": "0.24cm",
        "column_sep": "0.10cm",
    }
    base_tex = OUTPUT / "random_3_regular_iqp_n6.tex"
    augmented_tex = OUTPUT / "random_3_regular_plus_rzzz_iqp_n6.tex"
    base_builder.save_tikz(base_tex, params=base_angles, **tikz_options)
    augmented_builder.save_tikz(
        augmented_tex, params=augmented_angles, **tikz_options
    )

    metadata = {
        "source_experiment": str(RESULTS.parent.parent / "iqp_random_cubic_rzzz_variance.ipynb"),
        "source_circuit_table": str(RESULTS / "sampled_circuits.csv"),
        "n_qubits": N_QUBITS,
        "sample": SAMPLE,
        "graph_component_seed": int(row["graph_component_seed"]),
        "networkx_graph_seed": int(row["networkx_graph_seed"]),
        "triple_component_seed": int(row["triple_component_seed"]),
        "angle_component_seed": int(row["angle_component_seed"]),
        "edges": [list(edge) for edge in edges],
        "triples": [list(triple) for triple in triples],
        "base_parameter_count": len(base_angles),
        "augmented_parameter_count": len(augmented_angles),
        "shared_base_angles_bitwise_identical": True,
        "qasp_export_api": "CircuitBuilder.save_tikz(..., standalone=True, symbolic_parameters=True)",
    }
    (OUTPUT / "representative_circuits_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(base_tex)
    print(augmented_tex)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
