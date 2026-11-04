"""Export exact representative QASP policy circuits from the BP experiment."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import pandas as pd
import pennylane as qml
import torch

from qasp.circuits.constructor import CircuitBuilder
from qasp.circuits.templates import IQPAnsatz
from qasp.feature_maps.templates import NeuralEmbedding
from qasp.rl.model import draw_model
from qasp.rl.policies import PQCBornVariance


RESULTS = Path(
    "./tests/results/"
    "iqp_random_cubic_rzzz_neural_embedding_barren_plateau"
)
OUTPUT = RESULTS / "representative_circuits"
FIXED_STATE = torch.tensor([0.20, -0.30, 0.40, -0.50], dtype=torch.float32)
FEATURES = 4


class ThreeLocalNeuralIQPAnsatz:
    """Exact QASP-compatible ansatz used by the executed experiment."""

    def __init__(self, n_qubits, edges, triples, feature_map):
        self.n_qubits = int(n_qubits)
        self.n_layers = 1
        self.feature_map = feature_map
        self.reuploading = False
        self.include_single_qubit = True
        self.edges = [tuple(map(int, edge)) for edge in edges]
        self.triples = [tuple(map(int, triple)) for triple in triples]
        self.z_gates = None
        self.parity_gates = None
        self.CNOT_Gates = None
        self._block_size = self.n_qubits + len(self.edges) + len(self.triples)

    def get_param_shapes(self):
        return [(self.n_qubits * int(self.feature_map.features),), (self._block_size,)]

    def __call__(self, inputs=None, input_scaling_params=None, params=None):
        for wire in range(self.n_qubits):
            qml.Hadamard(wires=wire)
        self.feature_map(inputs, input_scaling_params=input_scaling_params)
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


def parse_tuples(value: str, arity: int):
    parsed = [tuple(map(int, item.split("-"))) for item in value.split(";") if item]
    assert all(len(item) == arity for item in parsed)
    return parsed


def make_policy(n, edges, triples=None):
    feature_map = NeuralEmbedding(n_qubits=n, rotations="Z", features=FEATURES)
    if triples is None:
        ansatz = IQPAnsatz(
            n_qubits=n,
            n_layers=1,
            imprimitive_pattern=edges,
            feature_map=feature_map,
            reuploading=False,
            include_single_qubit=True,
        )
    else:
        ansatz = ThreeLocalNeuralIQPAnsatz(n, edges, triples, feature_map)
    builder = CircuitBuilder(n_qubits=n)
    builder.set_ansatz(ansatz)
    model = PQCBornVariance(
        circuit_builder=builder,
        n_qubits=n,
        n_actions=2,
        init="random_-pi_pi",
        input_scaling_init="random_-pi_pi",
        policy="majority-modulo",
        device="lightning.qubit",
        diff_method="best",
        softmax_activation=False,
        output_scaling=None,
        training_method="probs",
        torchrl=False,
        measurement_qubits=list(range(n)),
        majority_blocks=[list(range(n))],
        strict_majority_blocks=False,
        parity_cnot_decomposition=False,
    )
    # draw_model historically reads this convenience attribute. The circuit itself
    # still comes from the model's exact CircuitBuilder.
    model.fm = feature_map
    return model


def indexed_uniform(shape, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    tensor = torch.empty(shape, dtype=torch.float32)
    return tensor.uniform_(-math.pi, math.pi, generator=generator)


def set_recorded_initialization(model, parameter_seed, scaling_seed):
    with torch.no_grad():
        model.qlayer.params.copy_(
            indexed_uniform(tuple(model.qlayer.params.shape), parameter_seed)
        )
        model.qlayer.input_scaling_params.copy_(
            indexed_uniform(
                tuple(model.qlayer.input_scaling_params.shape), scaling_seed
            )
        )


def labels(n, edges, triples):
    result = [rf"\phi_{{{wire + 1}}}(\mathbf{{s}})" for wire in range(n)]
    result += [rf"\alpha_{{{wire + 1}}}" for wire in range(n)]
    result += [rf"\beta_{{{u + 1},{v + 1}}}" for u, v in edges]
    result += [
        rf"\gamma_{{{u + 1},{v + 1},{w + 1}}}" for u, v, w in (triples or [])
    ]
    return result


def export_one(name, model, n, edges, triples):
    source = draw_model(
        model=model,
        policy_class="PQCBorn",
        n_qubits=n,
        num_inputs=FEATURES,
        inputs=FIXED_STATE,
        output="tikz",
        tikz_options={
            "standalone": True,
            "wire_labels": {wire: rf"q_{{{wire + 1}}}" for wire in range(n)},
            "parameter_labels": labels(n, edges, triples),
            "row_sep": "0.23cm",
            "column_sep": "0.08cm",
            "gate_colors": {
                "single": "DCECF8",
                "two": "F7D7B5",
                "multi": "CBECE8",
                "measurement": "DCE8D5",
            },
        },
    )
    # Preserve QASP's generated gate/wire layout; only select compact typography.
    # A paper can scale the standalone PDF with \includegraphics[width=\linewidth].
    source = source.replace(r"\begin{document}", "\\begin{document}\n\\scriptsize")
    destination = OUTPUT / f"{name}.tex"
    destination.write_text(source, encoding="utf-8")
    return destination


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    circuits = pd.read_csv(RESULTS / "sampled_circuits.csv")
    row = circuits[(circuits["n_qubits"] == 6) & (circuits["topology"] == 0)].iloc[0]
    edges = parse_tuples(row["edges"], 2)
    triples = parse_tuples(row["triples"], 3)
    assert len(edges) == 9 and len(triples) == 6
    assert len(set(edges)) == len(edges)
    assert len(set(triples)) == len(triples)
    assert all(edge in combinations(range(6), 2) for edge in edges)

    base = make_policy(6, edges)
    augmented = make_policy(6, edges, triples)
    set_recorded_initialization(base, row["parameter_seed"], row["scaling_seed"])
    set_recorded_initialization(augmented, row["parameter_seed"], row["scaling_seed"])
    assert torch.equal(
        base.qlayer.input_scaling_params,
        augmented.qlayer.input_scaling_params,
    )
    assert torch.equal(base.qlayer.params, augmented.qlayer.params[: len(base.qlayer.params)])

    base_tex = export_one(
        "neural_embedding_random_3_regular_iqp_n6", base, 6, edges, None
    )
    augmented_tex = export_one(
        "neural_embedding_random_3_regular_plus_rzzz_iqp_n6",
        augmented,
        6,
        edges,
        triples,
    )
    metadata = {
        "source_notebook": str(
            Path("./tests")
            / "iqp_random_cubic_rzzz_neural_embedding_barren_plateau.ipynb"
        ),
        "source_record": {key: row[key].item() if hasattr(row[key], "item") else row[key] for key in row.index},
        "fixed_state": FIXED_STATE.tolist(),
        "feature_map": "QASP NeuralEmbedding(rotations='Z', features=4)",
        "policy": "QASP PQCBornVariance majority-modulo, deterministic ties to action 0",
        "qasp_api": "qasp.rl.model.draw_model(model=..., inputs=..., output='tikz', tikz_options=...)",
        "parameter_pairing_checks": {
            "input_scaling_bitwise_equal": True,
            "shared_rz_rzz_prefix_bitwise_equal": True,
        },
        "files": {
            "tikz": [str(base_tex), str(augmented_tex)],
            "pdf": [
                str(base_tex.with_suffix(".pdf")),
                str(augmented_tex.with_suffix(".pdf")),
            ],
            "png_300_dpi": [
                str(base_tex.with_suffix(".png")),
                str(augmented_tex.with_suffix(".png")),
            ],
        },
        "latex_dependencies": ["standalone", "xcolor", "quantikz", "tikz/fit"],
        "paper_integration": "Include the compiled PDF at width=\\linewidth (single column) or width=\\textwidth inside figure* (two columns).",
    }
    (OUTPUT / "circuit_export_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
