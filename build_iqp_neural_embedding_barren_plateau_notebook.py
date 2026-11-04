from pathlib import Path
import textwrap

import nbformat as nbf


OUTPUT = Path("iqp_random_cubic_rzzz_neural_embedding_barren_plateau.ipynb")


def md(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.10"},
}

nb["cells"] = [
    md(
        r"""
        # Barren-plateau diagnostic for NeuralEmbedding random graph-IQP policies

        This notebook repeats the binary-majority variance experiment with a trainable
        four-feature `NeuralEmbedding` that applies one data-dependent (R_Z) rotation
        to every qubit. It compares matched random circuit ensembles:

        1. connected random 3-regular graph IQP: (R_Z+R_{ZZ});
        2. the same graph and common parameters plus (n) random (R_{ZZZ}) gates.

        Every width uses 100 independently sampled topology pairs and 20 parameter
        initializations per topology, for exactly 2,000 QASP policy realizations per
        condition. Each conditional experiment is executed through
        `qasp.barren_plateau_diagnoser` on a `qasp.rl.policies.PQCBornVariance` model.

        **Contents**

        1. [Protocol](#protocol)
        2. [QASP policy construction](#policy-construction)
        3. [Pilot and paired-initialization checks](#checks)
        4. [Barren-plateau diagnostic sweep](#sweep)
        5. [Results and figures](#results)
        6. [Validation and interpretation](#validation)
        """
    ),
    code(
        """
        from pathlib import Path
        from itertools import combinations
        import gc
        import hashlib
        import importlib.metadata as importlib_metadata
        import json
        import math
        import os
        import platform
        import subprocess
        import sys
        import time

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np
        import pandas as pd
        import pennylane as qml
        import qasp
        import torch
        from IPython.display import display

        from qasp import barren_plateau_diagnoser
        from qasp.circuits import CircuitBuilder, IQPAnsatz
        from qasp.feature_maps import NeuralEmbedding
        from qasp.rl.policies import PQCBornVariance

        QUBITS = [4, 6, 8, 10, 12, 14, 16]
        N_TOPOLOGIES = int(os.environ.get("QASP_BP_N_TOPOLOGIES", "100"))
        SAMPLES_PER_TOPOLOGY = int(os.environ.get("QASP_BP_SAMPLES_PER_TOPOLOGY", "20"))
        N_SAMPLES_PER_POINT = N_TOPOLOGIES * SAMPLES_PER_TOPOLOGY
        BOOTSTRAP_REPLICATES = int(os.environ.get("QASP_BP_BOOTSTRAPS", "2000"))
        RESULTS_NAME = os.environ.get(
            "QASP_BP_RESULTS_NAME",
            "iqp_random_cubic_rzzz_neural_embedding_barren_plateau",
        )
        BASE_SEED = 20261023
        FIGURE_DPI = 300
        DEVICE = "lightning.qubit"
        FEATURES = 4
        FIXED_STATE = torch.tensor([0.20, -0.30, 0.40, -0.50], dtype=torch.float32)

        cwd = Path.cwd().resolve()
        if cwd.name == "tests" and (cwd.parent / "qasp").is_dir():
            PROJECT_ROOT = cwd.parent
            NOTEBOOK_DIR = cwd
        elif (cwd / "qasp").is_dir() and (cwd / "tests").is_dir():
            PROJECT_ROOT = cwd
            NOTEBOOK_DIR = cwd / "tests"
        else:
            raise RuntimeError("Run from the QASP repository or QASP/tests directory.")

        RESULTS_DIR = NOTEBOOK_DIR / "results" / RESULTS_NAME
        FIGURES_DIR = RESULTS_DIR / "figures"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        try:
            QASP_VERSION = importlib_metadata.version("qasp")
        except importlib_metadata.PackageNotFoundError:
            QASP_VERSION = "local checkout"

        mpl.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": FIGURE_DPI,
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

        print(f"Python: {sys.version.split()[0]} ({sys.executable})")
        print(f"QASP: {QASP_VERSION}; PennyLane: {qml.__version__}; Torch: {torch.__version__}")
        print(f"Device: {DEVICE}; widths: {QUBITS}")
        print(f"Topology pairs per width: {N_TOPOLOGIES}")
        print(f"Diagnoser samples per topology: {SAMPLES_PER_TOPOLOGY}")
        print(f"Total policy realizations per plotted point: {N_SAMPLES_PER_POINT}")
        print(f"Fixed NeuralEmbedding input: {FIXED_STATE.tolist()}")
        print(f"Results: {RESULTS_DIR}")
        """
    ),
    md(
        r"""
        <a id="protocol"></a>
        ## Protocol

        For each (n), the notebook samples connected simple 3-regular graphs (G).
        The augmented condition uses the same (G), the same neural-embedding weights,
        and the same (R_Z/R_{ZZ}) angles at every matched draw, then adds (n)
        distinct random three-qubit supports and independent (R_{ZZZ}) angles.

        The four-dimensional input
        
        
        \[
        s_0=(0.20,-0.30,0.40,-0.50)
        \]

        is fixed across every width and sample. A nonzero input is required because
        `NeuralEmbedding(rotations="Z")` would be inactive at (s_0=0).

        Both ansatz angles and neural projection weights are sampled from
        \(\operatorname{Unif}[-\pi,\pi]\). The policy is QASP's exact binary
        `majority-modulo` Born policy on all measured wires. Even-weight ties are
        assigned deterministically to action 0 by explicitly using one block that
        contains all (n) qubits.

        `barren_plateau_diagnoser` reports finite-sample **cost concentration** for
        \(C=\pi_\theta(0\mid s_0)\). It is not by itself a gradient-variance theorem;
        an exponentially small cost variance is a warning signal, while a formal
        barren-plateau claim conventionally requires controlled gradient scaling.
        """
    ),
    md(
        r"""
        <a id="policy-construction"></a>
        ## QASP policy construction

        The base condition uses QASP's built-in `IQPAnsatz`. The augmented ansatz
        implements the same one-layer (H-D_Z-H) program and adds three-local
        diagonal Pauli rotations. Both are attached to QASP `CircuitBuilder` objects
        and executed through `PQCBornVariance`, so decoding and probability handling
        come from `qasp.rl` rather than a custom statevector simulator.
        """
    ),
    code(
        """
        class ThreeLocalNeuralIQPAnsatz:
            # QASP-compatible one-layer H-D_Z-H program with Z, ZZ and ZZZ generators.
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
                scale_count = self.n_qubits * int(self.feature_map.features)
                return [(scale_count,), (self._block_size,)]

            def __call__(self, inputs=None, input_scaling_params=None, params=None):
                if params.ndim != 1 or len(params) != self._block_size:
                    raise ValueError(
                        f"params must be 1D of length {self._block_size}; got {params.shape}"
                    )
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


        def connected_cubic_edges(n, rng):
            for attempt in range(1, 10_001):
                nx_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
                graph = nx.random_regular_graph(3, n, seed=nx_seed)
                if nx.is_connected(graph):
                    return sorted(tuple(sorted(edge)) for edge in graph.edges()), nx_seed, attempt
            raise RuntimeError(f"Failed to sample a connected cubic graph for n={n}.")


        def distinct_triples(n, rng):
            candidates = list(combinations(range(n), 3))
            selected = rng.choice(len(candidates), size=n, replace=False)
            return sorted(candidates[int(index)] for index in selected)


        def make_builder(ansatz):
            builder = CircuitBuilder(n_qubits=ansatz.n_qubits)
            builder.set_ansatz(ansatz)
            return builder


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
            builder = make_builder(ansatz)
            return PQCBornVariance(
                circuit_builder=builder,
                n_qubits=n,
                n_actions=2,
                init="random_-pi_pi",
                input_scaling_init="random_-pi_pi",
                policy="majority-modulo",
                device=DEVICE,
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


        class IndexedUniform:
            # Deterministic per-sample streams make common parameter prefixes exactly paired.
            def __init__(self, seed, label):
                self.seed = int(seed)
                self.counter = 0
                self.__name__ = label

            def __call__(self, tensor):
                generator = torch.Generator(device=tensor.device)
                generator.manual_seed(self.seed + self.counter)
                tensor.uniform_(-math.pi, math.pi, generator=generator)
                self.counter += 1
                return tensor


        def paired_initializers(parameter_seed, scaling_seed):
            return (
                IndexedUniform(parameter_seed, "paired_uniform_minus_pi_pi"),
                IndexedUniform(parameter_seed, "paired_uniform_minus_pi_pi"),
                IndexedUniform(scaling_seed, "paired_uniform_minus_pi_pi_scales"),
                IndexedUniform(scaling_seed, "paired_uniform_minus_pi_pi_scales"),
            )


        def topology_seeds(n, topology_index):
            sequence = np.random.SeedSequence([BASE_SEED, n, topology_index])
            children = sequence.spawn(5)
            return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
        """
    ),
    md(
        r"""
        <a id="checks"></a>
        ## Pilot and paired-initialization checks

        The following assertions verify that the base and augmented initializers
        produce bitwise-identical neural weights and a bitwise-identical prefix for
        the shared (R_Z/R_{ZZ}) parameters at every diagnoser sample. It also runs
        a small end-to-end QASP diagnostic before the production sweep.
        """
    ),
    code(
        """
        def assert_paired_prefixes():
            base_params = torch.empty(15)
            augmented_params = torch.empty(21)
            base_scales = torch.empty(24)
            augmented_scales = torch.empty(24)
            bp, ap, bs, a_s = paired_initializers(12345, 54321)
            for _ in range(5):
                bp(base_params); ap(augmented_params)
                bs(base_scales); a_s(augmented_scales)
                assert torch.equal(base_params, augmented_params[:15])
                assert torch.equal(base_scales, augmented_scales)
            return True


        pairing_ok = assert_paired_prefixes()
        pilot_rng = np.random.default_rng(111)
        pilot_edges, _, _ = connected_cubic_edges(4, pilot_rng)
        pilot_model = make_policy(4, pilot_edges)
        pilot = barren_plateau_diagnoser(
            pilot_model,
            n_samples=4,
            initialization=IndexedUniform(222, "pilot_uniform"),
            input_scaling_init=IndexedUniform(333, "pilot_uniform_scales"),
            state=FIXED_STATE,
            output_index=0,
            parameter_scope="quantum",
            model_type="policy",
            seed=444,
            return_samples=True,
        )
        assert pilot["cost"] == "action_probability[0]"
        assert pilot["n_samples"] == 4
        assert np.isfinite(pilot["cost_samples"]).all()
        assert ((pilot["cost_samples"] >= 0) & (pilot["cost_samples"] <= 1)).all()
        print(f"Paired common-prefix check: {pairing_ok}")
        print(f"Pilot parameter count: {pilot['n_parameters']}")
        print(f"Pilot action-0 samples: {np.round(pilot['cost_samples'], 6)}")
        del pilot_model, pilot
        gc.collect()
        """
    ),
    md(
        r"""
        <a id="sweep"></a>
        ## Barren-plateau diagnostic sweep

        Each topology is diagnosed independently with the public QASP function.
        Raw action-probability samples are retained, together with the diagnoser's
        within-topology variance and local-(Z) expectation summaries. The aggregate
        variance is then computed over all 2,000 realizations at a given width.
        """
    ),
    code(
        """
        raw_rows = []
        topology_rows = []
        circuit_rows = []
        sweep_start = time.perf_counter()

        for n in QUBITS:
            width_start = time.perf_counter()
            for topology_index in range(N_TOPOLOGIES):
                graph_seed, triple_seed, parameter_seed, scaling_seed, diagnostic_seed = topology_seeds(
                    n, topology_index
                )
                graph_rng = np.random.default_rng(graph_seed)
                triple_rng = np.random.default_rng(triple_seed)
                edges, nx_seed, attempts = connected_cubic_edges(n, graph_rng)
                triples = distinct_triples(n, triple_rng)
                edge_text = ";".join("-".join(map(str, edge)) for edge in edges)
                triple_text = ";".join("-".join(map(str, triple)) for triple in triples)

                base_policy = make_policy(n, edges)
                augmented_policy = make_policy(n, edges, triples)
                base_param_init, aug_param_init, base_scale_init, aug_scale_init = paired_initializers(
                    parameter_seed, scaling_seed
                )

                base_result = barren_plateau_diagnoser(
                    base_policy,
                    n_samples=SAMPLES_PER_TOPOLOGY,
                    initialization=base_param_init,
                    input_scaling_init=base_scale_init,
                    state=FIXED_STATE,
                    output_index=0,
                    parameter_scope="quantum",
                    model_type="policy",
                    seed=diagnostic_seed,
                    ddof=1,
                    return_samples=True,
                )
                augmented_result = barren_plateau_diagnoser(
                    augmented_policy,
                    n_samples=SAMPLES_PER_TOPOLOGY,
                    initialization=aug_param_init,
                    input_scaling_init=aug_scale_init,
                    state=FIXED_STATE,
                    output_index=0,
                    parameter_scope="quantum",
                    model_type="policy",
                    seed=diagnostic_seed,
                    ddof=1,
                    return_samples=True,
                )

                conditions = [
                    ("3-regular: RZ + RZZ", base_result, 0),
                    ("3-regular: RZ + RZZ + RZZZ", augmented_result, n),
                ]
                for condition, result, n_rzzz in conditions:
                    costs = np.asarray(result["cost_samples"], dtype=float)
                    topology_rows.append({
                        "n_qubits": n,
                        "topology": topology_index,
                        "condition": condition,
                        "n_diagnoser_samples": SAMPLES_PER_TOPOLOGY,
                        "n_parameters": int(result["n_parameters"]),
                        "cost_mean": float(result["mean"]),
                        "cost_variance": float(result["cost_variance"]),
                        "expectation_value_mean": float(result["expectation_value_mean"]),
                        "expectation_value_variance": float(result["expectation_value_variance"]),
                        "n_RZ_embedding": n,
                        "n_RZ_trainable": n,
                        "n_RZZ": len(edges),
                        "n_RZZZ": n_rzzz,
                    })
                    for sample_index, value in enumerate(costs):
                        raw_rows.append({
                            "n_qubits": n,
                            "topology": topology_index,
                            "sample": sample_index,
                            "condition": condition,
                            "action_0_probability": float(value),
                            "graph_seed": graph_seed,
                            "triple_seed": triple_seed,
                            "parameter_seed": parameter_seed,
                            "scaling_seed": scaling_seed,
                        })

                circuit_rows.append({
                    "n_qubits": n,
                    "topology": topology_index,
                    "graph_seed": graph_seed,
                    "networkx_graph_seed": nx_seed,
                    "graph_rejection_attempts": attempts,
                    "triple_seed": triple_seed,
                    "parameter_seed": parameter_seed,
                    "scaling_seed": scaling_seed,
                    "diagnostic_seed": diagnostic_seed,
                    "edges": edge_text,
                    "triples": triple_text,
                    "graph_hash": hashlib.sha256(edge_text.encode()).hexdigest()[:16],
                    "triple_hash": hashlib.sha256(triple_text.encode()).hexdigest()[:16],
                })

                del base_policy, augmented_policy, base_result, augmented_result
                gc.collect()

            elapsed = time.perf_counter() - width_start
            print(f"n={n:2d}: completed {N_TOPOLOGIES} paired topologies in {elapsed:.1f} s")

        total_simulation_seconds = time.perf_counter() - sweep_start
        raw_df = pd.DataFrame(raw_rows)
        topology_df = pd.DataFrame(topology_rows)
        circuits_df = pd.DataFrame(circuit_rows)
        assert len(raw_df) == 2 * len(QUBITS) * N_SAMPLES_PER_POINT
        assert len(topology_df) == 2 * len(QUBITS) * N_TOPOLOGIES
        assert raw_df["action_0_probability"].between(0, 1).all()
        print(f"Total diagnostic time: {total_simulation_seconds:.1f} s")
        display(topology_df.head())
        """
    ),
    md(
        r"""
        <a id="results"></a>
        ## Results and figures

        Confidence intervals resample complete topology blocks, retaining all
        diagnoser draws belonging to a selected topology. This treats topology,
        neural weights, and ansatz angles as parts of the reported ensemble.
        """
    ),
    code(
        """
        def block_bootstrap_variance(matrix, rng, n_bootstrap):
            n_topologies = matrix.shape[0]
            values = np.empty(n_bootstrap, dtype=float)
            for index in range(n_bootstrap):
                selected = rng.integers(0, n_topologies, size=n_topologies)
                values[index] = np.var(matrix[selected].reshape(-1), ddof=1)
            return values


        summary_rows = []
        for n in QUBITS:
            matrices = {}
            for condition in raw_df["condition"].unique():
                part = raw_df.loc[
                    (raw_df["n_qubits"] == n) & (raw_df["condition"] == condition)
                ].sort_values(["topology", "sample"])
                matrix = part["action_0_probability"].to_numpy().reshape(
                    N_TOPOLOGIES, SAMPLES_PER_TOPOLOGY
                )
                matrices[condition] = matrix
                means = matrix.mean(axis=1)
                grand_mean = matrix.mean()
                within_population = np.mean((matrix - means[:, None]) ** 2)
                between_population = np.mean((means - grand_mean) ** 2)
                total_population = np.mean((matrix - grand_mean) ** 2)
                assert np.isclose(total_population, within_population + between_population)

                bootstrap_rng = np.random.default_rng(BASE_SEED + 10_000 + n + len(summary_rows))
                boot = block_bootstrap_variance(matrix, bootstrap_rng, BOOTSTRAP_REPLICATES)
                summary_rows.append({
                    "n_qubits": n,
                    "condition": condition,
                    "n_topologies": N_TOPOLOGIES,
                    "samples_per_topology": SAMPLES_PER_TOPOLOGY,
                    "n_policy_realizations": matrix.size,
                    "mean_action_0_probability": grand_mean,
                    "cost_variance": np.var(matrix.reshape(-1), ddof=1),
                    "bootstrap_ci_low": np.quantile(boot, 0.025),
                    "bootstrap_ci_high": np.quantile(boot, 0.975),
                    "within_topology_population_variance": within_population,
                    "between_topology_population_variance": between_population,
                    "total_population_variance": total_population,
                    "between_topology_fraction": between_population / total_population,
                    "n_parameters": int(topology_df.loc[
                        (topology_df["n_qubits"] == n)
                        & (topology_df["condition"] == condition), "n_parameters"
                    ].iloc[0]),
                })

            base_name = "3-regular: RZ + RZZ"
            augmented_name = "3-regular: RZ + RZZ + RZZZ"
            rng = np.random.default_rng(BASE_SEED + 20_000 + n)
            differences = np.empty(BOOTSTRAP_REPLICATES)
            for index in range(BOOTSTRAP_REPLICATES):
                selected = rng.integers(0, N_TOPOLOGIES, size=N_TOPOLOGIES)
                differences[index] = (
                    np.var(matrices[augmented_name][selected].reshape(-1), ddof=1)
                    - np.var(matrices[base_name][selected].reshape(-1), ddof=1)
                )
            for row in summary_rows[-2:]:
                row["paired_variance_difference_augmented_minus_base"] = (
                    np.var(matrices[augmented_name].reshape(-1), ddof=1)
                    - np.var(matrices[base_name].reshape(-1), ddof=1)
                )
                row["paired_difference_ci_low"] = np.quantile(differences, 0.025)
                row["paired_difference_ci_high"] = np.quantile(differences, 0.975)

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = RESULTS_DIR / "variance_summary.csv"
        raw_csv = RESULTS_DIR / "diagnoser_action_probability_samples.csv"
        topology_csv = RESULTS_DIR / "diagnoser_topology_results.csv"
        circuits_csv = RESULTS_DIR / "sampled_circuits.csv"
        summary_df.to_csv(summary_csv, index=False)
        raw_df.to_csv(raw_csv, index=False)
        topology_df.to_csv(topology_csv, index=False)
        circuits_df.to_csv(circuits_csv, index=False)
        display(summary_df)

        figure, axis = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        style = {
            "3-regular: RZ + RZZ": ("#7950C5", "o", "-", r"3-regular: $R_Z+R_{ZZ}$"),
            "3-regular: RZ + RZZ + RZZZ": ("#E07A2D", "s", "--", r"3-regular: $R_Z+R_{ZZ}+R_{ZZZ}$"),
        }
        for condition, (color, marker, linestyle, label) in style.items():
            part = summary_df.loc[summary_df["condition"] == condition].sort_values("n_qubits")
            lower = part["cost_variance"] - part["bootstrap_ci_low"]
            upper = part["bootstrap_ci_high"] - part["cost_variance"]
            axis.errorbar(
                part["n_qubits"],
                part["cost_variance"],
                yerr=np.vstack([lower, upper]),
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=5.8,
                capsize=2.5,
                elinewidth=1.0,
                label=label,
            )
        reference_x = np.asarray(QUBITS)
        axis.plot(
            reference_x,
            np.power(2.0, -reference_x),
            ":",
            color="#4D4D4D",
            linewidth=1.4,
            label=r"reference $2^{-n}$",
        )
        axis.set_yscale("log")
        axis.set_xticks(QUBITS)
        axis.set_xlabel(r"number of qubits, $n$")
        axis.set_ylabel(r"$\\operatorname{Var}_{G,T,\\Theta}[\\pi_{\\Theta}(0\\mid s_0)]$")
        axis.grid(True, which="major", alpha=0.22, linewidth=0.7)
        axis.grid(True, which="minor", alpha=0.10, linewidth=0.5)
        axis.legend(frameon=True, framealpha=0.92, edgecolor="#CCCCCC", loc="best")

        plot_png = FIGURES_DIR / "neural_embedding_majority_barren_plateau.png"
        plot_pdf = FIGURES_DIR / "neural_embedding_majority_barren_plateau.pdf"
        figure.savefig(plot_png, dpi=FIGURE_DPI, bbox_inches="tight")
        figure.savefig(plot_pdf, bbox_inches="tight")
        plt.show()
        print(f"Saved {plot_png}")
        print(f"Saved {plot_pdf}")
        """
    ),
    md(
        r"""
        <a id="validation"></a>
        ## Validation and interpretation

        The output reports joint variance over graph topology, triple supports,
        neural projection weights, and ansatz parameters. The within/between
        decomposition verifies the law of total variance at the population level.

        Because every circuit already contains independent trainable local (R_Z)
        angles uniformly distributed over a full (2\pi) period, adding another
        commuting neural (R_Z) phase cannot change the marginal distribution of
        the combined local phase. Consequently, agreement with the earlier
        angle-only experiment is expected under this initialization ensemble. The
        neural feature map still increases the trainable parameter count and is
        executed by the real QASP policy; the invariance is a property of the
        chosen full-period initialization, not evidence that feature maps are
        generally irrelevant.
        """
    ),
    code(
        """
        previous_summary_path = (
            NOTEBOOK_DIR / "results" / "iqp_random_cubic_rzzz_variance" / "variance_summary.csv"
        )
        comparison_df = None
        if previous_summary_path.exists():
            previous = pd.read_csv(previous_summary_path)[
                ["n_qubits", "condition", "empirical_variance"]
            ].copy()
            previous["condition"] = previous["condition"].replace({
                "random cubic: Z + ZZ": "3-regular: RZ + RZZ",
                "same cubic + n random ZZZ": "3-regular: RZ + RZZ + RZZZ",
            })
            comparison_df = summary_df.merge(previous, on=["n_qubits", "condition"], how="left")
            comparison_df["neural_to_angle_only_variance_ratio"] = (
                comparison_df["cost_variance"] / comparison_df["empirical_variance"]
            )
            comparison_df.to_csv(RESULTS_DIR / "angle_only_comparison.csv", index=False)
            display(comparison_df[[
                "n_qubits", "condition", "cost_variance", "empirical_variance",
                "neural_to_angle_only_variance_ratio"
            ]])

        def sha256(path):
            digest = hashlib.sha256()
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        metadata = {
            "experiment": "QASP majority-policy cost concentration with RZ NeuralEmbedding on random cubic and cubic-plus-RZZZ IQP circuits",
            "notebook": str(NOTEBOOK_DIR / "iqp_random_cubic_rzzz_neural_embedding_barren_plateau.ipynb"),
            "qubits": QUBITS,
            "n_topologies_per_width": N_TOPOLOGIES,
            "diagnoser_samples_per_topology": SAMPLES_PER_TOPOLOGY,
            "policy_realizations_per_condition_and_width": N_SAMPLES_PER_POINT,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "base_seed": BASE_SEED,
            "fixed_state": FIXED_STATE.tolist(),
            "feature_map": {
                "class": "qasp.feature_maps.NeuralEmbedding",
                "features": FEATURES,
                "rotations": "Z",
                "one_rotation_per_qubit": True,
            },
            "policy": {
                "class": "qasp.rl.policies.PQCBornVariance",
                "decoder": "majority-modulo",
                "n_actions": 2,
                "measurement_qubits": "all",
                "majority_blocks": "one block containing all qubits",
                "even_tie_rule": "action 0",
                "training_method": "probs",
            },
            "diagnostic": {
                "function": "qasp.barren_plateau_diagnoser",
                "cost": "action_probability[0]",
                "parameter_scope": "quantum",
                "state_mode": "fixed",
                "ddof_within_topology": 1,
                "interpretation": "finite-sample cost concentration, not a standalone gradient-variance barren-plateau proof",
            },
            "initialization": "paired Uniform[-pi, pi] streams for ansatz and NeuralEmbedding weights",
            "pairing": "same graph, same NeuralEmbedding weights, and bitwise-identical common RZ/RZZ parameters; augmented condition adds n RZZZ angles",
            "graph_ensemble": "connected simple NetworkX random 3-regular graphs",
            "triple_ensemble": "n distinct unordered triples sampled uniformly without replacement",
            "device": DEVICE,
            "shots": None,
            "custom_statevector_simulator": False,
            "total_simulation_seconds": total_simulation_seconds,
            "software": {
                "python": sys.version,
                "qasp": QASP_VERSION,
                "pennylane": qml.__version__,
                "torch": torch.__version__,
                "networkx": nx.__version__,
                "platform": platform.platform(),
            },
        }
        metadata_path = RESULTS_DIR / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\\n", encoding="utf-8")

        artifact_paths = [
            summary_csv, raw_csv, topology_csv, circuits_csv,
            plot_png, plot_pdf, metadata_path,
        ]
        comparison_path = RESULTS_DIR / "angle_only_comparison.csv"
        if comparison_path.exists():
            artifact_paths.append(comparison_path)
        manifest = [
            {"path": str(path.relative_to(RESULTS_DIR)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifact_paths
        ]
        manifest_path = RESULTS_DIR / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\\n", encoding="utf-8")

        assert len(summary_df) == 2 * len(QUBITS)
        assert summary_df["n_policy_realizations"].eq(N_SAMPLES_PER_POINT).all()
        assert np.isfinite(summary_df.select_dtypes(include=[np.number])).all().all()
        assert abs(
            summary_df["total_population_variance"]
            - summary_df["within_topology_population_variance"]
            - summary_df["between_topology_population_variance"]
        ).max() < 1e-12
        print(f"Wrote metadata: {metadata_path}")
        print(f"Wrote manifest: {manifest_path}")
        print("All validation checks passed.")
        """
    ),
]

nbf.write(nb, OUTPUT)
print(OUTPUT.resolve())
