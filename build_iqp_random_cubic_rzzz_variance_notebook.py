from pathlib import Path
import textwrap

import nbformat as nbf


OUTPUT = Path("iqp_random_cubic_rzzz_variance.ipynb")


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
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
        # Binary-majority variance on random cubic IQP and three-local IQP circuits

        This notebook extends the complete-graph experiment to two matched random-circuit
        ensembles for \(n\in\{4,6,8,10,12,14,16\}\):

        1. a connected random 3-regular graph-IQP circuit with local \(R_Z\) and graph-edge \(R_{ZZ}\) gates;
        2. the **same sampled graph and the same local/two-body angles**, augmented with exactly \(n\)
           distinct random three-qubit \(R_{ZZZ}\) gates and independent three-body angles.

        Every plotted point contains 2,000 independently sampled circuit instances, not 2,000 angle
        assignments on one fixed topology. Execution uses native QASP `CircuitBuilder` QNodes on
        PennyLane `lightning.qubit` in the `torchlane` environment.

        **Contents**

        1. [Protocol](#protocol)
        2. [QASP circuit templates](#qasp-templates)
        3. [Random-circuit Monte Carlo](#monte-carlo)
        4. [Variance comparison](#variance-comparison)
        5. [Validation and reproducibility](#validation)
        """
    ),
    code(
        """
        from pathlib import Path
        from itertools import combinations
        import hashlib
        import importlib.metadata as importlib_metadata
        import json
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
        from IPython.display import display
        from qasp.circuits import CircuitBuilder, IQPAnsatz

        QUBITS = [4, 6, 8, 10, 12, 14, 16]
        N_CIRCUITS = 2_000
        BOOTSTRAP_REPLICATES = 2_000
        BASE_SEED = 20261022
        FIGURE_DPI = 300
        DEVICE = "lightning.qubit"

        cwd = Path.cwd().resolve()
        if cwd.name == "tests" and (cwd.parent / "qasp").is_dir():
            PROJECT_ROOT = cwd.parent
            NOTEBOOK_DIR = cwd
        elif (cwd / "qasp").is_dir() and (cwd / "tests").is_dir():
            PROJECT_ROOT = cwd
            NOTEBOOK_DIR = cwd / "tests"
        else:
            raise RuntimeError("Run this notebook from the QASP repository or QASP/tests directory.")

        RESULTS_DIR = NOTEBOOK_DIR / "results" / "iqp_random_cubic_rzzz_variance"
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
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

        print(f"Python: {sys.version.split()[0]} (torchlane interpreter: {sys.executable})")
        print(f"QASP: {QASP_VERSION}; PennyLane: {qml.__version__}; NetworkX: {nx.__version__}")
        print(f"Device: {DEVICE}; random circuits per n: {N_CIRCUITS}")
        print(f"Results: {RESULTS_DIR}")
        """
    ),
    md(
        r"""
        <a id="protocol"></a>
        ## Protocol

        For sample \(r\) at size \(n\), we first draw a simple connected 3-regular graph
        \(G_r=(V,E_r)\) with NetworkX's `random_regular_graph`, rejecting disconnected draws.
        The base circuit is

        \[
        U_{2,r}=H^{\otimes n}
          \left[\prod_{i=1}^{n}R_{Z_i}(\alpha_i)\right]
          \left[\prod_{(i,j)\in E_r}R_{Z_iZ_j}(\beta_{ij})\right]
          H^{\otimes n}.
        \]

        We then draw \(n\) distinct unordered triples uniformly without replacement from
        \(\binom{V}{3}\) and construct the matched augmented circuit

        \[
        U_{3,r}=H^{\otimes n}
          \left[\prod_iR_{Z_i}(\alpha_i)\right]
          \left[\prod_{(i,j)\in E_r}R_{Z_iZ_j}(\beta_{ij})\right]
          \left[\prod_{(i,j,k)\in T_r}R_{Z_iZ_jZ_k}(\gamma_{ijk})\right]
          H^{\otimes n}.
        \]

        All angles are independent \(\operatorname{Unif}[-\pi,\pi]\), except that the base and
        augmented member of a pair deliberately share \(\boldsymbol\alpha,\boldsymbol\beta\).
        This removes avoidable Monte Carlo noise from the comparison.

        The binary majority decoder uses every qubit and assigns an even-weight tie to action 0:
        \(a(x)=0\) for \(|x|\le n/2\), otherwise \(a(x)=1\). We estimate the joint topology-and-angle
        variance \(\operatorname{Var}_{G,T,\theta}[\pi_0]\); action 1 has the same variance.
        """
    ),
    md(
        r"""
        <a id="qasp-templates"></a>
        ## QASP circuit templates

        The two-local condition uses QASP's built-in `IQPAnsatz`. The augmented condition uses the
        small QASP-compatible template below because the current built-in template accepts one- and
        two-body diagonal generators only. Both are passed to the same QASP `CircuitBuilder`, compiled
        to native PennyLane QNodes, and measured with exact computational-basis probabilities.
        """
    ),
    code(
        """
        class ThreeLocalIQPAnsatz:
            # One-layer QASP-compatible H-D_Z-H template with Z, ZZ and ZZZ generators.
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

            def get_param_shapes(self):
                return [(self._block_size,)]

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


        def build_probability_qnode(ansatz):
            builder = CircuitBuilder(n_qubits=ansatz.n_qubits)
            builder.set_ansatz(ansatz)
            builder.set_measurement("probs", wires=list(range(ansatz.n_qubits)))
            builder.build()
            return builder, builder.compile(device=DEVICE, diff_method=None)


        def majority_zero_mask(n):
            basis_indices = np.arange(2**n, dtype=np.uint64)
            weights = np.fromiter(
                (int(index).bit_count() for index in basis_indices), dtype=np.int16
            )
            return (weights <= n // 2).astype(np.float64)


        def majority_bin_fraction(n):
            from math import comb
            return sum(comb(n, k) for k in range(n // 2 + 1)) / (2**n)


        def complete_graph_reference(n):
            p = majority_bin_fraction(n)
            return (2.0 ** (-n)) * p * (1.0 - p)


        def variance_mc_se(values):
            values = np.asarray(values, dtype=np.float64)
            variance = values.var(ddof=1)
            fourth = np.mean((values - values.mean()) ** 4)
            correction = ((len(values) - 3) / (len(values) - 1)) * variance**2
            mc_se = np.sqrt(max(fourth - correction, 0.0) / len(values))
            return variance, mc_se


        def sample_connected_cubic_edges(n, rng):
            if n < 4 or (3 * n) % 2:
                raise ValueError("A simple 3-regular graph requires even n >= 4.")
            attempts = 0
            while True:
                attempts += 1
                graph_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
                graph = nx.random_regular_graph(3, n, seed=graph_seed)
                if nx.is_connected(graph):
                    edges = sorted(tuple(sorted(map(int, edge))) for edge in graph.edges())
                    assert len(edges) == 3 * n // 2
                    assert all(degree == 3 for _, degree in graph.degree())
                    return edges, graph_seed, attempts


        def sample_distinct_triples(n, rng):
            candidates = np.asarray(list(combinations(range(n), 3)), dtype=np.int16)
            selected = rng.choice(len(candidates), size=n, replace=False)
            triples = sorted(tuple(map(int, candidates[index])) for index in selected)
            assert len(triples) == n and len(set(triples)) == n
            return triples


        def encode_edges(edges):
            return ";".join(f"{i}-{j}" for i, j in edges)


        def encode_triples(triples):
            return ";".join(f"{i}-{j}-{k}" for i, j, k in triples)


        def short_hash(text):
            return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


        for n in QUBITS:
            assert len(majority_zero_mask(n)) == 2**n
            assert np.isclose(majority_zero_mask(n).mean(), majority_bin_fraction(n))
        print("QASP-compatible templates and majority masks verified.")
        """
    ),
    md(
        r"""
        <a id="monte-carlo"></a>
        ## Random-circuit Monte Carlo

        Each sample uses deterministic component seeds derived from `(BASE_SEED, n, sample)`.
        The complete edge and triple lists are saved, together with topology hashes and rejection
        counts, so every random circuit can be reconstructed exactly.
        """
    ),
    code(
        """
        raw_rows = []
        circuit_rows = []
        total_start = time.perf_counter()

        for n in QUBITS:
            mask = majority_zero_mask(n)
            n_edges = 3 * n // 2
            start_n = time.perf_counter()

            for sample_index in range(N_CIRCUITS):
                seed_sequence = np.random.SeedSequence([BASE_SEED, n, sample_index])
                graph_child, triple_child, angle_child = seed_sequence.spawn(3)
                graph_component_seed = int(graph_child.generate_state(1, dtype=np.uint32)[0])
                triple_component_seed = int(triple_child.generate_state(1, dtype=np.uint32)[0])
                angle_component_seed = int(angle_child.generate_state(1, dtype=np.uint32)[0])
                graph_rng = np.random.default_rng(graph_component_seed)
                triple_rng = np.random.default_rng(triple_component_seed)
                angle_rng = np.random.default_rng(angle_component_seed)

                edges, networkx_graph_seed, graph_attempts = sample_connected_cubic_edges(
                    n, graph_rng
                )
                triples = sample_distinct_triples(n, triple_rng)
                edge_text = encode_edges(edges)
                triple_text = encode_triples(triples)

                base_ansatz = IQPAnsatz(
                    n_qubits=n,
                    n_layers=1,
                    imprimitive_pattern=edges,
                    include_single_qubit=True,
                )
                _, base_qnode = build_probability_qnode(base_ansatz)

                augmented_ansatz = ThreeLocalIQPAnsatz(n, edges, triples)
                _, augmented_qnode = build_probability_qnode(augmented_ansatz)

                base_parameter_count = n + n_edges
                base_angles = angle_rng.uniform(-np.pi, np.pi, size=base_parameter_count)
                triple_angles = angle_rng.uniform(-np.pi, np.pi, size=n)
                augmented_angles = np.concatenate([base_angles, triple_angles])
                assert np.array_equal(augmented_angles[:base_parameter_count], base_angles)

                base_probabilities = np.asarray(
                    base_qnode(params=base_angles), dtype=np.float64
                )
                augmented_probabilities = np.asarray(
                    augmented_qnode(params=augmented_angles), dtype=np.float64
                )
                pi0_base = float(base_probabilities @ mask)
                pi0_augmented = float(augmented_probabilities @ mask)

                raw_rows.append({
                    "n_qubits": n,
                    "sample": sample_index,
                    "pi0_cubic_Z_ZZ": pi0_base,
                    "pi0_cubic_Z_ZZ_plus_n_ZZZ": pi0_augmented,
                    "paired_probability_difference": pi0_augmented - pi0_base,
                })
                circuit_rows.append({
                    "n_qubits": n,
                    "sample": sample_index,
                    "graph_component_seed": graph_component_seed,
                    "networkx_graph_seed": networkx_graph_seed,
                    "graph_rejection_attempts": graph_attempts,
                    "triple_component_seed": triple_component_seed,
                    "angle_component_seed": angle_component_seed,
                    "n_edges": len(edges),
                    "n_triples": len(triples),
                    "graph_hash": short_hash(edge_text),
                    "triple_hash": short_hash(triple_text),
                    "edges": edge_text,
                    "triples": triple_text,
                })

            elapsed_n = time.perf_counter() - start_n
            print(
                f"n={n:2d}: completed {N_CIRCUITS} matched random-circuit pairs "
                f"in {elapsed_n:.1f}s"
            )

        total_simulation_seconds = time.perf_counter() - total_start
        raw_df = pd.DataFrame(raw_rows)
        circuits_df = pd.DataFrame(circuit_rows)
        print(f"Total native-QASP simulation time: {total_simulation_seconds:.1f}s")
        """
    ),
    code(
        """
        summary_rows = []
        for n in QUBITS:
            part = raw_df.loc[raw_df["n_qubits"] == n]
            base_values = part["pi0_cubic_Z_ZZ"].to_numpy()
            augmented_values = part["pi0_cubic_Z_ZZ_plus_n_ZZZ"].to_numpy()
            base_variance, base_mc_se = variance_mc_se(base_values)
            augmented_variance, augmented_mc_se = variance_mc_se(augmented_values)

            bootstrap_rng = np.random.default_rng(BASE_SEED + 100_000 + n)
            bootstrap_indices = bootstrap_rng.integers(
                0, N_CIRCUITS, size=(BOOTSTRAP_REPLICATES, N_CIRCUITS)
            )
            bootstrap_difference = (
                augmented_values[bootstrap_indices].var(axis=1, ddof=1)
                - base_values[bootstrap_indices].var(axis=1, ddof=1)
            )
            variance_difference = augmented_variance - base_variance
            ci_low, ci_high = np.quantile(bootstrap_difference, [0.025, 0.975])
            shared = {
                "n_qubits": n,
                "n_random_circuits": N_CIRCUITS,
                "theory_mean": majority_bin_fraction(n),
                "complete_graph_reference": complete_graph_reference(n),
                "variance_difference_augmented_minus_base": variance_difference,
                "paired_bootstrap_ci_low": ci_low,
                "paired_bootstrap_ci_high": ci_high,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
            summary_rows.extend([
                {
                    **shared,
                    "condition": "random cubic: Z + ZZ",
                    "n_RZ": n,
                    "n_RZZ": 3 * n // 2,
                    "n_RZZZ": 0,
                    "n_parameters": n + 3 * n // 2,
                    "sample_mean": base_values.mean(),
                    "empirical_variance": base_variance,
                    "mc_se": base_mc_se,
                },
                {
                    **shared,
                    "condition": "same cubic + n random ZZZ",
                    "n_RZ": n,
                    "n_RZZ": 3 * n // 2,
                    "n_RZZZ": n,
                    "n_parameters": n + 3 * n // 2 + n,
                    "sample_mean": augmented_values.mean(),
                    "empirical_variance": augmented_variance,
                    "mc_se": augmented_mc_se,
                },
            ])

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = RESULTS_DIR / "variance_summary.csv"
        raw_csv = RESULTS_DIR / "paired_action_probability_samples.csv"
        circuits_csv = RESULTS_DIR / "sampled_circuits.csv"
        summary_df.to_csv(summary_csv, index=False)
        raw_df.to_csv(raw_csv, index=False)
        circuits_df.to_csv(circuits_csv, index=False)

        display(summary_df.round({
            "theory_mean": 8,
            "sample_mean": 8,
            "empirical_variance": 10,
            "mc_se": 10,
            "variance_difference_augmented_minus_base": 10,
            "paired_bootstrap_ci_low": 10,
            "paired_bootstrap_ci_high": 10,
        }))
        print(f"Saved {summary_csv}")
        print(f"Saved {raw_csv}")
        print(f"Saved {circuits_csv}")
        """
    ),
    md(
        r"""
        <a id="variance-comparison"></a>
        ## Variance comparison

        Error bars are twice the plug-in Monte Carlo standard error. The dotted curve is the simple
        exponential reference \(2^{-n}=1/2^n\); it is not asserted as the theory curve for either
        random sparse ensemble.
        """
    ),
    code(
        """
        # Match the compact, title-free visual language used by the paper's
        # existing policy-variance panels.  Purple/orange are the QASP paper
        # palette; the neutral dotted line is a visual reference only.
        figure, axis = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        colors = {
            "random cubic: Z + ZZ": "#7950C5",
            "same cubic + n random ZZZ": "#E07A2D",
        }
        markers = {
            "random cubic: Z + ZZ": "o",
            "same cubic + n random ZZZ": "s",
        }
        linestyles = {
            "random cubic: Z + ZZ": "-",
            "same cubic + n random ZZZ": "--",
        }
        labels = {
            "random cubic: Z + ZZ": r"3-regular: $R_Z+R_{ZZ}$",
            "same cubic + n random ZZZ": r"3-regular: $R_Z+R_{ZZ}+R_{ZZZ}$",
        }

        for condition in colors:
            part = summary_df.loc[summary_df["condition"] == condition].sort_values("n_qubits")
            axis.errorbar(
                part["n_qubits"],
                part["empirical_variance"],
                yerr=2.0 * part["mc_se"],
                fmt=markers[condition],
                linestyle=linestyles[condition],
                markersize=5.8,
                capsize=2.5,
                elinewidth=1.0,
                linewidth=1.8,
                color=colors[condition],
                label=labels[condition],
            )

        reference_part = summary_df.loc[
            summary_df["condition"] == "random cubic: Z + ZZ"
        ].sort_values("n_qubits")
        reference_x = reference_part["n_qubits"].to_numpy()
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
        axis.set_ylabel(r"$\\operatorname{Var}_{G,T,\\theta}[\\pi_{\\theta}(0)]$")
        axis.grid(True, which="major", alpha=0.22, linewidth=0.7)
        axis.grid(True, which="minor", alpha=0.10, linewidth=0.5)
        axis.legend(
            frameon=True,
            framealpha=0.92,
            edgecolor="#CCCCCC",
            loc="lower left",
            handlelength=2.8,
        )

        plot_png = FIGURES_DIR / "majority_random_cubic_rzzz_variance.png"
        plot_pdf = FIGURES_DIR / "majority_random_cubic_rzzz_variance.pdf"
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
        ## Validation and reproducibility

        The checks enforce 2,000 distinct circuit samples per size, graph regularity/count identities,
        exactly \(n\) distinct triples per augmented circuit, paired topology/angle construction,
        normalized policy probabilities, expected parameter counts, and complete seed/topology records.
        """
    ),
    code(
        """
        assert len(raw_df) == len(QUBITS) * N_CIRCUITS
        assert len(circuits_df) == len(QUBITS) * N_CIRCUITS
        assert len(summary_df) == 2 * len(QUBITS)
        assert raw_df["pi0_cubic_Z_ZZ"].between(-1e-10, 1.0 + 1e-10).all()
        assert raw_df["pi0_cubic_Z_ZZ_plus_n_ZZZ"].between(-1e-10, 1.0 + 1e-10).all()
        assert (circuits_df["n_edges"] == 3 * circuits_df["n_qubits"] // 2).all()
        assert (circuits_df["n_triples"] == circuits_df["n_qubits"]).all()
        assert circuits_df[[
            "graph_component_seed", "networkx_graph_seed",
            "triple_component_seed", "angle_component_seed",
        ]].notna().all().all()
        assert summary_df.select_dtypes(include=[np.number]).apply(np.isfinite).all().all()

        for row in circuits_df.itertuples(index=False):
            edges = [tuple(map(int, item.split("-"))) for item in row.edges.split(";")]
            triples = [tuple(map(int, item.split("-"))) for item in row.triples.split(";")]
            graph = nx.Graph(edges)
            graph.add_nodes_from(range(row.n_qubits))
            assert nx.is_connected(graph)
            assert all(degree == 3 for _, degree in graph.degree())
            assert len(triples) == row.n_qubits == len(set(triples))

        def git_text(*args):
            return subprocess.check_output(
                ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()

        metadata = {
            "experiment": "binary majority variance on matched random cubic and cubic-plus-ZZZ IQP circuits",
            "notebook": str(NOTEBOOK_DIR / "iqp_random_cubic_rzzz_variance.ipynb"),
            "results_directory": str(RESULTS_DIR),
            "qubits": QUBITS,
            "n_random_circuits_per_qubit_count": N_CIRCUITS,
            "base_seed": BASE_SEED,
            "angle_distribution": "independent Uniform[-pi, pi]",
            "graph_ensemble": "NetworkX simple random 3-regular graph conditioned on connectedness",
            "three_local_ensemble": "n distinct unordered triples sampled uniformly without replacement",
            "pairing": "same graph and identical RZ/RZZ angles; augmented condition adds n RZZZ gates",
            "policy": "majority-modulo",
            "n_actions": 2,
            "action": 0,
            "tie_rule": "action 0 when Hamming weight equals n/2",
            "execution": {
                "backend": "native QASP CircuitBuilder / PennyLane lightning.qubit",
                "base_template": "QASP IQPAnsatz",
                "augmented_template": "self-contained QASP-compatible H-D_Z-H ansatz",
                "measurement": "exact computational-basis probabilities",
                "shots": None,
                "diff_method": None,
                "custom_statevector_simulator": False,
            },
            "statistic": "unbiased sample variance over independently sampled circuit topology and angles",
            "uncertainty": "plug-in Monte Carlo SE; paired bootstrap CI for variance difference",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "total_simulation_seconds": total_simulation_seconds,
            "software": {
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "qasp": QASP_VERSION,
                "pennylane": qml.__version__,
                "networkx": nx.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": mpl.__version__,
            },
            "git": {
                "branch": git_text("branch", "--show-current"),
                "commit": git_text("rev-parse", "HEAD"),
                "dirty": bool(git_text("status", "--porcelain")),
            },
        }
        metadata_path = RESULTS_DIR / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\\n")

        durable_files = [summary_csv, raw_csv, circuits_csv, plot_png, plot_pdf, metadata_path]
        manifest_rows = [
            {
                "path": str(path.relative_to(RESULTS_DIR)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in durable_files
        ]
        manifest_path = RESULTS_DIR / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_rows, indent=2) + "\\n")
        display(pd.DataFrame(manifest_rows))

        paired_summary = summary_df.loc[
            summary_df["condition"] == "random cubic: Z + ZZ",
            [
                "n_qubits", "variance_difference_augmented_minus_base",
                "paired_bootstrap_ci_low", "paired_bootstrap_ci_high",
            ],
        ]
        display(paired_summary)
        print("All random-circuit, pairing, probability, and reproducibility checks passed.")
        """
    ),
]

nbf.write(nb, OUTPUT)
print(OUTPUT.resolve())
