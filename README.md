# IQP policy experiments

Notebooks and recorded outputs for **Trainability–Simulability Tradeoff in
Parameterized IQP-based Reinforcement Learning Policies**, by André Sequeira.

**This repository contains the paper's experiments, not the QASP software.**
Install QASP separately from [INESCTEC/qasp](https://github.com/INESCTEC/qasp).
Only experiment-specific model, training and analysis helpers are included here.

## Browse the experiments

| Experiment | Notebook |
|---|---|
| IQP rank/light-cone variance, decoder comparisons, majority and repeated-depth tests | [Variance experiments](tests/iqp_variance_experiments.ipynb) |
| Majority Fourier coefficients | [Fourier validation](tests/majority_block_fourier_validation.ipynb) |
| CartPole decoder and depth comparisons | [CartPole policies](tests/test_policy_training_efficiency.ipynb) |
| Random cubic graphs with neural encoding and RZZZ extensions | [Neural-embedding variance](tests/iqp_random_cubic_rzzz_neural_embedding_barren_plateau.ipynb) |
| Majority versus valid local readouts and MLPs | [Local-sector controls](tests/majority_local_sector_ablation.ipynb) |
| Five-qubit gradient sectors | [Five-qubit diagnostics](tests/majority_gradient_spectrum_during_training.ipynb) |
| Ten-qubit gradient sectors | [Ten-qubit diagnostics](tests/majority_gradient_spectrum_during_training_10q.ipynb) |
| Ten-qubit three-body CartPole extension | [RZZZ comparison](tests/cartpole_10q_majority_rzzz_ablation.ipynb) |
| Supplementary finite-shot comparison of the small models | [Finite-shot experiment](tests/test_policy_training_efficiency_finite_shots_log2_directions.ipynb) |
| Connected 21-qubit IQP versus tuned MLP sizes | [Connected IQP/MLPs](connected_iqp_mlp_size_study.ipynb) |
| Uniform-initialization follow-up | [Initialization comparison](connected_iqp_uniform_initialization.ipynb) |
| Analytical runtime comparison | [Training runtime](connected_iqp_runtime.ipynb) |

See the [experiment catalogue](docs/EXPERIMENTS.md) for corresponding data folders.
Notebooks retain saved outputs where available; the three connected-model reporting
notebooks display the archived results without retraining.

![Connected IQP and MLP learning curves](results/connected_iqp_mlp_study/paper_assets/connected_iqp_training_curves.png)

## Install and run

For viewing notebooks on GitHub, no installation is necessary. For execution:

```bash
git clone https://github.com/andre-sequeira10/iqp-policy-experiments.git
cd iqp-policy-experiments
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab
```

The requirements install QASP from its official repository at the recorded public
commit. No QASP package is bundled here. Historical runs used study-specific
versions documented with their results; the pinned public dependency is not a
claim of bitwise equivalence to every historical local checkout.

**Use a working copy before running training cells.** Some notebooks take substantial
time, write results, or resume existing runs. Reading saved outputs does not rerun
an experiment.

### Inspect results without installing QASP or training

```bash
python -m pip install -r requirements-analysis.txt
python tools/verify_archive.py
python tools/replot_connected.py
```

The plot utility checks 80 saved return arrays against their AUC/convergence records
and writes new figures under `generated/`, leaving archived data unchanged.
The integrity check requires only the Python standard library.

## What is recorded?

- `tests/results/`: available variance samples, decoder controls, gradient-sector
  data, three-body comparison results, figures, configurations and checkpoints.
- `results/connected_iqp_mlp_study/`: the paper's MLP development grid and connected
  IQP/test records, initial-policy audits and figure assets.
- `results/connected_iqp_uniform_initialization/`: the uniform-initialization
  follow-up, including its less favorable result.
- `results/iqp_simulator_optimization/full_training_inference_only/`: the accepted
  analytical runtime comparison.
- Root Python helpers implement the paper-specific connected circuit and analysis;
  they are not the QASP library.

Standalone raw records for the earliest variance/CartPole notebooks remain incomplete
or unaudited. Saved notebook outputs are retained. See
[reproduction details](docs/REPRODUCIBILITY.md) and [validation](docs/VALIDATION.md).

## Scope and interpretation

The current repository excludes unrelated model searches, disconnected-model runs,
connected-model SPSA studies removed from the manuscript, and superseded runtime
experiments. The small-model finite-shot study remains because it is reported in
the supplementary experiments. Shared summary files were filtered to remove
disconnected-only entries; retained numerical results were not recomputed.

The connected 21-qubit model is efficiently classically simulable. Its original
initialization gives normalized AUC 0.9319, versus 0.8680 for the 135-parameter MLP;
the uniform-initialization follow-up gives 0.7615. This is an initialization-sensitive,
quantum-inspired policy comparison, not a quantum computational advantage.
Policy-variance laws are initialization statements, not guarantees of RL convergence.

## Provenance and safe use

[REPOSITORY_PROVENANCE.json](REPOSITORY_PROVENANCE.json) records source provenance
and packaging changes. The root SHA-256 manifest inventories the current files;
nested manifests are historical and can refer to original paths/hashes.

PyTorch `.pt` checkpoints may contain pickle data. Load only trusted files.
The quick-start archive verifier and plotting utility do not deserialize checkpoints.

QASP is independently licensed by its maintainers. A separate license for this
experiment material has not yet been selected. Please cite the manuscript and
QASP and record the repository commit used. No archival DOI is claimed.

The prior, over-inclusive upload remains in Git history for recovery; QASP and
out-of-scope experiments are absent from the current tree.
