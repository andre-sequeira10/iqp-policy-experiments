# Experiment catalogue

Legacy filenames containing `ablation` are retained to preserve links to saved
records; the manuscript describes these as controlled comparisons. Figure numbers
can change between manuscript revisions, so studies are mapped by content.

## Manuscript studies

| Study | Entry point | Saved evidence |
|---|---|---|
| Exact rank/graph variance and decoder comparisons | `tests/iqp_variance_experiments.ipynb` | Notebook outputs; a complete independent raw-run archive has not been established |
| Majority Fourier coefficients | `tests/majority_block_fourier_validation.ipynb` | `tests/results/majority_block_fourier_validation/` |
| Original CartPole policy/depth comparisons | `tests/test_policy_training_efficiency.ipynb` | Notebook outputs; separate raw coverage not fully audited |
| Repeated-IQP/trainability investigations | `tests/trainability_born_paper.ipynb`, `tests/iqp_variance_experiments.ipynb` | Notebook outputs; inspect actual notebook settings |
| Random cubic graph neural embeddings and added RZZZ gates | `tests/iqp_random_cubic_rzzz_neural_embedding_barren_plateau.ipynb` | Matching `tests/results/` folder |
| Majority versus valid local readouts and MLPs | `tests/majority_local_sector_ablation.ipynb` | Matching folder: 120 run records plus returns/checkpoints |
| Five-qubit gradient sectors | `tests/majority_gradient_spectrum_during_training.ipynb` | Matching folder |
| Ten-qubit gradient sectors | `tests/majority_gradient_spectrum_during_training_10q.ipynb` | Matching folder |
| Ten-qubit three-body CartPole extension | `tests/cartpole_10q_majority_rzzz_ablation.ipynb` | Matching folder |
| Earlier finite-shot policy comparison | `tests/test_policy_training_efficiency_finite_shots_log2_directions.ipynb` | Notebook outputs; not the later connected-model SPSA study |
| Connected 21-qubit policy versus tuned MLP sizes | `connected_iqp_mlp_study.py`, `connected_iqp_mlp_size_study.ipynb` | `results/connected_iqp_mlp_study/` |
| Uniform-initialization follow-up | `connected_iqp_uniform_test.py` | `results/connected_iqp_uniform_initialization/` |
| Accepted analytical runtime optimization audit | `benchmark_optimized_iqp_training.py`, `test_optimized_iqp.py` | `results/iqp_simulator_optimization/full_training_inference_only/` |

The original and uniform connected-model experiments are different initialization
conditions. The uniform follow-up reused the examined test cohort and is exploratory.
The repeated-variance notebook inventory is preserved, but no claim is made that
every saved cell output has been regenerated with the bundled snapshot.

## Historical / exploratory material (not additional confirmatory evidence)

| Material | Entry points / data |
|---|---|
| Earlier architecture and initialization search | `majority_iqp_model_search.py`, `majority_iqp_model_search.ipynb`, `results/majority_iqp_model_search/` |
| Disconnected block controls | `majority_iqp_block_model.py`; retained records inside the connected study |
| Connected IQP finite-shot baseline | `connected_iqp_finite_shot_test.py`, `results/connected_iqp_finite_shots/` |
| SPSA K tuning | `connected_iqp_spsa_k_study.py`, `results/connected_iqp_spsa_k_study/` |
| Final finite-shot comparison | `finite_spsa_mlp_comparison.py`, `final_finite_iqp_report.py`, `results/finite_spsa_mlp_comparison/` |
| Backend timing experiments | `benchmark_connected_iqp_backend.py`, `results/connected_backend_benchmark/` |
| Other runtime variants, including superseded compiled-gradient work | `results/iqp_simulator_optimization/` (use only `full_training_inference_only/` for the accepted paper runtime comparison) |
| Cubic graphs without neural embeddings; all-to-all variance | `tests/iqp_random_cubic_rzzz_variance.ipynb`, `tests/iqp_majority_all_to_all_variance.ipynb` |
| Extra three-/four-body rotations and initialization variants | `tests/cartpole_10q_*` notebooks and matching data folders |
| Six-qubit even-majority control | `tests/cartpole_6q_even_majority_vs_parity.ipynb` and matching data |
| Older general policy training notebooks | `tests/test_policy_training.ipynb`, `tests/test_policy_training_efficiency_finite_shots.ipynb` |

Not all exploratory conditions completed the same number of seeds. Consult the
saved protocol and actual run files; a directory name is not evidence of completion.
Root-level `build_*` scripts are historical notebook/figure generators, not required
to use the already-generated notebooks. Inspect their output paths before use.
