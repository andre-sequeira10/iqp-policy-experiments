# Paper experiment catalogue

Figure numbers can change; studies are mapped by content. Legacy filenames using
`ablation` refer to controlled comparisons in the manuscript.

| Paper study | Notebook | Recorded outputs |
|---|---|---|
| Rank, graph, decoder, majority and repeated-IQP variance laws | `tests/iqp_variance_experiments.ipynb` | Saved notebook outputs; complete standalone raw coverage not established |
| Majority Fourier coefficients | `tests/majority_block_fourier_validation.ipynb` | Matching folder under `tests/results/` |
| CartPole policy/depth comparisons | `tests/test_policy_training_efficiency.ipynb` | Saved notebook outputs; standalone raw coverage not fully audited |
| Random cubic neural embeddings and added RZZZ gates | `tests/iqp_random_cubic_rzzz_neural_embedding_barren_plateau.ipynb` | Matching `tests/results/` folder |
| Majority versus valid local readouts and MLPs | `tests/majority_local_sector_ablation.ipynb` | Matching folder; 120 run records, returns and checkpoints |
| Five-qubit gradient sectors | `tests/majority_gradient_spectrum_during_training.ipynb` | Matching folder |
| Ten-qubit gradient sectors | `tests/majority_gradient_spectrum_during_training_10q.ipynb` | Matching folder |
| Ten-qubit three-body CartPole extension | `tests/cartpole_10q_majority_rzzz_ablation.ipynb` | Matching folder |
| Supplementary small-model finite-shot comparison | `tests/test_policy_training_efficiency_finite_shots_log2_directions.ipynb` | Saved notebook outputs |
| Connected IQP versus tuned MLPs | `connected_iqp_mlp_size_study.ipynb` | `results/connected_iqp_mlp_study/` |
| Uniform-initialization follow-up | `connected_iqp_uniform_initialization.ipynb` | `results/connected_iqp_uniform_initialization/` |
| Analytical runtime comparison | `connected_iqp_runtime.ipynb` | `results/iqp_simulator_optimization/full_training_inference_only/` |

The uniform follow-up is exploratory but retained because it is reported in the
paper. MLP development records are retained because they document baseline selection.
Disconnected-only conditions and unrelated architecture searches are excluded.

The connected model requires small custom helpers: `connected_majority_iqp.py`,
`paper_blocks.py`, `paper_training.py`, `paper_statistics.py`, its study drivers,
and optimization/analysis utilities. These are experiment code, not QASP source.
