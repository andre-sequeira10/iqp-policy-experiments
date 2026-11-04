"""Package the complete search and locked-validation results for inspection."""
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parent
nb=nbf.v4.new_notebook()
nb.metadata={'kernelspec':{'display_name':'torchlane','language':'python','name':'python3'},
             'language_info':{'name':'python'}}
nb.cells=[
    nbf.v4.new_markdown_cell(r'''# Neural-embedding majority IQP: development and fresh-seed validation

The encoder remains $\phi_i(s)=w_i^Ts+b_i$, with all four feature weights
independently initialized from $\mathrm{Unif}[0,2\pi)$ and trained. No
CartPole controller or feature-specific weights are inserted. The decoder is
binary majority on all measured qubits. Only the commuting circuit block,
its gate initialization, and the qubit count are varied.

The second locked model uses **21 qubits in seven disjoint entangled triples**.
For the partition $B_j=\{3j-2,3j-1,3j\}$, its circuit is
$$U(s)=H^{\otimes21}D(s)H^{\otimes21},$$
$$D(s)=\prod_{i=1}^{21}R_{Z_i}(w_i^Ts+b_i)
\prod_{j=1}^{7}\left[\prod_{\{i,k\}\subset B_j}R_{Z_iZ_k}(\beta_{ik})
R_{\prod_{i\in B_j}Z_i}(\gamma_j)\right].$$
Here $R_P(t)=e^{-itP/2}$. All gates inside $D$ commute. Initialize
$b_i=\pi/2$ and each $\beta_{ik},\gamma_j$ independently from
$\mathcal N(0,0.5^2)$. All **133 parameters** (84 feature weights,
21 biases, 21 pair angles and 7 triple angles) remain trainable.
The action probability is $\pi(0|s)=\Pr[|x|\leq10]$, and
$\pi(1|s)=1-\pi(0|s)$. This is global majority, not majority per block.
The affine encoder and binary majority are retained; the qubit count,
interaction block and gate initialization change. This initialization is
**not** the full-period uniform ensemble of the paper's variance theorem.

Each triple's four possible Hamming-weight probabilities are computed from
its exact eight-amplitude state. Convolving these seven count distributions
gives the exact global-majority policy. This model is therefore **efficiently
classically simulable**, despite its 21-qubit description. No sampling
hardness or quantum advantage is asserted.

The comparison uses the unchanged 65-parameter 4–9–2 tanh MLP, identical
REINFORCE settings, 500 episodes with 200 steps, and 100 frozen evaluation
episodes per validation seed. The user-approved freezing correction is
applied to **both** models: at least 100 completed episodes are required
before a trailing-100 mean of at least 195 can freeze learning. The check
occurs at five-episode optimizer-batch boundaries. The reported solve time
is the first full rolling-100 mean of at least 195 (checked each episode);
unsolved runs receive restricted episode 501. Neither metric measures wall-clock speed.

The first locked seven-qubit model failed fresh-seed validation under the
legacy freezing protocol; its results are retained below. Further development
used eight seeds and a corrected-protocol comparison of three finalists.
The second model was locked before examining **20 new matched seeds**.
Every scored candidate is retained. The two primary endpoints are normalized
training-return AUC (mean return divided by 200 over 500 episodes) and
restricted solve episode. We use paired-bootstrap intervals and two-sided
exact paired sign-randomization tests, Holm-adjusted across two endpoints
and multiplied by two to account conservatively for two locked comparisons.
Both endpoints must improve under this rule. This does not make the initial
search a preregistered study; it documents prospective locks within an adaptive search.
These results concern sample efficiency against this fixed baseline, not
wall-clock speed, classical hardness, or quantum advantage. The IQP model
has more trainable parameters than the baseline, so this is **not a
capacity-matched comparison**. The initialization audit separates starting
performance from trained performance.'''),
    nbf.v4.new_code_cell('''from pathlib import Path
import json
from IPython.display import display, Image
ROOT = Path.cwd()
RESULTS = ROOT / "results" / "majority_iqp_model_search"
lock = json.loads((RESULTS / "locked_model_round2.json").read_text())
lock'''),
    nbf.v4.new_markdown_cell('## Complete development record\nNo trial is discarded because of its score.'),
    nbf.v4.new_code_cell('''for phase in ["development_v1", "development_v2", "development_strict"]:
    print(phase)
    display(json.loads((RESULTS / (phase + "_summary.json")).read_text()))'''),
    nbf.v4.new_markdown_cell('## First locked validation: unsuccessful\nRetained in full; this result informed subsequent development. It used the old freezing rule and must not be pooled with the corrected-protocol experiment.'),
    nbf.v4.new_code_cell('''validation = json.loads((RESULTS / "validation_summary.json").read_text())
validation'''),
    nbf.v4.new_code_cell('display(Image(filename=str(RESULTS / "validation_comparison.png")))'),
    nbf.v4.new_markdown_cell('## Second locked validation: corrected protocol\nInterpret both primary endpoints and report terminal performance separately. Positive differences mean higher IQP AUC or fewer IQP episodes to solve.'),
    nbf.v4.new_code_cell('''validation_round2 = json.loads((RESULTS / "validation_round2_strict_summary.json").read_text())
validation_round2'''),
    nbf.v4.new_code_cell('''from IPython.display import Markdown
q = validation_round2["summary"][lock["model"]]
c = validation_round2["summary"]["mlp"]
rows = ["| Metric | Majority IQP | Classical MLP |", "|---|---:|---:|"]
for key, label in [("auc", "Normalized training AUC"),
                   ("restricted_solve_episode", "Episodes to rolling-100 mean ≥195"),
                   ("heldout", "Frozen held-out return")]:
    rows.append(f"| {label} | {q[key]['mean']:.3f} | {c[key]['mean']:.3f} |")
display(Markdown("\\n".join(rows)))
print("Both locked primary endpoints passed:", validation_round2["passes_locked_success_rule"])'''),
    nbf.v4.new_code_cell('display(Image(filename=str(RESULTS / "validation_round2_strict_comparison.png")))'),
    nbf.v4.new_markdown_cell('## Starting-policy audit\nThese evaluations were not used to select the model. An advantage at initialization contributes to training AUC and should not be attributed entirely to faster parameter optimization.'),
    nbf.v4.new_code_cell('''import numpy as np
audit = json.loads((RESULTS / "validation_round2_strict_initial_policy_audit.json").read_text())
for name in sorted({r["model"] for r in audit}):
    group = [r for r in audit if r["model"] == name]
    print(name, "initial mean:", np.mean([r["initial_mean_return"] for r in group]),
          "trained mean:", np.mean([r["trained_mean_return"] for r in group]))'''),
    nbf.v4.new_markdown_cell('## Numerical verification\nCircuit probabilities were checked against PennyLane and gradients against finite differences. The following check compares a complete optimizer batch against the unmodified QASP trainer.'),
    nbf.v4.new_code_cell('json.loads((RESULTS / "training_equivalence_check.json").read_text())'),
    nbf.v4.new_code_cell('json.loads((RESULTS / "strict_window_regression_checks.json").read_text())'),
    nbf.v4.new_code_cell('json.loads((RESULTS / "block_circuit_checks.json").read_text())'),
    nbf.v4.new_code_cell('''integrity = json.loads((RESULTS / "validation_round2_strict_integrity.json").read_text())
print("Audited runs:", integrity["runs_checked"])
integrity["versions"]'''),
    nbf.v4.new_markdown_cell('''## Reproduction

Use the `torchlane` environment and the scripts beside this notebook:

```bash
python majority_iqp_model_search.py preflight
python test_majority_iqp_training_equivalence.py
python majority_iqp_block_model.py
python test_strict_window_trainer.py
python majority_iqp_model_search.py run --phase validation_round2_strict --models iqpblocksbalanced_21_0.5 mlp --seeds 601 607 613 617 619 631 641 643 647 653 659 661 673 677 683 691 701 709 719 727 --jobs 3
python analyze_majority_iqp_search.py validation_round2_strict
python audit_majority_initial_policy.py validation_round2_strict --jobs 3 --episodes 100
python verify_majority_iqp_results.py
```

Completed runs are reused. Per-seed checkpoints contain both initial and
final tensors; raw training and evaluation returns are saved separately.
Historical development and the first locked validation remain in their
original directories; the corrected protocol uses the `strict` suffix.
The interrupted initial pilot directory `development` is excluded because
of a checkpoint-naming bug; scored runs use the corrected `development_v1`.
Do not treat development-seed performance as confirmatory evidence.''')]
nbf.write(nb,root/'majority_iqp_model_search.ipynb')
print(root/'majority_iqp_model_search.ipynb')
