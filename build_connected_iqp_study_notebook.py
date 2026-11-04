"""Package the locked experiment, including all MLP sizes and uncertainty."""
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parent
nb=nbf.v4.new_notebook()
nb.metadata={'kernelspec':{'display_name':'torchlane','language':'python','name':'python3'},'language_info':{'name':'python'}}
nb.cells=[nbf.v4.new_markdown_cell(r'''# Connected majority IQP and a tuned MLP size study

## Circuit: genuine inter-block connections

Keep 21 qubits, the trainable affine neural encoder $\phi_i(s)=w_i^Ts+b_i$,
the three $R_{ZZ}$ and one $R_{ZZZ}$ gate inside each triple, and global
binary majority over all 21 outcomes. Add the six trainable bridges
$$R_{Z_3Z_4}(\beta_1),\ R_{Z_6Z_7}(\beta_2),\ldots,
R_{Z_{18}Z_{19}}(\beta_6).$$
The full circuit remains $H^{\otimes21}D(s)H^{\otimes21}$ with commuting
diagonal gates in $D$. The graph is connected; there are **139 parameters**:
84 neural feature weights, 21 biases, 21 internal pair angles, seven
three-body angles, and six bridge angles. All are trainable.
All pre-existing parameters have exactly the disconnected model's
initialization. Bridge angles independently start from $\mathcal N(0,0.5^2)$.
The disconnected 133-parameter circuit is included as a paired control.

**Why this is not a product of triples.** Across any former block boundary,
all gates other than its bridge act entirely on one side of the cut and
commute with that bridge. They can therefore be moved after it without
changing Schmidt coefficients. Applied to the initial $|+\rangle$ state,
$$R_{ZZ}(\beta)=\cos(\beta/2)I\otimes I-i\sin(\beta/2)Z\otimes Z$$
gives two orthogonal product components. Thus the reduced-state purity is
$$\operatorname{Tr}\rho_{\rm left}^{2}
=\cos^4(\beta/2)+\sin^4(\beta/2)=1-\tfrac12\sin^2\beta.$$
It is below one unless $\beta$ is an integer multiple of $\pi$; this holds
independently of the encoded state. Initial and trained bridge purities are
reported below. A full 21-qubit state-vector reference also verifies the result.

**Still classically simulable.** The connected chain has bond dimension two
in a matrix-product-state representation (four in the doubled probability
network). Exact count-generating-function contraction and inverse FFT compute
global majority probabilities without factorizing the blocks. Complexity is
polynomial for this chain; connecting blocks is not a classical-hardness proof.
No Fourier truncation or product-state approximation is used.

## Prospective comparison and selection

MLPs have architecture Linear(4,h) → Tanh → Linear(h,2), with softmax actions.
Widths $h\in\{1,2,4,6,9,12,16,19\}$ give 9, 16, 30, 44, 65, 86, 114, 135
parameters. The largest is approximately matched to the connected IQP's 139.
Learning rates 0.003, 0.01, 0.03 and 0.1 are tested on **eight development
seeds per configuration**. The 0.1 rate was added during development, before
any final-test runs. For each width, lock the rate maximizing development AUC.
Also lock the best-AUC and fastest-convergence configurations from development.
The latter is tested additionally if distinct from the width-selected models.

All selected MLP sizes, the fixed connected IQP, and the disconnected control
are evaluated on **20 untouched test seeds**. No test-driven architecture or
learning-rate selection is permitted. Each run uses 500 training episodes,
200 steps maximum, the same corrected full-100-window freezing rule, the same
REINFORCE settings, and 100 independent frozen-policy evaluation episodes.
The IQP learning rate remains 0.03; this study does not search IQP bridge strength.

Primary outcomes: normalized training-return AUC and first episode reaching
a rolling-100 mean ≥195. Runs not reaching the criterion are assigned 501
and retained in the restricted mean. This measures sample efficiency, not
wall-clock efficiency. Paired exact sign-randomization tests receive one Holm
correction across every selected MLP's two primary IQP comparisons.

For saturation, nonsignificance alone is insufficient. Before test, practical
equivalence margins were set to 0.02 AUC and 25 restricted solve episodes.
Width 19 is compared with widths 9, 12 and 16 using conservative 98.3333%
paired-bootstrap intervals; both intervals must lie inside their margins.
Otherwise saturation is **not established**, even if the mean curve looks flat.

The search is finite and task-specific. Parameter counts do not match
expressivity or initialization priors; the IQP had prior architecture selection.
Neither superiority to this grid nor inter-block entanglement demonstrates
quantum advantage. All development scores, test failures and checkpoints remain
available. The paper is unchanged.
'''),nbf.v4.new_code_cell('''from pathlib import Path
import json
from IPython.display import display, Image, Markdown
ROOT=Path.cwd()/'results'/'connected_iqp_mlp_study'
r=json.loads((ROOT/'test_analysis.json').read_text())
print('Development-selected best AUC:',r['selection']['best_auc_mlp'])
print('Development-selected fastest convergence:',r['selection']['best_solve_mlp'])
print('Audited test runs:',r['audited_test_runs'])'''),
nbf.v4.new_markdown_cell('## Held-out size curve and selected comparisons'),
nbf.v4.new_code_cell('''lines=['| Model | Parameters | LR | AUC | Restricted solve episode | Solved | Held-out return | Initial return |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for name,s in r['summary'].items():
    lines.append(f"| {name} | {s['parameters']} | {s['lr']} | {s['auc']['mean']:.4f} | {s['solve_episode']['mean']:.1f} | {s['solved']}/20 | {s['heldout']['mean']:.2f} | {s['initial_return']:.2f} |")
display(Markdown('\\n'.join(lines)))'''),
nbf.v4.new_code_cell("display(Image(filename=str(ROOT/'mlp_size_scaling.png')))"),
nbf.v4.new_code_cell("display(Image(filename=str(ROOT/'locked_comparison.png')))"),
nbf.v4.new_markdown_cell('## Primary paired comparisons\nPositive differences favor connected IQP, including fewer episodes to the criterion. Confidence intervals are ordinary paired-bootstrap intervals; p-values are Holm-adjusted over all selected MLP comparisons and both primary endpoints.'),
nbf.v4.new_code_cell('''lines=['| MLP | AUC difference [95% CI] | Adjusted p | Episodes saved [95% CI] | Adjusted p |', '|---|---:|---:|---:|---:|']
for name,c in r['connected_minus_mlp'].items():
    a=c['auc'];t=c['solve_episode']
    lines.append(f"| {name} | {a['mean_difference']:.4f} [{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}] | {a['holm_p']:.4g} | {t['mean_difference']:.1f} [{t['ci95'][0]:.1f}, {t['ci95'][1]:.1f}] | {t['holm_p']:.4g} |")
display(Markdown('\\n'.join(lines)))'''),
nbf.v4.new_markdown_cell('## Is there evidence of saturation?'),
nbf.v4.new_code_cell("r['saturation']"),
nbf.v4.new_markdown_cell('## Effect of connecting the blocks\nThis paired control comparison is exploratory and separate from the primary MLP comparisons.'),
nbf.v4.new_code_cell("r['connected_minus_disconnected_exploratory']"),
nbf.v4.new_code_cell("r['interblock_cut_purities']"),
nbf.v4.new_markdown_cell('## Complete development grid'),
nbf.v4.new_code_cell("display(Image(filename=str(ROOT/'development_grid.png')))"),
nbf.v4.new_code_cell("r['selection']['development_means']"),
nbf.v4.new_markdown_cell('## Numerical and training verification'),
nbf.v4.new_code_cell("json.loads((ROOT/'circuit_verification.json').read_text())"),
nbf.v4.new_code_cell("json.loads((ROOT/'training_equivalence.json').read_text())"),
nbf.v4.new_markdown_cell('''## Reproduction

Use the `torchlane` environment and preserve the stage separation:

```bash
python connected_majority_iqp.py
python test_connected_study_trainer.py
python connected_iqp_mlp_study.py run --phase development --jobs 4
python connected_iqp_mlp_study.py lock
python connected_iqp_mlp_study.py run --phase test --jobs 4
python analyze_connected_iqp_study.py audit-initial --jobs 4
python analyze_connected_iqp_study.py analyze
```

Completed runs are reused, and a saved development-selection lock is not
overwritten. Raw per-episode returns, initial/final model checkpoints and
training logs are saved under the study results directory.''')]
nbf.write(nb,root/'connected_iqp_mlp_size_study.ipynb')
print(root/'connected_iqp_mlp_size_study.ipynb')
