# Packaging validation — 2026-11-04

## Executed checks

- Parsed all published Python files successfully.
- Validated all 12 retained paper notebooks against the nbformat schema.
- Executed the three connected-model reporting notebooks against saved records,
  saving their rendered tables/figures without rerunning training.
- Parsed ordinary Python notebook cells; cells using notebook magics are excluded
  from plain Python parsing.
- Scanned file contents for common GitHub/OpenAI/AWS credential patterns and
  private-key headers: no matches. This is not a comprehensive security audit.
- Checked that no file approaches GitHub's 100 MiB regular-file limit.
- Recomputed the connected IQP and three displayed MLP curves from 80 saved
  test-seed return arrays. Each per-run AUC and convergence metric matched its JSON
  record; no models were retrained.
- Visually inspected the regenerated three-panel learning-curve figure.
- Smoke-tested reference versus optimized connected-IQP probabilities and parameter
  gradients for 3, 9 and 21 qubits: exact equality on the tested input/initialization.
- After extracting/renaming the helper modules, compared model state, outputs and
  parameter gradients against the previous committed implementation: exact equality
  on 3-, 9- and 21-qubit tests. Shared trainer definitions are AST-identical.
- Verified that all 1,444 retained numerical data/checkpoint files (NPZ, PT,
  Parquet and CSV) retain their previous SHA-256 hashes.
- Verified the final root SHA-256 inventory locally before publication.

Recomputed normalized AUCs:

| Condition | Mean AUC | Mean restricted episodes to criterion |
|---|---:|---:|
| Connected IQP, 139 parameters | 0.9319010 | 171.45 |
| MLP, 30 parameters | 0.8468635 | 263.65 |
| MLP, 86 parameters, fastest-development choice | 0.8590565 | 261.60 |
| MLP, 135 parameters | 0.8679550 | 236.50 |

The no-training plotting environment used Python 3.12, NumPy 2.5.4 and Matplotlib
3.11.2. Notebook validation used nbformat 5.11.1; model smoke tests used PyTorch
2.7.0 in an isolated environment. This is distinct from the historical training
environment. An initial attempt to use the author's existing environment failed
with duplicate OpenMP runtimes; that environment was not altered and the smoke
test was rerun in the isolated environment instead.

## Not established by these checks

No full training-notebook execution, complete multi-seed training rerun, full dependency
lock resolution, complete independent statistical audit, or guarantee of bitwise
cross-platform training reproduction is claimed. The original checkpoint files
were not deserialized during archive preparation. Full training with the external
public QASP dependency has not been rerun across the entire notebook suite.
