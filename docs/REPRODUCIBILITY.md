# Reproduction guide

## External QASP dependency

Install `requirements.txt` from a separate environment. It obtains QASP from
https://github.com/INESCTEC/qasp at public commit
`4aeec0ba2714676b86abe499c39acdb9b910a6de`. No QASP package is included here.

Historical studies used different local QASP revisions and library versions.
`requirements-historical.txt` records one observed package configuration, not a
resolved lock for all experiments. The pinned public dependency and complete
training suite have not been jointly rerun across all notebooks.

## Reading and reproducing recorded outputs

- GitHub renders saved notebook outputs directly.
- `python tools/verify_archive.py` checks the root manifest without importing QASP.
- `python tools/replot_connected.py` reproduces learning curves from saved returns,
  checks AUCs and convergence metrics, and writes only to `generated/`.
- The connected, uniform-initialization and runtime reporting notebooks read saved
  records and figures; their displayed outputs are not new training runs.

Notebook root setup recognizes `REPOSITORY_PROVENANCE.json`; it no longer requires
a local QASP source directory. Install QASP before running quantum/training cells.

## Training

Make a working copy first: historical notebook cells can write to
`tests/results/`, and connected scripts use `results/`. Some scripts reuse existing
JSON records. Reusing a record is not a training reproduction.

For the connected study, inspect the protocol and development selection, then use:

```bash
python connected_iqp_mlp_study.py status --phase test
# In a fresh working results directory with its saved development selection:
python connected_iqp_mlp_study.py run --phase test --models iqp_connected --jobs 1
```

The paper comparison uses development seeds 1001–1008 and test seeds 2001–2020,
500 episodes, a 200-step limit, and a complete trailing 100-episode success window.
AUC is `sum(returns)/(200*500)`; unsolved runs receive 501 episodes to criterion.
The uniform follow-up reuses this examined test cohort and is exploratory.

The cubic neural-embedding notebook supports `QASP_BP_N_TOPOLOGIES`,
`QASP_BP_SAMPLES_PER_TOPOLOGY`, `QASP_BP_BOOTSTRAPS` and `QASP_BP_RESULTS_NAME`.
Reduced smoke-test settings do not reproduce the paper's statistical precision.

## Provenance, limitations and safety

The root manifest is authoritative for this version. Nested manifests and software
metadata document historical runs and may refer to earlier paths or source hashes.
Original scientific arrays and checkpoints were retained. Mixed connected-study
summaries were filtered to exclude the removed disconnected condition; primary
IQP-versus-MLP statistics were retained unchanged.

Separate raw-data coverage for early notebooks is incomplete or unaudited.
No platform-independent lock or bitwise full-training reproduction is claimed.
Hardware, library versions and numerical precision can affect trajectories.
Some historical JSON files contain NaN values accepted by Python but not strict
JSON readers. Load pickle-based checkpoints only if trusted.

See `PACKAGING_CHANGES.md` for the scope correction and `VALIDATION.md` for checks
actually performed. The original source repositories and manuscript data were
not deleted; excluded publication-copy files remain recoverable in Git history.
