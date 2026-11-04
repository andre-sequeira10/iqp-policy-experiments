# Reproducibility guide

## Three different levels of reproduction

1. **Archive integrity:** `python tools/verify_archive.py` checks release hashes,
   required files, notebook structure, and absence of oversized GitHub files.
2. **Analysis reproduction:** `python tools/replot_connected.py` checks the saved
   20-seed return arrays against per-run metrics and redraws the main learning
   curves. It does not claim to re-estimate every bootstrap interval in the paper.
3. **Training reproduction:** requires the historical package environment, the
   particular study protocol and substantial computation. The full research suite
   has not been rerun as part of preparing this repository.

## Connected study

`connected_iqp_mlp_study.py` contains the model grid, training protocol, development
selection and test execution. The study uses development seeds 1001–1008 and test
seeds 2001–2020, 500 episodes and a 200-step limit. The strict stopping safeguard
requires a complete 100-episode window. The normalized episode AUC is
`sum(returns) / (200 * 500)`; unsolved runs receive 501 for the restricted number of
episodes to criterion. See `results/connected_iqp_mlp_study/protocol.json` and
`development_selection.json` for the recorded decisions.

Read-only summary:

```bash
python connected_iqp_mlp_study.py status --phase test
```

Example full test execution **in a separate working copy with a fresh results
directory and the preserved development selection**:

```bash
python connected_iqp_mlp_study.py run --phase test --models iqp_connected --jobs 1
```

Existing JSON files cause runs to be reused. Running the command in the published
archive is therefore not evidence that training was repeated. Do not delete or
overwrite the evidence archive to force a rerun; make a working copy first.

`analyze_connected_iqp_study.py analyze` is a historical comprehensive analysis
script. It includes disconnected controls, writes analysis artifacts and loads
trusted checkpoints; it is not the safe, read-only quick start. The new plotting
utility excludes disconnected controls from the manuscript comparison.

## Notebook execution

Launch Jupyter inside the repository. A first-cell bootstrap sets the root and
import path. QASP-side notebooks then write under `tests/results/`; connected-study
scripts use root-level `results/`. Full notebook execution can train many policies
and overwrite summaries. Review configuration cells and run from a working copy.

The cubic neural-embedding notebook supports environment variables
`QASP_BP_N_TOPOLOGIES`, `QASP_BP_SAMPLES_PER_TOPOLOGY`, `QASP_BP_BOOTSTRAPS`, and
`QASP_BP_RESULTS_NAME`. Reduced values are useful for smoke tests but do not
reproduce the paper's statistical precision. The recorded paper settings use
100 topologies and 20 parameter samples per topology.

## Historical environments and provenance

The root QASP source was copied from the manuscript's historical review snapshot,
not from today's upstream main. Run metadata records additional historical commits
and package versions. No single Git revision or package list has been established
as sufficient to bitwise reproduce every notebook in this multi-stage project.

`REPOSITORY_PROVENANCE.json` records source hashes and the initial publication-copy
hashes. Later packaging fixes are documented in `PACKAGING_CHANGES.md`. The root
`MANIFEST_SHA256.json` is the authoritative final file inventory. Nested experiment
manifests are historical and may refer to original absolute paths or old hashes.
Portable notebook setup and path normalization change notebook/source hashes, not
the scientific values stored in arrays.

Saved tables and arrays were copied, not regenerated. Some historical JSON files
contain `NaN`, which Python's JSON reader accepts but strict JSON consumers may not.
PyTorch checkpoints were copied without deserialization and may retain historical
path strings internally. Review trust before loading pickle-based files.

## Known limitations

- Standalone raw-data coverage for the earliest variance and policy-efficiency
  notebooks is incomplete or unaudited; saved outputs are preserved.
- No fully resolved, platform-independent environment lock is available.
- GPU/CPU libraries, precision, operating system, thread scheduling and library
  versions may affect trajectories and runtime.
- Notebook generators are historical authoring utilities, not a promise that every
  generator reruns unchanged on every machine.
- The archived experiment tree contains earlier and superseded studies; use the
  catalogue and protocols, not the most favorable run, to select results.
- No DOI or independent reproduction is claimed by this release.

See `VALIDATION.md` for the checks actually executed during packaging.
