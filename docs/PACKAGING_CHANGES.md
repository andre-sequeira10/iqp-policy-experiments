# Publication-copy scope correction

The initial upload included QASP source and exploratory studies. At the author's
request, the current tree contains only the paper experiments and recorded outputs.

Changes:
- Removed the bundled `qasp/` package, its packaging metadata and license copy.
  QASP is now an external dependency from INESCTEC/qasp.
- Removed unrelated model searches, disconnected-only runs, connected-model
  finite-shot/SPSA studies and superseded runtime variants.
- Retained the supplementary small-model finite-shot study, uniform follow-up,
  MLP development records and accepted runtime audit because they support the paper.
- Removed non-paper appended AngleAnsatz/navigation cells from the CartPole
  notebook and disconnected-only displays from the connected study notebook.
- Extracted unchanged shared trainer/statistical/block helper definitions into
  `paper_training.py`, `paper_statistics.py` and `paper_blocks.py`.
- Restricted connected-study drivers and analysis to connected IQP and MLPs.
- Updated source-path audits to use the installed QASP module, and notebook root
  checks to use the experiment-repository marker.
- Filtered disconnected-only entries from mixed summaries, without changing the
  retained primary comparison values. Scientific arrays/checkpoints were not altered.
- Added reporting notebooks for the already-recorded uniform and runtime studies.
- Updated README, catalogue, requirements, provenance and integrity inventory.

The original local research folders were not modified. Removed publication-copy
files are backed up locally and remain recoverable from the previous Git commit.
No Git history rewrite or force push was performed.
