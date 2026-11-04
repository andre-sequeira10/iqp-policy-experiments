# Changes made when preparing the repository

The original local QASP checkout and research workspace were not modified.

- Copied the dedicated experiment notebooks, research scripts and available saved
  result directories; excluded unrelated QASP application studies, caches, logs,
  LaTeX build intermediates, hidden files and manuscript source.
- Included the historical manuscript-review QASP Python source snapshot.
- Normalized author-machine paths in text files. Added notebook root setup and
  normalized notebook cell IDs to format 4.5. Scientific outputs were retained.
- Redirected the majority Fourier notebook/generator's figure export to its local
  results folder, rather than an external manuscript directory.
- Made the connected-study analysis source-audit path relative to the script.
- Replaced placeholder project URLs in historical package metadata.
- Added README, study catalogue, upstream audit, provenance and limitations.
- Added a standard-library archive verifier and a no-training plotting utility.
- Added a GitHub Actions integrity workflow. Local validation is recorded in
  `VALIDATION.md`; remote CI status should be checked on GitHub.
- Preserved QASP's upstream license notice separately. Experiment-specific
  licensing is undecided until the owner chooses it.

No model, loss, optimizer, seed schedule, experimental hyperparameter, numerical
array or checkpoint was deliberately changed. Historical nested manifests may
refer to pre-portability hashes; the root manifest inventories this release.
