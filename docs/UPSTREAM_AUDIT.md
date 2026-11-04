# Public QASP experiment audit

Inspected on 2026-11-04 using the GitHub recursive tree API. The response was not
truncated. Repository: https://github.com/INESCTEC/qasp

Main commit: `4aeec0ba2714676b86abe499c39acdb9b910a6de`.

## Related material already present

- `examples/cartpole_policies_and_variance.ipynb`
- `examples/iqp_simulator.ipynb`
- `examples/iqp_tn_simulator.py`
- IQP simulator, tensor-network, importance-sampling and parallel-backward tests.
- `tests/test_majority_modulo_sparse_projectors.py`
- Deucalion IQP barren-plateau and CartPole parity scripts.

## Dedicated paper studies missing from that tree

The dedicated notebooks for IQP variance, majority Fourier validation, local-sector
controls, gradient spectra, random cubic neural embeddings, and three-body CartPole
extensions were absent. The connected 21-qubit/MLP size study, its custom model and
analysis scripts, uniform follow-up, and research-workspace SPSA/runtime studies
were also absent as dedicated study implementations.

This is a file/tree audit, not a claim that the public examples contain no related
functionality. See `EXPERIMENTS.md` for the concrete local entry points collected
in this repository. The original QASP repository was not modified.
