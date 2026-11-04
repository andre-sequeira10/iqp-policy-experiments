"""Reinforcement-learning models, trainers, and Pauli simulators.

Exports are loaded lazily so the reusable Pauli-propagation engines can also be
used by classifiers without importing optional TorchRL and environment stacks.
"""

from importlib import import_module


_EXPORTS = {
    "policy_variance": (".metrics", "policy_variance"),
    "compute_DLA_generators": (".metrics", "compute_DLA_generators"),
    "fourier_analysis": (".metrics", "fourier_analysis"),
    "hadamard_decomposition": (".metrics", "hadamard_decomposition"),
    "cz_decomposition": (".metrics", "cz_decomposition"),
    "cnot_decomposition": (".metrics", "cnot_decomposition"),
    "_policy_expressivity": (".metrics", "_policy_expressivity"),
    "_sample_worker_batch": (".metrics", "_sample_worker_batch"),
    "create_model": (".model", "create_model"),
    "draw_model": (".model", "draw_model"),
    "PQCBorn": (".policies", "PQCBorn"),
    "PQDQN": (".qfunctions", "PQDQN"),
    "create_qfunction": (".qfunctions", "create_qfunction"),
    "draw_qfunction": (".qfunctions", "draw_qfunction"),
    "DeepQLearning": (".torchrl", "DeepQLearning"),
    "PQCSoftmax": (".policies", "PQCSoftmax"),
    "PQCBornVariance": (".policies", "PQCBornVariance"),
    "LearnablePQCBorn": (".policies", "LearnablePQCBorn"),
    "FastTruncatedPauliProp": (".pp_sim", "FastTruncatedPauliProp"),
    "DifferentiablePauliProp": (
        ".differentiable_pp_sim",
        "DifferentiablePauliProp",
    ),
    "PQCBornWrapper": (".torchrl", "PQCBornWrapper"),
    "REINFORCE": (".torchrl", "REINFORCE"),
    "BanditAWRConfig": (".torchrl", "BanditAWRConfig"),
    "ContextualBanditTrainer": (".torchrl", "ContextualBanditTrainer"),
    "WindyTargetNavigationEnv": (".envs", "WindyTargetNavigationEnv"),
    "PQCWrapper": (".trl", "PQCWrapper"),
    # Preserve the historical final binding from qasp.rl.trl.
    "TorchRLTrainer": (".trl", "TorchRLTrainer"),
}


def __getattr__(name):
    """Load and cache an RL export on first access, keeping optional trainer dependencies lazy."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
