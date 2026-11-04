"""Quantum action-value functions for discrete-action reinforcement learning."""

from .drawing import draw_qfunction
from .models import PQDQN
from .observables import predefined_observables


def create_qfunction(config, n_qubits=None, n_actions=None):
    """Construct a PQDQN from a policy-style configuration dictionary."""
    return PQDQN.from_config(config, n_qubits=n_qubits, n_actions=n_actions)


__all__ = ["PQDQN", "create_qfunction", "draw_qfunction", "predefined_observables"]
