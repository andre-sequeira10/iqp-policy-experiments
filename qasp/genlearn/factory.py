"""Dictionary-based construction of quantum circuit Born machines."""

from .models import PQCBM


def create_qcbm(config, n_qubits=None, data=None):
    """Build PQCBM from a copied config, optionally overriding width/data.

    Dimensions can live in config; explicit n_qubits must agree. Registered
    ansatz names/classes, feature maps, device options and loss settings follow
    the PQCBM constructor. No caller dictionary or CircuitBuilder is mutated.
    """
    return PQCBM.from_config(config, n_qubits=n_qubits, data=data)


__all__ = ["create_qcbm"]
