"""Legacy score/Fisher kernel over measured computational-basis bit strings.

QFKernel precomputes log-probability scores at fixed parameters; the newer
FidelityKernel and ProjectedKernel APIs live in quantum_kernels. This legacy
path enumerates all outcomes and requires nonzero probabilities for finite
scores; its normalize argument currently does not alter the output."""
import pennylane as qml
from pennylane import numpy as np
def _bits_to_int(bits):
    """Convert a bit sequence to its most-significant-bit-first integer basis-state index."""
    return int("".join(str(int(b)) for b in bits), 2)

class QFKernel:
    """Fixed-parameter Fisher score kernel s(x).T Freg^-1 s(y).

    The input objects are bit strings, not arbitrary continuous feature vectors.
    Construction computes the full probability Jacobian and Fisher matrix;
    zero probabilities can produce nonfinite scores and outcome enumeration
    scales exponentially in the measured qubit count."""
    def __init__(
        self,
        circuit: qml.QNode,
        params:         np.ndarray,
        *,
        fisher: str = "quantum",
        lambda_reg: float = 1e-2,
    ):
        """Precompute outcome scores and regularized Fisher geometry.

        circuit is a probability-returning QNode called positionally with params.
        fisher='quantum' selects quantum Fisher; other values select classical
        Fisher. lambda_reg adds a diagonal ridge before later linear solves.
        Scores are Jacobian rows divided by outcome probabilities, without clipping."""
        self.params = params

        # 1) Build F
        if fisher == "quantum":
            fim = qml.gradients.quantum_fisher(circuit)(params)
        else:
            fim = qml.gradients.classical_fisher(circuit)(params)

        # 2) Regularize + invert + symmetrize
        self.Freg = fim + lambda_reg * np.eye(fim.shape[0])
        #self.Finv = np.linalg.pinv(self.Freg)

        # 3) Precompute scores for every basis state
        jac = qml.jacobian(circuit)(params)  # shape (2ⁿ, …)
        probs = circuit(params)              # shape (2ⁿ,)
        self.scores = np.array([
            jac[i].flatten() / probs[i]
            for i in range(len(probs))
        ])

    def gram(self, X, Y=None, normalize: bool = True):
        """Return a NumPy score-kernel matrix for rows of binary X and optional Y.

        Y=None uses X on both axes. Each entry solves the regularized Fisher system
        and takes a score inner product. normalize is retained for compatibility but
        is not used; the output is not diagonal-normalized."""
        X = np.asarray(X, dtype=int)
        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y, dtype=int)

        m, n = len(X), len(Y)
        K = np.zeros((m, n))

        for i, x in enumerate(X):
            si = self.scores[_bits_to_int(x)]
            for j, y in enumerate(Y):
                sj = self.scores[_bits_to_int(y)]
                v = np.linalg.solve(self.Freg, sj)      # solves  Freg · v = sj  ⇒  v = Freg⁻¹ sj
                k = np.dot(si, v)                  # siᵀ · v = siᵀ Freg⁻¹ sj            
                K[i, j] = k 

        return K

    __call__ = gram