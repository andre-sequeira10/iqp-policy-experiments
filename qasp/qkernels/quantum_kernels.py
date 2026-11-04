"""Fidelity and projected quantum kernels built on QASP circuit components.

High-level quantum kernel classes built **on top of the QASP circuit
infrastructure** (:class:`qasp.circuits.constructor.CircuitBuilder` and the
ansatz / feature-map templates).  Each kernel assembles its circuit through the
same ``set_ansatz`` -> ``set_measurement`` -> ``build``/``build_kernel`` ->
``compile`` -> ``create_torchlayer`` pipeline used by the rest of the package,
so device selection, distributed (MPI / ``lightning.kokkos``) execution, the
differentiation method and the parameter-initialisation strategies all behave
exactly as in the reinforcement-learning and VQC modules.

Two kernels are provided:

1. ``FidelityKernel`` -- global or localized overlap fidelity. By default it
   measures the global all-zero projector,

       k(x, y) = |<0| U(x)^dagger U(y) |0>|^2,

   A user can instead provide a list of zero-state projectors. A label such as
   ``"P 0 2"`` denotes

       P_{0,2} = |00><00|_{0,2} tensor I_{rest}.

   On the overlap state ``|phi> = U(x')^dagger U(x)|0>`` the kernel averages
   the probability that each selected subsystem is in its all-zero state:

       k(x', x) = (1/K) sum_l <phi| P_{s_l} |phi>.

   A projector over every qubit recovers global fidelity, while smaller
   subsets localize it. The self-similarity ``k(x, x) = 1`` always holds.

2. ``ProjectedKernel`` -- the projected quantum kernel of Huang et al.

       k(x, y) = exp( -gamma * sum_l w_l ( f_l(x) - f_l(y) )^2 ),

   with single-state features ``f_l(x) = <0|U(x)^dagger O_l U(x)|0>``.

Encoding circuit ``U``
----------------------
``U(x)`` is the QASP feature map applied ``reps`` times (wrapped in a parameter
free :class:`EncodingAnsatz`).  If an ``ansatz`` is supplied the encoding becomes
the *trainable* circuit ``Ansatz(FeatureMap(x); theta)`` -- the angles live in the
``TorchLayer`` produced by ``create_torchlayer`` and can be optimised, e.g. by
kernel-target alignment (see :mod:`qasp.qkernels.kta`)."""

from typing import Optional, Sequence, List

import numpy as _np
import pennylane as qml
from pennylane import numpy as np
import torch

from qasp.circuits.constructor import CircuitBuilder
from qasp.feature_maps import NeuralEmbedding


# ---------------------------------------------------------------------------
# Feature-map / ansatz resolution helpers
# ---------------------------------------------------------------------------
def _resolve_feature_map(feature_map, feature_map_config, n_qubits):
    """Return an instantiated feature-map object from instance/class/string."""
    if hasattr(feature_map, "name") and not isinstance(feature_map, type):
        if getattr(feature_map, "n_qubits", n_qubits) != n_qubits:
            raise ValueError("feature_map.n_qubits must match n_qubits.")
        return feature_map

    cfg = dict(feature_map_config or {})
    cfg.setdefault("n_qubits", n_qubits)
    if cfg["n_qubits"] != n_qubits:
        raise ValueError("feature_map_config['n_qubits'] must match n_qubits.")

    if isinstance(feature_map, str):
        if feature_map == "AngleEmbedding":
            from qasp.feature_maps.templates import AngleEmbedding
            feature_map = AngleEmbedding
        elif feature_map == "NeuralEmbedding":
            from qasp.feature_maps.templates import NeuralEmbedding
            feature_map = NeuralEmbedding
        else:
            raise ValueError(f"Unknown feature map '{feature_map}'.")

    if cfg.get("features") is None and isinstance(cfg.get("rotations"), str):
        cfg["features"] = n_qubits

    return feature_map(**cfg)


def _resolve_ansatz(ansatz, ansatz_config, n_qubits, feature_map):
    """Instantiate a variational ansatz bound to ``feature_map``."""
    if ansatz is None:
        return None
    if not isinstance(ansatz, (str, type)):
        return ansatz
    if isinstance(ansatz, str):
        from qasp.circuits.templates import IQPAnsatz, AngleAnsatz
        ansatz_classes = {"IQPAnsatz": IQPAnsatz, "AngleAnsatz": AngleAnsatz}
        if ansatz not in ansatz_classes:
            raise ValueError(
                f"Unknown ansatz '{ansatz}'. Choose from {sorted(ansatz_classes)}."
            )
        ansatz = ansatz_classes[ansatz]
    cfg = dict(ansatz_config or {})
    cfg.setdefault("n_qubits", n_qubits)
    if cfg["n_qubits"] != n_qubits:
        raise ValueError("ansatz_config['n_qubits'] must match n_qubits.")
    return ansatz(**cfg, feature_map=feature_map)


class EncodingAnsatz:
    """Parameter-free ``ansatz`` that only applies a feature map ``reps`` times.

    This lets :class:`CircuitBuilder` treat a plain feature-map encoding exactly
    like a variational ansatz (same ``__call__(inputs, input_scaling_params,
    params)`` interface), which is what the non-trainable kernels use.
    """

    def __init__(self, n_qubits, feature_map=None, reps=1):
        """Store a feature map and repetition count for an encoding-only circuit with no variational parameter block."""
        self.n_qubits = n_qubits
        self.feature_map = feature_map
        self.reps = reps
        self.n_layers = 0
        self.z_gates = None
        self.parity_gates = None
        self.CNOT_Gates = None
        self._block_size = 0

    def __call__(self, inputs=None, input_scaling_params=None, params=None):
        """Queue the feature map reps times, forwarding fixed input scales when enabled; params is unused."""
        for _ in range(self.reps):
            if self.feature_map is not None and self.feature_map.input_scaling:
                self.feature_map(inputs, input_scaling_params=input_scaling_params)
            else:
                self.feature_map(inputs)

    def get_param_shapes(self):
        # no variational parameters
        """Return [(0,)] to satisfy CircuitBuilder without allocating variational parameters."""
        return [(0,)]


# ---------------------------------------------------------------------------
# Observable helpers
# ---------------------------------------------------------------------------
def parse_pauli_observables(labels: Sequence[str]) -> List[qml.operation.Operator]:
    """Parse Pauli-string labels into PennyLane observables (mirrors the
    parsing used by :class:`CircuitBuilder`)."""
    obs = []
    for item in labels:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("Observable labels must be non-empty strings.")
        signal = 1.0
        parts = item.split()
        pauli_string = parts[0]
        if pauli_string.startswith("-"):
            signal = -1.0
            pauli_string = pauli_string[1:]
        wire_indices = list(map(int, parts[1:]))
        if not pauli_string or set(pauli_string) - set("IXYZ"):
            raise ValueError(f"Invalid Pauli word in observable '{item}'.")
        if len(pauli_string) != len(wire_indices):
            raise ValueError(
                f"Observable '{item}' needs one wire per Pauli character."
            )
        if len(set(wire_indices)) != len(wire_indices):
            raise ValueError(f"Observable '{item}' contains duplicate wires.")
        wire_map = {wire: idx for idx, wire in enumerate(wire_indices)}
        pauli_word = qml.pauli.string_to_pauli_word(pauli_string, wire_map=wire_map)
        obs.append(signal * pauli_word)
    return obs


def default_local_observables(n_qubits: int, paulis: str = "XYZ") -> List[str]:
    """Per-qubit Pauli labels, e.g. ``["X 0", "Y 0", "Z 0", ...]``."""
    return [f"{p} {q}" for q in range(n_qubits) for p in paulis]


def default_z_observables(n_qubits: int) -> List[str]:
    """Per-qubit single-Z labels ``["Z 0", "Z 1", ...]``."""
    return [f"Z {q}" for q in range(n_qubits)]


def default_fidelity_projectors(n_qubits: int) -> List[str]:
    """The global all-zero projector label for ``n_qubits`` wires."""
    return ["P " + " ".join(str(q) for q in range(n_qubits))]


def _projector_wire_subsets(
    labels: Sequence[str], n_qubits: int
) -> List[List[int]]:
    """Validate ``P`` labels and return their projected wire subsets."""
    if not labels:
        raise ValueError("FidelityKernel needs at least one projector.")

    subsets = []
    for item in labels:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("Projector labels must be non-empty strings.")
        parts = item.split()
        if parts[0] != "P":
            raise ValueError(
                "FidelityKernel projectors must use labels such as "
                f"'P 0' or 'P 0 2'; got '{item}'."
            )
        if len(parts) == 1:
            raise ValueError(f"Projector '{item}' must contain at least one wire.")
        try:
            wires = [int(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"Projector '{item}' has a non-integer wire.") from exc
        if len(set(wires)) != len(wires):
            raise ValueError(f"Projector '{item}' contains duplicate wires.")
        if any(wire < 0 or wire >= n_qubits for wire in wires):
            raise ValueError(f"Projector '{item}' contains an out-of-range wire.")
        subsets.append(wires)
    return subsets


def parse_fidelity_projectors(
    labels: Sequence[str], n_qubits: int
) -> List[qml.operation.Operator]:
    """Parse zero-projector labels into PennyLane projectors.

    For example, ``"P 0 2"`` becomes
    ``|00><00|`` on wires ``[0, 2]`` with the identity on all other wires.
    Leading and repeated whitespace is accepted.
    """
    return [
        qml.Projector([0] * len(wires), wires=wires)
        for wires in _projector_wire_subsets(labels, n_qubits)
    ]


# ---------------------------------------------------------------------------
# Base class -- wires everything through CircuitBuilder
# ---------------------------------------------------------------------------
class _BaseQuantumKernel:
    """Common machinery: resolves the encoding, then assembles / compiles the
    circuit through :class:`CircuitBuilder`.

    Args:
        feature_map / feature_map_config: the QASP feature map.
        n_qubits (int): number of qubits.
        reps (int): feature-map re-uploading depth (non-trainable encoding).
        ansatz / ansatz_config: optional variational ansatz -> trainable kernel.
        init (str): TorchLayer angle-initialisation strategy (trainable only).
        input_scaling_params (array, optional): fixed scaling params for
            input-scaling feature maps in the non-trainable case.
        device (str), diff_method (str), shots (int), mpi (bool): forwarded to
            :meth:`CircuitBuilder.compile` exactly as in the RL / VQC modules.
        n_ancillas (int): forwarded to :class:`CircuitBuilder`.
    """

    def __init__(
        self,
        feature_map=None,
        feature_map_config: Optional[dict] = None,
        n_qubits: int = 1,
        reps: int = 1,
        ansatz=None,
        ansatz_config: Optional[dict] = None,
        init="random_-pi_pi",
        input_scaling_params=None,
        device: str = "default.qubit",
        diff_method: str = "best",
        shots: Optional[int] = None,
        mpi: bool = False,
        n_ancillas: int = 0,
        device_kwargs: Optional[dict] = None,
        input_scaling_init=None,
        cutting=None,
    ):
        """Resolve common kernel components without yet assembling measurement QNodes.

        feature_map/feature_map_config define input encoding; n_qubits sets register
        size and reps repeats an encoding-only circuit. Supplying ansatz plus its
        config creates trainable parameters initialized by init. input_scaling_init
        independently initializes scales; trainable models default to init.
        Encoding-only scaled maps accept fixed input_scaling_params or sample
        fixed scales once with input_scaling_init. Explicit scale arrays win.
        Device, shots,
        diff_method, mpi, device_kwargs and n_ancillas pass to CircuitBuilder.
        The input feature count is taken from the map, falling back to n_qubits."""
        self.cutting = cutting
        if feature_map is None:
            raise ValueError("A feature_map must be provided to build the kernel.")
        if not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError("n_qubits must be a positive integer.")
        if not isinstance(reps, int) or reps < 1:
            raise ValueError("reps must be a positive integer.")
        self.n_qubits = n_qubits
        self.reps = reps
        self.fm = _resolve_feature_map(feature_map, feature_map_config, n_qubits)
        self.fm_input_scaling = bool(getattr(self.fm, "input_scaling", False))

        self.trainable = ansatz is not None
        if self.trainable:
            self.ansatz = _resolve_ansatz(ansatz, ansatz_config, n_qubits, self.fm)
        else:
            self.ansatz = EncodingAnsatz(n_qubits, feature_map=self.fm, reps=reps)

        self.init = init
        self.input_scaling_init = input_scaling_init
        self.device = device
        self.diff_method = diff_method
        self.shots = shots
        self.mpi = mpi
        self.device_kwargs = device_kwargs
        self.n_ancillas = n_ancillas
        self._fixed_isp = (
            None if input_scaling_params is None
            else _np.asarray(input_scaling_params, dtype=float)
        )
        if not self.fm_input_scaling and input_scaling_init is not None:
            raise ValueError("input_scaling_init requires an input-scaling feature map.")
        if self.fm_input_scaling and not self.trainable:
            # EncodingAnsatz shares the same scale block across all repetitions.
            count = (self.fm.features if getattr(self.fm, "features", None) is not None
                     else n_qubits)
            if isinstance(self.fm, NeuralEmbedding):
                count *= (n_qubits if isinstance(self.fm.rotations, str)
                          else len(self.fm.rotations))
            if self._fixed_isp is None and input_scaling_init is not None:
                from qasp.initialization import sample_parameters

                self._fixed_isp = sample_parameters((count,), input_scaling_init)
            if self._fixed_isp is not None and (
                self._fixed_isp.shape != (count,) or not _np.isfinite(self._fixed_isp).all()
            ):
                raise ValueError(f"input_scaling_params must have shape {(count,)} and finite values.")
        if self.fm_input_scaling and not self.trainable and self._fixed_isp is None:
            raise ValueError(
                "input_scaling_params are required for a non-trainable "
                "input-scaling feature map."
            )

        self.builder = None
        self.qnode = None
        self.qlayer = None
        self.nfeat = (self.fm.features
                      if getattr(self.fm, "features", None) is not None
                      else n_qubits)

    def _data(self, values, name="X"):
        """Convert kernel inputs to a finite two-dimensional float array."""
        data = _np.asarray(values, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[1] != self.nfeat:
            raise ValueError(
                f"{name} must have shape (n_samples, {self.nfeat}); "
                f"got {data.shape}."
            )
        if not _np.isfinite(data).all():
            raise ValueError(f"{name} contains non-finite values.")
        return data

    # ------------------------------------------------------------------
    def _assemble(self, meas_type, observables=None, overlap=True):
        """Build an overlap or state circuit, configure measurements, compile its QNode and allocate a TorchLayer for trainable kernels."""
        self.builder = CircuitBuilder(n_qubits=self.n_qubits,
                                      n_ancillas=self.n_ancillas)
        self.builder.set_ansatz(self.ansatz)
        if meas_type == "expectation":
            self.builder.set_measurement("expectation",
                                         wires=range(self.n_qubits),
                                         observables=observables)
        else:
            self.builder.set_measurement(meas_type, wires=range(self.n_qubits))

        if overlap:
            self.builder.build_kernel(for_torchlayer=self.trainable)
        else:
            self.builder.build()

        self.qnode = self.builder.compile(
            device=self.device, diff_method=self.diff_method,
            shots=self.shots, mpi=self.mpi,
            device_kwargs=self.device_kwargs,
            cutting=self.cutting,
        )
        if self.trainable:
            self.qlayer = self.builder.create_torchlayer(
                initialization=self.init, input_scaling_init=self.input_scaling_init,
                input_scaling_params=self._fixed_isp,
            )

    # -- trainable parameters ----------------------------------------------
    def parameters(self):
        """Trainable angle tensors (the TorchLayer weights), or ``[]``."""
        if self.trainable and self.qlayer is not None:
            return list(self.qlayer.parameters())
        return []

    # -- low-level evaluation ----------------------------------------------
    def _x_tensor(self, x):
        """Convert a numeric feature vector through NumPy to a CPU float32 tensor; this is not an input-gradient-preserving conversion."""
        return torch.as_tensor(_np.asarray(x, dtype=float), dtype=torch.float32)

    def _overlap_out(self, a, b):
        """Run the overlap circuit on encodings ``(a, b)`` and return the raw
        measurement (torch tensor if trainable, else numpy/autograd)."""
        if self.trainable:
            concat = torch.cat([self._x_tensor(a), self._x_tensor(b)])
            return self.qlayer(concat)
        stacked = _np.stack([_np.asarray(a, dtype=float),
                             _np.asarray(b, dtype=float)])
        if self.fm_input_scaling:
            return self.qnode(stacked, input_scaling_params=self._fixed_isp)
        return self.qnode(stacked)

    def _state_out(self, x):
        """Run the single-state circuit on encoding ``x`` (projected kernel)."""
        if self.trainable:
            return self.qlayer(self._x_tensor(x))
        if self.fm_input_scaling:
            return self.qnode(_np.asarray(x, dtype=float),
                              input_scaling_params=self._fixed_isp)
        return self.qnode(_np.asarray(x, dtype=float))

    # -- drawing ------------------------------------------------------------
    def _draw_overlap(self, x, y, style, decimals):
        """Render the paired-input overlap circuit with current weights; generate random example inputs when omitted."""
        import matplotlib.pyplot as plt  # noqa: F401
        if x is None:
            x = np.random.random(self.nfeat)
        if y is None:
            y = np.random.random(self.nfeat)
        if self.trainable:
            concat = torch.cat([self._x_tensor(x), self._x_tensor(y)])
            params = self.qlayer.state_dict()["params"]
            if self.fm_input_scaling:
                isp = self.qlayer.state_dict()["input_scaling_params"]
                return qml.draw_mpl(self.qnode, decimals=decimals, style=style)(
                    concat, params=params, input_scaling_params=isp)
            return qml.draw_mpl(self.qnode, decimals=decimals, style=style)(
                concat, params=params)
        stacked = _np.stack([_np.asarray(x, dtype=float),
                             _np.asarray(y, dtype=float)])
        if self.fm_input_scaling:
            return qml.draw_mpl(self.qnode, decimals=decimals, style=style)(
                stacked, input_scaling_params=self._fixed_isp)
        return qml.draw_mpl(self.qnode, decimals=decimals, style=style)(stacked)

    def _draw_state(self, x, style, decimals):
        """Render the single-input feature circuit with current weights and optional random example input."""
        import matplotlib.pyplot as plt  # noqa: F401
        if x is None:
            x = np.random.random(self.nfeat)
        if self.trainable:
            params = self.qlayer.state_dict()["params"]
            if self.fm_input_scaling:
                isp = self.qlayer.state_dict()["input_scaling_params"]
                return qml.draw_mpl(self.qnode, decimals=decimals, style=style)(
                    self._x_tensor(x), params=params, input_scaling_params=isp)
            return qml.draw_mpl(self.qnode, decimals=decimals, style=style)(
                self._x_tensor(x), params=params)
        if self.fm_input_scaling:
            return qml.draw_mpl(self.qnode, decimals=decimals, style=style)(
                _np.asarray(x, dtype=float), input_scaling_params=self._fixed_isp)
        return qml.draw_mpl(self.qnode, decimals=decimals, style=style)(
            _np.asarray(x, dtype=float))

    def gram(self, X, Y=None, **kwargs):
        """Abstract Gram-matrix interface implemented by FidelityKernel and ProjectedKernel."""
        raise NotImplementedError

    __call__ = gram


def _to_np(out):
    """Convert outputs to NumPy, explicitly detaching Torch tensors and moving them to CPU."""
    if isinstance(out, torch.Tensor):
        return out.detach().cpu().numpy()
    return _np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# 1) Fidelity kernel  (global or localized overlap)
# ---------------------------------------------------------------------------
class FidelityKernel(_BaseQuantumKernel):
    r"""Global or localized fidelity from zero-state projectors.

    On ``|phi> = U(y)^dagger U(x)|0>`` and projected subsets ``s_l``,

        k(x, y) = (1/K) sum_l <phi|
            (|0><0|^{s_l} tensor I_{complement(s_l)}) |phi>.

    With no ``projectors`` argument, the single projector contains every
    qubit and the result is the global fidelity
    ``|<0|U(y)^dagger U(x)|0>|^2``. For example, on two qubits,
    ``projectors=["P 0", "P 1"]`` computes

        1/2 * (<phi|(|0><0| tensor I)|phi>
             + <phi|(I tensor |0><0|)|phi>).

    A label ``"P 0 2"`` jointly projects qubits 0 and 2 onto ``|00>``.
    Self-similarity is one. Averages involving proper subsets are localized
    similarities and are not guaranteed to be PSD for arbitrary encodings.

    Args:
        projectors (list[str], optional): zero-projector labels such as
            ``["P 0", "P 1"]`` or ``["P 0 1", "P 0 2"]``. If omitted,
            a single projector over all qubits is used.
    """

    def __init__(self, projectors=None, **kwargs):
        """Build an overlap circuit with zero-state projector measurements.

        projectors accepts the labels understood by parse_fidelity_projectors;
        None uses default_fidelity_projectors(n_qubits). A pairwise kernel value is
        the mean over these projectors, so local projectors are not automatically the
        full global fidelity. Common encoding/device options pass to the base class."""
        super().__init__(**kwargs)
        labels = (
            default_fidelity_projectors(self.n_qubits)
            if projectors is None
            else projectors
        )
        self.projector_labels = [item.strip() for item in labels]
        self.projector_wires = _projector_wire_subsets(
            self.projector_labels, self.n_qubits
        )
        self.projectors = parse_fidelity_projectors(
            self.projector_labels, self.n_qubits
        )
        self.n_projectors = len(self.projectors)
        self._assemble("expectation", observables=self.projectors, overlap=True)

    def kernel(self, x, y) -> float:
        """Return the Python-float mean projector overlap for feature vectors x and y; this convenience path detaches gradients."""
        return float(_to_np(self._overlap_out(x, y)).mean())

    def gram(self, X, Y=None, normalize: bool = False, return_tensor: bool = False):
        """Return the (len(X), len(Y)) overlap Gram matrix, or a square matrix if Y=None.

        Rows of X/Y are feature vectors. normalize=True divides by the square root
        of self-kernel products. return_tensor=True preserves circuit-parameter
        gradients only for trainable kernels; otherwise a NumPy array is returned.
        Input conversion does not promise gradients with respect to X/Y."""
        if self.trainable and return_tensor:
            return self._gram_torch(X, Y, normalize)
        return self._gram_numpy(X, Y, normalize)

    def _gram_numpy(self, X, Y, normalize):
        """Evaluate pairwise overlaps into a detached NumPy matrix, exploiting symmetry when Y is omitted and optionally normalizing the diagonal."""
        X = self._data(X)
        symmetric = Y is None
        Y = X if symmetric else self._data(Y, "Y")
        m, n = len(X), len(Y)
        K = _np.zeros((m, n))
        if symmetric:
            for i in range(m):
                K[i, i] = self.kernel(X[i], X[i])
                for j in range(i + 1, m):
                    K[i, j] = K[j, i] = self.kernel(X[i], X[j])
        else:
            for i in range(m):
                for j in range(n):
                    K[i, j] = self.kernel(X[i], Y[j])
        if normalize:
            dx = _np.array([self.kernel(x, x) for x in X])
            dy = dx if symmetric else _np.array([self.kernel(y, y) for y in Y])
            denom = _np.sqrt(_np.outer(dx, dy))
            denom[denom == 0] = 1.0
            K = K / denom
        return K

    def _gram_torch(self, X, Y, normalize):
        """Evaluate differentiable trainable-kernel pairs, symmetrize square matrices and optionally normalize self-overlaps."""
        X = self._data(X)
        symmetric = Y is None
        Y = X if symmetric else self._data(Y, "Y")

        def ov(a, b):
            """Average the projector outputs of one paired-input overlap without detaching its graph."""
            return self._overlap_out(a, b).mean()

        rows = [torch.stack([ov(X[i], Y[j]) for j in range(len(Y))])
                for i in range(len(X))]
        K = torch.stack(rows)
        if symmetric:
            K = 0.5 * (K + K.T)
        if normalize:
            dx = torch.stack([ov(x, x) for x in X])
            dy = dx if symmetric else torch.stack([ov(y, y) for y in Y])
            denom = torch.sqrt(torch.outer(dx, dy)).clamp_min(1e-12)
            K = K / denom
        return K

    __call__ = gram

    def draw(self, x=None, y=None, style: str = "pennylane", decimals: int = 2):
        """Return PennyLane Matplotlib figure/axes for the overlap circuit; optional x/y select inputs, style and decimals control presentation."""
        return self._draw_overlap(x, y, style, decimals)


# ---------------------------------------------------------------------------
# 2) Projected quantum kernel (Huang et al.)
# ---------------------------------------------------------------------------
class ProjectedKernel(_BaseQuantumKernel):
    """Projected quantum kernel with exponential post-processing
    ``k(x, y) = exp( -gamma * sum_l w_l (f_l(x) - f_l(y))^2 )``."""

    def __init__(self, gamma: float = 1.0, observables=None, weights=None,
                 **kwargs):
        """Construct an RBF kernel on measured quantum expectation features.

        gamma must be finite/nonnegative. observables defaults to local X, Y and Z
        on every data wire. weights supplies one finite nonnegative distance weight
        per observable (default ones). Other options configure the shared feature
        map, ansatz and device. The kernel is exp(-gamma * weighted squared distance)."""
        super().__init__(**kwargs)
        if observables is None:
            observables = default_local_observables(self.n_qubits, "XYZ")
        if not observables:
            raise ValueError("ProjectedKernel needs at least one observable.")
        self.observable_labels = list(observables)
        self.gamma = float(gamma)
        if not _np.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("gamma must be finite and non-negative.")
        if weights is None:
            weights = _np.ones(len(self.observable_labels))
        self.weights = _np.asarray(weights, dtype=float).reshape(-1)
        if len(self.weights) != len(self.observable_labels):
            raise ValueError("weights must match the number of observables.")
        if not _np.isfinite(self.weights).all() or (self.weights < 0).any():
            raise ValueError("weights must be finite and non-negative.")
        self._assemble("expectation", observables=self.observable_labels,
                       overlap=False)

    # -- features -----------------------------------------------------------
    def features(self, X) -> _np.ndarray:
        """Return a detached NumPy array of shape (samples, observables); input rows are feature vectors and trainable circuits use current weights."""
        X = self._data(X)
        if self.trainable:
            return self._features_torch(X).detach().cpu().numpy()
        F = _np.array([_to_np(self._state_out(x))
                       for x in X])
        if F.ndim == 1:
            F = F.reshape(len(X), -1)
        return F

    def _features_torch(self, X) -> torch.Tensor:
        """Stack per-sample expectation features as double-precision tensors with parameter gradients retained."""
        rows = []
        for x in self._data(X):
            o = self._state_out(x)
            o = (torch.stack([torch.as_tensor(v) for v in o])
                 if isinstance(o, (list, tuple)) else torch.as_tensor(o).flatten())
            rows.append(o.double())
        return torch.stack(rows)

    def gram(self, X, Y=None, return_tensor: bool = False):
        """Return the weighted-RBF Gram matrix between rows of X and Y.

        With Y=None reuse the X feature table and symmetrize the square matrix.
        return_tensor=True retains circuit-parameter gradients for trainable
        kernels; all other paths return NumPy arrays. Tiny negative squared
        distances from roundoff are clipped to zero before exponentiation."""
        symmetric = Y is None
        if self.trainable and return_tensor:
            Fx = self._features_torch(X)
            Fy = Fx if symmetric else self._features_torch(Y)
            W = torch.as_tensor(self.weights, dtype=Fx.dtype, device=Fx.device)
            xx = torch.einsum("ij,ij,j->i", Fx, Fx, W)
            yy = torch.einsum("ij,ij,j->i", Fy, Fy, W)
            cross = (Fx * W) @ Fy.T
            d2 = (xx[:, None] + yy[None, :] - 2.0 * cross).clamp_min(0.0)
            K = torch.exp(-self.gamma * d2)
            if symmetric:
                K = 0.5 * (K + K.T)
            return K

        Fx = self.features(X)
        Fy = Fx if symmetric else self.features(Y)
        W = self.weights
        xx = _np.einsum("ij,ij,j->i", Fx, Fx, W)
        yy = _np.einsum("ij,ij,j->i", Fy, Fy, W)
        cross = (Fx * W) @ Fy.T
        d2 = _np.maximum(xx[:, None] + yy[None, :] - 2.0 * cross, 0.0)
        K = _np.exp(-self.gamma * d2)
        if symmetric:
            K = 0.5 * (K + K.T)
        return K

    __call__ = gram

    def draw(self, x=None, style: str = "pennylane", decimals: int = 2):
        """Return a Matplotlib rendering of the feature circuit; use optional input x, PennyLane style and numeric precision."""
        return self._draw_state(x, style, decimals)
