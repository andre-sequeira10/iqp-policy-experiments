# qasp/__init__.py

# 1. Circuits
"""QASP: reusable quantum circuits and application-level learning models.

Subpackages cover encodings, circuits/drawing, classification, quantum kernels,
Born learning, optimization, reinforcement learning and diagnostics.
Optional application exports are loaded lazily; importing an export can still
require its optional dependency. See docs/package_guide.md for the API map and
implementation boundaries."""
from .initialization import (
    Initializer, Normal, Uniform, Beta, TruncatedNormal, LogNormal,
    Gamma, Exponential, Laplace, Rademacher, Constant, Zeros, Ones, Choice,
)

from .circuits import (
    AngleAnsatz,
    CircuitBuilder,
    IQPAnsatz,
    UQC,
    circuit_to_tikz,
    save_tikz,
    tape_to_tikz,
)

# 2. Feature Maps (Note: ensure your folder is named 'feature_maps')
from .feature_maps import AngleEmbedding, NeuralEmbedding

# 3. Models / Generative Learning
from .genlearn import BornMachine, PQCBM, PauliMMD, create_qcbm, draw_qcbm

# 4. Kernels
from .qkernels import (
    QFKernel,
    FidelityKernel,
    ProjectedKernel,
    KernelTargetAlignment,
    align_kernel,
    create_kernel,
    draw_kernel,
    target_alignment,
    target_matrix,
)

# 5. Quantum Classifiers
from .classifiers import (
    ClassProbabilityDecoder,
    ClassifierLoss,
    ClassifierTrainer,
    PQCClassifier,
    ProbabilityDecoder,
    QuantumClassifier,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    binary_cross_entropy,
    brier_score,
    categorical_cross_entropy,
    classification_loss,
    classification_metrics,
    confusion_matrix,
    create_classifier,
    draw_classifier,
    f1_score,
    hinge_loss,
    log_loss,
    mean_squared_error,
    precision_score,
    pr_auc_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy,
)

# 6. Combinatorial optimization
from .optimization import (
    AVAILABLE_OPTIMIZERS,
    PENNYLANE_OPTIMIZERS,
    SCIPY_OPTIMIZERS,
    TORCH_OPTIMIZERS,
    ClassicalOptimizer,
    LRQAOA,
    LinearRampQAOA,
    OptimizationResult,
    PCE,
    PauliCorrelationEncoding,
    PauliTerm,
    QAOA,
    QUBO,
    RQAOA,
    RecursiveQAOA,
    RecursiveQAOAResult,
    WarmStartQAOA,
    build_pauli_correlation_encoding,
    create_optimizer,
    draw_optimization,
    draw_pce,
    draw_qaoa,
    draw_recursive_qaoa,
)

# 7. RL, Training, and Simulation
#
# Import these on first access so circuit, kernel, and generative workflows do
# not require the complete RL dependency stack merely to import QASP.
_RL_EXPORTS = {
    "policy_variance",
    "compute_DLA_generators",
    "fourier_analysis",
    "hadamard_decomposition",
    "cz_decomposition",
    "cnot_decomposition",
    "_policy_expressivity",
    "_sample_worker_batch",
    "create_model",
    "draw_model",
    "PQCBorn",
    "PQCSoftmax",
    "PQDQN",
    "create_qfunction",
    "draw_qfunction",
    "DeepQLearning",
    "PQCBornVariance",
    "LearnablePQCBorn",
    "FastTruncatedPauliProp",
    "DifferentiablePauliProp",
    "PQCBornWrapper",
    "TorchRLTrainer",
    "REINFORCE",
    "BanditAWRConfig",
    "ContextualBanditTrainer",
    "PQCWrapper",
    "WindyTargetNavigationEnv",
}

_LAZY_EXPORTS = {
    "barren_plateau_diagnoser": (
        ".diagnostics",
        "barren_plateau_diagnoser",
    ),
}


def __getattr__(name):
    """Import and cache a supported lazy public export; raise AttributeError for unknown names."""
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module_name, attribute = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attribute)
        globals()[name] = value
        return value
    if name in _RL_EXPORTS:
        from importlib import import_module

        value = getattr(import_module(".rl", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Initializer", "Normal", "Uniform", "Beta", "TruncatedNormal", "LogNormal",
    "Gamma", "Exponential", "Laplace", "Rademacher", "Constant", "Zeros", "Ones", "Choice",
    "CircuitBuilder", "IQPAnsatz", "AngleAnsatz", "UQC",
    "circuit_to_tikz", "save_tikz", "tape_to_tikz",
    "AngleEmbedding", "NeuralEmbedding",
    "BornMachine", "PQCBM", "PauliMMD", "create_qcbm", "draw_qcbm",
    "QFKernel", "FidelityKernel", "ProjectedKernel",
    "create_kernel", "draw_kernel", "KernelTargetAlignment",
    "target_alignment", "target_matrix", "align_kernel",
    "PQCClassifier", "QuantumClassifier", "ClassifierTrainer",
    "ClassProbabilityDecoder", "ProbabilityDecoder", "create_classifier",
    "draw_classifier",
    "ClassifierLoss", "classification_loss", "mean_squared_error",
    "hinge_loss", "binary_cross_entropy", "categorical_cross_entropy",
    "accuracy_score", "balanced_accuracy_score", "precision_score",
    "recall_score", "f1_score", "confusion_matrix", "roc_auc_score",
    "pr_auc_score", "average_precision_score",
    "classification_metrics", "top_k_accuracy", "log_loss", "brier_score",
    "barren_plateau_diagnoser",
    "QUBO", "PauliTerm", "QAOA", "WarmStartQAOA", "RQAOA",
    "RecursiveQAOA", "LRQAOA", "LinearRampQAOA", "PCE",
    "PauliCorrelationEncoding", "OptimizationResult", "RecursiveQAOAResult",
    "build_pauli_correlation_encoding", "ClassicalOptimizer",
    "create_optimizer", "AVAILABLE_OPTIMIZERS",
    "SCIPY_OPTIMIZERS", "PENNYLANE_OPTIMIZERS", "TORCH_OPTIMIZERS",
    "draw_optimization", "draw_qaoa", "draw_pce", "draw_recursive_qaoa",
    "policy_variance", "compute_DLA_generators", "fourier_analysis", 
    "hadamard_decomposition", "cz_decomposition", "cnot_decomposition", 
    "_policy_expressivity", "_sample_worker_batch", "create_model", 
    "PQDQN", "create_qfunction", "draw_qfunction", "DeepQLearning",
    "draw_model", "PQCBorn", "PQCSoftmax", "PQCBornVariance", 
    "LearnablePQCBorn", "FastTruncatedPauliProp", "DifferentiablePauliProp",
    "PQCBornWrapper",
    "TorchRLTrainer", "REINFORCE", "BanditAWRConfig", 
    "ContextualBanditTrainer", "PQCWrapper", "WindyTargetNavigationEnv"
]
