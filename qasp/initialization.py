"""Reusable initializers for quantum parameters and feature-map input scales.

Use ``init=Normal(0, 0.2), input_scaling_init=Uniform(0.8, 1.2)`` in model
constructors or their configuration dictionaries. Initializers are immutable,
stateless specifications: Torch uses its current RNG (or a supplied generator),
while NumPy-based models supply their own seeded RNG. Reusing a specification
does not reuse a random sample.

``initializer(tensor)`` fills a floating-point Torch tensor in place, just like
``torch.nn.init``. ``initializer.sample(shape, rng=...)`` returns a NumPy array
for explicit-parameter APIs, including QFKernel and custom PennyLane QNodes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def _positive(value, name):
    value = _finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _bounds(low, high):
    low, high = _finite(low, "low"), _finite(high, "high")
    if low >= high:
        raise ValueError("low must be less than high.")
    return low, high


def _shape(shape):
    shape = (shape,) if isinstance(shape, Integral) else tuple(shape)
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in shape):
        raise ValueError("shape must contain nonnegative integers.")
    return tuple(int(n) for n in shape)


class Initializer(ABC):
    """Distribution specification supporting both NumPy and Torch backends."""

    def sample(self, shape, *, rng=None):
        """Return finite float64 NumPy values; rng may be a numpy.random.Generator."""
        shape = _shape(shape)
        values = np.asarray(self._sample(shape, np.random if rng is None else rng), dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Initializer must produce finite values with shape {shape}.")
        return values

    def __call__(self, tensor, *, generator=None):
        """Initialize a floating-point Torch tensor in place, preserving dtype/device."""
        import torch

        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise TypeError("Initializer expects a floating-point Torch tensor.")
        with torch.no_grad():
            self._fill(tensor, generator)
            if not torch.isfinite(tensor).all():
                raise ValueError("Initializer produced non-finite values.")
        return tensor

    @abstractmethod
    def _sample(self, shape, rng):
        """Sample NumPy values for a validated shape."""

    def _fill(self, tensor, generator):
        """Use a local NumPy sampler seeded from Torch when no native fill exists.

        This avoids private Torch distribution APIs and global RNG swapping.
        Beta, Gamma, Laplace and TruncatedNormal sample on CPU, then copy to the
        target device/dtype. A supplied Torch generator controls the seed without
        changing either global RNG. Native implementations override this method.
        """
        import torch

        if tensor.numel() == 0:
            return
        seed = torch.randint(2**63 - 1, (), device=tensor.device,
                             generator=generator, dtype=torch.int64).item()
        values = self.sample(tuple(tensor.shape), rng=np.random.default_rng(seed))
        tensor.copy_(torch.as_tensor(values, device=tensor.device, dtype=tensor.dtype))


@dataclass(frozen=True)
class Normal(Initializer):
    """Normal distribution with mean mu and standard deviation std (not variance)."""

    mu: float = 0.0
    std: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "mu", _finite(self.mu, "mu"))
        object.__setattr__(self, "std", _finite(self.std, "std"))
        if self.std < 0:
            raise ValueError("std must be nonnegative.")

    def _sample(self, shape, rng):
        return rng.normal(self.mu, self.std, size=shape)

    def _fill(self, tensor, generator):
        tensor.normal_(self.mu, self.std, generator=generator)


@dataclass(frozen=True)
class Uniform(Initializer):
    """Uniform distribution on [low, high); equal bounds give a constant."""

    low: float = 0.0
    high: float = 2 * math.pi

    def __post_init__(self):
        object.__setattr__(self, "low", _finite(self.low, "low"))
        object.__setattr__(self, "high", _finite(self.high, "high"))
        if self.low > self.high:
            raise ValueError("low must be less than or equal to high.")

    def _sample(self, shape, rng):
        return rng.uniform(self.low, self.high, size=shape)

    def _fill(self, tensor, generator):
        tensor.uniform_(self.low, self.high, generator=generator)


@dataclass(frozen=True)
class Beta(Initializer):
    """Beta(alpha, beta) rescaled to [low, high]. Both shapes must be positive.

    Equal shapes > 1 concentrate near the midpoint; shapes < 1 concentrate near
    the endpoints. Unequal shapes give a skewed distribution. The mean is
    low + (high - low) * alpha / (alpha + beta).
    """

    alpha: float = 1.0
    beta: float = 1.0
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "alpha", _positive(self.alpha, "alpha"))
        object.__setattr__(self, "beta", _positive(self.beta, "beta"))
        low, high = _bounds(self.low, self.high)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def _sample(self, shape, rng):
        return self.low + (self.high - self.low) * rng.beta(self.alpha, self.beta, size=shape)


@dataclass(frozen=True)
class TruncatedNormal(Initializer):
    """Normal(mu, std) conditioned on low <= x <= high, without clipping.

    Bounds are actual parameter values, not standard deviations from mu.
    mu/std describe the underlying normal; truncation changes its moments.
    Requires std > 0 and finite low < high. Uses SciPy's tail-stable sampler.
    """

    mu: float = 0.0
    std: float = 1.0
    low: float = -2.0
    high: float = 2.0

    def __post_init__(self):
        object.__setattr__(self, "mu", _finite(self.mu, "mu"))
        object.__setattr__(self, "std", _positive(self.std, "std"))
        low, high = _bounds(self.low, self.high)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def _sample(self, shape, rng):
        from scipy.stats import truncnorm

        return truncnorm.rvs((self.low - self.mu) / self.std,
                             (self.high - self.mu) / self.std,
                             loc=self.mu, scale=self.std, size=shape, random_state=rng)


@dataclass(frozen=True)
class LogNormal(Normal):
    """Positive exp(Normal(mu, std)); mu/std are in log space.

    The arithmetic mean is exp(mu + std**2 / 2), and the median is exp(mu).
    Use mu=-std**2/2 for input scales with mean one.
    """

    def _sample(self, shape, rng):
        return rng.lognormal(self.mu, self.std, size=shape)

    def _fill(self, tensor, generator):
        tensor.log_normal_(self.mu, self.std, generator=generator)


@dataclass(frozen=True)
class Gamma(Initializer):
    """Positive Gamma(concentration, rate), using rate rather than scale.

    Mean = concentration / rate; variance = concentration / rate**2.
    Both parameters must be positive. NumPy's scale is 1/rate.
    """

    concentration: float = 1.0
    rate: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "concentration", _positive(self.concentration, "concentration"))
        object.__setattr__(self, "rate", _positive(self.rate, "rate"))

    def _sample(self, shape, rng):
        return rng.gamma(self.concentration, scale=1 / self.rate, size=shape)


@dataclass(frozen=True)
class Exponential(Initializer):
    """Nonnegative exponential distribution with positive rate; mean = 1/rate."""

    rate: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "rate", _positive(self.rate, "rate"))

    def _sample(self, shape, rng):
        return rng.exponential(scale=1 / self.rate, size=shape)

    def _fill(self, tensor, generator):
        tensor.exponential_(self.rate, generator=generator)


@dataclass(frozen=True)
class Laplace(Initializer):
    """Symmetric Laplace distribution with location mu and positive scale.

    Mean = mu; standard deviation = sqrt(2) * scale. Has a sharper center and
    heavier tails than a normal distribution with the same variance.
    """

    mu: float = 0.0
    scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "mu", _finite(self.mu, "mu"))
        object.__setattr__(self, "scale", _positive(self.scale, "scale"))

    def _sample(self, shape, rng):
        return rng.laplace(self.mu, self.scale, size=shape)


@dataclass(frozen=True)
class Rademacher(Initializer):
    """Equiprobable -scale or +scale, with positive scale (default 1)."""

    scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "scale", _positive(self.scale, "scale"))

    def _sample(self, shape, rng):
        return rng.choice((-self.scale, self.scale), size=shape)

    def _fill(self, tensor, generator):
        import torch

        bits = torch.randint(2, tensor.shape, device=tensor.device, generator=generator)
        tensor.copy_(bits).mul_(2).sub_(1).mul_(self.scale)


@dataclass(frozen=True)
class Constant(Initializer):
    """Fill every parameter with value."""

    value: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "value", _finite(self.value, "value"))

    def _sample(self, shape, rng):
        return np.full(shape, self.value)

    def _fill(self, tensor, generator):
        tensor.fill_(self.value)


class Zeros(Constant):
    """Fill every parameter with zero."""

    def __init__(self):
        super().__init__(0.0)


class Ones(Constant):
    """Fill every parameter with one."""

    def __init__(self):
        super().__init__(1.0)


@dataclass(frozen=True)
class Choice(Initializer):
    """Sample uniformly from a nonempty sequence of finite scalar values."""

    values: tuple

    def __post_init__(self):
        values = tuple(_finite(value, "choice") for value in self.values)
        if not values:
            raise ValueError("values must contain at least one choice.")
        object.__setattr__(self, "values", values)

    def _sample(self, shape, rng):
        return rng.choice(self.values, size=shape)

    def _fill(self, tensor, generator):
        import torch

        indices = torch.randint(len(self.values), tensor.shape,
                                generator=generator, device=tensor.device)
        tensor.copy_(tensor.new_tensor(self.values)[indices])


_ALIASES = {
    "random_0_2pi": Uniform(),
    "uniform": Uniform(),
    "random_-pi_pi": Uniform(-math.pi, math.pi),
    "random_0_pi": Uniform(0, math.pi),
    "random_-1_1": Uniform(-1, 1),
    "random_0_1": Uniform(0, 1),
    "random_half_1": Uniform(0.5, 1),
    "normal_0_1": Normal(),
    "normal_half_0_15": Normal(0.5, 0.15),
    "small_normal": Normal(0, 0.1),
    "zeros": Zeros(),
    "ones": Ones(),
    "constant": Constant(math.pi),
    "discrete_0_pi_over_2": Choice((0, math.pi / 2)),
}


def resolve_initializer(spec, *, default=None, legacy_discrete_pi=False):
    """Resolve an object, legacy name or Torch-style callable; reject typos.

    legacy_discrete_pi preserves CircuitBuilder's historical 0/pi distribution
    for its misnamed discrete alias. Use Choice to specify unambiguous values.
    """
    if spec is None:
        spec = Uniform() if default is None else default
    if isinstance(spec, str):
        name = spec.lower()
        if legacy_discrete_pi and name == "discrete_0_pi_over_2":
            return Choice((0, math.pi))
        if name not in _ALIASES:
            raise ValueError(f"Unknown initialization {spec!r}. Use an initializer object "
                             f"or one of {', '.join(sorted(_ALIASES))}.")
        return _ALIASES[name]
    if isinstance(spec, Initializer) or callable(spec):
        return spec
    raise TypeError("Initialization must be an Initializer, a supported string, or a callable.")


def initialize_tensor(tensor, spec, *, generator=None, legacy_discrete_pi=False):
    """Fill a Torch tensor, validating the result of a custom tensor callable.

    Custom callables may mutate the tensor and return None, or return an array
    of exactly the same shape. They own their randomness; a supplied generator
    is forwarded only to QASP Initializer instances.
    """
    import torch

    initializer = resolve_initializer(spec, legacy_discrete_pi=legacy_discrete_pi)
    if isinstance(initializer, Initializer):
        return initializer(tensor, generator=generator)
    with torch.no_grad():
        value = initializer(tensor)
        if value is not None:
            value = torch.as_tensor(value, device=tensor.device, dtype=tensor.dtype)
            if value.shape != tensor.shape:
                raise ValueError(f"Initializer must return shape {tuple(tensor.shape)}.")
            tensor.copy_(value)
        if not torch.isfinite(tensor).all():
            raise ValueError("Initializer produced non-finite values.")
    return tensor


def sample_parameters(shape, spec, *, rng=None):
    """Sample a NumPy parameter array from any supported initialization spec.

    Torch-style custom callables receive a temporary CPU float64 tensor. Their
    randomness is user-controlled; built-in distributions use the supplied RNG.
    """
    initializer = resolve_initializer(spec)
    if isinstance(initializer, Initializer):
        return initializer.sample(shape, rng=rng)
    import torch

    tensor = torch.empty(_shape(shape), dtype=torch.float64)
    return initialize_tensor(tensor, initializer).numpy().copy()


__all__ = [
    "Initializer", "Normal", "Uniform", "Beta", "TruncatedNormal", "LogNormal",
    "Gamma", "Exponential", "Laplace", "Rademacher", "Constant", "Zeros", "Ones", "Choice",
]
