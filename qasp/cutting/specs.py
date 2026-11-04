"""Public, immutable inputs to manual or learned circuit-cut placement."""
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol


@dataclass(frozen=True)
class WireCut:
    """Split ``wire`` immediately after the zero-based source operation index."""
    wire: object
    after: int


@dataclass(frozen=True)
class CutConfig:
    """Cut selection and resource limits, independent of any placement algorithm.

    ``partitions`` selects every supported gate crossing the disjoint wire groups.
    Alternatively, ``gate_indices`` selects source-tape operation occurrences.
    A planner receives the source tape and this configuration and returns a new
    configuration with explicit selections. Selection is discrete, not autodiff.
    Width includes coherent measurement-record ancillas. Enumeration is exact;
    finite device shots still introduce statistical error. PP defaults to no
    pruning, unlike some legacy QASP model defaults.
    """
    partitions: tuple = ()
    gate_indices: tuple = ()
    wire_cuts: tuple = ()
    max_fragment_wires: int | None = None
    max_settings: int = 4096
    max_probability_wires: int = 12
    planner: Callable | None = field(default=None, repr=False, compare=False)
    pp_options: dict = field(default_factory=dict, compare=False)
    probability_policy: str = "strict"
    seed: int | None = 0

    def __post_init__(self):
        object.__setattr__(self, "partitions", tuple(tuple(p) for p in self.partitions))
        object.__setattr__(self, "gate_indices", tuple(self.gate_indices))
        object.__setattr__(self, "wire_cuts", tuple(
            WireCut(**w) if isinstance(w, dict) else w for w in self.wire_cuts))
        object.__setattr__(self, "pp_options", dict(self.pp_options))
        if self.planner is not None and not callable(self.planner):
            raise TypeError("planner must be callable.")
        if len(set(self.gate_indices)) != len(self.gate_indices):
            raise ValueError("gate_indices must not contain duplicates.")
        if any(type(i) is not int or i < 0 for i in self.gate_indices):
            raise ValueError("gate_indices must be nonnegative integers.")
        if any(not isinstance(w, WireCut) or type(w.after) is not int or w.after < 0
               for w in self.wire_cuts):
            raise ValueError("wire_cuts must contain WireCut(wire, after>=0).")
        wires = [w for p in self.partitions for w in p]
        if any(not p for p in self.partitions) or len(wires) != len(set(wires)):
            raise ValueError("partitions must be nonempty disjoint wire groups.")
        for name in ("max_settings", "max_probability_wires", "max_fragment_wires"):
            value = getattr(self, name)
            if value is None and name == "max_fragment_wires":
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.probability_policy not in {"strict", "project"}:
            raise ValueError("probability_policy must be 'strict' or 'project'.")

    def selected(self, **kwargs):
        """Return a planner-selected configuration without recursively invoking it."""
        return replace(self, planner=None, **kwargs)


class CutPlanner(Protocol):
    """Extension point for graph search, optimization or learned cut policies."""
    def __call__(self, tape, config: CutConfig) -> CutConfig: ...


def as_config(value):
    """Normalize the public dictionary or dataclass configuration."""
    if isinstance(value, CutConfig):
        return value
    if isinstance(value, dict):
        return CutConfig(**value)
    raise TypeError("cutting must be a CutConfig or a dictionary.")
