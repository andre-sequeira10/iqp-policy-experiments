"""Feature encoders that enqueue PennyLane rotations in an active quantum tape.

AngleEmbedding maps input components directly to angles; NeuralEmbedding uses
trainable linear projections. Both are supplied to ansatz templates through
``feature_map``; they are not standalone QNodes and return no measurements."""
import pennylane as qml
from pennylane import numpy as np
from typing import Union, Sequence, Tuple, List

class AngleEmbedding:
    """Encode features as single-qubit or Pauli-product rotations.

    String rotations such as ``"YZ"`` apply each axis to the full feature vector.
    A sequence such as ``["Z 0", "ZZ 0 1"]`` pairs one input component with each
    Pauli word and its wire indices. Optional input scaling multiplies components
    before encoding. Entanglers are appended after the rotations."""
    def __init__(
        self,
        n_qubits: int,
        rotations: list = None,
        input_scaling: bool = False,
        features= None,
        imprimitive_pattern = None,
        imprimitive = "CNOT"
    ):
        """Store the encoding layout without allocating a simulator.

        Args:
            n_qubits: Number of data wires, indexed from zero.
            rotations: Axis string or sequence of ``"PAULI wire ..."`` specifications.
                Supply an explicit supported value before calling the encoder.
            input_scaling: Multiply inputs by caller-supplied trainable scales.
            features: Input dimension used by CircuitBuilder to allocate scales.
            imprimitive_pattern: None, ``chain``, ``ring``, ``all_to_all``, or an
                explicit iterable of wire pairs. Prefer concrete pairs to callbacks.
            imprimitive: ``"CZ"`` selects CZ; other values select CNOT.

        No shape validation is performed here; inputs and rotation specifications must
        agree when gates are queued. String encodings require at most n_qubits features."""
        self.n_qubits = n_qubits
        self.rotations = rotations
        self.input_scaling = input_scaling
        self.features = features
        self.imprimitive_pattern = imprimitive_pattern
        self.name = "AngleEmbedding"
        self.imprimitive = qml.CZ if imprimitive == "CZ" else qml.CNOT

    def __call__(self, inputs=None , input_scaling_params=None) -> None:
        """Queue encoding rotations and optional entanglers; return None.

        ``inputs`` is a feature vector (or PennyLane-compatible broadcast input for
        string rotations). If scaling is enabled, ``input_scaling_params`` must
        broadcast against it. Explicit Pauli specifications consume one input each;
        the implementation pairs entries using zip, so supply matching lengths."""
        if self.input_scaling:
            inputs = qml.math.multiply(inputs, input_scaling_params)

        if type(self.rotations) is str:

            for r in self.rotations:
                
                qml.AngleEmbedding(
                    features=inputs,
                    wires=range(self.n_qubits),
                    rotation=r,
                )
        else:
            for (index,pf) in zip(range(len(inputs)), self.rotations):
                "get the first element of the list and convert it to a string"
                parts = pf.split(" ")
                pauli = parts[0]
                wires = parts[1:]
                "get the second element of the list and convert it to a list of integers"
                qubits = [int(i) for i in wires]
                "apply the rotation to the qubit"
                qml.PauliRot(theta=inputs[index],pauli_word=pauli, wires=qubits)
        
        if self.imprimitive_pattern is not None:
            if isinstance(self.imprimitive_pattern, str):
                if self.imprimitive_pattern == "all_to_all":
                    self.imprimitive_edges = [
                        (i, j)
                        for i in range(self.n_qubits)
                        for j in range(i + 1, self.n_qubits)
                    ]
                elif self.imprimitive_pattern == "ring":
                    self.imprimitive_edges = [
                        (i, (i + 1) % self.n_qubits)
                        for i in range(self.n_qubits)
                    ]
                elif self.imprimitive_pattern == "chain":
                    self.imprimitive_edges = [
                        (i, i + 1)
                        for i in range(self.n_qubits - 1)
                    ]
            else:
                #self.imprimitive_pattern = "manual"


                self.imprimitive_edges = self.imprimitive_pattern
                if callable(self.imprimitive_pattern):
                    self.imprimitive_pattern = self.imprimitive_pattern(self.n_qubits)

            # if (len(imprimitive_pattern) == 0):
            #     self.include_imprimitive = False


            # print(imprimitive_pattern)

            #self.imprimitive_edges = list(self.imprimitive_pattern)
            for (i, j) in self.imprimitive_edges:
                        self.imprimitive(wires=[i, j])

class NeuralEmbedding:
    """Encode learned linear projections of a feature vector as Pauli rotations.

    For string rotations, each wire uses its own length-features weight block,
    shared across the requested axes. For explicit Pauli words, each word uses
    one weight block. There is no bias in these projections."""
    def __init__(
        self,
        n_qubits: int,
        rotations: Union [None, str] = None,
        features= None,
        imprimitive_pattern = None, 
        imprimitive = "CNOT"
    ):
        """Store a trainable projection encoder without allocating its weights.

        Args:
            n_qubits: Number of data wires.
            rotations: Axis string or explicit Pauli-word/wire specifications.
            features: Input dimension and length of each projection weight block.
            imprimitive_pattern: Optional chain, ring, all_to_all or explicit edges.
            imprimitive: ``"CZ"`` selects CZ; other values select CNOT.

        ``input_scaling`` is always enabled. CircuitBuilder owns the flat trainable
        weights; this object only queues operations using weights passed to __call__."""
        self.n_qubits = n_qubits
        self.rotations = rotations
        self.input_scaling = True
        self.features = features
        self.name = "NeuralEmbedding"
        self.imprimitive_pattern = imprimitive_pattern
        self.imprimitive = qml.CZ if imprimitive == "CZ" else qml.CNOT

    def __call__(self, inputs=None , input_scaling_params=None) -> None:
        """Queue rotations with angles equal to input/weight dot products.

        Supply one feature vector and a flat input_scaling_params array with a
        length-features block per wire (axis strings) or Pauli specification. The
        current implementation sums all product entries and is intended for a single
        sample; evaluate batches sample by sample. Returns None."""

        if type(self.rotations) is str:
            
            for i in range(len(self.rotations)):
                for n in range(self.n_qubits):

                    inputsScaled = qml.math.multiply(inputs, input_scaling_params[n*self.features: n*self.features + self.features]).sum()
                    qml.PauliRot(theta=inputsScaled,pauli_word=self.rotations[i], wires=n)

        else:
            for (index,pf) in zip(range(len(self.rotations)), self.rotations):
                "get the first element of the list and convert it to a string"
                parts = pf.split(" ")
                pauli = parts[0]
                wires = parts[1:]
                "get the second element of the list and convert it to a list of integers"
                qubits = [int(i) for i in wires]
                "apply the rotation to the qubit"
                inputsScaled = qml.math.multiply(inputs, input_scaling_params[index*self.features: index*self.features + self.features]).sum()

                qml.PauliRot(theta=inputsScaled,pauli_word=pauli, wires=qubits)
        
                
        if self.imprimitive_pattern is not None:
            if isinstance(self.imprimitive_pattern, str):
                if self.imprimitive_pattern == "all_to_all":
                    self.imprimitive_edges = [
                        (i, j)
                        for i in range(self.n_qubits)
                        for j in range(i + 1, self.n_qubits)
                    ]
                elif self.imprimitive_pattern == "ring":
                    self.imprimitive_edges = [
                        (i, (i + 1) % self.n_qubits)
                        for i in range(self.n_qubits)
                    ]
                elif self.imprimitive_pattern == "chain":
                    self.imprimitive_edges = [
                        (i, i + 1)
                        for i in range(self.n_qubits - 1)
                    ]
            else:
                #self.imprimitive_pattern = "manual"

                self.imprimitive_edges = self.imprimitive_pattern
                if callable(self.imprimitive_pattern):
                    self.imprimitive_pattern = self.imprimitive_pattern(self.n_qubits)

            # if (len(imprimitive_pattern) == 0):
            #     self.include_imprimitive = False


            # print(imprimitive_pattern)

            #self.imprimitive_edges = list(self.imprimitive_pattern)
            for (i, j) in self.imprimitive_edges:
                        self.imprimitive(wires=[i, j])
            
        
        
