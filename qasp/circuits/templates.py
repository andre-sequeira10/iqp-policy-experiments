"""Gate-queueing IQP and axis/Pauli-word ansatz templates.

Templates combine optional feature maps, reuploading and entanglers. They do
not allocate devices or trainable tensors: get_param_shapes describes the flat
weights consumed by CircuitBuilder and its TorchLayer. Angles are in radians."""
import pennylane as qml
from pennylane import numpy as np
from typing import Union, Sequence, Tuple, List
import torch
import functools


class IQPAnsatz:
    """Layered Hadamard, optional RZ and trainable ZZ circuit template.

    Each layer has one angle per edge and, when enabled, one RZ angle per
    qubit. Initial and inter-layer Hadamards are queued by this template.
    Feature-map reuploading is optional. Call inside a PennyLane quantum
    function; this object neither creates a device nor owns trainable tensors."""
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 1,
        imprimitive_pattern: Union[str, Sequence[Tuple[int, int]]] = "all_to_all",
        feature_map: int = None,
        reuploading: bool = False,
        include_single_qubit: bool = True,
    ):
        """Configure the IQP gate layout without allocating parameters.

        ``imprimitive_pattern`` is chain, ring, all_to_all, explicit wire pairs,
        or a callable producing pairs from n_qubits. ``include_single_qubit``
        controls RZ gates; ZZ edges always have trainable angles. ``feature_map``
        is an embedding instance, encoded initially and between layers when
        ``reuploading=True``. Flat params have length
        ``n_layers * (num_edges + n_qubits)`` with RZ gates, otherwise
        ``n_layers * num_edges``. See get_param_shapes for input-scale storage."""
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map = feature_map
        self.reuploading = reuploading
        self.include_single_qubit = include_single_qubit
        self.z_gates = None
        self.parity_gates = None
        self.CNOT_Gates = None
        # determine edges & broadcast pattern
        if isinstance(imprimitive_pattern, str):
            self.pattern = imprimitive_pattern
            if self.pattern == "all_to_all":
                self.edges = [
                    (i, j)
                    for i in range(n_qubits)
                    for j in range(i + 1, n_qubits)
                ]
            elif self.pattern == "ring":
                self.edges = [
                    (i, (i + 1) % n_qubits)
                    for i in range(n_qubits)
                ]
            elif self.pattern == "chain":
                self.edges = [
                    (i, i + 1)
                    for i in range(n_qubits - 1)
                ]
        else:
            self.pattern = "pairwise"

            if callable(imprimitive_pattern):
                imprimitive_pattern = imprimitive_pattern(n_qubits)

            # if (len(imprimitive_pattern) == 0):
            #     self.include_imprimitive = False

            self.edges = list(imprimitive_pattern)
        self.num_edges = len(self.edges)
        #self._block_size = self.n_qubits + self.num_edges
        if self.include_single_qubit:
            self._block_size = self.n_qubits + self.num_edges
        else:
            self._block_size = self.num_edges

    def __call__(self, inputs=None, input_scaling_params=None, params=None) -> None:
        """Apply the IQP ansatz gates using a flat parameter vector."""
        total = self.n_layers * self._block_size
        if params.ndim != 1 or len(params) != total:
            raise ValueError(
                f"params must be 1D of length {total}, got shape {params.shape}"
            )
        # initial Hadamards
        #qml.broadcast(qml.Hadamard, pattern="single", wires=range(self.n_qubits))

        for n in range(self.n_qubits):
            qml.Hadamard(wires=n)

        # apply each layer
        if self.feature_map is not None:
            if self.feature_map.input_scaling:
                if self.feature_map.name == "AngleEmbedding":
                    self.feature_map(inputs, input_scaling_params=input_scaling_params[:self.feature_map.features])
                elif self.feature_map.name == "NeuralEmbedding":
                    if type(self.feature_map.rotations) is str:
                        self.feature_map(inputs, input_scaling_params=input_scaling_params[:self.n_qubits*self.feature_map.features])
                    else:
                        self.feature_map(inputs, input_scaling_params=input_scaling_params[:len(self.feature_map.rotations)*self.feature_map.features])

            else:
                self.feature_map(inputs)
        for l in range(self.n_layers):
            start = l * self._block_size
            if self.feature_map is not None:
                if self.feature_map.name == "AngleEmbedding":
                    start_features = (l+1) * self.feature_map.features
                elif self.feature_map.name == "NeuralEmbedding":
                    if type(self.feature_map.rotations) is str:
                        start_features = (l+1) * self.n_qubits * self.feature_map.features
                    else:
                        start_features = (l+1) * len(self.feature_map.rotations) * self.feature_map.features
            #phase = params[start : start + self.n_qubits]
            #ent   = params[start + self.n_qubits : start + self._block_size]
            if self.include_single_qubit:
                phase = params[start : start + self.n_qubits]
                ent   = params[start + self.n_qubits : start + self._block_size]
            else:
                phase = None
                ent   = params[start : start + self.num_edges]
            # phase layer (Z rotations)
            #qml.broadcast(
            #    qml.RZ,
            #    parameters=phase,
            #    wires=range(self.n_qubits),
            #    pattern="single",
            #)

            '''
            if (self.reuploading or l == 0):
                for n in range(self.n_qubits):
                    # qml.RZ(phase[n], wires=n, id = 'input-encoding gate')
                    qml.RZ(phase[n], wires=n)
            else:
                for n in range(self.n_qubits):
                    qml.RZ(phase[n], wires=n)
            '''
            if self.include_single_qubit:

                if (self.reuploading or l == 0):
                    for n in range(self.n_qubits):
                        qml.RZ(phase[n], wires=n)
                else:
                    for n in range(self.n_qubits):
                        qml.RZ(phase[n], wires=n)

            # entangling ZZ layer
            for (i, j), phi in zip(self.edges, ent):
                qml.PauliRot(phi, "ZZ", wires=[i, j])

            # final Hadamards
            
            #qml.broadcast(qml.Hadamard, pattern="single", wires=range(self.n_qubits))
            for n in range(self.n_qubits):
                qml.Hadamard(wires=n)

            if self.feature_map is not None and self.reuploading and l < self.n_layers - 1:
                if self.feature_map.input_scaling:
                    if self.feature_map.name == "AngleEmbedding":
                        self.feature_map(inputs, input_scaling_params=input_scaling_params[start_features: start_features + self.feature_map.features])
                    elif self.feature_map.name == "NeuralEmbedding":
                        if type(self.feature_map.rotations) is str:
                            self.feature_map(inputs, input_scaling_params=input_scaling_params[start_features: start_features + self.n_qubits*self.feature_map.features])
                        else:
                            self.feature_map(inputs, input_scaling_params=input_scaling_params[start_features: start_features + len(self.feature_map.rotations)*self.feature_map.features])
                else:
                    self.feature_map(inputs)

        if (self.z_gates != None):
            for n in range(self.n_qubits):
                if self.z_gates[n] == True:
                    qml.Z(wires=n)

            for l in range(self.n_layers-1, -1, -1):
                start = l * self._block_size
                if self.feature_map is not None:
                    if self.feature_map.name == "AngleEmbedding":
                        start_features = (l+1) * self.feature_map.features
                    elif self.feature_map.name == "NeuralEmbedding":
                        if type(self.feature_map.rotations) is str:
                            start_features = (l+1) * self.n_qubits * self.feature_map.features
                        else:
                            start_features = (l+1) * len(self.feature_map.rotations) * self.feature_map.features
                phase = params[start : start + self.n_qubits]
                ent   = params[start + self.n_qubits : start + self._block_size]

                if self.feature_map is not None and self.reuploading and l < self.n_layers - 1:
                    if self.feature_map.input_scaling:
                        if self.feature_map.name == "AngleEmbedding":
                            self.feature_map(inputs, input_scaling_params=input_scaling_params[start_features: start_features + self.feature_map.features])
                        elif self.feature_map.name == "NeuralEmbedding":
                            if type(self.feature_map.rotations) is str:
                                self.feature_map(inputs, input_scaling_params=input_scaling_params[start_features: start_features + self.n_qubits*self.feature_map.features])
                            else:
                                self.feature_map(inputs, input_scaling_params=input_scaling_params[start_features: start_features + len(self.feature_map.rotations)*self.feature_map.features])
                    else:
                        self.feature_map(inputs)

                for n in range(self.n_qubits):
                    qml.Hadamard(wires=n)

                for (i, j), phi in reversed(list(zip(self.edges, ent))):
                    qml.PauliRot(phi, "ZZ", wires=[i, j])

                if (self.reuploading or l == 0):
                    for n in range(self.n_qubits):
                        qml.RZ(phase[n], wires=n, id = 'input-encoding gate')
                else:
                    for n in range(self.n_qubits):
                        qml.RZ(phase[n], wires=n) 
            
            for n in range(self.n_qubits):
                qml.Hadamard(wires=n)
                 
        if self.parity_gates != None:
            for (i,j) in self.parity_gates:
                qml.CNOT(wires=[i,j])

        elif self.CNOT_Gates != None:
            for (i,j) in self.CNOT_Gates:
                qml.CNOT(wires=[i,j])


    def get_param_shapes(self) -> Tuple[int]:
        """Return the flat trainable tensor shapes expected by CircuitBuilder.

        The final tuple is ``(n_layers * _block_size,)`` for variational angles.
        If input scaling is enabled, a preceding tuple reserves scales for every
        layer: features per layer for AngleEmbedding; n_qubits * features for
        axis-based NeuralEmbedding; or len(rotations) * features for explicit
        NeuralEmbedding words. Reservation includes layers even when reuploading
        is disabled; not every allocated input scale is then used."""
        if self.feature_map is not None and self.feature_map.input_scaling:
           
            if self.feature_map.name == "AngleEmbedding":
                return [(self.n_layers*self.feature_map.features,), (self.n_layers * self._block_size,)]
            elif self.feature_map.name == "NeuralEmbedding":
                if type(self.feature_map.rotations) is str:
                    return [(self.n_layers*self.n_qubits*self.feature_map.features,), (self.n_layers * self._block_size,)]
                else:
                    return [(self.n_layers*len(self.feature_map.rotations)*self.feature_map.features,), (self.n_layers * self._block_size,)]
        else:
            return [(self.n_layers * self._block_size,)]


class AngleAnsatz:
    """Layered trainable rotations with optional fixed CZ/CNOT entanglers.

    For an axis string such as "YZ", each layer has n_qubits * len(rotations)
    angles. An explicit Pauli-word list has one angle per entry per layer.
    Fixed entanglers add no trainable parameters. Supported string patterns are
    chain, ring and all_to_all; explicit/callable edge lists are also accepted.
    Feature encoding may precede the circuit and be reuploaded between layers.
    Call this object inside a PennyLane quantum function with flat weights;
    use get_param_shapes for variational and optional input-scale dimensions."""
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 1,
        imprimitive_pattern: Union[str, Sequence[Tuple[int, int]]] = "all_to_all",
        feature_map: int = None,
        reuploading: bool = False,
        imprimitive: str = "CZ",
        include_imprimitive: bool = False,
        rotations: Union[str, Sequence[str]] = "ZY",
        include_final_imprimitive: bool = True,
        include_feature_map_first: bool = True,
        include_hadamards_initial: bool = False
    ):
        """Configure layered Pauli rotations and optional fixed entanglers.

        ``rotations`` gives axes in order (e.g. "YZ") or explicit Pauli-rotation
        specifications. ``imprimitive_pattern`` selects chain/ring/all_to_all,
        explicit pairs, or a callable of n_qubits. ``imprimitive`` selects the
        fixed entangler; ``include_imprimitive`` enables it and
        ``include_final_imprimitive`` controls the final layer's entangler.
        ``feature_map`` is an embedding instance; ``reuploading`` repeats encoding
        between layers. ``include_feature_map_first`` controls initial encoding,
        and ``include_hadamards_initial`` adds initial Hadamards.

        The template queues gates when called and owns no Torch parameters.
        Use get_param_shapes rather than inferring dimensions from wire count."""
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map = feature_map
        self.reuploading = reuploading
        self.imprimitive = qml.CZ if imprimitive == "CZ" else qml.CNOT
        self.include_imprimitive = include_imprimitive
        self.rotations = rotations
        self.z_gates = None
        self.parity_gates = None
        self.CNOT_Gates = None
        self.include_final_imprimitive = include_final_imprimitive
        self.include_feature_map_first = include_feature_map_first
        self.include_hadamards_initial = include_hadamards_initial
        # determine edges & broadcast pattern
        if isinstance(imprimitive_pattern, str):
            self.imprimitive_pattern = imprimitive_pattern
            if self.imprimitive_pattern == "all_to_all":
                self.imprimitive_edges = [
                    (i, j)
                    for i in range(n_qubits)
                    for j in range(i + 1, n_qubits)
                ]
            elif self.imprimitive_pattern == "ring":
                self.imprimitive_edges = [
                    (i, (i + 1) % n_qubits)
                    for i in range(n_qubits)
                ]
            elif self.imprimitive_pattern == "chain":
                self.imprimitive_edges = [
                    (i, i + 1)
                    for i in range(n_qubits - 1)
                ]
        else:
            self.imprimitive_pattern = "manual"


            if callable(imprimitive_pattern):
                imprimitive_pattern = imprimitive_pattern(n_qubits)

            # if (len(imprimitive_pattern) == 0):
            #     self.include_imprimitive = False


            # print(imprimitive_pattern)

            self.imprimitive_edges = list(imprimitive_pattern)
        
        self._block_size=0
        if isinstance(self.rotations, str):
            for r in rotations:
                self._block_size += self.n_qubits
        else:
            self._block_size = len(self.rotations)

    def __call__(self, inputs=None, input_scaling_params=None, params=None) -> None:
        """Apply the Angle ansatz gates using a flat parameter vector."""
        total = self.n_layers * self._block_size
        if params.ndim != 1 or len(params) != total:
            raise ValueError(
                f"params must be 1D of length {total}, got shape {params.shape}"
            )
        # apply each layer
        if self.include_hadamards_initial:
            for n in range(self.n_qubits):
                qml.Hadamard(wires=n)

        if self.feature_map is not None:
            if self.include_feature_map_first:
                if self.feature_map.input_scaling:
                    if self.feature_map.name == "AngleEmbedding":
                        self.feature_map(inputs, input_scaling_params=input_scaling_params[:self.feature_map.features])
                    elif self.feature_map.name == "NeuralEmbedding":
                        if type(self.feature_map.rotations) is str:
                            self.feature_map(inputs, input_scaling_params=input_scaling_params[:self.n_qubits*self.feature_map.features])
                        else:
                            self.feature_map(inputs, input_scaling_params=input_scaling_params[:len(self.feature_map.rotations)*self.feature_map.features])

                else:
                    self.feature_map(inputs)

        for l in range(self.n_layers):
            start = l * self._block_size
            if self.feature_map is not None:
                if self.feature_map.name == "AngleEmbedding":
                    if self.include_feature_map_first:
                        start_features = (l+1) * self.feature_map.features
                    else:
                        start_features = l * self.feature_map.features
                elif self.feature_map.name == "NeuralEmbedding":
                    if type(self.feature_map.rotations) is str:
                        if self.include_feature_map_first:
                            start_features = (l+1) * self.n_qubits * self.feature_map.features
                        else:
                            start_features = l * self.n_qubits * self.feature_map.features 
                    else:
                        if self.include_feature_map_first:
                            start_features = (l+1) * len(self.feature_map.rotations) * self.feature_map.features
                        else:
                            start_features = l * len(self.feature_map.rotations) * self.feature_map.features

            phase = params[start : start + self._block_size]
            if type(self.rotations) is str:
            
                #set python counter to 0
                counter = 0
                if (self.reuploading or l == 0):
                    for r in self.rotations:
                        for n in range(self.n_qubits):
                            qml.PauliRot(
                                theta=phase[counter],
                                pauli_word=r,
                                wires=[n],
                                id='input-encoding gate'
                            )
                            counter += 1
                else:
                    for r in self.rotations:
                        for n in range(self.n_qubits):
                            qml.PauliRot(
                                theta=phase[counter],
                                pauli_word=r,
                                wires=[n]
                            )
                            counter += 1
            else:
                for (index,pf) in zip(range(len(self.rotations)), self.rotations):
                    "get the first element of the list and convert it to a string"
                    parts = pf.split(" ")
                    pauli = parts[0]
                    wires = parts[1:]
                    "get the second element of the list and convert it to a list of integers"
                    qubits = [int(i) for i in wires]
                    "apply the rotation to the qubit"
                    if (self.reuploading or l == 0):
                        if pauli == "WC":
                            qml.WireCut(wires=qubits)
                        else:
                            qml.PauliRot(theta=phase[index],pauli_word=pauli, wires=qubits, id = 'input-encoding gate')
                    else:
                        if pauli == "WC":
                            qml.WireCut(wires=qubits)
                        else:
                            qml.PauliRot(theta=phase[index],pauli_word=pauli, wires=qubits)
            # entangling layer
            if self.include_imprimitive:
                if l < self.n_layers - 1 or self.include_final_imprimitive:
                    for (i, j) in self.imprimitive_edges:
                        self.imprimitive(wires=[i, j])

            if self.feature_map is not None and (self.reuploading or (self.include_feature_map_first==False)and l==0) and l < self.n_layers - 1:
                if self.feature_map.input_scaling:
                    if self.feature_map.name == "AngleEmbedding":
                        self.feature_map(inputs, input_scaling_params=input_scaling_params[start_features: start_features + self.feature_map.features])
                    elif self.feature_map.name == "NeuralEmbedding":
                        if type(self.feature_map.rotations) is str:
                            self.feature_map(inputs, input_scaling_params=input_scaling_params[start_features: start_features + self.n_qubits*self.feature_map.features])
                        else:
                            self.feature_map(inputs, input_scaling_params=input_scaling_params[start_features: start_features + len(self.feature_map.rotations)*self.feature_map.features])
                else:
                    self.feature_map(inputs)

        if (self.z_gates != None):
            for n in range(self.n_qubits):
                if self.z_gates[n] == True:
                    qml.Z(wires=n)

            for l in range(self.n_layers-1,-1,-1):
                start = l * self._block_size
                if self.feature_map is not None:
                    if self.feature_map.name == "AngleEmbedding":
                        if self.include_feature_map_first:
                            start_features = (l+1) * self.feature_map.features
                        else:
                            start_features = l * self.feature_map.features
                    elif self.feature_map.name == "NeuralEmbedding":
                        if type(self.feature_map.rotations) is str:
                            if self.include_feature_map_first:
                                start_features = (l+1) * self.n_qubits * self.feature_map.features
                            else:
                                start_features = l * self.n_qubits * self.feature_map.features
                        else:
                            if self.include_feature_map_first:

                                start_features = (l+1) * len(self.feature_map.rotations) * self.feature_map.features
                            else:
                                start_features = l * len(self.feature_map.rotations) * self.feature_map.features
                phase = params[start : start + self._block_size]

                if self.feature_map is not None and self.reuploading and l < self.n_layers - 1:
                    if self.feature_map.input_scaling:
                        if self.feature_map.name == "AngleEmbedding":
                            self.feature_map(inputs, input_scaling_params=input_scaling_params[start_features: start_features + self.feature_map.features])
                        elif self.feature_map.name == "NeuralEmbedding":
                            if type(self.feature_map.rotations) is str:
                                self.feature_map(inputs, input_scaling_params=input_scaling_params[start_features: start_features + self.n_qubits*self.feature_map.features])
                            else:
                                self.feature_map(inputs, input_scaling_params=input_scaling_params[start_features: start_features + len(self.feature_map.rotations)*self.feature_map.features])
                    else:
                        self.feature_map(inputs)

                if self.include_imprimitive:
                    if l < self.n_layers - 1 or self.include_final_imprimitive:
                        for (i, j) in self.imprimitive_edges[::-1]:
                            self.imprimitive(wires=[i, j])

                if type(self.rotations) is str:
                
                    #set python counter to 0
                    counter = 0
                    if (self.reuploading or l == 0):
                        for r in self.rotations:
                            for n in range(self.n_qubits):
                                qml.PauliRot(
                                    theta=phase[counter],
                                    pauli_word=r,
                                    wires=[n],
                                    id='input-encoding gate'
                                )
                                counter += 1
                    else:
                        for r in self.rotations:
                            for n in range(self.n_qubits):
                                qml.PauliRot(
                                    theta=phase[counter],
                                    pauli_word=r,
                                    wires=[n]
                                )
                                counter += 1
                else:
                    for (index,pf) in zip(range(len(self.rotations)), self.rotations):
                        "get the first element of the list and convert it to a string"
                        parts = pf.split(" ")
                        pauli = parts[0]
                        wires = parts[1:]
                        "get the second element of the list and convert it to a list of integers"
                        qubits = [int(i) for i in wires]
                        "apply the rotation to the qubit"
                        if (self.reuploading or l == 0):
                            qml.PauliRot(theta=phase[index],pauli_word=pauli, wires=qubits, id = 'input-encoding gate')
                        else:
                            qml.PauliRot(theta=phase[index],pauli_word=pauli, wires=qubits)                    
        elif self.parity_gates != None:
            for (i,j) in self.parity_gates:
                qml.CNOT(wires=[i,j])

        elif self.CNOT_Gates != None:
            for (i,j) in self.CNOT_Gates:
                qml.CNOT(wires=[i,j])

    def get_param_shapes(self) -> Tuple[int]:
        """Return the flat trainable tensor shapes expected by CircuitBuilder.

        The final tuple is ``(n_layers * _block_size,)`` for variational angles.
        If input scaling is enabled, a preceding tuple reserves scales for every
        layer: features per layer for AngleEmbedding; n_qubits * features for
        axis-based NeuralEmbedding; or len(rotations) * features for explicit
        NeuralEmbedding words. Reservation includes layers even when reuploading
        is disabled; not every allocated input scale is then used."""
        if self.feature_map is not None and self.feature_map.input_scaling:
            
            if self.feature_map.name == "AngleEmbedding":
                return [(self.n_layers*self.feature_map.features,), (self.n_layers * self._block_size,)]
            elif self.feature_map.name == "NeuralEmbedding":
                if type(self.feature_map.rotations) is str:
                    return [(self.n_layers*self.n_qubits*self.feature_map.features,), (self.n_layers * self._block_size,)]
                else:
                    return [(self.n_layers*len(self.feature_map.rotations)*self.feature_map.features,), (self.n_layers * self._block_size,)]
        else:
            return [(self.n_layers * self._block_size,)]
        

    


            