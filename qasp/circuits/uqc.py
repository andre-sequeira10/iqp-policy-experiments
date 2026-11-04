"""Legacy illustrative universal-quantum-classifier circuit function.

This file lacks its qml/torch imports and uses the old qml.broadcast API.
It is not a standalone supported circuit template; prefer AngleAnsatz and
NeuralEmbedding for the composable implementation."""
def UQC(state, w, alpha, varphi, n_qubits=2, n_layers=1, entanglement="chain"):
    """Queue the legacy data-dependent RY/RZ circuit and CNOT entanglers.

    For each layer, RY uses 2*varphi[layer]; each wire's RZ angle is
    2*dot(state[0], w[layer][wire]) + 2*alpha[layer][wire]. n_layers/n_qubits
    set loop sizes and entanglement is the old broadcast pattern. Returns
    None. Requires externally supplied qml/torch globals and a compatible
    PennyLane broadcast API; it is retained as a historical example."""
    for l in range(n_layers):
        qml.broadcast(qml.RY, wires=range(n_qubits), pattern="single", parameters=2*varphi[l])
        for q in range(n_qubits):
            inner_p = torch.dot(state[0], w[l][q])
            r = 2*inner_p + 2*alpha[l][q]
            qml.RZ(r, wires=q)  
        qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern=entanglement)

