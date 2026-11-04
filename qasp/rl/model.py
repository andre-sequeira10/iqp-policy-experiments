#from qulacs import Observable
"""Dictionary-based RL model creation and circuit drawing adapters.

create_model selects the current Born decoder, softmax policy or PQDQN.
Policy construction retains historical shallow-copy semantics; copy nested
configuration dictionaries before reusing them across widths."""
import numpy as np
import pennylane as qml

import os


def create_model(policy_class, policy_config_kwargs, n_qubits):#, observables=None):
    """Create an RL model from a selector, component dictionary and wire count.

    Selectors are PQCBorn (constructs PQCBornVariance), PQCSoftmax and PQDQN.
    For policies, resolve registered ansatz/feature-map names and callable
    n_actions, n_layers and features using n_qubits. Optional parity CNOTs
    configure ancilla readout. The PQDQN route delegates to create_qfunction.

    Returns the constructed Torch module. Policy configs are only shallow
    copied: nested ansatz_config/feature_map_config dictionaries are modified
    in place. Use deepcopy when reusing one configuration at multiple widths.
    Pass a supported selector and complete component dictionaries."""
    if policy_class == "PQDQN":
        from qasp.rl.qfunctions import create_qfunction
        return create_qfunction(policy_config_kwargs, n_qubits=n_qubits)
    if policy_class == "PQCBorn":
        from qasp.rl.policies import PQCBorn, PQCBornVariance
        # policy = PQCBorn
        policy = PQCBornVariance
    elif policy_class == "PQCSoftmax":
        from qasp.rl.policies import PQCSoftmax
        policy = PQCSoftmax


    # Resolve n_actions if it's a callable
    resolved_config = policy_config_kwargs.copy()
    if callable(resolved_config.get("n_actions")):
        resolved_config["n_actions"] = resolved_config["n_actions"](n_qubits)
    if callable(resolved_config["ansatz_config"].get("n_layers")):
        resolved_config["ansatz_config"]["n_layers"] = resolved_config["ansatz_config"]["n_layers"](n_qubits)
    if callable(resolved_config["feature_map_config"].get("features")):
        resolved_config["feature_map_config"]["features"] = resolved_config["feature_map_config"]["features"](n_qubits)
    
    if resolved_config["ansatz"] == "IQPAnsatz":
        from qasp.circuits.templates import IQPAnsatz
        resolved_config["ansatz"] = IQPAnsatz
    elif resolved_config["ansatz"] == "AngleAnsatz":
        from qasp.circuits.templates import AngleAnsatz
        resolved_config["ansatz"] = AngleAnsatz

    if resolved_config["feature_map"] == "AngleEmbedding":
        from qasp.feature_maps.templates import AngleEmbedding
        resolved_config["feature_map"] = AngleEmbedding
    elif resolved_config["feature_map"] == "NeuralEmbedding":
        from qasp.feature_maps.templates import NeuralEmbedding
        resolved_config["feature_map"] = NeuralEmbedding
    
    if type(resolved_config["feature_map_config"]["rotations"]) == str and resolved_config["feature_map"] == "AngleEmbedding":
        resolved_config["feature_map_config"]["features"] = n_qubits 


    resolved_config["feature_map_config"]["n_qubits"] = n_qubits
    resolved_config["ansatz_config"]["n_qubits"] = n_qubits

    if resolved_config.get("policy") is not None:
        if (resolved_config["policy"] == "parity-like" and policy_class == "PQCBorn"):
            if resolved_config["parity_cnot_decomposition"] == True:
                n_ancillas = int(np.ceil(np.log2(resolved_config["n_actions"])))
                gates = []
                if (resolved_config.get("measurement_qubits") is not None):
                    qubits_to_use = resolved_config["measurement_qubits"]

                    for i in range(n_ancillas):
                        list_qubits  = qubits_to_use[:-i] if i  !=0 else qubits_to_use
                        for value in list_qubits:
                            gates.append((value,n_qubits+n_ancillas-1-i))

                
                else:
                    for value in range(n_ancillas):
                        for i in range(n_qubits - value):
                            gates.append((i,n_qubits+n_ancillas-1-value))


                resolved_config["n_ancillas"] = n_ancillas
                resolved_config["CNOT_gates"] = gates
                measure_qubits = [n_qubits+n_ancillas-1-i for i in range(n_ancillas)]
                resolved_config["measurement_qubits"] = measure_qubits

            pqc_model = policy(**resolved_config, n_qubits=n_qubits)
            return pqc_model
        

    if policy_class == "PQCBorn":
        pqc_model = policy(**resolved_config, n_qubits=n_qubits)
    elif policy_class == "PQCSoftmax":
        pqc_model = policy(**resolved_config, n_qubits=n_qubits)

    return pqc_model


def draw_model(
    num_inputs=1,
    policy_class=None,
    policy_config_kwargs=None,
    n_qubits=None,
    model=None,
    inputs=None,
    output="matplotlib",
    tikz_options=None,
):
    """Draw a QASP policy or PQDQN as Matplotlib (default) or Quantikz/TikZ.

    Set ``output="tikz"`` to return LaTeX source. ``tikz_options`` is passed
    to :func:`qasp.circuits.circuit_to_tikz` and can contain options such as
    ``standalone``, ``wire_labels``, and ``symbolic_parameters``.
    For PQDQN, omitted inputs use the configured feature count and zero values.
    """

    from qasp.rl.qfunctions import PQDQN, draw_qfunction

    if isinstance(model, PQDQN) or (model is None and policy_class == "PQDQN"):
        return draw_qfunction(
            model,
            config=policy_config_kwargs,
            n_qubits=n_qubits,
            inputs=inputs,
            output=output,
            tikz_options=tikz_options,
        )

    pqc_model = None

    if model is not None:
        pqc_model = model
    else:
        if (model is None) and (policy_class is None or policy_config_kwargs is None or n_qubits is None):
            raise ValueError("If model is not provided, policy_class, policy_config_kwargs, and n_qubits must be specified.")
        if policy_class == "PQCBorn":
            from qasp.rl.policies import PQCBorn, PQCBornVariance
            # policy = PQCBorn
            policy = PQCBornVariance
        elif policy_class == "PQCSoftmax":
            from qasp.rl.policies import PQCSoftmax
            policy = PQCSoftmax
    
    
        # Resolve n_actions if it's a callable
        resolved_config = policy_config_kwargs.copy()
        if callable(resolved_config.get("n_actions")):
            resolved_config["n_actions"] = resolved_config["n_actions"](n_qubits)
        if callable(resolved_config["ansatz_config"].get("n_layers")):
            resolved_config["ansatz_config"]["n_layers"] = resolved_config["ansatz_config"]["n_layers"](n_qubits)
        if callable(resolved_config["feature_map_config"].get("features")):
            resolved_config["feature_map_config"]["features"] = resolved_config["feature_map_config"]["features"](n_qubits)
        
        if resolved_config["ansatz"] == "IQPAnsatz":
            from qasp.circuits.templates import IQPAnsatz
            resolved_config["ansatz"] = IQPAnsatz
        elif resolved_config["ansatz"] == "AngleAnsatz":
            from qasp.circuits.templates import AngleAnsatz
            resolved_config["ansatz"] = AngleAnsatz
    
        if resolved_config["feature_map"] == "AngleEmbedding":
            from qasp.feature_maps.templates import AngleEmbedding
            resolved_config["feature_map"] = AngleEmbedding
        elif resolved_config["feature_map"] == "NeuralEmbedding":
            from qasp.feature_maps.templates import NeuralEmbedding
            resolved_config["feature_map"] = NeuralEmbedding
        
        if type(resolved_config["feature_map_config"]["rotations"]) == str:
            resolved_config["feature_map_config"]["features"] = n_qubits 
    
    
        resolved_config["feature_map_config"]["n_qubits"] = n_qubits
        resolved_config["ansatz_config"]["n_qubits"] = n_qubits
    
        if resolved_config.get("policy") is not None:
            if (resolved_config["policy"] == "parity-like" and policy_class == "PQCBorn"):
                if resolved_config["parity_cnot_decomposition"] == True:

                    n_ancillas = int(np.ceil(np.log2(resolved_config["n_actions"])))
                    gates = []
                    if (resolved_config.get("measurement_qubits") is not None):
                        qubits_to_use = resolved_config["measurement_qubits"]
        
                        for i in range(n_ancillas):
                            list_qubits  = qubits_to_use[:-i] if i  !=0 else qubits_to_use
                            for value in list_qubits:
                                gates.append((value,n_qubits+n_ancillas-1-i))
        
                    
                    else:
                        for value in range(n_ancillas):
                            for i in range(n_qubits - value):
                                gates.append((i,n_qubits+n_ancillas-1-value))
        
        
                    resolved_config["n_ancillas"] = n_ancillas
                    resolved_config["CNOT_gates"] = gates
                    measure_qubits = [n_qubits+n_ancillas-1-i for i in range(n_ancillas)]
                    resolved_config["measurement_qubits"] = measure_qubits
    
                pqc_model = policy(**resolved_config, n_qubits=n_qubits)
        
        
        
        if (pqc_model is None):
    
            if policy_class == "PQCBorn":
                pqc_model = policy(**resolved_config, n_qubits=n_qubits)
            elif policy_class == "PQCSoftmax":
                pqc_model = policy(**resolved_config, n_qubits=n_qubits)


    if output not in {"matplotlib", "mpl", "tikz", "tex", "latex"}:
        raise ValueError("output must be 'matplotlib' or 'tikz'")

    draw_inputs = inputs if inputs is not None else np.random.random(num_inputs)
    circuit_kwargs = {
        "inputs": draw_inputs,
        "params": pqc_model.qlayer.state_dict()["params"],
    }
    if pqc_model.fm.input_scaling is not None:
        if pqc_model.fm.input_scaling == True:
            circuit_kwargs["input_scaling_params"] = pqc_model.qlayer.state_dict()[
                "input_scaling_params"
            ]

    if output in {"tikz", "tex", "latex"}:
        from qasp.circuits import circuit_to_tikz

        circuit_source = getattr(pqc_model, "circuit_builder", pqc_model.circuit)
        return circuit_to_tikz(
            circuit_source,
            circuit_kwargs=circuit_kwargs,
            **(tikz_options or {}),
        )

    fig, ax = qml.draw_mpl(
        pqc_model.circuit, decimals=2, style="sketch"
    )(**circuit_kwargs)

    return fig, ax
