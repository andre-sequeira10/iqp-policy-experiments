"""Legacy policy-output sampling and exploratory Lie-closure utilities.

policy_variance measures output-probability variation, not gradient variance.
For reproducible cost-concentration diagnostics prefer qasp.diagnostics. DLA helpers
construct heuristic generators rather than deriving exact reachable circuit
algebras; fourier_analysis is a legacy misnomer, not a Fourier transform."""
import itertools
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import ray
import pennylane as qml
from itertools import combinations
import copy
import matplotlib.pyplot as plt
import os
from qasp.feature_maps.templates import AngleEmbedding , NeuralEmbedding

#os.environ["OMP_NUM_THREADS"] = "1"

# Worker: does batch_size samples per thread
def _sample_worker_batch(policy_class, policy_config_kwargs, n_qubits, action, batch_size, state = None):
    # Dynamic imports inside worker thread (safe)
    """Instantiate a configured Born/softmax policy and collect selected action probabilities.

    Configuration callables are evaluated at n_qubits. state=None draws normal
    inputs; action=None samples an action from the current distribution. In the
    configured-policy branch, parameters are reinitialized after each sample;
    the fallback samples inputs with one parameter initialization. Returns a
    list of Python floats and uses global Torch RNG state."""
    if policy_class == "PQCBorn":
        from qasp.rl.policies import PQCBorn, PQCBornVariance
        policy = PQCBorn
        policyVar = PQCBornVariance
    elif policy_class == "PQCSoftmax":
        from qasp.rl.policies import PQCSoftmax
        policy = PQCSoftmax

    # Resolve n_actions if it's a callable
    resolved_config = copy.deepcopy(policy_config_kwargs)
    if callable(resolved_config.get("n_actions")):
        resolved_config["n_actions"] = resolved_config["n_actions"](n_qubits)
    if callable(resolved_config["ansatz_config"].get("n_layers")):
        resolved_config["ansatz_config"]["n_layers"] = resolved_config["ansatz_config"]["n_layers"](n_qubits)
    if callable(resolved_config["ansatz_config"].get("rotations")):
        resolved_config["ansatz_config"]["rotations"] = resolved_config["ansatz_config"]["rotations"](n_qubits)

    if callable(resolved_config["feature_map_config"].get("features")):
        resolved_config["feature_map_config"]["features"] = resolved_config["feature_map_config"]["features"](n_qubits)
    
    if resolved_config["ansatz"] == "IQPAnsatz":
        from qasp.circuits.templates import IQPAnsatz
        resolved_config["ansatz"] = IQPAnsatz
    elif resolved_config["ansatz"] == "AngleAnsatz":
        from qasp.circuits.templates import AngleAnsatz
        resolved_config["ansatz"] = AngleAnsatz

    if resolved_config["feature_map"] == "AngleEmbedding":
        resolved_config["feature_map"] = AngleEmbedding
    elif resolved_config["feature_map"] == "NeuralEmbedding":
        resolved_config["feature_map"] = NeuralEmbedding
    
    if type(resolved_config["feature_map_config"]["rotations"]) == str and resolved_config["feature_map"] == AngleEmbedding:
        resolved_config["feature_map_config"]["features"] = n_qubits 

    resolved_config["n_actions"] = min(resolved_config["n_actions"], 2**n_qubits)

    resolved_config["feature_map_config"]["n_qubits"] = n_qubits
    resolved_config["ansatz_config"]["n_qubits"] = n_qubits



    if resolved_config.get("policy") is not None or policy_class == "PQCSoftmax":
        probs = [0]*batch_size

        if policy_class == "PQCBorn":

            if (resolved_config["policy"] == "parity-like" and policy_class == "PQCBorn"):
                n_ancillas = int(np.ceil(np.log2(resolved_config["n_actions"])))
                gates = []
                if (resolved_config.get("measurement_qubits") is not None):
                    if callable(resolved_config["measurement_qubits"]):
                        resolved_config["measurement_qubits"] = resolved_config["measurement_qubits"](n_qubits)

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
                #measure_qubits = [n_qubits+n_ancillas-1-i for i in range(n_ancillas)]
                #resolved_config["measurement_qubits"] = measure_qubits

            elif (resolved_config["policy"] == "hidden-like" or resolved_config["policy"] == "contiguous-like" and policy_class == "PQCBorn"):
                if callable(resolved_config["measurement_qubits"]):
                    resolved_config["measurement_qubits"] = resolved_config["measurement_qubits"](n_qubits)
                    #print("measurement_qubits:", resolved_config["measurement_qubits"])
            #pqc_model = policyVar(**resolved_config, n_qubits=n_qubits)
            #draw model 
            #qml.draw_mpl(pqc_model.c, decimals=2)(inputs=np.random.random(n_qubits), params=pqc_model.qlayer.state_dict()["params"])       
        elif policy_class == "PQCSoftmax":
            if callable(resolved_config.get("observables")):
                resolved_config["observables"] = resolved_config["observables"](n_qubits)
                #observables = resolved_config["observables"]
                #print("measurement_qubits:", resolved_config["measurement_qubits"])
        #show fig
        #plt.show()
        def init_binary_minus_one_plus_one(
            tensor: torch.Tensor,
        ) -> torch.Tensor:
            """Fill a head tensor in place with independent values -1/n_qubits or +1/n_qubits."""
            with torch.no_grad():
                bits = torch.randint(
                    low=0,
                    high=2,
                    size=tensor.shape,
                    device=tensor.device,
                )

                signs = (1/n_qubits)*(2 * bits.to(tensor.dtype) - 1)
                tensor.copy_(signs)

            return tensor
        
        def weights_init(module):
            # ========================================================
            # Quantum TorchLayer
            # ========================================================
            """Reinitialize supported quantum weights and affine-head weights/bias in place."""
            if isinstance(module, qml.qnn.TorchLayer):
                from qasp.initialization import initialize_tensor

                initialize_tensor(module.params, resolved_config["init"])
                if hasattr(module, "input_scaling_params"):
                    scale_init = resolved_config.get("input_scaling_init")
                    initialize_tensor(
                        module.input_scaling_params,
                        resolved_config["init"] if scale_init is None else scale_init,
                    )

            # ========================================================
            # Classical softmax head
            # ========================================================
            elif isinstance(module, torch.nn.Linear):
                init_binary_minus_one_plus_one(module.weight)

                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)


        if policy_class == "PQCBorn":
                pqc_model = policyVar(**resolved_config, n_qubits=n_qubits)
        else:
                pqc_model = policy(**resolved_config, n_qubits=n_qubits)

        for b in range(batch_size):
            #if policy_class == "PQCBorn":
            #    pqc_model = policyVar(**resolved_config, n_qubits=n_qubits)
            #else:
            #    pqc_model = policy(**resolved_config, n_qubits=n_qubits, observables=observables)
           
            if state is None:
                obs = torch.randn((pqc_model.feature_map_config["features"],))
            else:
                obs = state
            #p = pqc_model.action_probs(obs, temperature=None)
            p = pqc_model.forward(obs, temperature=None)

            #softmax check
            if policy_class == "PQCSoftmax":
                dist = torch.distributions.Categorical(logits=p)
            else:
                dist = torch.distributions.Categorical(probs=p)

            if action is None:
                sampled_action = dist.sample()
                prob = dist.probs[sampled_action]
            else:
                prob = dist.probs[action]

            probs[b] += prob.item()
            
            #reset parameters
            pqc_model.apply(weights_init)
            #print(f"Sample {b+1}/{batch_size} for n_qubits {n_qubits} completed.", flush=True)

        return probs
    
    
    if policy_class == "PQCBorn":
        pqc_model = policy(**resolved_config, n_qubits=n_qubits)
    elif policy_class == "PQCSoftmax":
        pqc_model = policy(**resolved_config, n_qubits=n_qubits)

    
    # Perform batch_size samples in one go
    probs = []
    for _ in range(batch_size):
        if state is None:
            obs = torch.randn((pqc_model.feature_map_config["features"],))
        else:
            obs = state
        p = pqc_model.action_probs(obs, temperature=None)

        if action is not None:
            prob = p[action]
        else:
            dist = torch.distributions.Categorical(probs=p)
            sampled_action = dist.sample()
            prob = dist.probs[sampled_action]

        probs.append(prob.item())
    return probs

# Efficient variance computation with ThreadPool batching
def policy_variance(policy_class, policy_config_kwargs, action=None, qubit_list=None,
                    n_samples=5000, batch_size=100,observables=None, parallel = False, num_remotes=None, state=None):
    """Estimate population variance of selected action probabilities across widths.

    policy_class selects PQCBorn or PQCSoftmax; policy_config_kwargs configures
    its circuit. qubit_list is required. n_samples determines requested draws;
    action=None measures the probability of a sampled action, rather than a
    fixed output. state=None also randomizes embedded inputs.

    Returns (variances, samples_by_width). Serial samples are stored as a
    single nested list per width; Ray samples are flat. parallel=True may
    initialize Ray and uses floor(n_samples/num_remotes) samples per worker,
    dropping a remainder. batch_size and observables are retained legacy
    arguments but are not used. No gradient or uncertainty estimate is computed;
    prefer qasp.diagnostics for controlled initialization/cost sweeps."""
    variances = []
    variances_dict = {}



    if parallel:
        # Initialize Ray if parallel execution is enabled
        if not ray.is_initialized():
            ray.init(

                include_dashboard=False,

                ignore_reinit_error=True,

            )

        if num_remotes is None:
            # Get number of remote workers available
            num_remotes = max(1, int(ray.available_resources().get('CPU', 0)) - 2)  # Leave some CPUs free
    else:
        # Get number of logical CPU cores available
        max_threads = multiprocessing.cpu_count()

        #print("max_threads:", max_threads)

        # Set OpenMP threads to maximum available
        #os.environ["OMP_NUM_THREADS"] = str(max_threads)

    for i,n_qubits in enumerate(qubit_list):
        print(f"- Starting sampling for n_qubits: {n_qubits}" , flush=True)
        # num_remotes = max(1,int(ray.available_resources().get('CPU', 0)))
        # print("num_remotes:", num_remotes)

        variance_qubit = []
        if (parallel):
            @ray.remote
            def sampler(samples):
                """Collect one Ray worker's action-probability samples for the current width."""
                results = []
                if (policy_class == "PQCSoftmax"):
                    # for _ in range(samples):
                    results.extend(_sample_worker_batch(policy_class, policy_config_kwargs, n_qubits, action, samples, state))
                else:
                    # for _ in range(samples):
                    results.extend(_sample_worker_batch(policy_class, policy_config_kwargs, n_qubits, action, samples, state))
                return results

            probsPar = ray.get([sampler.remote(n_samples//num_remotes) for _ in range(num_remotes)])
            #variance_qubit.extend(probsPar)
            variance_qubit = list(itertools.chain.from_iterable(probsPar))
            variances.append(np.var(variance_qubit))
            variances_dict[n_qubits] = variance_qubit
        else:
            if policy_class == "PQCSoftmax":

                # Create a ThreadPoolExecutor with the number of available CPU cores
                variance_qubit.append(
                    _sample_worker_batch(policy_class, policy_config_kwargs, n_qubits, action, n_samples, state))
            else:
                # Create a ThreadPoolExecutor with the number of available CPU cores
                variance_qubit.append(
                    _sample_worker_batch(policy_class, policy_config_kwargs, n_qubits, action, n_samples, state))

            print(f"-- Finished sampling for n_qubits: {n_qubits}" , flush=True)
            variances.append(np.var(variance_qubit))
            variances_dict[n_qubits] = variance_qubit

    return variances, variances_dict

def compute_DLA_generators(policy_class, policy_config_kwargs, qubit_list,observables=None, parallel=False):
    
    """Return heuristic Lie-closure dimensions for the requested qubit widths.

    Constructs configured policies and calls _policy_expressivity, optionally
    via Ray (which may be initialized here). Returns one integer per width.
    Despite the name, this returns closure sizes, not the generator operators.
    The fixed-gate generator heuristic is not a proof of circuit expressivity."""
    num_gates = []

    if (parallel):
        # Initialize Ray if parallel execution is enabled
        if not ray.is_initialized():
            ray.init()

        @ray.remote
        def sampler(n_qubits):
            """Compute one width's heuristic closure dimension in a Ray worker."""
            results = _policy_expressivity(policy_class, policy_config_kwargs,n_qubits, observables)
            return len(results)
        probsPar = ray.get([sampler.remote(n_qubits) for n_qubits in qubit_list])
    else:
        for n_qubits in qubit_list:
            print(f"- Starting expressivity computation for n_qubits: {n_qubits}")
            results = _policy_expressivity(policy_class, policy_config_kwargs,n_qubits, observables)
            num_gates.append(len(results))
            print(f"-- Finished expressivity computation for n_qubits: {n_qubits}")
        probsPar = num_gates

    return probsPar


def _policy_expressivity(policy_class, policy_config_kwargs, n_qubits, observables = None):
    """Build a policy, construct heuristic rotation/entangler generators and return their Lie closure.

    Supports axis-string rotations and chain/ring/all_to_all or explicit edges.
    Nested configuration dictionaries may be mutated; progress is printed.
    The returned closure is for these chosen generators, not a certified
    reachable algebra for the original finite-depth circuit."""
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
    if callable(resolved_config["ansatz_config"].get("rotations")):
        resolved_config["ansatz_config"]["rotations"] = resolved_config["ansatz_config"]["rotations"](n_qubits)

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
            
    if policy_class == "PQCBorn":
        pqc_model = policy(**resolved_config, n_qubits=n_qubits)
    elif policy_class == "PQCSoftmax":
        pqc_model = policy(**resolved_config, n_qubits=n_qubits, observables=observables)

    generators = []
    for n_qubits in range(pqc_model.feature_map_config["n_qubits"]):
        for rot in pqc_model.feature_map_config["rotations"]:
            match rot:
                case "Y":
                    generators.append(qml.Y(n_qubits))
                case "X":
                    generators.append(qml.X(n_qubits))
                case "Z":
                    generators.append(qml.Z(n_qubits))
                case _:
                    raise ValueError("Unknown rotation type")

        for rot in pqc_model.ansatz_config["rotations"]:
            match rot:
                case "Y":
                    generators.append(qml.Y(n_qubits))
                case "X":
                    generators.append(qml.X(n_qubits))
                case "Z":
                    generators.append(qml.Z(n_qubits))
                case _:
                    raise ValueError("Unknown rotation type")


    imprimitive_pattern = pqc_model.ansatz_config["imprimitive_pattern"]
    n_qubits = pqc_model.ansatz_config["n_qubits"]
    if isinstance(imprimitive_pattern, str):
        if imprimitive_pattern == "all_to_all":
            imprimitive_edges = [
                (i, j)
                for i in range(n_qubits)
                for j in range(i + 1, n_qubits)
            ]
        elif imprimitive_pattern == "ring":
            imprimitive_edges = [
                (i, (i + 1) % n_qubits)
                for i in range(n_qubits)
            ]
        elif imprimitive_pattern == "chain":
            imprimitive_edges = [
                (i, i + 1)
                for i in range(n_qubits - 1)
            ]
    else:

        if (callable(imprimitive_pattern)):
            imprimitive_pattern = imprimitive_pattern(n_qubits)



        # if (len(imprimitive_pattern) == 0):
        #     pqc_model.ansatz_config["include_imprimitive"] = False

        imprimitive_edges = list(imprimitive_pattern)

    if (pqc_model.ansatz_config.get("imprimitive") == None):
        pqc_model.ansatz_config["imprimitive"] = "CZ"

    print(pqc_model.ansatz_config["include_imprimitive"])


    if (pqc_model.ansatz_config["include_imprimitive"]):
        if (pqc_model.ansatz_config["imprimitive"] == "CNOT"):
            for (control, target) in imprimitive_edges:
                generators.extend(cnot_decomposition(control, target))
        elif (pqc_model.ansatz_config["imprimitive"] == "CZ"):
            for (control, target) in imprimitive_edges:
                generators.extend(cz_decomposition(control, target))
        else:
            raise ValueError("Unknown imprimitive type")

    print("Imprimitive edges", imprimitive_edges, "Generators:", generators)

    #TODO ATUALIZAR, ISTO AQUI FOI SO PARA TETSAR UMA COISA. pAORVEITAR E TAMBÉM RETIRRAR AS COISAS DA VARIANCIAS DAQUI
    closure = qml.lie_closure(generators)
    return closure

    return pqc_model, resolved_config, generators, closure

def fourier_analysis(policy_class, policy_config_kwargs, n_qubits, action, batch_size, observables = None):
    """Legacy helper returning (model, config, generators, Lie closure).

    Despite its name this performs no Fourier transform. action and batch_size
    are unused, nested configs are mutated, and unqualified legacy imports
    prevent ordinary installed-package use. Prefer explicit circuit analysis."""
    if policy_class == "PQCBorn":
        from rl.policies import PQCBorn
        policy = PQCBorn
    elif policy_class == "PQCSoftmax":
        from rl.policies import PQCSoftmax
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
    
    if policy_class == "PQCBorn":
        pqc_model = policy(**resolved_config, n_qubits=n_qubits)
    elif policy_class == "PQCSoftmax":
        pqc_model = policy(**resolved_config, n_qubits=n_qubits, observables=observables)
        # Print the gates of the circuit

    generators = []
    for n_qubits in range(pqc_model.feature_map_config["n_qubits"]):
        for rot in pqc_model.feature_map_config["rotations"]:
            match rot:
                case "Y":
                    generators.append(qml.Y(n_qubits))
                case "X":
                    generators.append(qml.X(n_qubits))
                case "Z":
                    generators.append(qml.Z(n_qubits))
                case _:
                    raise ValueError("Unknown rotation type")

        for rot in pqc_model.ansatz_config["rotations"]:
            match rot:
                case "Y":
                    generators.append(qml.Y(n_qubits))
                case "X":
                    generators.append(qml.X(n_qubits))
                case "Z":
                    generators.append(qml.Z(n_qubits))
                case _:
                    raise ValueError("Unknown rotation type")


    imprimitive_pattern = pqc_model.ansatz_config["imprimitive_pattern"]
    n_qubits = pqc_model.ansatz_config["n_qubits"]
    if imprimitive_pattern == "all_to_all":
        imprimitive_edges = [
            (i, j)
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        ]
    elif imprimitive_pattern == "ring":
        imprimitive_edges = [
            (i, (i + 1) % n_qubits)
            for i in range(n_qubits)
        ]
    elif imprimitive_pattern == "chain":
        imprimitive_edges = [
            (i, i + 1)
            for i in range(n_qubits - 1)
        ]

    if (pqc_model.ansatz_config["include_imprimitive"]):
        if (pqc_model.ansatz_config["imprimitive"] == "CNOT"):
            for (control, target) in imprimitive_edges:
                generators.extend(cnot_decomposition(control, target))
        elif (pqc_model.ansatz_config["imprimitive"] == "CZ"):
            for (control, target) in imprimitive_edges:
                generators.extend(cz_decomposition(control, target))
        else:
            raise ValueError("Unknown imprimitive type")


    closure = qml.lie_closure(generators)

    return pqc_model, resolved_config, generators, closure



def hadamard_decomposition(qubit):
    """Return X and Y as heuristic Lie generators; this is not a gate-by-gate Hadamard decomposition."""
    return [qml.X(qubit), qml.Y(qubit)]

def cz_decomposition(control, target):
    """Return the legacy CZ generator heuristic (including duplicates), not an executable CZ decomposition."""
    h_decompose_ctrl = hadamard_decomposition(control)
    h_decompose_target = hadamard_decomposition(target)

    multi_h = []
    for h1 in h_decompose_ctrl:
        for h2 in h_decompose_target:
            multi_h.append(h1@h2)

    decomposition = []
    decomposition.extend(multi_h)
    decomposition.append(qml.I(control)@ qml.Z(target))
    decomposition.extend(multi_h)

    return decomposition

def cnot_decomposition(control, target):
    """Return the legacy CNOT generator heuristic, not a unitary gate decomposition with angles."""
    h_decompose_ctrl = hadamard_decomposition(control)

    multi_h = []
    for h1 in h_decompose_ctrl:
            multi_h.append(h1@qml.I(target))

    decomposition = []
    decomposition.extend(multi_h)
    decomposition.extend(cz_decomposition(control, target))
    decomposition.extend(multi_h)

    return decomposition
