
"""Quantum policy models and computational-basis action decoders.

PQCSoftmax.forward returns expectation-based logits, while PQCBornVariance
returns decoded action probabilities. The PQCBorn factory selector chooses
PQCBornVariance; the class named PQCBorn is an older sampling interface.
LearnablePQCBorn is an experimental adaptive-measurement prototype. Q-functions
with unconstrained values live in qasp.rl.qfunctions, not in this module."""
from importlib.metadata import requires
import math
import numpy as onp
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import functools
from copy import deepcopy
from qasp.circuits.constructor import CircuitBuilder
from pennylane.pauli import PauliWord, pauli_word_to_matrix, string_to_pauli_word
from itertools import combinations
from pennylane.pauli import PauliWord, PauliSentence
from scipy.sparse import csr_matrix

#import FastTruncatedPauliProp from path root circuits/pp_sim.py \
from qasp.rl.pp_sim import FastTruncatedPauliProp
from qasp.rl.differentiable_pp_sim import DifferentiablePauliProp


class PQCBorn(nn.Module):
    """Legacy Born sampler with (action, log_probability) forward output.

    The name differs from create_model("PQCBorn", ...), which constructs the
    current PQCBornVariance class. Use the latter for decoder-consistent
    probability outputs. This class retains older local-expectation and weight
    setter paths; see their method documentation before using them."""


    def __init__(self, circuit_builder=None, ansatz=None, ansatz_config=None, feature_map=None, feature_map_config=None, n_qubits=1, n_actions=2, init="normal_0_1", policy="contiguous-like", device="default.qubit", diff_method="best" , softmax_activation=None, output_scaling=None, training_method="probs", torchrl=False , measurement_qubits=None,z_gates=None, mpi=False, input_scaling_init=None, cutting=None):
        """Build the legacy Born policy and optional local-expectation sampler.

        Supply a builder or ansatz/feature_map classes with configs. policy selects
        contiguous-like or parity-like decoding, and measurement_qubits selects
        wires. init and input_scaling_init select ansatz and feature-map
        initializers; device/diff_method/mpi configure the quantum layer.
        training_method is probs or local_pauli_decomposition; torchrl enables
        batched postprocessing only for compatible paths. Optional softmax_activation
        and output_scaling modify the decoded weights. Prefer PQCBornVariance for
        current decoder semantics and probability-returning forward calls."""
        self.cutting = cutting
        super().__init__()   
        self.mpi = mpi
        self.torchrl = torchrl
        self.circuit_builder = circuit_builder
        self.circuit_builder_copy = deepcopy(circuit_builder)

        if self.circuit_builder is None and ansatz is None:
            raise ValueError("Either circuit_builder or ansatz must be provided.")
        
        self.ansatz = ansatz
        self.ansatz_config = ansatz_config
        self.feature_map = feature_map
        self.feature_map_config = feature_map_config
        self.n_qubits=n_qubits
        self.n_actions=n_actions
        if self.circuit_builder is not None:
            self.n_layers=self.circuit_builder.ansatz.n_layers
        else:
            self.n_layers=self.ansatz_config["n_layers"]
        self.init = init
        self.input_scaling_init = input_scaling_init
        self.diff_method = diff_method
        self.policy = policy
        self.policy_type = policy
        self.softmax_activation = softmax_activation
        self.output_scaling = output_scaling
        self.training_method = training_method
        if self.output_scaling:
            self.output_scaling_weight = nn.Parameter(torch.ones(1, requires_grad=True))
        self.device=device
        self.diff_method=diff_method
        self.measurement_qubits = measurement_qubits
        self.z_gates = z_gates

        self.build_policy()

    def build_policy(self):
        """Configure and compile the legacy probability/expectation circuit and its sampler copy."""
        if self.circuit_builder is None:

            self.fm = self.feature_map(**self.feature_map_config)

            ans = self.ansatz(**self.ansatz_config, feature_map=self.fm)

            self.shapes = ans.get_param_shapes()

            self.circuit_builder = CircuitBuilder(n_qubits=self.n_qubits)
            self.circuit_builder.set_ansatz(ans)           # pass the IQPAnsatz instance
            self.circuit_builder_copy = deepcopy(self.circuit_builder)


        if self.policy == "contiguous-like":
            if self.measurement_qubits is None:
                self.measurement_qubits = list(range(int(np.log2(self.n_actions))))
        else:
            if (self.z_gates != None):
                self.circuit_builder.set_z_gates(self.z_gates)
                self.ansatz.z_gates = self.z_gates
                self.circuit_builder.ansatz.z_gates = self.z_gates

            if self.measurement_qubits is None:
                self.measurement_qubits = list(range(self.n_qubits))

        if self.training_method == "probs":
            self.circuit_builder.set_measurement("probs", wires=self.measurement_qubits)
        elif self.training_method == "local_pauli_decomposition":
            if self.torchrl:
                raise ValueError("TorchRL does not support local pauli decomposition.")
            if self.measurement_qubits is None:
                self.measurement_qubits = list(range(self.n_qubits))
            observables = [qml.PauliZ(i) for i in self.measurement_qubits]
            self.circuit_builder.set_measurement("expectation", wires=self.measurement_qubits, observables=observables)

        self.c = self.circuit_builder.build()
        
        self.circuit = self.circuit_builder.compile(device = self.device , diff_method=self.diff_method, mpi=self.mpi, cutting=self.cutting)
        
        self.circuit_builder_copy.set_measurement("sample", wires=self.measurement_qubits)
        self.c_copy = self.circuit_builder_copy.build()

        self.circuit_copy = self.circuit_builder_copy.compile(device = self.device , diff_method=self.diff_method, mpi=self.mpi, cutting=self.cutting)

        self.qlayer = self.circuit_builder.create_torchlayer(initialization=self.init, input_scaling_init=self.input_scaling_init)
        if self.training_method == "local_pauli_decomposition":
            self.qlayer_copy = self.circuit_builder_copy.create_torchlayer(initialization=self.init, input_scaling_init=self.input_scaling_init)
            self.qlayer_copy.load_state_dict(self.qlayer.state_dict())

    def update_sampler_weights(self):
        """Copy current quantum weights to the local-decomposition sampler; requires that sampler to exist."""
        self.qlayer_copy.load_state_dict(self.qlayer.state_dict())

    def set_weights(self, params=None, input_scaling_params=None):
        """Legacy setter that calls set_weights on the compiled circuit.

        Ordinary PennyLane QNodes lack that method. Prefer
        ``model.qlayer.load_state_dict(...)`` or the builder's set_weights with
        both parameter groups when input scaling is enabled."""
        if params is not None:

            self.circuit.set_weights(params=params)
        if input_scaling_params is not None:

            self.circuit.set_weights(input_scaling_params=input_scaling_params)

    def get_weights(self):
        """Return the quantum layer's live named weight mapping, excluding classical head parameters."""
        return self.qlayer.qnode_weights
    
    def get_meyer_wallach(self, x):
        #TO DO
        """Unimplemented legacy placeholder; returns None without evaluating entanglement."""
        pass
    
    def post_process_policy(self, probs, policy_type="contiguous-like", n_actions=2, n_qubits=1):
        """Aggregate basis probabilities with the legacy contiguous/parity mapping, optionally on a batch."""
        if self.torchrl:
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)  # Ensure batched input
            
            batch_size, n_states = probs.shape

            if n_actions == 2**n_qubits:
                return probs

            if policy_type == "contiguous-like":
                base_size = n_states // n_actions
                remainder = n_states % n_actions
                sizes = [base_size + 1 if i < remainder else base_size for i in range(n_actions)]
                splits = torch.split(probs, sizes, dim=1)
                reduced = torch.stack([split.sum(dim=1) for split in splits], dim=1)  # [B, n_actions]
                return reduced

            elif policy_type == "parity-like":
                policy = torch.zeros((batch_size, n_actions), device=probs.device)
                counter = 0 
                for i in range(n_states):
                    bitstring = np.binary_repr(i, width=n_qubits)
                    a = []
                    for m in range(int(np.log2(n_actions))):
                        if m == 0:
                            a.append(bitstring.count("1") % 2)
                        else:
                            a.append(bitstring[:-m].count("1") % 2)
                    idx = int("".join(str(x) for x in a)[::-1], 2)
                    if idx >= self.n_actions:
                        idx = counter % self.n_actions
                        counter+=1
                    policy[:, idx] += probs[:, i]
                    
                return policy

            else:
                raise NotImplementedError(f"Policy type '{policy_type}' not supported in TorchRL mode.")
        
        else:
            # Non-batched version for standalone training
            if n_actions == 2**n_qubits:
                return probs

            if policy_type == "contiguous-like":
                tensor_size = 2**n_qubits
                base_size = tensor_size // n_actions
                remainder = tensor_size % n_actions

                sizes = [base_size] * n_actions
                for i in range(remainder):
                    sizes[i] += 1

                splits = torch.split(probs, sizes)
                return torch.stack([torch.sum(split) for split in splits])

            elif policy_type == "parity-like":
                policy = torch.zeros(n_actions, device=probs.device)
                counter=0
                for i in range(len(probs)):
                    bitstring = np.binary_repr(i, width=n_qubits)
                    a = []
                    for m in range(int(np.log2(n_actions))):
                        if m == 0:
                            a.append(bitstring.count("1") % 2)
                        else:
                            a.append(bitstring[:-m].count("1") % 2)
                    idx = int("".join(str(x) for x in a)[::-1], 2)
                    if idx >= self.n_actions:
                        idx = counter % self.n_actions
                        counter += 1

                    policy[idx] += probs[i]
                return policy

            else:
                raise NotImplementedError(f"Policy type '{policy_type}' not supported.")   

    def post_process_sample(self, sample):
        """
        Post-process the sample to get the action: put sample as list of bits -> integer
        """
        # Convert the sample to a list of bits
        bitstring = "".join(str(int(b)) for b in sample)

        if self.policy == "contiguous-like":
            action = int(bitstring, 2)

        elif self.policy == "parity-like":
            a=[]
            for m in range(int(np.log2(self.n_actions))):
                if m == 0:
                    a.append(bitstring.count("1") % 2)
                else:
                    a.append(bitstring[:-m].count("1") % 2)

            action = int("".join(str(x) for x in a)[::-1],2)

        return action
    
    def post_process_expval(self, action, exp_val):
        """
        Post-process the sample to get the action: put sample as list of bits -> integer
        """
        # Convert the sample to a list of bits
        action_bin = np.binary_repr(action, width=len(self.measurement_qubits))
        expvalue = 0
        for (i,b) in zip(range(len(self.measurement_qubits)), action_bin):
            expvalue += 0.5 + (0.5*(-1)**int(b))*exp_val[i]

        return torch.log(expvalue/len(self.measurement_qubits))
    
    def forward(self, x, temperature=None):

        """Sample and return (action, log_probability) for a single observation.

        This legacy method does not return a probability vector. Its probs branch
        does not apply tanh; local_pauli_decomposition uses a separate sampler.
        Prefer PQCBornVariance for the current probability interface."""
        if self.training_method == "probs":
            probs = self.qlayer(x)
            action_probs = probs
            if (self.policy == "contiguous-like"):
                action_probs = self.post_process_policy(probs, policy_type=self.policy, n_actions=self.n_actions, n_qubits=self.n_qubits)

            if self.softmax_activation:
                if temperature is not None:
                        action_probs = action_probs*temperature
                    
                    #nn functional softmax
                if self.output_scaling:
                    action_probs *= self.output_scaling_weight

                action_probs = torch.nn.functional.softmax(action_probs, dim=-1)

            dist = torch.distributions.Categorical(probs=action_probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action)
        elif self.training_method == "local_pauli_decomposition":
            sample = self.qlayer_copy(x)

            if self.policy_type == "contiguous-like" and self.n_actions==2:
                sample = [sample]

            action = self.post_process_sample(sample)
            exp_val = self.qlayer(x)
            log_prob = self.post_process_expval(action,exp_val)
            return action, log_prob

    def action_probs(self, x, temperature=None):
        """Apply tanh, execute the layer and return decoded weights with optional softmax (legacy interface)."""
        x = torch.tanh(x)           # each entry ∈ (-1, 1)
        probs = self.qlayer(x)
        action_probs = probs
        if (self.policy == "contiguous-like"):
            action_probs = self.post_process_policy(probs, policy_type=self.policy, n_actions=self.n_actions, n_qubits=len(self.measurement_qubits))
        # Process each sample separately
        if self.softmax_activation:
            if temperature is not None:
                    action_probs = action_probs*temperature
                
                #nn functional softmax
            if self.output_scaling:
                action_probs *= self.output_scaling_weight

            action_probs = torch.nn.functional.softmax(action_probs, dim=-1)

        return action_probs


class PQCSoftmax(nn.Module):

    """Expectation-based action logits with an optional biased affine head.

    forward executes the complete statevector or differentiable-PP model and
    returns scores; use Categorical(logits=...) for a distribution. action_probs
    is an older shortcut that bypasses the head/PP route. Parameters are Torch
    registered and may be trained jointly with the head. See __init__ for the
    legacy fc_layer=None convention and simulator approximation settings."""

    def __init__(

        self,

        circuit_builder=None,

        ansatz=None,

        ansatz_config=None,

        feature_map=None,

        feature_map_config=None,

        n_qubits=1,

        n_actions=2,

        init="normal_0_1",

        device="default.qubit",

        diff_method="best",

        softmax_activation=None,

        output_scaling=None,

        torchrl=False,

        observables=None,

        shots=None,

        fc_layer=None,

        spsa_num_directions=1,

        pp_truncation=None,

        pp_topM=20000,

        pp_tau=1e-10,

        pp_prune_every=5,

        pp_prune=True,

        mpi=False,

        input_scaling_init=None,

        cutting=None,
    ):

        """Construct an expectation-score policy with an optional affine action head.

        Supply a builder or ansatz/feature-map classes/configs, plus an observable
        list. When fc_layer is None, its length must equal n_actions; any non-None
        value creates a biased Linear layer (including False, a legacy convention).
        Weights initialize to +/-1/n_qubits and biases to zero. output_scaling=True
        adds a trainable scalar; a numeric value fixes scaling in forward.

        init accepts an initializer object, legacy name or tensor callable;
        input_scaling_init separately initializes feature-map scales (default init).
        init/device/diff_method/shots/mpi and spsa_num_directions configure the
        circuit. lightning.pp selects differentiable Pauli propagation; pp_truncation
        sets Pauli-weight cutoff and pp_topM/tau/prune_every/prune control pruning.
        These approximations can change learning. softmax_activation and torchrl
        are legacy configuration fields; forward returns scores regardless.
        Use Categorical(logits=model(x)) to form the action distribution."""
        self.cutting = cutting
        super().__init__()

        self.torchrl = torchrl

        self.circuit_builder = circuit_builder

        self.circuit_builder_copy = deepcopy(circuit_builder)

        if self.circuit_builder is None and ansatz is None:

            raise ValueError("Either circuit_builder or ansatz must be provided.")

        self.ansatz = ansatz

        self.ansatz_config = ansatz_config

        self.feature_map = feature_map

        self.feature_map_config = feature_map_config

        self.n_qubits = n_qubits

        self.n_actions = n_actions

        if self.circuit_builder is not None:

            self.n_layers = self.circuit_builder.ansatz.n_layers

        else:

            self.n_layers = self.ansatz_config["n_layers"]

        self.init = init
        self.input_scaling_init = input_scaling_init

        self.diff_method = diff_method

        self.softmax_activation = softmax_activation

        self.output_scaling = output_scaling

        if self.output_scaling:

            self.output_scaling_weight = nn.Parameter(torch.ones(1, requires_grad=True))

        self.device = device

        self.mpi = mpi

        self.diff_method = diff_method

        self.observables = observables

        self.shots = shots

        self.fc_layer = fc_layer

        self.spsa_num_directions = spsa_num_directions

        # Pauli propagation configuration

        self.pp_truncation = pp_truncation

        self.pp_topM = pp_topM

        self.pp_tau = pp_tau

        self.pp_prune_every = pp_prune_every

        self.pp_prune = pp_prune

        if self.device == "lightning.pp" and self.cutting is None:

            self.pp_engine = DifferentiablePauliProp(

                topM=self.pp_topM,

                tau=self.pp_tau,

                prune_every=self.pp_prune_every,

            )

        else:

            self.pp_engine = None

        if self.fc_layer is None:

            if len(self.observables) != self.n_actions:

                raise ValueError("Observables must be equal to the number of actions.")

        else:

            self.fc_layer = nn.Linear(len(self.observables), self.n_actions,bias=True)

            #def init_softmax_head(fc_layer, gain=0.1):
            #    
            #    nn.init.normal_(fc_layer.weight, mean=0.0, std=1)
            def init_softmax_head(fc_layer):
                """Initialize head weights to equiprobable +/-1/n_qubits and bias to zero in place."""
                with torch.no_grad():
                    bits = torch.randint(
                        low=0,
                        high=2,
                        size=fc_layer.weight.shape,
                        device=fc_layer.weight.device,
                    )

                    fc_layer.weight.copy_(
                        (1/n_qubits) * (2 * bits.to(fc_layer.weight.dtype) - 1 )
                    )

                    nn.init.zeros_(fc_layer.bias)
                    #nn.init.normal_(fc_layer.bias, mean=0.0, std=0.1)
                #nn.init.zeros_(fc_layer.bias)

            init_softmax_head(self.fc_layer)

        self.policy_type = "softmax"

        self.build_policy()

    def build_policy(self):

        """Build observable expectations, compile the backend/tape route and create quantum parameters."""
        if self.circuit_builder is None:

            self.fm = self.feature_map(**self.feature_map_config)

            ans = self.ansatz(**self.ansatz_config, feature_map=self.fm)

            self.shapes = ans.get_param_shapes()

            self.ans = ans

            self.fm = self.fm

            self.circuit_builder = CircuitBuilder(n_qubits=self.n_qubits)

            self.circuit_builder.set_ansatz(ans)

        self.circuit_builder.set_measurement(

            "expectation",

            wires=range(self.n_qubits),

            observables=self.observables,

        )

        self.c = self.circuit_builder.build()

        if self.device == "lightning.pp" and self.cutting is None:

            # Your custom CircuitBuilder.compile should return:

            #   circuit, ansatz_tape_compiled

            #

            # where ansatz_tape_compiled(inputs, params, input_scaling_params)

            # returns a PennyLane tape.

            self.circuit, self.ansatz_tape_compiled = self.circuit_builder.compile(

                device=self.device,

                diff_method=self.diff_method,

                shots=self.shots,

                spsa_num_directions=self.spsa_num_directions,

                mpi=self.mpi,

                cutting=self.cutting,
            )

        else:

            self.circuit = self.circuit_builder.compile(

                device=self.device,

                diff_method=self.diff_method,

                shots=self.shots,

                spsa_num_directions=self.spsa_num_directions,

                mpi=self.mpi,

                cutting=self.cutting,
            )

        self.qlayer = self.circuit_builder.create_torchlayer(initialization=self.init, input_scaling_init=self.input_scaling_init)

    def set_weights(self, weights):

        """Strictly load a quantum-layer state dictionary; this does not load the classical head."""
        self.qlayer.load_state_dict(weights)

    def get_weights(self):

        """Return the quantum layer's live named weight mapping, excluding classical head parameters."""
        return self.qlayer.qnode_weights

    def get_meyer_wallach(self, x):

        # TO DO

        """Unimplemented legacy placeholder; returns None without evaluating entanglement."""
        pass

    def _get_pp_output(self, x):

        """

        Differentiable Pauli propagation path.

        Important:

            Use named_parameters(), not state_dict(), otherwise PyTorch may lose

            the autograd connection to the trainable parameters.

        """

        if not hasattr(self, "ansatz_tape_compiled"):

            raise RuntimeError(

                "device='lightning.pp' requires CircuitBuilder.compile(...) to return "

                "(circuit, ansatz_tape_compiled)."

            )

        weights = dict(self.qlayer.named_parameters())

        if "params" not in weights:

            raise RuntimeError(

                "Could not find `params` in self.qlayer.named_parameters(). "

                f"Available parameters: {list(weights.keys())}"

            )

        params = weights["params"]

        input_scaling_params = weights.get("input_scaling_params", None)

        tape_kwargs = {"params": params}
        feature_map = getattr(self.circuit_builder.ansatz, "feature_map", None)
        if feature_map is not None:
            tape_kwargs["inputs"] = x
            if getattr(feature_map, "input_scaling", False):
                tape_kwargs["input_scaling_params"] = input_scaling_params
        tape = self.ansatz_tape_compiled(**tape_kwargs)

        out = self.pp_engine.execute_from_tape(

            tape,

            k=self.pp_truncation,

            prune=self.pp_prune,

        )

        return out.flatten()

    def forward(self, x, temperature=None):

        # x = torch.tanh(x)

        """Return unnormalized single-sample action logits, without input tanh.

        Execute expectations (or PP), scale them by output_scaling and the optional
        inverse-temperature multiplier, then apply the optional affine head.
        Statevector output is flattened, so this is not an arbitrary-batch API.
        Use softmax on the final axis or Categorical(logits=...) for probabilities."""
        if self.device == "lightning.pp" and self.cutting is None:

            out = self._get_pp_output(x)

        else:

            out = self.qlayer(x).flatten()

        if self.output_scaling is not None:

            if self.output_scaling is True:

                out = out * self.output_scaling_weight

            else:

                out = out * self.output_scaling

        if temperature is not None:

            out = out * temperature

        if self.fc_layer is not None:

            out = self.fc_layer(out)

        # This class returns logits/scores, not probabilities.

        # Your training code can apply Categorical(logits=out), softmax, etc.

        return out
    
    def action_probs(self, x, temperature=None):
        """Legacy tanh/softmax path over raw layer outputs.

        This method bypasses the affine head, the PP execution route and the
        trainable output_scaling_weight. It can return a different number of
        outputs from n_actions when a head is configured. For the complete model
        use softmax(self.forward(x), dim=-1), with input preprocessing as desired."""
        x = torch.tanh(x)           # each entry ∈ (-1, 1)
        out = self.qlayer(x).flatten()
        #check if output scaling is needed
        if self.output_scaling is not None:
            out = out*self.output_scaling
        if temperature is not None:
            out = out*temperature
        
        #nn functional softmax
        probs = torch.nn.functional.softmax(out, dim=-1)
        return probs

class _PolicyDecoder(nn.Module):
    """Precompute a basis-state-to-action map and aggregate with scatter-add.

    PennyLane orders probabilities as ``|00...0>, |00...1>, ..., |11...1>``;
    therefore bit position zero below is the most significant measured bit.
    """

    _ALIASES = {
        "contiguous": "contiguous-like",
        "contiguous-like": "contiguous-like",
        "parity": "parity-like",
        "parity-like": "parity-like",
        "hidden": "hidden-like",
        "hidden-like": "hidden-like",
        "majority_modulo": "majority-modulo",
        "majority-modulo": "majority-modulo",
    }

    def __init__(
        self,
        policy,
        n_actions,
        n_measured_qubits,
        majority_block_positions=None,
    ):
        """Validate the action partition and register basis-to-action indices and bit-order buffers."""
        super().__init__()
        if policy not in self._ALIASES:
            supported = ", ".join(sorted(self._ALIASES))
            raise ValueError(f"Unknown policy '{policy}'. Supported values: {supported}.")

        self.policy = self._ALIASES[policy]
        self.n_actions = int(n_actions)
        self.n_measured_qubits = int(n_measured_qubits)
        if self.n_actions < 2:
            raise ValueError("n_actions must be at least 2.")
        if self.n_measured_qubits < 1:
            raise ValueError("At least one measurement qubit is required.")
        if self.n_actions > (1 << self.n_measured_qubits):
            raise ValueError(
                f"n_actions={self.n_actions} exceeds the {1 << self.n_measured_qubits} "
                "measured basis states."
            )

        self.n_code_bits = math.ceil(math.log2(self.n_actions))
        self.majority_block_positions = (
            None
            if majority_block_positions is None
            else tuple(tuple(int(p) for p in block) for block in majority_block_positions)
        )
        action_index = self._build_action_index()
        basis_weights = 1 << torch.arange(
            self.n_measured_qubits - 1, -1, -1, dtype=torch.long
        )
        self.register_buffer("action_index", action_index, persistent=True)
        self.register_buffer("basis_weights", basis_weights, persistent=False)

    def _basis_bits(self):
        """Enumerate basis indices and bits with the first measured wire as the most significant bit."""
        states = torch.arange(1 << self.n_measured_qubits, dtype=torch.long)
        shifts = torch.arange(
            self.n_measured_qubits - 1, -1, -1, dtype=torch.long
        )
        return states, torch.bitwise_and(
            torch.bitwise_right_shift(states[:, None], shifts[None, :]), 1
        )

    def _build_action_index(self):
        """Map every measured basis state to a contiguous, prefix-parity or strict-majority modulo action."""
        n_states = 1 << self.n_measured_qubits
        if self.policy == "contiguous-like":
            base, remainder = divmod(n_states, self.n_actions)
            sizes = torch.full((self.n_actions,), base, dtype=torch.long)
            sizes[:remainder] += 1
            return torch.repeat_interleave(
                torch.arange(self.n_actions, dtype=torch.long), sizes
            )

        _, bits = self._basis_bits()

        if self.policy in {"parity-like", "hidden-like"}:
            parity_bits = torch.stack(
                [
                    bits[:, : self.n_measured_qubits - offset]
                    .sum(dim=1)
                    .remainder(2)
                    for offset in range(self.n_code_bits)
                ],
                dim=1,
            )
            code_weights = 1 << torch.arange(
                self.n_code_bits - 1, -1, -1, dtype=torch.long
            )
            return (parity_bits * code_weights).sum(dim=1).remainder(self.n_actions)

        if self.policy == "majority-modulo":
            blocks = self.majority_block_positions
            if blocks is None:
                raise ValueError("majority_block_positions are required for majority-modulo.")
            if len(blocks) != self.n_code_bits:
                raise ValueError(
                    "majority-modulo requires exactly ceil(log2(n_actions)) "
                    f"={self.n_code_bits} blocks; received {len(blocks)}."
                )
            majority_bits = []
            for block in blocks:
                if not block:
                    raise ValueError("Majority blocks must be non-empty.")
                block_bits = bits[:, list(block)]
                majority_bits.append(
                    (block_bits.sum(dim=1) > (len(block) // 2)).to(torch.long)
                )
            majority_bits = torch.stack(majority_bits, dim=1)
            code_weights = 1 << torch.arange(
                self.n_code_bits - 1, -1, -1, dtype=torch.long
            )
            return (majority_bits * code_weights).sum(dim=1).remainder(self.n_actions)

        raise RuntimeError(f"Unhandled policy '{self.policy}'.")

    def forward(self, probabilities):
        """Aggregate probabilities while preserving all leading batch dimensions."""
        if probabilities.ndim < 1:
            raise ValueError("probabilities must have at least one dimension.")
        expected_states = self.action_index.numel()
        if probabilities.shape[-1] != expected_states:
            raise ValueError(
                f"Expected {expected_states} basis probabilities on the final axis, "
                f"received {probabilities.shape[-1]}."
            )

        leading_shape = probabilities.shape[:-1]
        flat = probabilities.reshape(-1, expected_states)
        result = flat.new_zeros((flat.shape[0], self.n_actions))
        indices = self.action_index.unsqueeze(0).expand(flat.shape[0], -1)
        result.scatter_add_(1, indices, flat)
        return result.reshape(*leading_shape, self.n_actions)

    def decode_samples(self, samples):
        """Decode one sample ``[n]`` or a sample batch ``[..., n]``."""
        samples = torch.as_tensor(samples, device=self.action_index.device)
        if samples.ndim < 1 or samples.shape[-1] != self.n_measured_qubits:
            raise ValueError(
                f"Expected samples with final dimension {self.n_measured_qubits}; "
                f"received shape {tuple(samples.shape)}."
            )
        if not torch.all((samples == 0) | (samples == 1)):
            raise ValueError("Samples must contain only binary values 0 and 1.")
        basis_index = (samples.to(torch.long) * self.basis_weights).sum(dim=-1)
        return self.action_index[basis_index]


class PQCBornVariance(nn.Module):
    """Born policy with fast, consistent classical action decoding.

    Supported policies
    ------------------
    ``"contiguous-like"``
        Contiguous, near-balanced groups of measured basis states.
    ``"parity-like"``
        Recursive prefix parities, interpreted most-significant parity first and
        reduced modulo ``n_actions``.
    ``"hidden-like"``
        Uses every second qubit by default and the same recursive-parity decoder.
    ``"majority-modulo"``
        Computes one majority bit per block, interprets the majority code as an
        integer, and reduces it modulo ``n_actions``.

    Majority-modulo is exact for ``training_method="probs"`` and
    ``training_method="sparse_projectors"``. The latter evaluates exact diagonal
    action projectors with PennyLane's standard Lightning adjoint path. Device
    VJPs are deliberately disabled because they return incorrect gradients for
    the multi-projector (for example, four-action) case in the tested stack.
    Its action projectors are Fourier dense, so truncated-Pauli modes are
    deliberately rejected instead of silently using an incorrect reconstruction.

    ``majority_blocks`` contains actual wire labels, not positions. Explicit
    blocks may have even size; an equal zero/one tie deterministically produces
    majority bit zero. If omitted, QASP creates maximally covering, nearly
    balanced odd blocks and leaves at most one measured qubit unused. Set
    ``strict_majority_blocks=True`` to require equal odd block sizes of at least
    three, matching the clean theorem assumptions.
    """

    _POLICY_ALIASES = _PolicyDecoder._ALIASES

    def __init__(
        self,
        circuit_builder=None,
        ansatz=None,
        ansatz_config=None,
        feature_map=None,
        feature_map_config=None,
        n_qubits=1,
        n_actions=2,
        init="normal_0_1",
        policy="contiguous-like",
        device="default.qubit",
        diff_method="best",
        softmax_activation=None,
        output_scaling=None,
        training_method="probs",
        torchrl=False,
        measurement_qubits=None,
        z_gates=None,
        n_ancillas=0,
        CNOT_gates=None,
        parity_cnot_decomposition=False,
        shots=None,
        pp_truncation=2,
        enable_mpi=None,
        cut_circuit=False,
        spsa_num_directions=1,
        majority_blocks=None,
        strict_majority_blocks=False,
        mpi=None,
        input_scaling_init=None,
        cutting=None,
    ):
        """Build a Born policy with explicit basis-to-action decoding.

        Supply a builder or embedding/ansatz classes and configs. n_actions>=2;
        policy selects contiguous-like, parity-like, hidden-like or majority-modulo
        (aliases are accepted). measurement_qubits sets physical wires. Majority
        blocks are disjoint physical-wire lists; defaults use balanced odd blocks
        and may leave wires unused. strict_majority_blocks requests equal blocks
        of size >=3 with no unused wires; explicit even blocks are not separately
        rejected by the current strict check, and ties decode to zero.

        training_method chooses probs, expectation, local_pauli_decomposition or
        majority-only sparse_projectors. Sparse projectors require analytic adjoint
        on lightning.qubit/kokkos, without MPI/cutting. Majority's probability/
        sparse-projector paths do not support lightning.pp; PQDQN majority Pauli
        features are a different construction. PP truncation applies to supported
        expectation routes. parity_cnot_decomposition, CNOT_gates and n_ancillas
        configure optional ancilla readout.

        init accepts an initializer object, legacy name or tensor callable;
        input_scaling_init separately initializes feature-map scales (default init).
        init/device/diff_method/shots/mpi (legacy alias enable_mpi), cut_circuit
        and spsa_num_directions control execution. softmax_activation enables a
        post-decoder softmax with optional trainable output scaling; temperature
        in forward is an inverse-temperature multiplier. z_gates is retained
        metadata. The attached builder is mutable, not an immutable circuit copy."""
        self.cutting = cutting
        super().__init__()
        if circuit_builder is None and ansatz is None:
            raise ValueError("Either circuit_builder or ansatz must be provided.")
        if policy not in self._POLICY_ALIASES:
            supported = ", ".join(sorted(self._POLICY_ALIASES))
            raise ValueError(f"Unknown policy '{policy}'. Supported values: {supported}.")
        if int(n_actions) < 2:
            raise ValueError("n_actions must be at least 2.")

        self.torchrl = torchrl
        self.circuit_builder = circuit_builder
        self.circuit_builder_copy = deepcopy(circuit_builder)
        self.ansatz = ansatz
        self.ansatz_config = ansatz_config
        self.feature_map = feature_map
        self.feature_map_config = feature_map_config
        self.n_qubits = int(n_qubits)
        self.n_ancillas = int(n_ancillas)
        self.n_actions = int(n_actions)
        self.n_layers = (
            self.circuit_builder.ansatz.n_layers
            if self.circuit_builder is not None
            else self.ansatz_config["n_layers"]
        )
        self.init = init
        self.input_scaling_init = input_scaling_init
        self.diff_method = diff_method
        self.policy = self._POLICY_ALIASES[policy]
        self.policy_type = self.policy
        self.softmax_activation = softmax_activation
        self.output_scaling = output_scaling
        self.training_method = training_method
        self.device = device
        self.measurement_qubits = (
            None if measurement_qubits is None else list(measurement_qubits)
        )
        self.z_gates = z_gates
        self.CNOT_gates = CNOT_gates
        self.parity_cnot_decomposition = parity_cnot_decomposition
        self.shots = shots
        self.pp_truncation = pp_truncation
        from qasp.devices import resolve_mpi

        self.mpi = self.enable_mpi = resolve_mpi(mpi, enable_mpi)
        self.cut_circuit = cut_circuit
        self.spsa_num_directions = spsa_num_directions
        self.requested_majority_blocks = majority_blocks
        self.strict_majority_blocks = bool(strict_majority_blocks)

        if self.output_scaling:
            self.output_scaling_weight = nn.Parameter(torch.ones(1))

        self.cnot_table = {
            ("I", "I"): (("I", "I"), 1),
            ("I", "X"): (("I", "X"), 1),
            ("I", "Y"): (("Z", "Y"), 1),
            ("I", "Z"): (("Z", "Z"), 1),
            ("X", "I"): (("X", "X"), 1),
            ("X", "X"): (("X", "I"), 1),
            ("X", "Y"): (("Y", "Z"), 1),
            ("X", "Z"): (("Y", "Y"), -1),
            ("Y", "I"): (("Y", "X"), 1),
            ("Y", "X"): (("Y", "I"), 1),
            ("Y", "Y"): (("X", "Z"), -1),
            ("Y", "Z"): (("X", "Y"), 1),
            ("Z", "I"): (("Z", "I"), 1),
            ("Z", "X"): (("Z", "X"), 1),
            ("Z", "Y"): (("I", "Y"), 1),
            ("Z", "Z"): (("I", "Z"), 1),
        }

        self.build_policy()

    @staticmethod
    def _balanced_odd_block_sizes(n_items, n_blocks):
        """Use as many items as possible in balanced, positive odd blocks."""
        if n_blocks < 1 or n_items < n_blocks:
            raise ValueError(
                f"Cannot construct {n_blocks} non-empty blocks from {n_items} qubits."
            )
        used = n_items if (n_items - n_blocks) % 2 == 0 else n_items - 1
        pairs, extra = divmod((used - n_blocks) // 2, n_blocks)
        sizes = [1 + 2 * pairs + (2 if i < extra else 0) for i in range(n_blocks)]
        return sizes, n_items - used

    def _resolve_majority_blocks(self):
        """Validate/choose physical majority blocks, store unused wires and return measured-wire positions."""
        k = math.ceil(math.log2(self.n_actions))
        wires = list(self.measurement_qubits)
        wire_to_position = {wire: position for position, wire in enumerate(wires)}
        if len(wire_to_position) != len(wires):
            raise ValueError("measurement_qubits must not contain duplicates.")

        if self.requested_majority_blocks is None:
            sizes, n_unused = self._balanced_odd_block_sizes(len(wires), k)
            blocks = []
            start = 0
            for size in sizes:
                blocks.append(wires[start : start + size])
                start += size
            unused = wires[start:]
            assert len(unused) == n_unused
        else:
            blocks = [list(block) for block in self.requested_majority_blocks]
            if len(blocks) != k:
                raise ValueError(
                    f"majority_blocks must contain k=ceil(log2(n_actions))={k} "
                    f"blocks; received {len(blocks)}."
                )
            flattened = [wire for block in blocks for wire in block]
            unknown = [wire for wire in flattened if wire not in wire_to_position]
            if unknown:
                raise ValueError(
                    f"majority_blocks contain wires not in measurement_qubits: {unknown}."
                )
            if len(set(flattened)) != len(flattened):
                raise ValueError("majority_blocks must be disjoint.")
            unused = [wire for wire in wires if wire not in set(flattened)]

        sizes = [len(block) for block in blocks]
        if any(size == 0 for size in sizes):
            raise ValueError(
                "Every majority block must have positive size; "
                f"received sizes {sizes}."
            )
        if self.strict_majority_blocks and (
            len(set(sizes)) != 1 or sizes[0] < 3 or unused
        ):
            raise ValueError(
                "strict_majority_blocks=True requires equal odd block sizes >= 3 "
                f"and no unused measured wires; received sizes={sizes}, unused={unused}."
            )

        positions = [tuple(wire_to_position[wire] for wire in block) for block in blocks]
        self.majority_blocks = tuple(tuple(block) for block in blocks)
        self.unused_majority_qubits = tuple(unused)
        return positions

    def _configure_measurement_qubits(self):
        """Choose decoder-specific measured wires and require enough unique wires for the action codes."""
        k = math.ceil(math.log2(self.n_actions))
        if self.policy == "contiguous-like":
            if self.measurement_qubits is None:
                self.measurement_qubits = list(range(k))
        elif self.policy == "hidden-like":
            if self.measurement_qubits is None:
                self.measurement_qubits = list(range(0, self.n_qubits, 2))
        else:  # parity-like and majority-modulo
            if self.measurement_qubits is None:
                self.measurement_qubits = list(range(self.n_qubits))

        if not self.measurement_qubits:
            raise ValueError("measurement_qubits must be non-empty.")
        if len(set(self.measurement_qubits)) != len(self.measurement_qubits):
            raise ValueError("measurement_qubits must not contain duplicates.")
        if k > len(self.measurement_qubits):
            raise ValueError(
                f"Need at least ceil(log2(n_actions))={k} measured qubits; "
                f"received {len(self.measurement_qubits)}."
            )

    def _build_sparse_action_projectors(self):
        """Return exact projectors for actions ``0, ..., n_actions - 2``.

        The final action projector is omitted because its probability is exactly
        one minus the sum of the other action probabilities. PennyLane-Lightning
        0.42 requires a SparseHamiltonian used by the adjoint method to span all
        device wires, so unmeasured wires and ancillas are included as identities.
        """
        all_wires = tuple(range(self.n_qubits + self.n_ancillas))
        measured_wires = tuple(self.measurement_qubits)
        unknown_wires = sorted(set(measured_wires) - set(all_wires))
        if unknown_wires:
            raise ValueError(
                f"measurement_qubits contain wires outside the device: {unknown_wires}."
            )

        n_full_qubits = len(all_wires)
        dimension = 1 << n_full_qubits
        wire_position = {wire: position for position, wire in enumerate(all_wires)}
        full_basis_states = onp.arange(dimension, dtype=onp.uint64)
        measured_basis_index = onp.zeros(dimension, dtype=onp.uint64)

        # PennyLane uses big-endian basis ordering. Preserve measurement_qubits'
        # order because _PolicyDecoder uses the same order for majority blocks.
        for wire in measured_wires:
            shift = n_full_qubits - 1 - wire_position[wire]
            bit = (full_basis_states >> shift) & 1
            measured_basis_index = (measured_basis_index << 1) | bit

        measured_action_index = (
            self.policy_decoder.action_index.detach().cpu().numpy()
        )
        full_action_index = measured_action_index[
            measured_basis_index.astype(onp.int64, copy=False)
        ]

        projectors = []
        for action in range(self.n_actions - 1):
            diagonal_indices = onp.flatnonzero(full_action_index == action).astype(
                onp.int64, copy=False
            )
            values = onp.ones(diagonal_indices.size, dtype=onp.float64)
            matrix = csr_matrix(
                (values, (diagonal_indices, diagonal_indices)),
                shape=(dimension, dimension),
                dtype=onp.float64,
            )
            projectors.append(qml.SparseHamiltonian(matrix, wires=all_wires))

        return tuple(projectors)

    def build_policy(self):
        """Validate readout/backend compatibility, construct decoder and measurements, and compile trainable layers."""
        if self.circuit_builder is None:
            if self.feature_map is None or self.feature_map_config is None:
                raise ValueError(
                    "feature_map and feature_map_config are required when circuit_builder is None."
                )
            self.fm = self.feature_map(**self.feature_map_config)
            ans = self.ansatz(**self.ansatz_config, feature_map=self.fm)
            self.shapes = ans.get_param_shapes()
            self.circuit_builder = CircuitBuilder(
                n_qubits=self.n_qubits, n_ancillas=self.n_ancillas
            )
            self.circuit_builder.set_ansatz(ans)

        self._configure_measurement_qubits()

        if self.policy == "majority-modulo" and self.training_method not in {
            "probs",
            "sparse_projectors",
        }:
            raise ValueError(
                "majority-modulo requires training_method='probs' or "
                "training_method='sparse_projectors'."
            )
        if self.policy == "majority-modulo" and self.device == "lightning.pp" and self.cutting is None:
            raise ValueError(
                "majority-modulo requires full basis probabilities and is not compatible "
                "with the current lightning.pp expectation-only path."
            )
        if self.training_method == "sparse_projectors":
            if self.policy != "majority-modulo":
                raise ValueError(
                    "training_method='sparse_projectors' is currently available only "
                    "for majority-modulo."
                )
            if self.device not in {"lightning.qubit", "lightning.kokkos"}:
                raise ValueError(
                    "training_method='sparse_projectors' requires device="
                    "'lightning.qubit' or 'lightning.kokkos'."
                )
            if self.diff_method != "adjoint":
                raise ValueError(
                    "training_method='sparse_projectors' requires diff_method='adjoint'."
                )
            if self.shots is not None:
                raise ValueError(
                    "SparseHamiltonian measurements require analytic execution; "
                    "set shots=None."
                )
            if self.device == "lightning.kokkos" and self.enable_mpi:
                raise ValueError(
                    "SparseHamiltonian is not supported by lightning.kokkos with MPI "
                    "in PennyLane-Lightning 0.42; set enable_mpi=False."
                )
            if self.cut_circuit:
                raise ValueError(
                    "Sparse action projectors are not compatible with cut_circuit=True."
                )

        majority_positions = None
        if self.policy == "majority-modulo":
            majority_positions = self._resolve_majority_blocks()
        else:
            self.majority_blocks = tuple()
            self.unused_majority_qubits = tuple()

        self.policy_decoder = _PolicyDecoder(
            policy=self.policy,
            n_actions=self.n_actions,
            n_measured_qubits=len(self.measurement_qubits),
            majority_block_positions=majority_positions,
        )

        if self.policy == "parity-like" and self.parity_cnot_decomposition:
            self.circuit_builder.set_CNOT_gates(self.CNOT_gates)

        if self.training_method == "probs":
            self.circuit_builder.set_measurement("probs", wires=self.measurement_qubits)
            self._expectation_decoder = None
        elif self.training_method == "sparse_projectors":
            self.sparse_action_projectors = self._build_sparse_action_projectors()
            self.circuit_builder.set_measurement(
                "expectation",
                wires=range(self.n_qubits + self.n_ancillas),
                observables=self.sparse_action_projectors,
            )
            self._expectation_decoder = "sparse-action-projectors"
        elif self.training_method == "local_pauli_decomposition":
            if self.torchrl:
                raise ValueError("TorchRL does not support local_pauli_decomposition.")
            observables = [qml.PauliZ(i) for i in self.measurement_qubits]
            self.circuit_builder.set_measurement(
                "expectation", wires=self.measurement_qubits, observables=observables
            )
            self._expectation_decoder = None
        elif self.training_method == "expectation":
            if self.policy == "parity-like" and not self.parity_cnot_decomposition:
                labels = self.parity_observable_labels_no_ancilla(
                    self.measurement_qubits, self.n_actions, order="default"
                )
                self._expectation_n_bits = math.ceil(math.log2(self.n_actions))
                self._expectation_decoder = "logical-modulo"
            else:
                labels = self._all_z_observable_labels(self.measurement_qubits)
                self._expectation_n_bits = len(self.measurement_qubits)
                self._expectation_decoder = (
                    "logical-modulo"
                    if self.policy == "parity-like" and self.parity_cnot_decomposition
                    else "basis-policy"
                )
            self.circuit_builder.set_measurement(
                "expectation", wires=self.measurement_qubits, observables=labels
            )
            mask_order = self._combination_mask_order(self._expectation_n_bits)
            self.register_buffer(
                "_expectation_mask_order",
                torch.tensor(mask_order, dtype=torch.long),
                persistent=False,
            )
        else:
            raise ValueError(
                "training_method must be 'probs', 'sparse_projectors', "
                "'expectation', or 'local_pauli_decomposition'."
            )

        self.c = self.circuit_builder.build()
        if self.device == "lightning.pp" and self.cutting is None:
            self.circuit, self.ansatz_tape_compiled = self.circuit_builder.compile(
                device=self.device, diff_method=self.diff_method, shots=self.shots,
                mpi=self.enable_mpi,
                cutting=self.cutting,
            )
            self.pp_engine = DifferentiablePauliProp(
                topM=20000, tau=1e-10, prune_every=5
            )
        else:
            self.circuit = self.circuit_builder.compile(
                device=self.device,
                diff_method=self.diff_method,
                shots=self.shots,
                mpi=self.enable_mpi,
                cut_circuit=self.cut_circuit,
                spsa_num_directions=self.spsa_num_directions,
                # Lightning's device-native VJP produces incorrect gradients
                # for QNodes containing three or more SparseHamiltonian
                # measurements (for example |A|=4 uses three projectors).
                # The standard PennyLane adjoint path is still statevector
                # adjoint differentiation and agrees with full-probability
                # backpropagation.
                device_vjp=False,
                batch_obs=self.training_method != "sparse_projectors",
                cutting=self.cutting,
            )
        self.qlayer = self.circuit_builder.create_torchlayer(initialization=self.init, input_scaling_init=self.input_scaling_init)

        if self.training_method == "local_pauli_decomposition":
            self.circuit_builder_copy = deepcopy(self.circuit_builder)
            self.circuit_builder_copy.set_measurement(
                "sample", wires=self.measurement_qubits
            )
            self.c_copy = self.circuit_builder_copy.build()
            self.circuit_copy = self.circuit_builder_copy.compile(
                device=self.device,
                diff_method=self.diff_method,
                shots=self.shots,
                mpi=self.enable_mpi,
                cut_circuit=self.cut_circuit,
                spsa_num_directions=self.spsa_num_directions,
                cutting=self.cutting,
            )
            self.qlayer_copy = self.circuit_builder_copy.create_torchlayer(
                initialization=self.init, input_scaling_init=self.input_scaling_init
            )
            self.qlayer_copy.load_state_dict(self.qlayer.state_dict())

    @staticmethod
    def _all_z_observable_labels(wires):
        """List all nonempty measured-wire Z products, ordered by support size then combinations."""
        labels = []
        wires = list(wires)
        for size in range(1, len(wires) + 1):
            for combo in combinations(wires, size):
                labels.append(("Z" * size) + " " + " ".join(map(str, combo)))
        return labels

    @staticmethod
    def _combination_mask_order(n_bits):
        """Map size-ordered subsets to bit masks with the first logical bit most significant."""
        order = []
        for size in range(1, n_bits + 1):
            for combo in combinations(range(n_bits), size):
                mask = 0
                for position in combo:
                    mask |= 1 << (n_bits - 1 - position)
                order.append(mask)
        return order

    def parity_observable_labels_no_ancilla(
        self, measurement_qubits, n_actions, order="example"
    ):
        """Return products of the nested recursive-parity observables."""
        qubits = list(measurement_qubits)
        if not qubits:
            raise ValueError("measurement_qubits must be non-empty.")
        m = math.ceil(math.log2(n_actions))
        if len(qubits) < m:
            raise ValueError(
                f"Need at least ceil(log2(n_actions))={m} measurement qubits."
            )

        base_sets = [set(qubits[: len(qubits) - offset]) for offset in range(m)]
        labels = []
        for size in range(1, m + 1):
            for combo in combinations(range(m), size):
                wires = set()
                for index in combo:
                    wires ^= base_sets[index]
                ordered_wires = sorted(wires)
                if ordered_wires:
                    labels.append(
                        ("Z" * len(ordered_wires))
                        + " "
                        + " ".join(map(str, ordered_wires))
                    )

        if order == "by_size":
            labels.sort(key=lambda label: len(label.split()) - 1)
        elif order == "example" and m == 2:
            z_last = f"Z {qubits[-1]}"
            z_all = ("Z" * len(qubits)) + " " + " ".join(map(str, qubits))
            z_prefix = ("Z" * (len(qubits) - 1)) + " " + " ".join(
                map(str, qubits[:-1])
            )
            preferred = [z_last, z_all, z_prefix]
            labels = [label for label in preferred if label in labels] + [
                label for label in labels if label not in preferred
            ]
        elif order not in {"default", "example", "by_size"}:
            raise ValueError("order must be 'default', 'example', or 'by_size'.")
        return labels

    @staticmethod
    def _fwht(values):
        """Differentiable fast Walsh-Hadamard transform along the final axis."""
        transformed = values
        width = 1
        length = values.shape[-1]
        while width < length:
            blocks = transformed.reshape(*transformed.shape[:-1], -1, 2, width)
            left = blocks[..., 0, :]
            right = blocks[..., 1, :]
            transformed = torch.cat((left + right, left - right), dim=-1).reshape(
                *transformed.shape[:-1], length
            )
            width *= 2
        return transformed

    def _probabilities_from_z_moments(self, moments):
        """Reconstruct basis probabilities by Walsh transform; clamp negative values and renormalize."""
        n_bits = self._expectation_n_bits
        expected = (1 << n_bits) - 1

        # TorchLayer stacks multiple QNode measurements on axis zero. Move that
        # measurement axis to the end before applying the Walsh reconstruction.
        # The binary case has one measurement and already arrives with a final
        # singleton axis, which is why this was previously hidden by |A|=2.
        if expected > 1:
            if moments.ndim < 1 or moments.shape[0] != expected:
                raise ValueError(
                    f"Expected {expected} non-identity Z moments on axis zero; "
                    f"received shape {tuple(moments.shape)}."
                )
            moments = moments.movedim(0, -1)
            if moments.ndim >= 2 and moments.shape[-2] == 1:
                moments = moments.squeeze(-2)

        if moments.shape[-1] != expected:
            raise ValueError(
                f"Expected {expected} non-identity Z moments, got {moments.shape[-1]}."
            )
        full = moments.new_zeros(*moments.shape[:-1], expected + 1)
        full[..., 0] = 1
        indices = self._expectation_mask_order.expand(*moments.shape[:-1], -1)
        full = full.scatter(-1, indices, moments)
        probabilities = self._fwht(full) / (1 << n_bits)
        probabilities = probabilities.clamp_min(0)
        denominator = probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probabilities / denominator

    def _aggregate_modulo(self, probabilities):
        """Sum final-axis code probabilities into code % n_actions bins, preserving batch axes."""
        states = probabilities.shape[-1]
        action_index = torch.arange(states, device=probabilities.device).remainder(
            self.n_actions
        )
        flat = probabilities.reshape(-1, states)
        result = flat.new_zeros(flat.shape[0], self.n_actions)
        result.scatter_add_(
            1, action_index.unsqueeze(0).expand(flat.shape[0], -1), flat
        )
        return result.reshape(*probabilities.shape[:-1], self.n_actions)

    def _probabilities_from_sparse_projectors(self, projector_values):
        """Append the omitted final action probability and normalize roundoff."""
        n_projectors = self.n_actions - 1

        # TorchLayer stacks multiple QNode measurements on axis zero. Move that
        # measurement axis to the end so action probabilities use the same shape
        # convention as every other policy decoder.
        if n_projectors > 1:
            if projector_values.ndim < 1 or projector_values.shape[0] != n_projectors:
                raise ValueError(
                    f"Expected {n_projectors} sparse projector measurements on axis "
                    f"zero; received shape {tuple(projector_values.shape)}."
                )
            projector_values = projector_values.movedim(0, -1)
            if projector_values.ndim >= 2 and projector_values.shape[-2] == 1:
                projector_values = projector_values.squeeze(-2)

        last_probability = 1.0 - projector_values.sum(dim=-1, keepdim=True)
        probabilities = torch.cat((projector_values, last_probability), dim=-1)

        # Expectation values can overshoot [0, 1] at machine precision. Keep the
        # output valid for torch.distributions.Categorical without changing the
        # exact probabilities away from those numerical boundaries.
        probabilities = probabilities.clamp_min(0.0)
        eps = torch.finfo(probabilities.dtype).eps
        return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)

    def post_process_policy(
        self,
        probs,
        policy_type=None,
        n_actions=None,
        n_qubits=None,
        softmax_activation=None,
    ):
        """Convert circuit outputs to action probabilities without Python state loops.

        Legacy arguments are accepted for call-site compatibility, but the validated
        configuration stored by this module is authoritative.
        """
        if policy_type is not None:
            normalized = self._POLICY_ALIASES.get(policy_type)
            if normalized != self.policy:
                raise ValueError(
                    f"Configured policy is '{self.policy}', received '{policy_type}'."
                )
        if n_actions is not None and int(n_actions) != self.n_actions:
            raise ValueError(
                f"Configured n_actions={self.n_actions}, received {n_actions}."
            )

        if self.training_method == "probs":
            return self.policy_decoder(probs)
        if self.training_method == "sparse_projectors":
            return self._probabilities_from_sparse_projectors(probs)
        if self.training_method == "expectation":
            basis_probabilities = self._probabilities_from_z_moments(probs)
            if self._expectation_decoder == "logical-modulo":
                return self._aggregate_modulo(basis_probabilities)
            return self.policy_decoder(basis_probabilities)
        raise NotImplementedError(
            "Direct action-probability reconstruction is not defined for "
            "training_method='local_pauli_decomposition'."
        )

    def post_process_sample(self, sample):
        """Decode measured bits into actions, returning a Python int for a scalar or a tensor for batches."""
        actions = self.policy_decoder.decode_samples(sample)
        return int(actions.item()) if actions.ndim == 0 else actions

    def update_sampler_weights(self):
        """Synchronize local-decomposition sampler weights; raise RuntimeError if no sampler exists."""
        if not hasattr(self, "qlayer_copy"):
            raise RuntimeError(
                "A sampler copy exists only for local_pauli_decomposition."
            )
        self.qlayer_copy.load_state_dict(self.qlayer.state_dict())

    def set_weights(self, params=None, input_scaling_params=None):
        """Legacy setter that calls set_weights on the compiled circuit.

        Ordinary PennyLane QNodes lack that method. Prefer
        ``model.qlayer.load_state_dict(...)`` or the builder's set_weights with
        both parameter groups when input scaling is enabled."""
        if params is not None:
            self.circuit.set_weights(params=params)
        if input_scaling_params is not None:
            self.circuit.set_weights(input_scaling_params=input_scaling_params)

    def get_weights(self):
        """Return the quantum layer's live named weight mapping, excluding classical head parameters."""
        return self.qlayer.qnode_weights

    def get_meyer_wallach(self, x):
        """Raise NotImplementedError; entanglement evaluation is not implemented here."""
        raise NotImplementedError("Meyer-Wallach evaluation is not implemented.")

    def post_process_expval(self, action, exp_val):
        """Return log of the mean single-wire probability matching the action's bits.

        This local surrogate averages marginal probabilities; it is not the log
        of a reconstructed joint action probability for correlated measurements."""
        action_bin = format(int(action), f"0{len(self.measurement_qubits)}b")
        expvalue = exp_val.new_zeros(())
        for index, bit in enumerate(action_bin):
            expvalue = expvalue + 0.5 + 0.5 * ((-1) ** int(bit)) * exp_val[index]
        return torch.log(expvalue / len(self.measurement_qubits))

    def apply_cnot(self, wires, H, k=None):
        """Apply a CNOT to an operator in the Heisenberg picture."""
        new_H = PauliSentence()
        for pauli_word, coeff in H.items():
            op_pw_0 = pauli_word.get(wires[0], "I")
            op_pw_1 = pauli_word.get(wires[1], "I")
            (new_op_pw_0, new_op_pw_1), factor = self.cnot_table[
                (op_pw_0, op_pw_1)
            ]
            new_pw = pauli_word.copy()
            new_pw.update({wires[0]: new_op_pw_0, wires[1]: new_op_pw_1})
            new_pw = PauliWord(new_pw)
            if k is None or len(new_pw) <= k:
                new_H[new_pw] += factor * coeff
        return new_H

    def apply_single_qubit_rot(self, pauli, wire, param, H):
        """Apply a one-qubit Pauli rotation in the Heisenberg picture."""
        new_H = PauliSentence()
        rotation_word = PauliWord({wire: pauli})
        for pauli_word, coeff in H.items():
            if pauli_word.commutes_with(rotation_word):
                new_H[pauli_word] += coeff
            else:
                new_H[pauli_word] += qml.math.cos(param) * coeff
                new_pauli_word, factor = list((rotation_word @ pauli_word).items())[0]
                new_H[new_pauli_word] += (
                    qml.math.sin(param) * coeff * factor * 1j
                ).real
        return new_H

    @staticmethod
    def initial_state_expval(H):
        """Evaluate a Pauli sentence in the all-zero state."""
        expval = 0.0
        for pauli_word, coeff in H.items():
            if all(pauli in {"I", "Z"} for pauli in pauli_word.values()):
                expval += coeff
        return expval

    def execute_tape(self, x, k=None):
        """Classically estimate all tape observables by truncated Pauli propagation."""
        weights = self.qlayer.state_dict()
        tape_kwargs = {"params": weights["params"]}
        feature_map = getattr(self.circuit_builder.ansatz, "feature_map", None)
        if feature_map is not None:
            tape_kwargs["inputs"] = x
            if getattr(feature_map, "input_scaling", False):
                tape_kwargs["input_scaling_params"] = weights.get(
                    "input_scaling_params"
                )
        tape = self.ansatz_tape_compiled(**tape_kwargs)
        batch, _ = qml.transforms.decompose(
            tape, gate_set={qml.CNOT, qml.RX, qml.RY, qml.RZ}
        )
        decomposed = batch[0]
        expvals = torch.zeros(len(decomposed.measurements), dtype=torch.float32)
        for measurement_index, measurement in enumerate(decomposed.measurements):
            H = measurement.obs.pauli_rep
            for op in reversed(decomposed.operations):
                if isinstance(op, qml.CNOT):
                    H = self.apply_cnot(op.wires, H, k=k)
                elif isinstance(op, (qml.RZ, qml.RX, qml.RY)):
                    H = self.apply_single_qubit_rot(
                        op.name[-1], op.wires[0], op.data[0], H
                    )
                else:
                    raise NotImplementedError(
                        f"Pauli propagation does not support operation {op.name}."
                    )
            expvals[measurement_index] = self.initial_state_expval(H)
        return expvals

    def _circuit_output(self, x):
        """Evaluate the layer or trace live quantum parameters through differentiable truncated PP."""
        if self.device != "lightning.pp" or self.cutting is not None:
            return self.qlayer(x)

        weights = dict(self.qlayer.named_parameters())
        tape_kwargs = {"params": weights["params"]}
        feature_map = getattr(self.circuit_builder.ansatz, "feature_map", None)
        if feature_map is not None:
            tape_kwargs["inputs"] = x
            if getattr(feature_map, "input_scaling", False):
                tape_kwargs["input_scaling_params"] = weights.get(
                    "input_scaling_params"
                )
        tape = self.ansatz_tape_compiled(**tape_kwargs)
        return self.pp_engine.execute_from_tape(tape, k=self.pp_truncation, prune=True)

    def forward(self, x, temperature=None):
        """Return decoded action probabilities; apply optional post-decoder softmax.

        No input tanh is applied here. PP results are clamped/renormalized with a
        uniform fallback if mass vanishes. If softmax_activation is enabled,
        temperature multiplies the weights (inverse-temperature convention), and
        the final softmax uses n_actions times the scaled decoder probabilities."""
        output_qlayer = self._circuit_output(x)
        action_probs = self.post_process_policy(output_qlayer)

        if self.device == "lightning.pp" and self.cutting is None:
            action_probs = action_probs.clamp_min(0)
            denominator = action_probs.sum(dim=-1, keepdim=True)
            uniform = torch.full_like(action_probs, 1.0 / self.n_actions)
            action_probs = torch.where(
                denominator > 1e-8,
                action_probs / denominator.clamp_min(1e-8),
                uniform,
            )

        if self.softmax_activation:
            # Preserve the historical QASP interpretation: temperature is an
            # inverse-temperature multiplier rather than a divisor.
            if temperature is not None:
                action_probs = action_probs * temperature
            if self.output_scaling:
                action_probs = action_probs * self.output_scaling_weight
            action_probs = torch.softmax(self.n_actions * action_probs, dim=-1)
        return action_probs

    def action_probs(self, x, temperature=None):
        """Compatibility alias that shares exactly the same path as forward()."""
        return self.forward(x, temperature=temperature)

class LearnablePQCBorn(nn.Module):
    """Experimental policy with a sampled state-dependent measurement subset.

    A classical selector enumerates wire combinations and supplies a log
    probability for score-function training. Adaptive measurements and legacy
    postprocessing have restrictions documented on the constructor and methods;
    this prototype is not a drop-in replacement for PQCBornVariance."""


    def __init__(self, circuit_builder=None, ansatz=None, ansatz_config=None, feature_map=None, feature_map_config=None, n_qubits=1, n_actions=2, init="normal_0_1", device="default.qubit", diff_method="best" , softmax_activation=None, output_scaling=None, training_method="probs", torchrl=False, n_qubits_measure=None, mpi=False, input_scaling_init=None, cutting=None):
        """Configure an experimental sampled measurement-subset policy.

        n_qubits_measure chooses subset size; all combinations are enumerated.
        The selector uses an affine categorical distribution over subsets, and
        returns a score-function log probability. This path assumes an instantiated
        ansatz with a feature_map during setup; class-only configuration is not a
        complete supported path. Other arguments mirror the legacy Born policy.
        Use current fixed-decoder policies for validated training examples."""
        self.cutting = cutting
        super().__init__()   
        self.mpi = mpi
        self.torchrl = torchrl
        self.circuit_builder = circuit_builder
        self.circuit_builder_copy = deepcopy(circuit_builder)

        if self.circuit_builder is None and ansatz is None:
            raise ValueError("Either circuit_builder or ansatz must be provided.")
        
        self.ansatz = ansatz if self.circuit_builder is None else self.circuit_builder.ansatz
        self.ansatz_config = ansatz_config
        self.feature_map = feature_map if self.circuit_builder is None else self.circuit_builder.feature_map
        self.feature_map_config = feature_map_config
        self.n_qubits=n_qubits
        self.n_actions=n_actions
        self.policy_type = "LearnablePQCBorn"
        if self.circuit_builder is not None:
            self.n_layers=self.circuit_builder.ansatz.n_layers
        else:
            self.n_layers=self.ansatz_config["n_layers"]
        self.init = init
        self.input_scaling_init = input_scaling_init
        self.diff_method = diff_method
        self.softmax_activation = softmax_activation
        self.output_scaling = output_scaling
        self.training_method = training_method
        if self.output_scaling:
            self.output_scaling_weight = nn.Parameter(torch.ones(1, requires_grad=True))
        self.device=device
        self.diff_method=diff_method
        self.n_qubits_measure = n_qubits_measure
        n = list(range(self.n_qubits))
        k = self.n_qubits_measure
        self.measurement_combinations = list(combinations(n, k))
        self.num_measurements = len(self.measurement_combinations)

        class MeasurementSelector(nn.Module):
            """Sample a measured-wire subset from state-dependent categorical logits."""
            def __init__(self, input_dim, num_measurements, measurement_combinations):
                """Create an affine input-to-subset map and a trainable scalar logit multiplier."""
                super().__init__()
                self.linear = nn.Linear(input_dim, num_measurements)
                self.measurement_combinations = measurement_combinations
                self.output_scaling_selector = nn.Parameter(torch.ones(1, requires_grad=True))
            def forward(self, x):
                """Return a sampled wire tuple and its differentiable log probability for one input."""
                p = torch.nn.functional.softmax(self.output_scaling_selector*self.linear(x), dim=-1)

                dist = torch.distributions.Categorical(probs=p)
                action = dist.sample()

                return self.measurement_combinations[action.item()], dist.log_prob(action)

        self.measurement_selector = MeasurementSelector(self.ansatz.feature_map.features , self.num_measurements, self.measurement_combinations)

        self.measurement_qubits = [i for i in range(k)]

        self.build_policy()

    def build_policy(self):
        """Build a call-time measurement circuit and its TorchLayer for the experimental selector."""
        if self.circuit_builder is None:

            self.fm = self.feature_map(**self.feature_map_config)

            ans = self.ansatz(**self.ansatz_config, feature_map=self.fm)

            self.circuit_builder = CircuitBuilder(n_qubits=self.n_qubits)
            self.circuit_builder.set_ansatz(ans)           # pass the IQPAnsatz instance

        self.circuit_builder.set_measurement("probs", wires=range(self.n_qubits))
        self.c = self.circuit_builder.build_adaptive(for_torchlayer=True)

        self.circuit = self.circuit_builder.compile(device = self.device , diff_method=self.diff_method, mpi=self.mpi, cutting=self.cutting)
        
        self.qlayer = self.circuit_builder.create_torchlayer(initialization=self.init, input_scaling_init=self.input_scaling_init)

    def update_sampler_weights(self):
        """Legacy sampler synchronization; build_policy does not create the required qlayer_copy."""
        self.qlayer_copy.load_state_dict(self.qlayer.state_dict())

    def set_weights(self, params=None, input_scaling_params=None):
        """Legacy setter that calls set_weights on the compiled circuit.

        Ordinary PennyLane QNodes lack that method. Prefer
        ``model.qlayer.load_state_dict(...)`` or the builder's set_weights with
        both parameter groups when input scaling is enabled."""
        if params is not None:

            self.circuit.set_weights(params=params)
        if input_scaling_params is not None:

            self.circuit.set_weights(input_scaling_params=input_scaling_params)

    def get_weights(self):
        """Return the quantum layer's live named weight mapping, excluding classical head parameters."""
        return self.qlayer.qnode_weights
    
    def get_meyer_wallach(self, x):
        #TO DO
        """Unimplemented legacy placeholder; returns None without evaluating entanglement."""
        pass
    
    def post_process_policy(self, probs, n_actions=2, n_qubits=1):
        """Apply the prototype parity grouping to measured probabilities.

        The nonbatched branch iterates only n_qubits_measure entries, not all
        basis states; the result is not generally a normalized action distribution.
        This limitation is retained here rather than changing prototype behavior."""
        if self.torchrl:
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)  # Ensure batched input
            
            batch_size, n_states = probs.shape

            if n_actions == 2**n_qubits:
                return probs

            
            policy = torch.zeros((batch_size, n_actions), device=probs.device)
            counter=0
            for i in range(n_states):
                bitstring = np.binary_repr(i, width=n_qubits)
                a = []
                for m in range(int(np.ceil(np.log2(n_actions)))):
                    if m == 0:
                        a.append(bitstring.count("1") % 2)
                    else:
                        a.append(bitstring[:-m].count("1") % 2)
                idx = int("".join(str(x) for x in a)[::-1], 2)
                if idx >= n_actions:
                    idx = counter % n_actions
                    counter+=1
                policy[:, idx] += probs[:, i]
                

            return policy

        else:
            # Non-batched version for standalone training
            if n_actions == 2**n_qubits:
                return probs

            policy = torch.zeros(n_actions)
            
            counter=0
            for i in range(self.n_qubits_measure):
                bitstring = np.binary_repr(i, width=n_qubits)
                a = []
                for m in range(int(np.ceil(np.log2(n_actions)))):
                    if m == 0:
                        a.append(bitstring.count("1") % 2)
                    else:
                        a.append(bitstring[:-m].count("1") % 2)
                idx = int("".join(str(x) for x in a)[::-1], 2)
                if idx >= n_actions:
                    idx = counter % n_actions
                    counter+=1
                policy[idx] += probs[i]
        
        return policy

    def action_probs(self, x, temperature=None):
        """Apply tanh in place to input features, execute selected measurements and optionally softmax legacy weights."""
        x[:self.ansatz.feature_map.features] = torch.tanh(x[:self.ansatz.feature_map.features])
        #m = x[self.feature_map.features:]
        probs = self.qlayer(x)
        action_probs = self.post_process_policy(probs, n_actions=self.n_actions, n_qubits=self.n_qubits_measure)
        # Process each sample separately
        if self.softmax_activation:
            if temperature is not None:
                    action_probs = action_probs*temperature
                
                #nn functional softmax
            if self.output_scaling:
                action_probs *= self.output_scaling_weight

            action_probs = torch.nn.functional.softmax(action_probs, dim=-1)

        return action_probs
