"""Discrete-action RL trainers, baselines and a synthetic contextual bandit.

DeepQLearning consumes unconstrained Q-values; policy-gradient trainers consume
probabilities or logits according to their configuration. REINFORCE and
Reinforce are distinct implementations, not aliases. PPO is a direct Gymnasium/
Torch implementation; TorchRLTrainer is the optional external TorchRL adapter.
Simulator execution device and Torch tensor device are separate choices.

Trainer-specific preprocessing, stopping rules and truncation handling differ;
see each class before comparing experiments. No trainer promises convergence."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as D
from collections import deque
from typing import Any, Optional, Type, Union

from tensordict.nn import TensorDictModule
#from torchrl.modules import ProbabilisticActor
from tensordict import TensorDict



# -----------------------------------------------------------------------------
# Helper: wrap PQCBorn so it plugs into TorchRL’s TensorDict pipeline
# -----------------------------------------------------------------------------


class PQCBornWrapper(nn.Module):
    """Adapt a differentiable action_probs policy to the external TorchRL actor interface."""

    def __init__(self, pqc: "PQCBorn", obs_dim: int):
        """Store a probability-returning policy and observation-size metadata for the TorchRL adapter."""
        super().__init__()
        self.pqc = pqc
        self.obs_dim = obs_dim

    def forward(self, obs):
        # IMPORTANT: This must be differentiable w.r.t. pqc parameters.
        # If pqc.action_probs() internally detaches / converts to numpy,
        # gradients will be zero/no-op.
        """Delegate to action_probs without detaching so gradients can reach the quantum parameters."""
        return self.pqc.action_probs(obs)


# -----------------------------------------------------------------------------
# Trainer – manual REINFORCE *or* any TorchRL loss class
# -----------------------------------------------------------------------------


class TorchRLTrainer:
    """Episode-based training loop.

    If loss_class == "REINFORCE": manual REINFORCE (no critic) with grad-safe logp recomputation.
    Otherwise: TorchRL loss module.
    """

    def __init__(
        self,
        pqc: "PQCBorn",
        env_name: str,
        *,
        lr: float = 3e-3,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        loss_class: Union[str, Type[nn.Module]] = "REINFORCE",
        loss_kwargs: Optional[dict] = None,
        gamma: float = 0.99,
        max_steps: int = 1000,
        wandb: bool = False,
        wandb_project: str = "pqc_rl",
        **wandb_kwargs: Any,
    ) -> None:
        """Create a TorchRL environment, probabilistic actor and optimizer.

        pqc must provide differentiable action_probs. env_name identifies the Gym
        environment. loss_class='REINFORCE' or None enables the manual Monte Carlo
        loss; otherwise pass a loss class accepting actor_network plus loss_kwargs.
        lr/optimizer_class configure updates, gamma discounts rewards, and max_steps
        caps each rollout. wandb=True initializes a logging run with the supplied
        project/options. External TorchRL/TensorDict dependencies are required."""
        self.gamma = gamma
        self.max_steps = max_steps
        self.use_manual = loss_class == "REINFORCE" or loss_class is None
        self.loss_class = loss_class if not self.use_manual else None
        self.loss_kwargs = loss_kwargs or {}

        # ---------------- Logging ----------------
        self.wandb = wandb
        if self.wandb:
            import wandb as _wandb
            self._wandb = _wandb
            self._wandb.init(project=wandb_project, **wandb_kwargs)

        # ---------------- Environment ------------
        from torchrl.envs import GymEnv
        self.env = GymEnv(env_name, from_pixels=False)
        self.obs_dim = self._flat_obs_dim(self.env.observation_spec)

        # ---------------- Actor -------------------
        self.actor = self._build_actor(pqc, self.env.action_spec)

        # Keep a handle to the PQC (for grabbing any non-Parameter trainables)
        self.pqc = pqc

        # ---------------- Loss (optional) ---------
        if not self.use_manual:
            self.loss_module = self._build_torch_loss()

        # ---------------- Optimiser --------------
        self.lr = lr
        params = self._collect_optim_params()
        self.optim = optimizer_class(params, lr=self.lr)

        # ---------------- Stats ------------------
        self.frame_count = 0
        self._last_batch_loss_val: float = float("nan")

    # =====================================================================
    # Public API
    # =====================================================================

    def train(
        self,
        num_episodes: int = 500,
        *,
        batch_episodes: int = 1,
        eval_every: int = 10,
        eval_avg_reward: int = 20,
    ):
        """Collect num_episodes rollouts and update on batches of complete episodes.

        Returns None; recent_returns, frame_count and last loss are stored on the
        trainer. eval_every prints a rolling training-return mean over
        eval_avg_reward episodes, not an independent evaluation. A final incomplete
        batch is not optimized. Nonfinite losses are skipped; absence of nonzero
        finite parameter gradients raises RuntimeError. Gradients are clipped to 1."""
        self.recent_returns = deque(maxlen=eval_avg_reward)
        batch_loss_acc: list[torch.Tensor] = []

        for ep in range(1, num_episodes + 1):
            td = self.env.rollout(max_steps=self.max_steps, policy=self.actor)

            ep_return = td["next", "reward"].sum().item()
            self.recent_returns.append(ep_return)
            self.frame_count += td["next", "reward"].numel()

            if self.use_manual:
                ep_loss = self._manual_reinforce_loss(td)
            else:
                loss_dict = self.loss_module(td)
                ep_loss = torch.stack([v for v in loss_dict.values() if torch.is_tensor(v)]).sum()

            batch_loss_acc.append(ep_loss)

            if len(batch_loss_acc) == batch_episodes:
                self.optim.zero_grad(set_to_none=True)
                loss = torch.stack(batch_loss_acc).mean()

                if not torch.isfinite(loss):
                    print(f"[WARN] Non-finite loss at ep {ep}: {float(loss.detach().cpu().item())}")
                    batch_loss_acc.clear()
                    continue

                loss.backward()

                # ---- ASSERT: something must receive gradient ----
                got_grad = False
                for p in self._all_trainables():
                    if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0:
                        got_grad = True
                        break
                if not got_grad:
                    raise RuntimeError(
                        "No gradients reached PQC parameters. "
                        "Likely pqc.action_probs() detaches/uses numpy or qnode output is detached, "
                        "or your trainables are not leaf tensors requiring grad."
                    )

                torch.nn.utils.clip_grad_norm_(self._all_trainables(), max_norm=1.0)
                self.optim.step()

                self._last_batch_loss_val = float(loss.detach().cpu().item())
                batch_loss_acc.clear()

                if self.wandb:
                    self._wandb.log({
                        "train/total_loss": self._last_batch_loss_val,
                        "global_step": self.frame_count,
                    })

            if self.wandb:
                self._wandb.log({
                    "train/episode_reward": ep_return,
                    "train/total_loss": self._last_batch_loss_val,
                    "global_step": self.frame_count,
                })

            if ep % eval_every == 0:
                avg = sum(self.recent_returns) / len(self.recent_returns)
                print(f"Episode {ep:04d} | AvgReward(last {eval_avg_reward}) = {avg:.2f}")

                # (optional) quick check: print a single parameter value
                # so you can see it changing.
                self._debug_print_one_param()

                if self.wandb:
                    self._wandb.log({
                        "eval/avg_return_last20": avg,
                        "global_step": self.frame_count,
                    })

    # =====================================================================
    # Internal helpers
    # =====================================================================

    @staticmethod
    def _safe_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Replace nonfinite weights, clamp positive and normalize the final probability axis."""
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs.clamp_min(eps)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def _manual_reinforce_loss(self, td):
        """
        Manual REINFORCE with grad-safe logp recomputation.

        We DO NOT use td["sample_log_prob"] because rollout often detaches it.
        """
        r = td["next", "reward"].squeeze(-1)
        done = td["next", "done"].squeeze(-1).to(r.dtype)
        a = td["action"].squeeze(-1)          # [T]
        obs = td["observation"]               # [T, obs_dim] (typical)

        # Recompute probs WITH grad through the same wrapper used by the actor:
        td_in = TensorDict({"observation": obs}, batch_size=obs.shape[0])
        td_out = self.actor.module(td_in)     # TensorDictModule, should preserve graph
        probs = td_out["probs"]

        probs = self._safe_probs(probs)
        logp = torch.log(probs.gather(-1, a.unsqueeze(-1)).squeeze(-1))  # [T]

        # discounted returns with termination handling
        returns = torch.zeros_like(r)
        running = 0.0
        for t in reversed(range(len(r))):
            running = r[t] + self.gamma * running * (1.0 - done[t])
            returns[t] = running

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return -(returns.detach() * logp).sum()

    def _flat_obs_dim(self, obs_spec):
        """Infer feature count from a TorchRL observation spec, summing composite components."""
        from torchrl.data import CompositeSpec
        if isinstance(obs_spec, CompositeSpec):
            return sum(int(torch.tensor(s.shape).prod()) for s in obs_spec.values())
        return obs_spec.shape[-1]

    def _build_actor(self, pqc, action_spec):
        """Wrap action_probs in a TensorDictModule and categorical ProbabilisticActor."""
        wrapper = PQCBornWrapper(pqc, obs_dim=self.obs_dim)
        td_mod = TensorDictModule(wrapper, in_keys=["observation"], out_keys=["probs"])
        return ProbabilisticActor(
            module=td_mod,
            in_keys=["probs"],
            distribution_class=D.Categorical,
            return_log_prob=True,
        )

    def _build_torch_loss(self):
        """Instantiate the supplied loss class with actor_network and additional loss kwargs."""
        loss_cls = self.loss_class
        kwargs = {"actor_network": self.actor, **self.loss_kwargs}
        return loss_cls(**kwargs)

    # ------------------- Optimizer param collection -------------------

    def _collect_optim_params(self):
        """
        Build optimizer param list that includes:
          - standard nn.Parameters from actor (includes wrapper + pqc if properly registered)
          - PLUS any leaf tensors (NOT nn.Parameter) named 'params'/'input_scaling_params'
            inside pqc.qlayer (common if these were registered as buffers/plain tensors).
        """
        params = list(self.actor.parameters())

        # Add qlayer extra tensors if they exist and require grad
        qlayer = getattr(self.pqc, "qlayer", None)
        if qlayer is not None:
            for name in ("params", "input_scaling_params"):
                t = getattr(qlayer, name, None)
                if isinstance(t, torch.Tensor) and t.requires_grad:
                    # if it's already a Parameter, it will already be in actor.parameters()
                    if not isinstance(t, nn.Parameter):
                        # only leaf tensors can be optimized
                        if not t.is_leaf:
                            raise RuntimeError(
                                f"Found trainable tensor qlayer.{name} but it is not a leaf tensor; "
                                "make it an nn.Parameter in your model."
                            )
                        params.append(t)

        if len(params) == 0:
            raise RuntimeError("Optimizer got zero parameters — actor/pqc has no trainables.")

        return params

    def _all_trainables(self):
        """Return a flat list of tensors that optimizer steps over (Parameters + extra tensors)."""
        trainables: list[torch.Tensor] = []
        for p in self.actor.parameters():
            if p.requires_grad:
                trainables.append(p)

        qlayer = getattr(self.pqc, "qlayer", None)
        if qlayer is not None:
            for name in ("params", "input_scaling_params"):
                t = getattr(qlayer, name, None)
                if isinstance(t, torch.Tensor) and t.requires_grad and not isinstance(t, nn.Parameter):
                    trainables.append(t)

        return trainables

    def _debug_print_one_param(self):
        """Print one representative param value so you can see updates."""
        # Try common qlayer tensors first:
        qlayer = getattr(self.pqc, "qlayer", None)
        if qlayer is not None:
            for name in ("params", "input_scaling_params"):
                t = getattr(qlayer, name, None)
                if isinstance(t, torch.Tensor):
                    v = float(t.flatten()[0].detach().cpu().item())
                    print(f"  [debug] qlayer.{name}[0] = {v:.6f}")
                    return

        # fallback: any actor parameter
        for n, p in self.actor.named_parameters():
            if p.requires_grad:
                v = float(p.flatten()[0].detach().cpu().item())
                print(f"  [debug] actor.{n}[0] = {v:.6f}")
                return

# ======================================================================
# Low-complexity upgraded REINFORCE for PQC policies
# - EMA return baseline
# - batch-wise advantage normalization
# - entropy bonus
# - safe probability handling
# - optional CartPole feature expansion via normalize_mode="expand X"
# ======================================================================

# ======================================================================
# Low-complexity upgraded REINFORCE for PQC policies
# - EMA or value-network baseline
# - batch-wise advantage normalization
# - entropy bonus
# - safe probability handling
# - optional CartPole feature expansion via normalize_mode="expand X"
# - reward shaping for Acrobot ("dense" or "potential")
# ======================================================================

from dataclasses import dataclass, field
from typing import List, Sequence

import gymnasium as gym
import numpy as onp

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical

__all__ = ["REINFORCE"]


class ValueBaseline(nn.Module):
    """Small classical value network V(s)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """Construct a two-hidden-layer tanh MLP mapping input_dim features to one scalar value."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return baseline values with the final singleton output axis removed."""
        return self.net(x).squeeze(-1)


@dataclass
class REINFORCE:
    """Episode-batched policy gradients with an EMA or learned value baseline.

    Args:
        policy: Torch module returning logits or probabilities (set logits).
        env_name: Gymnasium name or environment instance. Name-based CartPole/
            Acrobot detection enables task-specific preprocessing and shaping.
        gamma, lr: Discount and Adam/AMSGrad learning rate.
        max_steps, batch_episodes: Episode cap and episodes per policy update.
        device: Torch device; defaults to CUDA if available, otherwise CPU.
        normalize_mode: none, clip, tanh or 'expand N'; affects named CartPole/
            Acrobot only. acrobot_use_angles converts six features to four.
        use_reward_shaping, shaping_coef, shaping_mode: Acrobot height-based
            dense or potential reward additions. Logged returns remain unshaped.
        normalize_returns: Standardize batch advantages, not rewards.
        entropy_coef: Categorical entropy bonus coefficient.
        baseline_type: ema or value. baseline_momentum controls EMA updates;
            baseline_hidden_dim/lr/updates_per_batch configure the value MLP.
        max_grad_norm: Gradient clipping limit; None disables clipping.
        fc_layer: Use separate optimizer groups for an existing policy head
            (lr=0.01) and quantum weights (lr); does not create a head.

    For named CartPole, optimization freezes when the available trailing
    up-to-100 training episodes average >=195, even before 100 are collected.
    Rollouts continue after freezing; this historical rule is not a CartPole-v1
    solving certificate. This class differs from Reinforce/ReinforceConfig."""

    policy: torch.nn.Module
    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    lr: float = 1e-3
    max_steps: int = 500
    batch_episodes: int = 5
    device: torch.device | str | None = None

    # If policy(state) returns logits, use logits=True.
    # If policy(state) returns probabilities, use logits=False.
    logits: bool = True

    # Preprocessing mode:
    #   "clip"       -> scale by hand-chosen factors, clip to [-3, 3]
    #   "tanh"       -> scale then tanh (bounded to [-2, 2])
    #   "none"/None  -> raw state
    #   "expand X"   -> nonlinear feature expansion to X features
    normalize_mode: str | None = "clip"

    # For Acrobot only:
    # If False, use the original Gymnasium 6D state:
    #   [cos(theta1), sin(theta1), cos(theta2), sin(theta2), thetaDot1, thetaDot2]
    #
    # If True, convert it to a 4D state:
    #   [theta1, theta2, thetaDot1, thetaDot2]
    acrobot_use_angles: bool = False

    # ------------------------------------------------------------------
    # Reward shaping (Acrobot only). Phi(s) = height of the free tip:
    #   Phi(s) = -cos(theta1) - cos(theta1 + theta2),  range [-2, +2].
    # The TRUE (unshaped) return is always what gets logged.
    #
    # shaping_mode:
    #   "dense" adds coef * Phi(s_next), changing the task objective.
    #   "potential" adds coef * (gamma * Phi(s_next) - Phi(s)); the
    #   terminal potential is zero on true termination, not time truncation.
    # Neither choice guarantees learning; finite-horizon handling and baseline
    # fitting affect the resulting update. Always report the unshaped return.
    # ------------------------------------------------------------------
    use_reward_shaping: bool = False
    shaping_coef: float = 1.0
    shaping_mode: str = "dense"

    # Whether to normalize advantages across the whole batch.
    normalize_returns: bool = True

    # Entropy regularization coefficient.
    entropy_coef: float = 0.01

    # Baseline selection:
    #   "ema"   -> scalar exponential moving average baseline
    #   "value" -> learned classical value network
    baseline_type: str = "ema"

    # EMA baseline parameters.
    baseline_momentum: float = 0.05
    baseline_value: float = field(default=0.0, init=False, repr=False)
    baseline_initialized: bool = field(default=False, init=False, repr=False)

    # Value baseline parameters.
    baseline_hidden_dim: int = 64
    baseline_lr: float = 1e-3
    baseline_updates_per_batch: int = 5
    value_baseline: nn.Module | None = field(default=None, init=False, repr=False)
    baseline_opt: torch.optim.Optimizer | None = field(default=None, init=False, repr=False)

    # Optional gradient clipping.
    max_grad_norm: float | None = 1.0

    opt: torch.optim.Optimizer = field(init=False, repr=False)

    fc_layer: bool = False

    def __post_init__(self):
        """Create/reuse the environment, move the policy and configure optimizers and selected baseline."""
        if isinstance(self.env_name, str):
            self.env = gym.make(self.env_name)
        else:
            self.env = self.env_name

        self.device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy.to(self.device)

        params = list(self.policy.parameters())

        qlayer = getattr(self.policy, "qlayer", None)
        if qlayer is not None:
            for name in ("params", "input_scaling_params"):
                t = getattr(qlayer, name, None)
                if (
                    isinstance(t, torch.Tensor)
                    and t.requires_grad
                    and not isinstance(t, nn.Parameter)
                ):
                    if not t.is_leaf:
                        raise RuntimeError(
                            f"Found trainable tensor qlayer.{name}, but it is not a leaf tensor. "
                            "Make it an nn.Parameter in your model."
                        )
                    params.append(t)

        if self.fc_layer:
            # verify if there are input scaling params in the policy
            if self.policy.fm.input_scaling == False:
                self.opt = Adam(
                    [
                        {
                            "params": self.policy.fc_layer.parameters(),
                            "lr": 1e-2,
                        },
                        {
                            "params": [self.policy.qlayer.params],
                            "lr": self.lr,
                        }
                    ],
                    amsgrad=True,
                )
            else:
                self.opt = Adam(
                    [
                        {
                            "params": self.policy.fc_layer.parameters(),
                            "lr": 1e-2,
                        },
                        {
                            "params": [self.policy.qlayer.params],
                            "lr": self.lr,
                        },
                        {
                            "params": [self.policy.qlayer.input_scaling_params],
                            "lr": self.lr,
                        }
                    ],
                    amsgrad=True,
                )
        else:
            self.opt = Adam(params, lr=self.lr, amsgrad=True)

        if self.baseline_type not in {"ema", "value"}:
            raise ValueError("baseline_type must be either 'ema' or 'value'")

        if self.shaping_mode not in {"dense", "potential"}:
            raise ValueError("shaping_mode must be either 'dense' or 'potential'")

        if self.baseline_type == "value":
            input_dim = self._get_processed_state_dim()
            self.value_baseline = ValueBaseline(
                input_dim=input_dim,
                hidden_dim=self.baseline_hidden_dim,
            ).to(self.device)
            self.baseline_opt = Adam(self.value_baseline.parameters(), lr=self.baseline_lr)

    def _is_cartpole(self) -> bool:
        """Identify CartPole only when env_name is a string with the corresponding prefix."""
        return isinstance(self.env_name, str) and self.env_name.startswith("CartPole")

    def _is_acrobot(self) -> bool:
        """Identify Acrobot only when env_name is a string with the corresponding prefix."""
        return isinstance(self.env_name, str) and self.env_name.startswith("Acrobot")

    def _raw_state_dim(self) -> int:
        """Infer observation width, accounting for optional six-to-four Acrobot angle conversion."""
        if self._is_acrobot() and self.acrobot_use_angles:
            return 4
        obs_space = self.env.observation_space
        if hasattr(obs_space, "shape") and obs_space.shape is not None:
            return int(obs_space.shape[0])
        raise ValueError("Could not infer observation dimension from environment.")

    def _get_processed_state_dim(self) -> int:
        """Validate preprocessing mode and return its output feature count for the value baseline."""
        mode = self.normalize_mode
        raw_dim = self._raw_state_dim()

        if mode is None or mode == "none" or mode == "clip" or mode == "tanh":
            return raw_dim

        if isinstance(mode, str) and mode.startswith("expand"):
            parts = mode.split()
            if len(parts) != 2:
                raise ValueError(
                    f'Invalid normalize_mode="{mode}". Expected format: "expand X"'
                )
            try:
                num_features = int(parts[1])
            except ValueError as e:
                raise ValueError(
                    f'Invalid normalize_mode="{mode}". X must be an integer.'
                ) from e
            if num_features < raw_dim:
                raise ValueError(
                    f'Invalid normalize_mode="{mode}". X must be >= {raw_dim}.'
                )
            return num_features

        raise ValueError(f"Unknown normalize_mode: {mode}")

    def _convert_acrobot_state(self, state: torch.Tensor) -> torch.Tensor:
        """Recover Acrobot angles with atan2 and retain velocities when angle conversion is enabled."""
        if not self._is_acrobot() or not self.acrobot_use_angles:
            return state

        if state.shape[0] != 6:
            raise ValueError(
                "Expected Acrobot state with 6 features: "
                "[cos(theta1), sin(theta1), cos(theta2), sin(theta2), thetaDot1, thetaDot2]. "
                f"Got shape {tuple(state.shape)}."
            )

        cos_theta1 = state[0]
        sin_theta1 = state[1]
        cos_theta2 = state[2]
        sin_theta2 = state[3]
        theta_dot1 = state[4]
        theta_dot2 = state[5]

        # Use atan2 to recover the full [-pi, pi] angle (asin only covers
        # [-pi/2, pi/2] and loses which half of the circle the link is in).
        theta1 = torch.atan2(sin_theta1, cos_theta1)
        theta2 = torch.atan2(sin_theta2, cos_theta2)

        return torch.stack([theta1, theta2, theta_dot1, theta_dot2])

    def _get_env_scale(self, state: torch.Tensor) -> torch.Tensor:
        """Return hand-selected CartPole/Acrobot normalization scales, or ones for other tasks."""
        if self._is_cartpole():
            return torch.tensor(
                [2.4, 3.0, 0.2095, 3.5],
                dtype=state.dtype,
                device=state.device,
            )

        if self._is_acrobot():
            if self.acrobot_use_angles:
                return torch.tensor(
                    [torch.pi, torch.pi, 4.0 * torch.pi, 9.0 * torch.pi],
                    dtype=state.dtype,
                    device=state.device,
                )

            return torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 4.0 * torch.pi, 9.0 * torch.pi],
                dtype=state.dtype,
                device=state.device,
            )

        return torch.ones_like(state)

    def _expand_features(self, x: torch.Tensor, num_features: int) -> torch.Tensor:
        """Take N ordered raw, square, cross, sine, cosine, tanh and absolute features; reject unavailable sizes."""
        d = x.shape[0]

        if num_features < d:
            raise ValueError(
                f"Requested expand {num_features}, but input dimension is {d}. "
                f"Need num_features >= {d}."
            )

        feats = []

        feats.extend([x[i] for i in range(d)])
        feats.extend([x[i] * x[i] for i in range(d)])

        for i in range(d):
            for j in range(i + 1, d):
                feats.append(x[i] * x[j])

        feats.extend([torch.sin(x[i]) for i in range(d)])
        feats.extend([torch.cos(x[i]) for i in range(d)])
        feats.extend([torch.tanh(x[i]) for i in range(d)])
        feats.extend([torch.abs(x[i]) for i in range(d)])

        if num_features > len(feats):
            raise ValueError(
                f"Requested expand {num_features}, but only {len(feats)} generic "
                f"features are currently defined for input dimension {d}."
            )

        return torch.stack(feats[:num_features])

    def _preprocess_state(self, state: torch.Tensor) -> torch.Tensor:
        """Convert optional angles, then apply configured task scaling/clipping/tanh or nonlinear expansion."""
        mode = self.normalize_mode

        state = self._convert_acrobot_state(state)

        scale = self._get_env_scale(state)
        x = state / scale
        if mode == "clip":
            return torch.clamp(x, -3.0, 3.0)

        if mode == "tanh":
            return 2 * torch.tanh(x)   # bounded to [-2, 2]

        if mode is None or mode == "none":
            return state

        if isinstance(mode, str) and mode.startswith("expand"):
            parts = mode.split()
            if len(parts) != 2:
                raise ValueError(
                    f'Invalid normalize_mode="{mode}". Expected format: "expand X"'
                )

            try:
                num_features = int(parts[1])
            except ValueError as e:
                raise ValueError(
                    f'Invalid normalize_mode="{mode}". X must be an integer.'
                ) from e

            x = torch.clamp(x, -3.14, 3.14)
            return self._expand_features(x, num_features)

        raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")

    # ------------------------------------------------------------------
    # Potential function for Acrobot reward shaping.
    # ------------------------------------------------------------------
    def _potential(self, raw_obs) -> float:
        """Height of the free tip for an Acrobot raw observation.

        raw_obs is the *unprocessed* 6D Gymnasium observation:
            [cos(theta1), sin(theta1), cos(theta2), sin(theta2), thetaDot1, thetaDot2]

        Phi(s) = -cos(theta1) - cos(theta1 + theta2),  range [-2, +2].
        (Goal is reached when this height exceeds 1.0.)
        """
        c1 = float(raw_obs[0])
        s1 = float(raw_obs[1])
        c2 = float(raw_obs[2])
        s2 = float(raw_obs[3])
        cos_12 = c1 * c2 - s1 * s2  # cos(theta1 + theta2)
        return -c1 - cos_12

    @staticmethod
    def _safe_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Replace nonfinite weights, clamp positive and normalize the final action axis."""
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs.clamp_min(eps)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def _update_baseline(self, baseline_target: float):
        """Initialize or update the scalar EMA baseline, with momentum weighting the new target."""
        if not self.baseline_initialized:
            self.baseline_value = float(baseline_target)
            self.baseline_initialized = True
        else:
            m = self.baseline_momentum
            self.baseline_value = (1.0 - m) * self.baseline_value + m * float(baseline_target)

    def _evaluate_baseline(self, states: list[torch.Tensor]) -> torch.Tensor:
        """Return one detached EMA/value prediction per trajectory state."""
        if self.baseline_type == "ema":
            return torch.full(
                (len(states),),
                fill_value=self.baseline_value,
                dtype=torch.float32,
                device=self.device,
            )

        assert self.value_baseline is not None
        with torch.no_grad():
            state_batch = torch.stack(states)
            return self.value_baseline(state_batch)

    def _update_value_baseline(self, states: list[torch.Tensor], targets: torch.Tensor) -> tuple[float, float]:
        """Fit the value network to detached discounted returns; return last loss and mean prediction."""
        if self.baseline_type != "value":
            return 0.0, 0.0

        assert self.value_baseline is not None
        assert self.baseline_opt is not None

        state_batch = torch.stack(states)
        last_loss = 0.0
        last_mean_pred = 0.0

        for _ in range(self.baseline_updates_per_batch):
            pred = self.value_baseline(state_batch)
            loss = nn.functional.mse_loss(pred, targets.detach())

            self.baseline_opt.zero_grad(set_to_none=True)
            loss.backward()

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.value_baseline.parameters(),
                    max_norm=self.max_grad_norm,
                )

            self.baseline_opt.step()
            last_loss = float(loss.detach().cpu().item())
            last_mean_pred = float(pred.detach().mean().cpu().item())

        return last_loss, last_mean_pred

    def train(self, num_episodes: int = 500, log_every: int = 10) -> List[float]:
        """Train num_episodes and return a list of unshaped episode returns.

        Policy updates use discounted (possibly shaped) returns minus the baseline,
        optional advantage normalization and entropy regularization. Final partial
        batches are included. log_every controls console output. See the class
        notes for the legacy CartPole optimization-freezing rule; no new independent
        evaluation episodes are collected by this method."""
        returns_hist: list[float] = []

        batch_logp: list[torch.Tensor] = []
        batch_adv: list[torch.Tensor] = []
        batch_entropy: list[torch.Tensor] = []

        baseline_states: list[torch.Tensor] = []
        baseline_targets: list[torch.Tensor] = []

        last_baseline_loss: float | None = None
        last_baseline_mean_pred: float | None = None

        solved = False
        for ep in range(1, num_episodes + 1):
            # ep_rewards are the (possibly shaped) rewards used for training;
            # ep_true_return is the real unshaped return used only for logging.
            ep_states, ep_logp, ep_rewards, ep_entropy, ep_true_return = self._run_episode()
            ep_return = ep_true_return
            returns_hist.append(ep_return)

            disc = self._discount(ep_rewards)

            if self.baseline_type == "ema":
                adv = disc - self.baseline_value
                self._update_baseline(disc.mean().item())
            else:
                baseline_pred = self._evaluate_baseline(ep_states)
                adv = disc - baseline_pred
                baseline_states.extend(ep_states)
                baseline_targets.append(disc.detach())

            batch_logp.append(ep_logp)
            batch_adv.append(adv)
            batch_entropy.append(ep_entropy)

            if len(batch_logp) == self.batch_episodes:
                logp = torch.cat(batch_logp)
                adv = torch.cat(batch_adv)
                entropy = torch.cat(batch_entropy)

                if self.normalize_returns and len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                loss = -(adv.detach() * logp).mean()
                if self.entropy_coef > 0:
                    loss = loss - self.entropy_coef * entropy.mean()

                # test if environment is cartpole and if so, if the average reward of last 100 eps are >=195 then do not optimize
                if self._is_cartpole() and not solved:
                    avg100 = onp.mean(returns_hist[-100:])
                    if avg100 < 195.0:
                        self._optimise_loss(loss)
                    else:
                        print(f"CartPole solved! Average return: {avg100:.2f}. Stopping optimization.", flush=True)
                        solved = True
                else:
                    if not solved:
                        self._optimise_loss(loss)

                if self.baseline_type == "value" and len(baseline_states) > 0 and not solved:
                    targets = torch.cat(baseline_targets)
                    last_baseline_loss, last_baseline_mean_pred = self._update_value_baseline(
                        baseline_states, targets
                    )
                    baseline_states.clear()
                    baseline_targets.clear()

                batch_logp.clear()
                batch_adv.clear()
                batch_entropy.clear()

            if ep % log_every == 0:
                avg20 = onp.mean(returns_hist[-20:])
                if self.baseline_type == "ema":  
                    baseline_info = f"ema={self.baseline_value:.1f}"
                else:
                    if last_baseline_loss is None:
                        baseline_info = "value_loss=NA"
                    else:
                        baseline_info = (
                            f"value_loss={last_baseline_loss:.4f}, "
                            f"value_mean={last_baseline_mean_pred:.1f}"
                        )
                print(
                    f"Episode {ep:04d} | avg(return_last20) = {avg20:.1f} | baseline = {baseline_info}", flush=True
                )

        if len(batch_logp) > 0:
            logp = torch.cat(batch_logp)
            adv = torch.cat(batch_adv)
            entropy = torch.cat(batch_entropy)

            if self.normalize_returns and len(adv) > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            loss = -(adv.detach() * logp).mean()
            if self.entropy_coef > 0:
                loss = loss - self.entropy_coef * entropy.mean()

            if not solved:
                self._optimise_loss(loss)

            if self.baseline_type == "value" and len(baseline_states) > 0 and not solved:
                targets = torch.cat(baseline_targets)
                last_baseline_loss, last_baseline_mean_pred = self._update_value_baseline(
                    baseline_states, targets
                )

        return returns_hist

    def _run_episode(self) -> tuple[list[torch.Tensor], torch.Tensor, list[float], torch.Tensor, float]:
        """Collect processed states, differentiable log-probs/entropies, training rewards and true episode return."""
        state, _ = self.env.reset()

        traj_states: list[torch.Tensor] = []
        traj_logp: list[torch.Tensor] = []
        traj_rewards: list[float] = []        # shaped rewards used for training
        traj_true_rewards: list[float] = []   # real env rewards used for logging
        traj_entropy: list[torch.Tensor] = []

        # Reward shaping is only defined for Acrobot; it's a no-op elsewhere.
        use_shaping = self.use_reward_shaping and self._is_acrobot()
        phi_s = self._potential(state) if use_shaping else 0.0

        for _ in range(self.max_steps):
            s = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            )

            if self._is_cartpole() or self._is_acrobot():
                s = self._preprocess_state(s)

            out = self.policy.forward(s)

            if self.logits:
                dist = Categorical(logits=out)
            else:
                # probs = self._safe_probs(out)
                dist = Categorical(probs=out)

            action = dist.sample()
            logp = dist.log_prob(action)
            entropy = dist.entropy()

            next_state, reward, terminated, truncated, _ = self.env.step(action.item())

            shaped_reward = float(reward)
            if use_shaping:
                if self.shaping_mode == "dense":
                    # Persistent height reward: rewards being high (incl. the
                    # goal state). Does not telescope, so the value baseline
                    # reduces its variance without cancelling its direction.
                    phi_next = self._potential(next_state)
                    shaped_reward += self.shaping_coef * phi_next
                    phi_s = phi_next
                else:  # "potential"
                    # Policy-invariant but telescopes -> cancelled by a value
                    # baseline. Phi(terminal)=0 only on a true terminal (goal);
                    # keep the real potential on truncation (time limit).
                    phi_next = 0.0 if terminated else self._potential(next_state)
                    shaped_reward += self.shaping_coef * (self.gamma * phi_next - phi_s)
                    phi_s = phi_next

            traj_states.append(s)
            traj_logp.append(logp)
            traj_rewards.append(shaped_reward)
            traj_true_rewards.append(float(reward))
            traj_entropy.append(entropy)

            state = next_state

            if terminated or truncated:
                break

        return (
            traj_states,
            torch.stack(traj_logp),
            traj_rewards,
            torch.stack(traj_entropy),
            float(sum(traj_true_rewards)),
        )

    def _discount(self, rewards: Sequence[float]) -> torch.Tensor:
        """Compute reward-to-go with zero tail bootstrap; return a float32 tensor on the trainer device."""
        R = 0.0
        disc: list[float] = []

        for r in reversed(rewards):
            R = r + self.gamma * R
            disc.insert(0, R)

        return torch.tensor(
            disc,
            dtype=torch.float32,
            device=self.device,
        )

    def _optimise_loss(self, loss: torch.Tensor):
        """Skip nonfinite loss; otherwise backpropagate, optionally clip gradients and apply one Adam step."""
        self.opt.zero_grad(set_to_none=True)

        if not torch.isfinite(loss):
            print("[WARN] Non-finite loss:", float(loss.detach().cpu().item()))
            return

        loss.backward()

        if self.max_grad_norm is not None:
            params_to_clip = list(self.policy.parameters())

            qlayer = getattr(self.policy, "qlayer", None)
            if qlayer is not None:
                for name in ("params", "input_scaling_params"):
                    t = getattr(qlayer, name, None)
                    if isinstance(t, torch.Tensor) and t.requires_grad:
                        params_to_clip.append(t)

            torch.nn.utils.clip_grad_norm_(
                params_to_clip,
                max_norm=self.max_grad_norm,
            )
            #print(f"  grad_norm = {float(total_norm):.2e}")

        self.opt.step()

#### 

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn


def _to_tensor_obs(obs: Any, device: torch.device) -> torch.Tensor:
    """Convert a Gym/Gymnasium obs into a flat float32 torch tensor."""
    if isinstance(obs, torch.Tensor):
        return obs.to(device=device, dtype=torch.float32).view(-1)

    if isinstance(obs, dict):
        parts = []
        for k in sorted(obs.keys()):
            v = np.asarray(obs[k], dtype=np.float32).reshape(-1)
            parts.append(v)
        x = np.concatenate(parts, axis=0)
        return torch.as_tensor(x, device=device, dtype=torch.float32)

    x = np.asarray(obs, dtype=np.float32).reshape(-1)
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def _safe_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Sanitize a probability vector to be a valid simplex."""
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = probs.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


@dataclass
class BanditAWRConfig:
    # Optim / data
    """Settings for advantage-weighted contextual-bandit regression.

    lr controls Adam; batch_size and updates_per_epoch control replay updates.
    Weights are exp((reward - EMA baseline)/temperature), capped at weight_clip.
    baseline_beta weights the previous baseline. entropy_coef adds exploration
    regularization; grad_clip_norm and eps provide numerical controls. device
    chooses tensor placement, and seed initializes global Torch/NumPy RNGs.
    These fields configure ContextualBanditTrainer, not multi-step DQN."""
    lr: float = 1e-2
    batch_size: int = 64
    updates_per_epoch: int = 10

    # AWR / REPS-style weighting
    temperature: float = 0.2          # smaller -> more greedy weights
    weight_clip: float = 20.0         # cap exp weights
    baseline_beta: float = 0.95       # EMA reward baseline

    # Regularization / stability
    entropy_coef: float = 0.0         # optional entropy bonus
    grad_clip_norm: float = 1.0
    eps: float = 1e-8

    # System
    device: Optional[str] = None
    seed: int = 0


class ContextualBanditTrainer:
    """
    Contextual bandit trainer using Advantage-Weighted Regression (AWR).

    Key property for your PQC:
    - PennyLane TorchLayer often fails on batched inputs [B, obs_dim].
      This trainer therefore evaluates the policy ONE OBS AT A TIME in update().

    Bandit assumption:
    - one decision per episode: observe s, pick a, receive r.
      If your env is multi-step, we only use the first step.
    """

    def __init__(self, env, policy: nn.Module, n_actions: int, cfg: BanditAWRConfig):
        """Store environment/policy/config, seed Torch and NumPy, move the policy and allocate Adam plus replay lists."""
        self.env = env
        self.policy = policy
        self.n_actions = n_actions
        self.cfg = cfg

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy.to(self.device)

        params = list(self.policy.parameters())
        if len(params) == 0:
            raise ValueError("Policy has no parameters(). Ensure PQCBorn exposes trainable torch params.")
        self.opt = torch.optim.Adam(params, lr=cfg.lr)

        # reward baseline (EMA)
        self.baseline = 0.0

        # replay buffer for (s, a, r)
        self._obs_buf: List[torch.Tensor] = []
        self._act_buf: List[int] = []
        self._rew_buf: List[float] = []

    # ---------------- Gym/Gymnasium adapters ----------------

    def _env_reset(self):
        """Return observations from either old Gym reset output or Gymnasium (observation, info)."""
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:  # gymnasium (obs, info)
            return out[0]
        return out

    def _env_step(self, action: int):
        """Normalize old/new Gym step outputs to (observation, float reward, done, info)."""
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:  # gymnasium
            obs2, r, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs2, float(r), done, info
        obs2, r, done, info = out  # old gym
        return obs2, float(r), bool(done), info

    # ---------------- Policy eval (batch-safe) ----------------

    def _policy_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate policy.forward() in a way that works even if the underlying
        TorchLayer can't handle batch dims.

        Returns:
          - obs [obs_dim]      -> probs [A]
          - obs [B, obs_dim]   -> probs [B, A] (computed by looping)
        """
        if obs.ndim == 1:
            p = self.policy.forward(obs)
            return _safe_probs(p, eps=self.cfg.eps)

        # batched: loop (keeps gradients!)
        probs_list = []
        for i in range(obs.shape[0]):
            p = self.policy.forward(obs[i])
            probs_list.append(_safe_probs(p, eps=self.cfg.eps))
        return torch.stack(probs_list, dim=0)

    @torch.no_grad()
    def act(self, obs: Any, explore: bool = True) -> int:
        """Return a sampled action when explore=True, otherwise the largest-probability action."""
        s = _to_tensor_obs(obs, self.device)
        probs = self._policy_probs(s)  # [A]
        if explore:
            a = torch.distributions.Categorical(probs=probs).sample().item()
        else:
            a = int(torch.argmax(probs).item())
        return a

    # ---------------- Data collection ----------------

    def collect(self, n_episodes: int, explore: bool = True) -> Dict[str, float]:
        """Collect one-step bandit data: (s, a, r) from n_episodes."""
        rews = []
        for _ in range(n_episodes):
            obs = self._env_reset()
            a = self.act(obs, explore=explore)

            _obs2, r, _done, _info = self._env_step(a)  # one step only

            self._obs_buf.append(_to_tensor_obs(obs, self.device))
            self._act_buf.append(int(a))
            self._rew_buf.append(float(r))
            rews.append(float(r))

            # update baseline EMA
            self.baseline = self.cfg.baseline_beta * self.baseline + (1.0 - self.cfg.baseline_beta) * float(r)

        return {
            "mean_reward": float(np.mean(rews)) if rews else 0.0,
            "baseline": float(self.baseline),
            "buffer_size": float(len(self._rew_buf)),
        }

    def _sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample replay without replacement and return observation/action/reward tensors; reject an empty buffer."""
        n = len(self._rew_buf)
        if n == 0:
            raise RuntimeError("Buffer empty. Call collect() before update().")

        b = min(self.cfg.batch_size, n)
        idx = np.random.choice(n, size=b, replace=False)

        obs = torch.stack([self._obs_buf[i] for i in idx], dim=0)  # [B, obs_dim]
        act = torch.as_tensor([self._act_buf[i] for i in idx], device=self.device, dtype=torch.long)  # [B]
        rew = torch.as_tensor([self._rew_buf[i] for i in idx], device=self.device, dtype=torch.float32)  # [B]
        return obs, act, rew

    # ---------------- Learning ----------------

    def update(self) -> Dict[str, float]:
        """Run a few gradient updates using buffered data (AWR style)."""
        if len(self._rew_buf) == 0:
            return {"loss": 0.0, "entropy": 0.0}

        losses = []
        entropies = []

        for _ in range(self.cfg.updates_per_epoch):
            obs, act, rew = self._sample_batch()

            # IMPORTANT: batch-safe evaluation (loops internally if needed)
            probs = self._policy_probs(obs)  # [B, A]

            # log pi(a|s)
            logp = torch.log(probs.gather(-1, act.unsqueeze(-1)).squeeze(-1))  # [B]

            # advantages & weights
            adv = rew - float(self.baseline)
            w = torch.exp((adv / self.cfg.temperature).clamp(min=-50.0, max=50.0))
            w = w.clamp(max=self.cfg.weight_clip).detach()  # do not backprop through weights

            loss = -(w * logp).mean()

            if self.cfg.entropy_coef != 0.0:
                ent = -(probs * torch.log(probs)).sum(dim=-1).mean()
                loss = loss - self.cfg.entropy_coef * ent
                entropies.append(float(ent.detach().cpu().item()))

            self.opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip_norm)
            self.opt.step()

            losses.append(float(loss.detach().cpu().item()))

        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "buffer_size": float(len(self._rew_buf)),
        }

    def train(
        self,
        epochs: int = 50,
        episodes_per_epoch: int = 256,
        explore: bool = True,
        log_every: int = 1,
    ) -> List[float]:
        """
        Train for `epochs`. Each epoch:
          - collect `episodes_per_epoch` one-step samples
          - run `updates_per_epoch` SGD steps
        Returns history of mean rewards per epoch.
        """
        hist = []
        for ep in range(1, epochs + 1):
            stats = self.collect(episodes_per_epoch, explore=explore)
            upd = self.update()
            hist.append(stats["mean_reward"])

            if ep % log_every == 0:
                print(
                    f"Epoch {ep:04d} | mean_reward={stats['mean_reward']:.4f} "
                    f"| baseline={stats['baseline']:.4f} | loss={upd['loss']:.6f} "
                    f"| buf={int(upd['buffer_size'])}"
                )
        return hist



"""
reinforce.py
============
REINFORCE with baseline for PQC-based policies on OpenAI Gym environments.

Improvements over vanilla REINFORCE that matter specifically for PQC training:
  1. Variance-reduced returns    — value baseline V(s) subtracts a state-dependent
                                   (or global) estimate, reducing gradient variance
                                   (critical for PQCs prone to barren plateaus).
  2. Two baseline modes          — select via ReinforceConfig.baseline_type:
       "mean_return"  simple running average of past episode returns (no extra
                      parameters, no extra circuit evals — good starting point).
       "mlp"          small classical MLP fitted to MC returns each update step
                      (state-dependent, lower variance, recommended for harder envs).
  3. Entropy regularisation      — encourages exploration; prevents the PQC from
                                   collapsing to a deterministic policy too early.
  4. Return normalisation        — zero-mean unit-variance normalisation of
                                   advantages per episode; stabilises learning.
  5. Gradient clipping           — prevents occasional large parameter-shift
                                   gradients from destabilising the circuit.
  6. Episode batching            — averages gradients over N_BATCH episodes per
                                   update step, reducing update variance further.
  7. Early stopping              — stops when the solved criterion is met for
                                   SOLVED_WINDOW consecutive evaluation episodes.

Usage:
    from qasp import create_model
    policy = create_model("PQCBorn", policy_model_config, n_qubits=n_qubits)
    trainer = Reinforce(policy, env_name="CartPole-v1")
    trainer.train()
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import time


# ── Hyperparameters ───────────────────────────────────────────────────────────

@dataclass
class ReinforceConfig:
    # Environment
    """Configuration for the separate Reinforce Monte Carlo trainer.

    Environment fields set the name, episode cap and solved_score/window.
    n_episodes limits training, n_batch counts episodes per update and gamma
    discounts rewards. baseline_type is mean_return (EMA with baseline_momentum)
    or mlp (fit with lr_baseline). lr_policy, grad_clip and entropy_coef control
    policy updates; normalise_state enables online observation normalization.
    log_every/eval_every/n_eval_episodes control reports and greedy evaluations.
    fc_layer selects special optimizer groups for an already-existing head;
    it does not add an input layer and overrides ordinary learning rates.
    Create a fresh config per trainer because the trainer retains/may mutate it."""
    env_name:         str   = "CartPole-v1"
    max_episode_steps: int  = 500          # CartPole-v1 terminates at 500
    solved_score:     float = 475.0        # rolling mean to declare "solved"
    solved_window:    int   = 100          # episodes in rolling window

    # Training
    n_episodes:       int   = 2000         # max training episodes
    n_batch:          int   = 4            # episodes per gradient update (mini-batch REINFORCE)
    gamma:            float = 0.99         # discount factor

    # Baseline
    baseline_type:     str   = "mean_return"  # "mean_return" | "mlp"
    #   "mean_return" — exponential moving average of past episode returns.
    #                   No extra parameters.  A constant per-episode shift that
    #                   centres advantages around zero.  Good default for PQCs.
    #   "mlp"         — small classical network V(s) fitted to MC returns each
    #                   update.  State-dependent; lower variance on harder envs.
    baseline_momentum: float = 0.05   # EMA momentum for "mean_return"
    #                   b <- (1-m)*b + m*G after each episode.
    #                   Lower = slower adaptation; higher = tracks recent returns.
    #                   Typical range: 0.01 (very slow) - 0.2 (fast).

    # Optimiser
    lr_policy:        float = 0.01    # learning rate for PQC parameters
    lr_baseline:      float = 1e-3   # learning rate for MLP baseline (ignored for mean_return)
    grad_clip:        float = 1.0    # max gradient norm

    # Regularisation
    entropy_coef:     float = 0.01         # entropy bonus coefficient (0 = no bonus)

    # State preprocessing
    normalise_state:  bool  = True         # online normalise observations

    # Logging
    log_every:        int   = 10           # print every N episodes
    eval_every:       int   = 50           # run greedy evaluation every N episodes
    n_eval_episodes:  int   = 10           # episodes per evaluation
    fc_layer:       bool  = False         # use special optimizer groups for an existing head

# ── Classical value baseline (tiny MLP) ───────────────────────────────────────
'''
class ValueBaseline(nn.Module):
    """
    Small classical MLP that estimates V(s).

    Deliberately classical and fast:  PQC gradients are expensive (2 × n_params
    circuit evaluations per param-shift step) — offloading the baseline to a
    classical net costs almost nothing and halves gradient variance.

    Input:  raw (unnormalised) observation from gym
    Output: scalar V(s)
    """
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
'''

# ── Mean-return baseline ───────────────────────────────────────────────────────

class MeanReturnBaseline:
    """
    State-independent baseline: an exponential moving average (EMA) of past
    episode returns.

    V(s_t) = b  (same scalar for every timestep in the episode)

    where b is updated after each episode as:
        b ← (1 − momentum) · b  +  momentum · G_episode

    Why this works:
        A_t = G_t − b  centres the advantages around zero.  Even though the
        baseline ignores the state, it already removes most of the variance
        coming from the overall reward scale shifting during training.  This is
        often enough for PQC policies, where the circuit is the bottleneck and
        adding a state-dependent MLP adds negligible benefit vs. its complexity.

    Args:
        momentum: EMA smoothing factor in (0, 1).  Higher = faster adaptation
                  to recent returns but more noisy.  Default 0.05 works well for
                  CartPole where returns grow from ~20 to 500 over training.
    """

    def __init__(self, momentum: float = 0.05):
        """Initialize a scalar return baseline; momentum weights the next observed return."""
        self.momentum = momentum
        self.value    = 0.0       # current baseline estimate
        self._initialised = False

    def update(self, episode_return: float):
        """Call once per episode with the total undiscounted return."""
        if not self._initialised:
            self.value        = episode_return   # cold-start: no decay on first episode
            self._initialised = True
        else:
            self.value = (1.0 - self.momentum) * self.value + self.momentum * episode_return

    def get(self) -> float:
        """Return the current baseline estimate as a plain float."""
        return self.value


# ── Online running normaliser ──────────────────────────────────────────────────

class RunningNormaliser:
    """
    Welford online algorithm for mean/variance of observations.
    Keeps the PQC's RY angles in a sane range.
    """
    def __init__(self, shape):
        """Allocate float64 running means and regularized squared-deviation accumulators for shape."""
        self.n    = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2   = np.ones(shape,  dtype=np.float64)

    def update(self, x: np.ndarray):
        """Update Welford running statistics with one observation in place."""
        self.n  += 1
        delta    = x - self.mean
        self.mean += delta / self.n
        self.M2  += delta * (x - self.mean)

    @property
    def std(self):
        """Return regularized population standard deviation; M2 starts at one and variance is floored."""
        return np.sqrt(np.maximum(self.M2 / max(self.n, 1), 1e-8))

    def normalise(self, x: np.ndarray) -> np.ndarray:
        """Standardize an observation with current statistics without updating them."""
        return (x - self.mean) / self.std


# ── Trajectory storage ────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    """Mutable episode record of states, actions, rewards, log probabilities and baseline values."""
    states:   list = field(default_factory=list)
    actions:  list = field(default_factory=list)
    rewards:  list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values:   list = field(default_factory=list)   # V(s) from baseline

    @property
    def total_return(self):
        """Return the undiscounted sum of recorded episode rewards."""
        return sum(self.rewards)

    def discounted_returns(self, gamma: float) -> torch.Tensor:
        """
        G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
        Computed efficiently with a backwards cumsum.
        """
        R, returns = 0.0, []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)


# ── REINFORCE trainer ──────────────────────────────────────────────────────────

class Reinforce:
    """
    REINFORCE with learned value baseline for PQC policies.

    Args:
        policy:   Any nn.Module with .forward(state_tensor) → action probabilities.
                  Compatible with qasp's PQCBorn (parity-like or softmax output).
        config:   ReinforceConfig dataclass.  Defaults tuned for CartPole-v1.

    Example:
        policy  = create_model("PQCBorn", policy_model_config, n_qubits=n_qubits)
        trainer = Reinforce(policy)
        trainer.train()
    """

    def __init__(
        self,
        policy:   nn.Module,
        config:   ReinforceConfig = ReinforceConfig(),
        env_name: Optional[str]   = None,
    ):
        """Create the configured baseline, policy optimizer and observation normalizer.

        policy is a Torch module; config is retained (not copied), and env_name
        when supplied overwrites its name. A temporary Gym environment determines
        feature count and is closed. fc_layer=True uses fixed group rates 0.01 for
        the existing head and 0.1 for quantum/input-scale weights, overriding
        lr_policy. Supply a fresh ReinforceConfig to avoid sharing the default object."""
        self.policy = policy
        self.cfg    = config
        if env_name:
            self.cfg.env_name = env_name

        # Build a throwaway env to read obs shape
        _env         = gym.make(self.cfg.env_name)
        self.obs_dim = _env.observation_space.shape[0]
        _env.close()

        # ── Baseline ──────────────────────────────────────────────────────────
        if self.cfg.baseline_type == "mean_return":
            self.baseline      = MeanReturnBaseline(momentum=self.cfg.baseline_momentum)
            self.baseline_opt  = None   # no parameters to optimise
        elif self.cfg.baseline_type == "mlp":
            self.baseline      = ValueBaseline(self.obs_dim)
            self.baseline_opt  = torch.optim.Adam(
                self.baseline.parameters(), lr=self.cfg.lr_baseline
            )
        else:
            raise ValueError(
                f"Unknown baseline_type '{self.cfg.baseline_type}'. "
                "Choose 'mean_return' or 'mlp'."
            )

        if self.cfg.fc_layer == False:
            self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr_policy)
        else:
            self.policy_opt = torch.optim.Adam(

            [

            {

                "params": self.policy.fc_layer.parameters(),

                "lr": 1e-2,

            },

            {

                "params": [self.policy.qlayer.params],

                "lr": 1e-1,

            },

            {

                "params": [self.policy.qlayer.input_scaling_params],

                "lr": 1e-1,

            },

            ]

            )
        # Observation normaliser
        self.normaliser = RunningNormaliser(self.obs_dim)

        # Logging
        self.episode_returns: list = []
        self.eval_returns:    list = []
        self.rolling_window        = deque(maxlen=self.cfg.solved_window)


    # ── State preprocessing ────────────────────────────────────────────────────

    def preprocess(self, obs: np.ndarray, update_stats: bool = True) -> torch.Tensor:
        """
        Normalise and convert observation to torch tensor.
        PQC feature maps (NeuralEmbedding with rotations="Y") work best when
        inputs are in [−π, π].  We clip after normalisation to ±3σ.
        """
        if update_stats:
            self.normaliser.update(obs)
        normed = self.normaliser.normalise(obs)
        normed = np.clip(normed, -3.0, 3.0)
        return torch.tensor(normed, dtype=torch.float32)


    # ── Single episode rollout ─────────────────────────────────────────────────

    def rollout(self, env: gym.Env, greedy: bool = False) -> Trajectory:
        """
        Collect one episode of experience.

        Args:
            env:    Gym environment instance.
            greedy: If True, always pick argmax action (evaluation mode).
        """
        traj = Trajectory()
        obs, _ = env.reset()

        for _ in range(self.cfg.max_episode_steps):
            state = self.preprocess(obs, update_stats=not greedy)

            # Policy forward pass
            with torch.set_grad_enabled(not greedy):
                probs = self.policy.forward(state)

            dist   = Categorical(probs)
            action = dist.sample() if not greedy else probs.argmax()

            # Value estimate per step — depends on baseline type:
            #   MLP:         V(s_t) queried each step (state-dependent)
            #   mean_return: current scalar b used for every step (filled in
            #                after the episode ends, so we leave a placeholder)
            if self.cfg.baseline_type == "mlp":
                with torch.no_grad():
                    value = self.baseline(state)
                traj.values.append(value)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            traj.states.append(state)
            traj.actions.append(action)
            traj.rewards.append(float(reward))
            traj.log_probs.append(dist.log_prob(action))

            if done:
                break

        # For mean_return baseline: fill values with the current EMA scalar.
        # We do this AFTER the rollout so the same b applies to every timestep
        # in the episode (it is a per-episode, not per-step, estimate).
        if self.cfg.baseline_type == "mean_return":
            b = self.baseline.get()
            traj.values = [torch.tensor(b, dtype=torch.float32)] * len(traj.rewards)

        return traj


    # ── Advantage computation ──────────────────────────────────────────────────

    def compute_advantages(self, traj: Trajectory) -> torch.Tensor:
        """
        Advantage = discounted return − baseline estimate.
        Normalised to zero mean and unit variance within the episode.

        A_t = G_t − V(s_t)
        Â_t = (A_t − mean(A)) / (std(A) + ε)

        For "mean_return" baseline: V(s_t) = b for all t (same scalar).
          The normalisation step then mostly cancels b out, but the sign and
          relative magnitude of advantages across timesteps is preserved — which
          is exactly what the policy gradient needs.

        For "mlp" baseline: V(s_t) is state-dependent, so advantages carry
          genuine information about which states were above/below expectation.
        """
        returns   = traj.discounted_returns(self.cfg.gamma)
        values    = torch.stack(traj.values)
        advantages = returns - values.detach()

        # Normalise advantages (per episode)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages


    # ── Policy gradient loss ───────────────────────────────────────────────────

    def policy_loss(self, traj: Trajectory, advantages: torch.Tensor) -> torch.Tensor:
        """
        L_π = − Σ_t [ A_t · log π(a_t|s_t) ] − β · H[π(·|s_t)]

        The entropy term H[π] = −Σ_a π(a|s)·log π(a|s) encourages exploration.
        """
        log_probs = torch.stack(traj.log_probs)
        pg_loss   = -(advantages * log_probs).mean()

        # Entropy bonus: recompute probabilities for all states
        entropy = torch.tensor(0.0)
        if self.cfg.entropy_coef > 0:
            states = torch.stack(traj.states)
            # Forward pass over all states (batched if policy supports it,
            # otherwise loop — PQC policies are sequential by nature)
            h = []
            for s in states:
                probs = self.policy.forward(s)
                h.append(Categorical(probs).entropy())
            entropy = torch.stack(h).mean()

        return pg_loss - self.cfg.entropy_coef * entropy


    # ── Baseline loss ──────────────────────────────────────────────────────────

    def baseline_loss(self, traj: Trajectory, returns: torch.Tensor) -> Optional[torch.Tensor]:
        """
        MSE between V(s_t) and the MC return G_t.
        Returns None for mean_return baseline (no parameters to train).
        """
        if self.cfg.baseline_type == "mean_return":
            return None
        states = torch.stack(traj.states)
        values = self.baseline(states)
        return nn.functional.mse_loss(values, returns)


    # ── Update step ───────────────────────────────────────────────────────────

    def update(self, trajectories: list[Trajectory]):
        """Optimize a nonempty trajectory batch and update the baseline; return (policy_loss, baseline_loss) floats."""
        total_policy_loss = torch.tensor(0.0)
        total_baseline_loss = torch.tensor(0.0)

        for traj in trajectories:
            if self.cfg.baseline_type == "mean_return":
                b = self.baseline.get()
                traj.values = [torch.tensor(b, dtype=torch.float32)] * len(traj.rewards)

            returns, advantages = self.compute_advantages(traj)

            total_policy_loss = total_policy_loss + self.policy_loss(traj, advantages)

            bl = self.baseline_loss(traj, returns)
            if bl is not None:
                total_baseline_loss = total_baseline_loss + bl

            if self.cfg.baseline_type == "mean_return":
                self.baseline.update(traj.total_return)

        total_policy_loss = total_policy_loss / len(trajectories)

        self.policy_opt.zero_grad()
        total_policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.policy_opt.step()

        v_loss_val = 0.0
        if self.cfg.baseline_type == "mlp" and self.baseline_opt is not None:
            total_baseline_loss = total_baseline_loss / len(trajectories)
            self.baseline_opt.zero_grad()
            total_baseline_loss.backward()
            self.baseline_opt.step()
            v_loss_val = total_baseline_loss.item()

        return total_policy_loss.item(), v_loss_val


    # ── Greedy evaluation ──────────────────────────────────────────────────────

    def evaluate(self) -> float:
        """Run N greedy episodes and return the mean total return."""
        env   = gym.make(self.cfg.env_name)
        total = 0.0
        for _ in range(self.cfg.n_eval_episodes):
            traj  = self.rollout(env, greedy=True)
            total += traj.total_return
        env.close()
        return total / self.cfg.n_eval_episodes


    # ── Main training loop ────────────────────────────────────────────────────

    def train(self):
        """
        Main training loop.  Runs until solved or n_episodes exhausted.

        Returns:
            dict with 'episode_returns', 'eval_returns', 'solved' keys.
        """
        env     = gym.make(self.cfg.env_name)
        episode = 0
        start   = time.time()
        solved  = False

        print(f"\n{'='*60}")
        print(f"  REINFORCE  |  env={self.cfg.env_name}")
        print(f"  batch={self.cfg.n_batch}  gamma={self.cfg.gamma}  "
              f"entropy_coef={self.cfg.entropy_coef}")
        print(f"  baseline={self.cfg.baseline_type}", end="")
        if self.cfg.baseline_type == "mean_return":
            print(f"  momentum={self.cfg.baseline_momentum}")
        else:
            print(f"  lr_baseline={self.cfg.lr_baseline}")
        print(f"  lr_policy={self.cfg.lr_policy}  grad_clip={self.cfg.grad_clip}")
        print(f"  solved at mean>={self.cfg.solved_score} "
              f"over {self.cfg.solved_window} episodes")
        print(f"{'='*60}\n")

        while episode < self.cfg.n_episodes:

            # ── Collect a batch of episodes ───────────────────────────────────
            batch = []
            for _ in range(self.cfg.n_batch):
                traj = self.rollout(env)
                batch.append(traj)
                self.episode_returns.append(traj.total_return)
                self.rolling_window.append(traj.total_return)
                episode += 1

            # ── Update ───────────────────────────────────────────────────────
            pi_loss, v_loss = self.update(batch)

            # ── Logging ──────────────────────────────────────────────────────
            if episode % self.cfg.log_every == 0:
                rolling_mean = np.mean(self.rolling_window)
                batch_return = np.mean([t.total_return for t in batch])
                elapsed      = time.time() - start
                baseline_str = (
                    f"b={self.baseline.get():.1f}"
                    if self.cfg.baseline_type == "mean_return"
                    else f"V_loss={v_loss:.4f}"
                )
                print(
                    f"  ep={episode:5d} | "
                    f"return={batch_return:7.1f} | "
                    f"rolling={rolling_mean:7.2f} | "
                    f"pi_loss={pi_loss:.4f} | "
                    f"{baseline_str} | "
                    f"t={elapsed:.0f}s",
                    flush=True
                )

            # ── Periodic evaluation ───────────────────────────────────────────
            if episode % self.cfg.eval_every == 0:
                eval_return = self.evaluate()
                self.eval_returns.append((episode, eval_return))
                print(f"  {'─'*50}")
                print(f"  [EVAL] ep={episode}  mean_return={eval_return:.1f}")
                print(f"  {'─'*50}")

            # ── Solved check ──────────────────────────────────────────────────
            if (len(self.rolling_window) == self.cfg.solved_window and
                    np.mean(self.rolling_window) >= self.cfg.solved_score):
                rolling_mean = np.mean(self.rolling_window)
                print(f"\n  ✓ SOLVED at episode {episode} "
                      f"(rolling mean = {rolling_mean:.1f})\n")
                solved = True
                break

        env.close()
        return {
            "episode_returns": self.episode_returns,
            "eval_returns":    self.eval_returns,
            "solved":          solved,
            "n_episodes":      episode,
        }


from dataclasses import dataclass, field
from typing import List

import gymnasium as gym
import numpy as onp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

__all__ = ["REINFORCE", "PPO", "DeepQLearning"]


class ValueMLP(nn.Module):
    """Small default critic for low-dimensional control tasks like CartPole."""

    def __init__(self, obs_dim: int, hidden_sizes=(64, 64)):
        """Build a tanh MLP with hidden_sizes mapping obs_dim inputs to a scalar critic output."""
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return value predictions with all leading batch dimensions preserved."""
        return self.net(x).squeeze(-1)


@dataclass
class PPO:
    """Clipped policy optimization with GAE and a separate value network.

    policy returns logits (logits=True) or action probabilities. env_name is a
    Gymnasium name/instance. gamma and gae_lambda control advantage targets;
    actor_lr/critic_lr configure Adam. steps_per_rollout counts transitions,
    epochs repeats optimization, and minibatch_size sets update size. clip_eps
    bounds likelihood ratios; value_coef and entropy_coef weight loss terms.
    normalize_advantages standardizes the rollout advantages. normalize_mode
    applies only to named CartPole (clip, tanh or none). device selects Torch
    placement. max_grad_norm=None disables clipping. Supply value_net or use
    a default tanh MLP. fc_layer enables fixed rates for an existing head
    (0.01) and quantum weights (0.001), rather than creating a layer.

    Each rollout resets the environment, so an unfinished episode at its end
    is not carried into the next rollout. Termination and time-limit truncation
    both mask GAE bootstrapping. max_steps is retained but not enforced by the
    collector; use the environment's time limit. See train for return records
    and the full-window stopping check. This is distinct from TorchRLTrainer."""

    policy: nn.Module
    env_name: str = "CartPole-v1"

    gamma: float = 0.99
    gae_lambda: float = 0.95

    actor_lr: float = 3e-4
    critic_lr: float = 1e-3

    steps_per_rollout: int = 2048
    epochs: int = 10
    minibatch_size: int = 64

    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    max_steps: int = 500
    device: torch.device | str | None = None

    logits: bool = True

    normalize_mode: str = "clip"
    normalize_advantages: bool = True

    max_grad_norm: float | None = 0.5

    fc_layer: bool = False
    value_net: nn.Module | None = None

    actor_opt: torch.optim.Optimizer = field(init=False, repr=False)
    critic_opt: torch.optim.Optimizer = field(init=False, repr=False)

    def __post_init__(self):
        """Create/reuse the environment, place actor/critic on the tensor device and configure Adam optimizers."""
        if isinstance(self.env_name, str):
            self.env = gym.make(self.env_name)
        else:
            self.env = self.env_name

        self.device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy.to(self.device)

        obs_dim = self.env.observation_space.shape[0]

        if self.value_net is None:
            self.value_net = ValueMLP(obs_dim)
        self.value_net.to(self.device)

        actor_params = list(self.policy.parameters())

        qlayer = getattr(self.policy, "qlayer", None)
        if qlayer is not None:
            for name in ("params", "input_scaling_params"):
                t = getattr(qlayer, name, None)
                if (
                    isinstance(t, torch.Tensor)
                    and t.requires_grad
                    and not isinstance(t, nn.Parameter)
                ):
                    if not t.is_leaf:
                        raise RuntimeError(
                            f"Found trainable tensor qlayer.{name}, but it is not a leaf tensor. "
                            "Make it an nn.Parameter in your model."
                        )
                    actor_params.append(t)

        if self.fc_layer:
            actor_groups = []

            if hasattr(self.policy, "fc_layer"):
                actor_groups.append(
                    {"params": self.policy.fc_layer.parameters(), "lr": 1e-2}
                )

            if qlayer is not None and hasattr(qlayer, "params"):
                actor_groups.append({"params": [qlayer.params], "lr": 1e-3})

            if qlayer is not None and hasattr(qlayer, "input_scaling_params"):
                actor_groups.append(
                    {"params": [qlayer.input_scaling_params], "lr": 1e-3}
                )

            if not actor_groups:
                actor_groups = [{"params": actor_params, "lr": self.actor_lr}]

            self.actor_opt = Adam(actor_groups, amsgrad=True)
        else:
            self.actor_opt = Adam(actor_params, lr=self.actor_lr, amsgrad=True)

        self.critic_opt = Adam(
            self.value_net.parameters(), lr=self.critic_lr, amsgrad=True
        )

    def _normalize_cartpole_state(self, state: torch.Tensor) -> torch.Tensor:
        """Scale four CartPole features and clip to [-1,1], tanh, or return raw input for none."""
        scale = torch.tensor(
            [2.4, 3.0, 0.2095, 3.5],
            dtype=state.dtype,
            device=state.device,
        )

        x = state / scale

        if self.normalize_mode == "clip":
            return torch.clamp(x, -1.0, 1.0)

        if self.normalize_mode == "tanh":
            return torch.tanh(x)

        if self.normalize_mode is None or self.normalize_mode == "none":
            return state

        raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")

    @staticmethod
    def _safe_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Replace nonfinite action weights, clamp positive and normalize their final axis."""
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs.clamp_min(eps)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def _preprocess_state(self, state) -> torch.Tensor:
        """Convert one observation to float32 and optionally apply named-CartPole preprocessing."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if isinstance(self.env_name, str) and self.env_name.startswith("CartPole"):
            s = self._normalize_cartpole_state(s)
        return s

    def _policy_forward_batched(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """
        Safe batched forward for policies that may not support true TorchLayer batching.
        Falls back to per-sample evaluation.
        """
        if obs_batch.ndim == 1:
            out = self.policy(obs_batch.unsqueeze(0))
            return out.squeeze(0)

        outs = []
        for obs in obs_batch:
            o = self.policy(obs.unsqueeze(0))
            outs.append(o.squeeze(0))

        return torch.stack(outs, dim=0)

    def _dist_from_policy(self, obs_batch: torch.Tensor) -> Categorical:
        """Construct a categorical action distribution from batched logits or sanitized probabilities."""
        out = self._policy_forward_batched(obs_batch)

        if self.logits:
            return Categorical(logits=out)

        probs = self._safe_probs(out)
        return Categorical(probs=probs)

    def _get_solve_threshold(self) -> tuple[float | None, int]:
        """
        Returns:
            (threshold, window)

        threshold = None means unknown / unsupported environment, so no early stopping.
        """
        if not isinstance(self.env_name, str):
            return None, 100

        name = self.env_name

        if name == "CartPole-v0":
            return 195.0, 100

        if name == "CartPole-v1":
            return 475.0, 100

        return None, 100

    def _check_solved(self, episode_returns: List[float]) -> bool:
        """Require a full task-specific trailing episode window before testing its training-return threshold."""
        threshold, window = self._get_solve_threshold()

        if threshold is None:
            return False

        if len(episode_returns) < window:
            return False

        running_mean = float(onp.mean(episode_returns[-window:]))
        return running_mean >= threshold

    @torch.no_grad()
    def _collect_rollout(self):
        """Reset and collect a fixed transition batch with GAE targets and completed-episode statistics."""
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []

        ep_returns = []
        ep_lengths = []

        ep_ret = 0.0
        ep_len = 0

        state, _ = self.env.reset()

        for _ in range(self.steps_per_rollout):
            obs = self._preprocess_state(state)
            obs_b = obs.unsqueeze(0)

            dist = self._dist_from_policy(obs_b)
            value = self.value_net(obs_b).squeeze(0)

            action = dist.sample()
            logp = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            obs_buf.append(obs)
            act_buf.append(action.squeeze())
            logp_buf.append(logp.squeeze())
            rew_buf.append(float(reward))
            done_buf.append(float(done))
            val_buf.append(value.squeeze())

            ep_ret += float(reward)
            ep_len += 1
            state = next_state

            if done:
                ep_returns.append(ep_ret)
                ep_lengths.append(ep_len)
                ep_ret = 0.0
                ep_len = 0
                state, _ = self.env.reset()

        last_obs = self._preprocess_state(state).unsqueeze(0)
        last_value = self.value_net(last_obs).squeeze(0)

        obs = torch.stack(obs_buf)
        actions = torch.stack(act_buf).long()
        logp_old = torch.stack(logp_buf)
        values = torch.stack(val_buf)
        rewards = torch.tensor(rew_buf, dtype=torch.float32, device=self.device)
        dones = torch.tensor(done_buf, dtype=torch.float32, device=self.device)

        advantages, returns = self._compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_value=last_value,
        )

        if self.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "obs": obs,
            "actions": actions,
            "logp_old": logp_old,
            "advantages": advantages,
            "returns": returns,
            "values": values,
            "ep_returns": ep_returns,
            "ep_lengths": ep_lengths,
        }

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ):
        """Return (advantages, returns) for one-dimensional transition sequences.

        Backwards recursion uses gamma and gae_lambda. dones masks both termination
        and truncation; last_value bootstraps only an unfinished rollout boundary.
        Returns are advantages plus the recorded value predictions."""
        T = rewards.shape[0]
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    def _iterate_minibatches(self, batch_size: int):
        """Yield randomly permuted device-local indices, including the last smaller minibatch."""
        idx = torch.randperm(batch_size, device=self.device)
        for start in range(0, batch_size, self.minibatch_size):
            yield idx[start : start + self.minibatch_size]

    def train(self, total_updates: int = 200, log_every: int = 10):
        """Run up to total_updates PPO rollouts/optimization cycles and return a statistics dictionary.

        Records episode_returns/lengths, mean completed-episode return per update
        (zero if none), update_episode_counts and solved flags/indices. Stops when
        the known task's full-window training-return criterion is met. log_every
        prints progress; this is not a held-out evaluation. Nonfinite losses are
        skipped, and no parameter checkpoint is written automatically."""
        update_returns: List[float] = []
        episode_returns: List[float] = []
        episode_lengths: List[int] = []
        update_episode_counts: List[int] = []

        solved = False
        solved_at_episode: int | None = None
        solved_at_update: int | None = None

        for update in range(1, total_updates + 1):
            batch = self._collect_rollout()
            obs = batch["obs"]
            actions = batch["actions"]
            logp_old = batch["logp_old"]
            advantages = batch["advantages"]
            returns = batch["returns"]

            batch_size = obs.shape[0]

            for _ in range(self.epochs):
                for mb_idx in self._iterate_minibatches(batch_size):
                    mb_obs = obs[mb_idx]
                    mb_actions = actions[mb_idx]
                    mb_logp_old = logp_old[mb_idx]
                    mb_adv = advantages[mb_idx]
                    mb_returns = returns[mb_idx]

                    dist = self._dist_from_policy(mb_obs)
                    new_logp = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_logp - mb_logp_old)

                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.clip_eps,
                        1.0 + self.clip_eps,
                    ) * mb_adv

                    actor_loss = -torch.min(surr1, surr2).mean()

                    value_pred = self.value_net(mb_obs)
                    critic_loss = F.mse_loss(value_pred, mb_returns)

                    loss = (
                        actor_loss
                        + self.value_coef * critic_loss
                        - self.entropy_coef * entropy
                    )

                    if not torch.isfinite(loss):
                        print("[WARN] Non-finite PPO loss:", float(loss.detach().cpu()))
                        continue

                    self.actor_opt.zero_grad(set_to_none=True)
                    self.critic_opt.zero_grad(set_to_none=True)

                    loss.backward()

                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.policy.parameters(),
                            max_norm=self.max_grad_norm,
                        )
                        torch.nn.utils.clip_grad_norm_(
                            self.value_net.parameters(),
                            max_norm=self.max_grad_norm,
                        )

                    self.actor_opt.step()
                    self.critic_opt.step()

            batch_ep_returns = batch["ep_returns"]
            batch_ep_lengths = batch["ep_lengths"]

            episode_returns.extend(batch_ep_returns)
            episode_lengths.extend(batch_ep_lengths)
            update_episode_counts.append(len(batch_ep_returns))

            mean_ep_ret = (
                float(onp.mean(batch_ep_returns)) if batch_ep_returns else 0.0
            )
            update_returns.append(mean_ep_ret)

            if update % log_every == 0:
                avg10_updates = onp.mean(update_returns[-10:])
                avg20_episodes = (
                    onp.mean(episode_returns[-20:]) if len(episode_returns) > 0 else 0.0
                )
                total_eps = len(episode_returns)
                print(
                    f"Update {update:04d} | "
                    f"avg(update_return_last10) = {avg10_updates:.1f} | "
                    f"avg(episode_return_last20) = {avg20_episodes:.1f} | "
                    f"episodes_seen = {total_eps}"
                )

            if self._check_solved(episode_returns):
                solved = True
                solved_at_episode = len(episode_returns)
                solved_at_update = update
                print(f"Environment solved at episode {solved_at_episode} (update {solved_at_update}).")
                break

        return {
            "episode_returns": episode_returns,
            "update_returns": update_returns,
            "episode_lengths": episode_lengths,
            "update_episode_counts": update_episode_counts,
            "solved": solved,
            "solved_at_episode": solved_at_episode,
            "solved_at_update": solved_at_update,
        }



import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ContextualBanditEnv(gym.Env):
    """One-step synthetic Gymnasium bandit over a finite table of real contexts.

    Each action has a linear score, converted by a sigmoid to success probability.
    Reward modes expose normalized/raw dense rewards or Bernoulli rewards.
    This is a controllable test task, not a model of CartPole dynamics."""
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        n_contexts: int = 1000,
        seed: int | None = None,
        contexts: np.ndarray | None = None,
        reward_mode: str = "dense_normalized",
        base_reward_scale: float = 10.0,
        optimal_reward_bonus: float = 1.0,
        reward_shift: float = 0.0,
        temperature: float = 1.0,
        action_weights: np.ndarray | None = None,
        action_bias: np.ndarray | None = None,
    ):
        """Create or accept contexts and a linear action reward model.

        contexts has shape (n_contexts,n_features), action_weights has shape
        (n_actions,n_features), and action_bias has shape (n_actions,). Missing
        arrays are sampled with a local NumPy generator seeded by seed. temperature
        must be positive and divides action scores. reward_mode is dense_normalized,
        dense_raw, stochastic or deterministic (an alias of dense_raw).
        base_reward_scale, optimal_reward_bonus and reward_shift set rewards.
        Provided context values are shape-checked but not clipped to the declared
        [-1,1] observation range. Call reset before step."""
        super().__init__()

        if n_features <= 0:
            raise ValueError("n_features must be positive.")
        if n_actions <= 1:
            raise ValueError("n_actions must be at least 2.")
        if n_contexts <= 0:
            raise ValueError("n_contexts must be positive.")
        if reward_mode not in {
            "dense_normalized",
            "dense_raw",
            "stochastic",
            "deterministic",
        }:
            raise ValueError(
                "reward_mode must be one of: "
                "'dense_normalized', 'dense_raw', 'stochastic', 'deterministic'"
            )
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")

        self.n_features = n_features
        self.n_actions = n_actions
        self.n_contexts = n_contexts
        self.reward_mode = reward_mode
        self.base_reward_scale = float(base_reward_scale)
        self.optimal_reward_bonus = float(optimal_reward_bonus)
        self.reward_shift = float(reward_shift)
        self.temperature = float(temperature)

        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_actions)

        if contexts is None:
            self.contexts = self.rng.uniform(
                low=-1.0,
                high=1.0,
                size=(self.n_contexts, self.n_features),
            ).astype(np.float32)
        else:
            contexts = np.asarray(contexts, dtype=np.float32)
            expected_shape = (self.n_contexts, self.n_features)
            if contexts.shape != expected_shape:
                raise ValueError(
                    f"contexts must have shape {expected_shape}, got {contexts.shape}"
                )
            self.contexts = contexts

        if action_weights is None:
            self.action_weights = self.rng.normal(
                loc=0.0,
                scale=1.0,
                size=(self.n_actions, self.n_features),
            ).astype(np.float32)
        else:
            action_weights = np.asarray(action_weights, dtype=np.float32)
            if action_weights.shape != (self.n_actions, self.n_features):
                raise ValueError(
                    f"action_weights must have shape {(self.n_actions, self.n_features)}, "
                    f"got {action_weights.shape}"
                )
            self.action_weights = action_weights

        if action_bias is None:
            self.action_bias = self.rng.normal(
                loc=0.0,
                scale=0.25,
                size=(self.n_actions,),
            ).astype(np.float32)
        else:
            action_bias = np.asarray(action_bias, dtype=np.float32)
            if action_bias.shape != (self.n_actions,):
                raise ValueError(
                    f"action_bias must have shape {(self.n_actions,)}, got {action_bias.shape}"
                )
            self.action_bias = action_bias

        self.current_context_idx: int | None = None
        self.current_context: np.ndarray | None = None

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Apply the elementwise logistic function to action scores."""
        return 1.0 / (1.0 + np.exp(-x))

    def _compute_action_scores(self, context: np.ndarray) -> np.ndarray:
        """Return temperature-divided affine action scores for one context."""
        logits = (self.action_weights @ context + self.action_bias) / self.temperature
        return logits

    def _compute_action_success_probabilities(self, context: np.ndarray) -> np.ndarray:
        """Map action scores to sigmoid probabilities clipped away from zero and one."""
        logits = self._compute_action_scores(context)
        probs = self._sigmoid(logits)
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        return probs

    def get_optimal_actions(
        self,
        context: int | np.ndarray | None = None,
        return_probs: bool = False,
    ):
        """Return (best_action_indices, best_success_probability), optionally all probabilities.

        context is a table index, feature vector or None for the active context.
        Ties use numpy.isclose. With return_probs=True, append the full action
        probability vector. Requires reset when using the active-context default."""
        if context is None:
            if self.current_context is None:
                raise RuntimeError("No current context. Call reset() first or pass a context.")
            ctx = self.current_context
        elif isinstance(context, (int, np.integer)):
            ctx = self.contexts[int(context)]
        else:
            ctx = np.asarray(context, dtype=np.float32)
            if ctx.shape != (self.n_features,):
                raise ValueError(
                    f"context vector must have shape ({self.n_features},), got {ctx.shape}"
                )

        probs = self._compute_action_success_probabilities(ctx)
        best_prob = np.max(probs)
        optimal_actions = np.where(np.isclose(probs, best_prob))[0]

        if return_probs:
            return optimal_actions, float(best_prob), probs
        return optimal_actions, float(best_prob)

    def get_oracle_value(self) -> float:
        """Return the helper's mean nominal optimal reward across stored contexts.

        For dense_normalized this assumes normalized base reward 1. The step
        formula includes an epsilon denominator, so this nominal value need not
        be exact, especially for equal action probabilities. It also assumes
        nonnegative reward scale/bonus when interpreting the result as an optimum."""
        vals = []
        for i in range(self.n_contexts):
            _, _, probs = self.get_optimal_actions(i, return_probs=True)
            p_max = float(np.max(probs))
            p_min = float(np.min(probs))

            if self.reward_mode == "dense_normalized":
                base = 1.0
            elif self.reward_mode in {"dense_raw", "deterministic", "stochastic"}:
                base = p_max
            else:
                raise ValueError(f"Unknown reward_mode {self.reward_mode}")

            val = self.base_reward_scale * base + self.optimal_reward_bonus + self.reward_shift
            vals.append(val)

        return float(np.mean(vals))

    def get_reward_bounds(self) -> tuple[float, float]:
        """Return nominal (shift, scale+bonus+shift) bounds, assuming nonnegative scale and bonus."""
        min_r = self.reward_shift
        max_r = self.base_reward_scale + self.optimal_reward_bonus + self.reward_shift
        return float(min_r), float(max_r)

    def reset(self, *, seed: int | None = None, options=None):
        """Optionally reseed context selection, choose a table row and return its copy plus index metadata."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_context_idx = int(self.rng.integers(0, self.n_contexts))
        self.current_context = self.contexts[self.current_context_idx].copy()

        info = {"context_idx": self.current_context_idx}
        return self.current_context.copy(), info

    def step(self, action: int):
        """Reward one valid action and return (next_observation, reward, True, False, info).

        Reward uses the configured dense/Bernoulli base plus an optimal-action
        bonus and shift. The returned next observation is a random table row;
        reset must start the next one-step episode. The current context is not
        changed by this terminal transition. Invalid actions or missing reset raise."""
        if self.current_context is None:
            raise RuntimeError("Call reset() before step().")

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.n_actions - 1}].")

        probs = self._compute_action_success_probabilities(self.current_context)
        p = float(probs[action])

        optimal_actions, best_prob = self.get_optimal_actions(self.current_context)
        p_min = float(np.min(probs))
        p_max = float(np.max(probs))

        if self.reward_mode == "dense_normalized":
            base_reward = (p - p_min) / (p_max - p_min + 1e-8)
        elif self.reward_mode == "dense_raw":
            base_reward = p
        elif self.reward_mode == "stochastic":
            base_reward = float(self.rng.binomial(1, p))
        elif self.reward_mode == "deterministic":
            base_reward = p
        else:
            raise ValueError(f"Unknown reward_mode {self.reward_mode}")

        reward = self.base_reward_scale * base_reward

        if action in optimal_actions:
            reward += self.optimal_reward_bonus

        reward += self.reward_shift

        terminated = True
        truncated = False

        info = {
            "context_idx": self.current_context_idx,
            "chosen_action": int(action),
            "chosen_action_prob": p,
            "best_action_prob": best_prob,
            "min_action_prob": p_min,
            "base_reward": float(base_reward),
            "reward": float(reward),
            "optimal_actions": optimal_actions.tolist(),
            "all_action_probs": probs.tolist(),
        }

        next_context_idx = int(self.rng.integers(0, self.n_contexts))
        next_obs = self.contexts[next_context_idx].copy()

        return next_obs, float(reward), terminated, truncated, info

    def render(self):
        """Print active context, success probabilities, best actions and nominal reward bounds; return None."""
        if self.current_context is None:
            print("No active context. Call reset() first.")
            return

        optimal_actions, best_prob, probs = self.get_optimal_actions(
            self.current_context, return_probs=True
        )
        reward_min, reward_max = self.get_reward_bounds()

        print("Current context index:", self.current_context_idx)
        print("Current context:", self.current_context)
        print("Action probabilities:", probs)
        print("Optimal action(s):", optimal_actions, "| best probability:", best_prob)
        print("Oracle value:", self.get_oracle_value())
        print("Reward bounds:", (reward_min, reward_max))

    def save(self, path: str):
        """
        Save the environment definition to a .npz file.
        """
        np.savez_compressed(
            path,
            n_features=np.array(self.n_features, dtype=np.int64),
            n_actions=np.array(self.n_actions, dtype=np.int64),
            n_contexts=np.array(self.n_contexts, dtype=np.int64),
            reward_mode=np.array(self.reward_mode),
            base_reward_scale=np.array(self.base_reward_scale, dtype=np.float32),
            optimal_reward_bonus=np.array(self.optimal_reward_bonus, dtype=np.float32),
            reward_shift=np.array(self.reward_shift, dtype=np.float32),
            temperature=np.array(self.temperature, dtype=np.float32),
            contexts=self.contexts.astype(np.float32),
            action_weights=self.action_weights.astype(np.float32),
            action_bias=self.action_bias.astype(np.float32),
        )

    @classmethod
    def load(cls, path: str):
        """
        Load a saved environment from a .npz file.
        """
        data = np.load(path, allow_pickle=True)

        env = cls(
            n_features=int(data["n_features"]),
            n_actions=int(data["n_actions"]),
            n_contexts=int(data["n_contexts"]),
            seed=None,
            contexts=data["contexts"].astype(np.float32),
            reward_mode=str(data["reward_mode"].item()),
            base_reward_scale=float(data["base_reward_scale"]),
            optimal_reward_bonus=float(data["optimal_reward_bonus"]),
            reward_shift=float(data["reward_shift"]),
            temperature=float(data["temperature"]),
            action_weights=data["action_weights"].astype(np.float32),
            action_bias=data["action_bias"].astype(np.float32),
        )
        return env

class DeepQLearning:
    """Replay-based (optionally Double) DQN for Gymnasium discrete environments.

    ``qfunction`` is a PQDQN or a module mapping observations to action values.
    Target-network updates are counted in optimizer steps. Time-limit truncation
    ends a rollout but retains bootstrapping; only true termination masks it.
    ``observation_transform`` must be a deterministic function of an observation.
    Environment instances are caller-owned; environments created by name are
    closed by ``close()`` or the context manager. Evaluation uses a separate env.
    """

    def __init__(
        self,
        qfunction,
        env_name="CartPole-v1",
        *,
        lr=1e-3,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
        gamma=0.99,
        batch_size=64,
        replay_capacity=10000,
        learning_starts=256,
        train_frequency=1,
        gradient_steps=1,
        target_update_interval=100,
        double_dqn=True,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=10000,
        max_steps=500,
        max_grad_norm=10.0,
        seed=0,
        observation_transform=None,
    ):
        """Initialize replay-based DQN with an independent frozen target-network copy.

        Args:
            qfunction: Trainable module mapping (...,features) to (...,n_actions).
            env_name: Gymnasium name or caller-owned environment with zero-based
                Discrete actions matching the model's n_actions.
            lr, optimizer_class, optimizer_kwargs: Online-network optimizer settings.
            gamma: Reward discount in [0,1].
            batch_size, replay_capacity: Uniform replay minibatch and FIFO capacity.
            learning_starts: Environment frames before updates may begin; enough
                transitions for batch_size are also required.
            train_frequency, gradient_steps: Update cadence in frames and number
                of optimizer steps per update event.
            target_update_interval: Hard synchronization interval in optimizer steps.
            double_dqn: Select next actions online and evaluate with the target;
                False instead takes the target network maximum.
            epsilon_start, epsilon_end, epsilon_decay_steps: Linear exploration
                schedule indexed by cumulative environment frames.
            max_steps: Per-episode rollout/evaluation cap.
            max_grad_norm: Positive gradient-norm clipping limit.
            seed: Local exploration/replay RNG and per-episode reset-seed base;
                does not reseed model initialization or global Torch RNG.
            observation_transform: Deterministic preprocessing callable before
                flattening and conversion to model dtype/device.

        Invalid capacities, ranges and action spaces raise ValueError. No training
        starts in the constructor. Time-limit truncation retains Bellman bootstrap;
        only true termination removes it. Caller-owned environments remain open."""
        import copy
        import random
        import gymnasium as gym

        for name, value in {
            "batch_size": batch_size,
            "replay_capacity": replay_capacity,
            "train_frequency": train_frequency,
            "gradient_steps": gradient_steps,
            "target_update_interval": target_update_interval,
            "epsilon_decay_steps": epsilon_decay_steps,
            "max_steps": max_steps,
        }.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(learning_starts, int) or learning_starts < 0:
            raise ValueError("learning_starts must be a non-negative integer.")
        if replay_capacity < batch_size:
            raise ValueError("replay_capacity must be at least batch_size.")
        if not 0 <= gamma <= 1 or not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError(
                "Require gamma in [0,1] and 0 <= epsilon_end <= epsilon_start <= 1."
            )
        if lr <= 0 or max_grad_norm <= 0:
            raise ValueError("lr and max_grad_norm must be positive.")
        self.qfunction = qfunction
        # PennyLane circuit closures can retain original builder references after
        # deepcopy. TorchLayer passes the target's own parameters explicitly.
        self.target_network = copy.deepcopy(qfunction)
        self.target_network.eval()
        self.target_network.requires_grad_(False)
        self.optimizer = optimizer_class(
            qfunction.parameters(), lr=lr, **(optimizer_kwargs or {})
        )
        self._owns_env = isinstance(env_name, str)
        self.env_name = env_name if self._owns_env else None
        self.env = gym.make(env_name) if self._owns_env else env_name
        if (
            not isinstance(self.env.action_space, gym.spaces.Discrete)
            or self.env.action_space.start != 0
        ):
            if self._owns_env:
                self.env.close()
            raise ValueError("DeepQLearning requires zero-based Discrete actions.")
        if (
            getattr(qfunction, "n_actions", self.env.action_space.n)
            != self.env.action_space.n
        ):
            if self._owns_env:
                self.env.close()
            raise ValueError("Q-function n_actions does not match the environment.")
        self.n_actions = self.env.action_space.n
        self.gamma, self.batch_size = gamma, batch_size
        self.learning_starts, self.train_frequency = learning_starts, train_frequency
        self.gradient_steps, self.target_update_interval = (
            gradient_steps,
            target_update_interval,
        )
        self.double_dqn = double_dqn
        self.epsilon_start, self.epsilon_end = epsilon_start, epsilon_end
        self.epsilon_decay_steps, self.max_steps = epsilon_decay_steps, max_steps
        self.max_grad_norm, self.seed = max_grad_norm, seed
        self.observation_transform = observation_transform
        self.replay_buffer = deque(maxlen=replay_capacity)
        self._rng = random.Random(seed)
        self.frame_count = self.update_count = self.episode_count = 0
        self.history = []

    @property
    def epsilon(self):
        """Return exploration probability from the frame-indexed linear schedule, clamped at its final value."""
        fraction = min(1.0, self.frame_count / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def _observation(self, observation):
        """Apply the optional transform and flatten an observation using model parameter dtype/device."""
        if self.observation_transform is not None:
            observation = self.observation_transform(observation)
        parameter = next(self.qfunction.parameters())
        return torch.as_tensor(
            observation, dtype=parameter.dtype, device=parameter.device
        ).flatten()

    def _values(self, network, observation):
        """Evaluate a network and require finite (...,n_actions) output with matching batch axes."""
        values = network(observation)
        if values.shape != (*observation.shape[:-1], self.n_actions):
            raise ValueError("Q-function must return (..., n_actions) values.")
        if not torch.isfinite(values).all():
            raise FloatingPointError("Q-function produced non-finite values.")
        return values

    def _greedy(self, observation):
        """Return one argmax action under no_grad/eval mode and restore the online network's mode."""
        training = self.qfunction.training
        self.qfunction.eval()
        try:
            with torch.no_grad():
                return int(self._values(self.qfunction, observation).argmax(-1).item())
        finally:
            self.qfunction.train(training)

    def select_action(self, observation, *, explore=True):
        """Choose an epsilon-random action if explore=True, otherwise evaluate greedy Q-values; return int."""
        if explore and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_actions)
        return self._greedy(self._observation(observation))

    def sync_target(self):
        """Copy online parameters/buffers to the frozen target and retain target eval mode."""
        self.target_network.load_state_dict(self.qfunction.state_dict())
        self.target_network.eval()

    def _bellman_targets(self, rewards, next_observations, terminated):
        """Compute detached DQN/Double-DQN reward-plus-bootstrap targets, masking only true terminations."""
        with torch.no_grad():
            target_values = self._values(self.target_network, next_observations)
            if self.double_dqn:
                training = self.qfunction.training
                self.qfunction.eval()
                try:
                    actions = self._values(self.qfunction, next_observations).argmax(
                        -1, keepdim=True
                    )
                finally:
                    self.qfunction.train(training)
                continuation = target_values.gather(-1, actions).squeeze(-1)
            else:
                continuation = target_values.max(-1).values
            return rewards + self.gamma * (~terminated).to(rewards.dtype) * continuation

    def optimize(self):
        """Take one uniform replay update with smooth-L1 loss and gradient clipping.

        Returns a Python loss float, or None if replay has fewer than batch_size
        entries. Increments update_count and hard-syncs the target at the configured
        optimizer-step interval. Raises on nonfinite Q-values, loss or gradients."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self._rng.sample(list(self.replay_buffer), self.batch_size)
        parameter = next(self.qfunction.parameters())
        observations = torch.stack([b[0] for b in batch]).to(parameter)
        actions = torch.tensor([b[1] for b in batch], device=parameter.device)
        rewards = torch.tensor(
            [b[2] for b in batch], device=parameter.device, dtype=parameter.dtype
        )
        next_observations = torch.stack([b[3] for b in batch]).to(parameter)
        terminated = torch.tensor(
            [b[4] for b in batch], device=parameter.device, dtype=torch.bool
        )
        targets = self._bellman_targets(rewards, next_observations, terminated)
        self.qfunction.train()
        predicted = (
            self._values(self.qfunction, observations)
            .gather(-1, actions[:, None])
            .squeeze(-1)
        )
        loss = torch.nn.functional.smooth_l1_loss(predicted, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite DQN loss.")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.qfunction.parameters(), self.max_grad_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.sync_target()
        return float(loss.detach())

    def train(self, num_episodes=500, *, log_every=10):
        """Train additional episodes and return their per-episode record dictionaries.

        Each reset uses seed + cumulative episode_count. Records contain episode,
        return, steps, cumulative frames/updates, epsilon and mean update loss
        (None when no update occurred). Also appends them to history. CPU-cloned
        replay observations are retained up to capacity; repeated calls continue
        training and exploration counters. log_every=0 suppresses progress output.
        No early solved stop or automatic disk checkpoint is performed."""
        if not isinstance(num_episodes, int) or num_episodes < 1:
            raise ValueError("num_episodes must be a positive integer.")
        records = []
        for _ in range(num_episodes):
            observation, _ = self.env.reset(seed=self.seed + self.episode_count)
            state = self._observation(observation)
            total, losses = 0.0, []
            for step in range(1, self.max_steps + 1):
                action = (
                    self._rng.randrange(self.n_actions)
                    if self._rng.random() < self.epsilon
                    else self._greedy(state)
                )
                observation, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self._observation(observation)
                self.replay_buffer.append(
                    (
                        state.detach().cpu().clone(),
                        action,
                        float(reward),
                        next_state.detach().cpu().clone(),
                        bool(terminated),
                    )
                )
                state = next_state
                total += float(reward)
                self.frame_count += 1
                if (
                    self.frame_count >= self.learning_starts
                    and self.frame_count % self.train_frequency == 0
                ):
                    for _ in range(self.gradient_steps):
                        loss = self.optimize()
                        if loss is not None:
                            losses.append(loss)
                if terminated or truncated:
                    break
            self.episode_count += 1
            record = {
                "episode": self.episode_count,
                "return": total,
                "steps": step,
                "frames": self.frame_count,
                "updates": self.update_count,
                "epsilon": self.epsilon,
                "loss": sum(losses) / len(losses) if losses else None,
            }
            self.history.append(record)
            records.append(record)
            if log_every and self.episode_count % log_every == 0:
                recent = self.history[-log_every:]
                average = sum(r["return"] for r in recent) / len(recent)
                print(
                    f"Episode {self.episode_count:04d} | return={average:.1f} | epsilon={self.epsilon:.3f}",
                    flush=True,
                )
        return records

    def evaluate(self, num_episodes=10, *, seed=10000, env=None):
        """Return greedy unshaped returns from num_episodes fresh evaluation episodes.

        Resets use seed + episode_index; max_steps caps each episode. Training
        counters, replay and exploration RNG are untouched. env must differ from
        the training environment. If omitted, a new environment is made from the
        training name and closed afterward; instance-based training requires an
        explicit evaluation env, which remains caller-owned."""
        import gymnasium as gym

        if not isinstance(num_episodes, int) or num_episodes < 1:
            raise ValueError("num_episodes must be a positive integer.")
        owned = env is None
        if owned:
            if self.env_name is None:
                raise ValueError(
                    "Supply a separate evaluation env when training with an instance."
                )
            env = gym.make(self.env_name)
        if env is self.env:
            raise ValueError("Evaluation requires a separate environment.")
        returns = []
        try:
            for episode in range(num_episodes):
                observation, _ = env.reset(seed=seed + episode)
                total = 0.0
                for _ in range(self.max_steps):
                    action = self.select_action(observation, explore=False)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    if terminated or truncated:
                        break
                returns.append(total)
        finally:
            if owned:
                env.close()
        return returns

    def close(self):
        """Close only an environment created by this trainer from its name."""
        if self._owns_env:
            self.env.close()

    def __enter__(self):
        """Return this trainer for context-manager use."""
        return self

    def __exit__(self, *_):
        """Close trainer-owned environment resources without suppressing exceptions."""
        self.close()
