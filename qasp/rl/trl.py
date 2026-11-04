"""Historical TorchRL PPO prototype; use qasp.rl.torchrl for maintained trainers.

This module uses unqualified legacy imports and a fixed four-feature critic.
Its train loop lacks a total-frame stopping condition and references tqdm
without importing it. It is retained for provenance, not as a working public
training recipe."""
import torch
import torch.nn as nn
from torch.optim import Adam
from torchrl.envs import TransformedEnv, GymEnv
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from collections import defaultdict

class PQCWrapper(nn.Module):
    """Legacy dictionary-output wrapper around a configured quantum policy."""
    def __init__(self, policy_class, policy_config_kwargs):
        """Resolve legacy component imports by mutating the supplied config, then construct the policy."""
        super().__init__()
        if policy_class == "PQCBorn":
            from rl.policies import PQCBorn
            policy = PQCBorn
        elif policy_class == "PQCSoftmax":
            from rl.policies import PQCSoftmax
            policy = PQCSoftmax

        if policy_config_kwargs["ansatz"] == "IQPAnsatz":
            from circuits.templates import IQPAnsatz
            policy_config_kwargs["ansatz"] = IQPAnsatz
        elif policy_config_kwargs["ansatz"] == "AngleAnsatz":
            from circuits.templates import AngleAnsatz
            policy_config_kwargs["ansatz"] = AngleAnsatz

        if policy_config_kwargs["feature_map"] == "AngleEmbedding":
            from feature_maps.templates import AngleEmbedding
            policy_config_kwargs["feature_map"] = AngleEmbedding

        self.pqc_model = policy(**policy_config_kwargs)

    def forward(self, obs):
        """Return {"probs": policy.action_probs(obs)} for the legacy TensorDict adapter."""
        action_probs = self.pqc_model.action_probs(obs)
        return {"probs": action_probs}

class TorchRLTrainer:
    """Experimental PPO collector/critic scaffold with hard-coded four-feature input."""
    def __init__(self, env_name, policy_class, policy_config_kwargs , lr):
        """Create actor, four-input critic, normalized environment, 128-frame collector and PPO loss."""
        self.actor_module = TensorDictModule(
            module=PQCWrapper(policy_class=policy_class , policy_config_kwargs=policy_config_kwargs),
            in_keys=["observation"],
            out_keys=["probs"]
        )
        self.actor = ProbabilisticActor(
            module=self.actor_module,
            in_keys=["probs"],
            out_keys=["action"],
            distribution_class=Categorical,
            return_log_prob=True
        )

        value_net = nn.Sequential(
            nn.Linear(4,1),
        )
        self.value_module = ValueOperator(
            module=value_net,
            in_keys=["observation"],
        )

        self.base_env = GymEnv(env_name)
        self.env = TransformedEnv(
            self.base_env,
            Compose(
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(),
                StepCounter(),
            ),
        )

        self.env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
        self.collector = SyncDataCollector(self.env, self.actor, frames_per_batch=128)
        self.loss_module = ClipPPOLoss(actor_network=self.actor, critic_network=self.value_module)
        self.optimizer = Adam(self.loss_module.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)

        self.replay_buffer = ReplayBuffer()
        self.advantage_module = GAE(
            gamma=0.99, lmbda=0.95,
            value_network=self.value_module,
            average_gae=True
        )

    def train(self, total_frames=10_000, num_epochs=5, frames_per_batch=128, sub_batch_size=32, max_grad_norm=1.0):
        """Run the legacy PPO collector loop with minibatch updates and logging.

        Returns None. total_frames sizes the progress display but does not stop
        the collector; frames_per_batch affects updates, not collector creation.
        The missing tqdm import and legacy setup must be repaired before use.
        Use the separate PPO class in qasp.rl.torchrl for new experiments."""
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        logs = defaultdict(list)
        pbar = tqdm(total=total_frames)
        eval_str = ""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i, tensordict_data in enumerate(self.collector):
            for _ in range(num_epochs):
                self.advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                self.replay_buffer.extend(data_view.cpu())

                for _ in range(frames_per_batch // sub_batch_size):
                    subdata = self.replay_buffer.sample(sub_batch_size)
                    loss_vals = self.loss_module(subdata.to(device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            pbar.update(tensordict_data.numel())
            cum_reward_str = (
                f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
            )
            logs["step_count"].append(tensordict_data["step_count"].max().item())
            stepcount_str = f"step count (max): {logs['step_count'][-1]}"
            logs["lr"].append(self.optimizer.param_groups[0]["lr"])
            lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

            if i % 10 == 0:
                with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                    eval_rollout = self.env.rollout(1000, self.actor,auto_reset=True)
                    logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
                    logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                    eval_str = (
                        f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                        f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                        f"eval step-count: {logs['eval step_count'][-1]}"
                    )
                    del eval_rollout

            pbar.set_description(
                ", ".join([eval_str, cum_reward_str, stepcount_str, lr_str])
            )
            self.scheduler.step()
