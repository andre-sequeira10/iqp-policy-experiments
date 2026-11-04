"""Read-only pilot for a physics-aligned IQP initialization."""

import math
from pathlib import Path

import gymnasium as gym
import nbformat
import numpy as np
import torch


NOTEBOOK = Path(
    "./tests/"
    "cartpole_10q_graph_local_rzzz_fast_convergence.ipynb"
)
namespace = {}
notebook = nbformat.read(NOTEBOOK, 4)
code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
for source in code_cells[:4]:
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)

Model = namespace["ExactTorchCubicMajority"]
triples = namespace["SELECTED_TRIPLES"]
scale = torch.tensor([2.4, 3.0, 0.2095, 3.5], dtype=torch.float32)


def initialized_policy(angle_weight, angular_weight, seed=1):
    policy = Model(triples, seed)
    with torch.no_grad():
        weights = torch.zeros(10, 4)
        weights[:, 2] = angle_weight
        weights[:, 3] = angular_weight
        policy.input_scaling_params.copy_(weights.reshape(-1))
        policy.params.zero_()
        policy.params[:10].fill_(math.pi / 2)
        policy.rzzz_params.zero_()
    return policy


def evaluate(policy, pilot_seeds, episodes=20):
    returns = []
    env = gym.make("CartPole-v1")
    with torch.no_grad():
        for pilot_seed in pilot_seeds:
            for episode in range(episodes):
                state, _ = env.reset(seed=100_000 * pilot_seed + episode)
                generator = torch.Generator().manual_seed(200_000 * pilot_seed + episode)
                total = 0.0
                for _ in range(200):
                    x = torch.clamp(torch.as_tensor(state, dtype=torch.float32) / scale,
                                    -3.0, 3.0)
                    action = int(torch.multinomial(policy(x), 1, generator=generator).item())
                    state, reward, terminated, truncated, _ = env.step(action)
                    total += reward
                    if terminated or truncated:
                        break
                returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns, ddof=1))


pilot_seeds = (11, 13, 17)
for angle_weight in (0.4, 0.8, 1.2, 1.6):
    for angular_weight in (0.1, 0.2, 0.4, 0.8):
        policy = initialized_policy(angle_weight, angular_weight)
        mean, std = evaluate(policy, pilot_seeds)
        print(angle_weight, angular_weight, mean, std, flush=True)
