"""Small self-contained Gymnasium environments for policy benchmarks."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class WindyTargetNavigationEnv(gym.Env):
    """Two-dimensional inertial navigation with three or four discrete actions.

    The agent controls a point mass using equally spaced thrust directions while
    compensating for persistent, time-varying wind. An episode contains several
    randomly placed waypoints, so a useful policy must react to the target,
    velocity, and wind rather than memorize a fixed action sequence.

    The six-dimensional observation contains normalized target displacement,
    velocity, and wind components. This makes the environment a convenient
    match for six-qubit feature maps. Rewards combine progress toward the
    current waypoint, a small time/velocity penalty, waypoint bonuses, and a
    final completion bonus.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_actions: int = 3,
        max_episode_steps: int = 180,
        n_waypoints: int = 3,
        dt: float = 0.2,
        thrust: float = 1.0,
        drag: float = 0.92,
        max_speed: float = 1.6,
        max_wind: float = 0.35,
        wind_persistence: float = 0.96,
        wind_noise: float = 0.035,
        target_min_distance: float = 2.0,
        target_max_distance: float = 4.0,
        target_tolerance: float = 0.32,
    ):
        """Configure a six-observation, discrete-thrust Gymnasium navigation task.

        n_actions is 3 or 4 evenly spaced thrust directions. Episodes end after
        n_waypoints targets or max_episode_steps. dt integrates acceleration,
        drag damps velocity, and max_speed bounds its norm. Wind follows a noisy
        persistent process bounded by max_wind; wind_persistence and wind_noise
        control its updates. Targets are sampled at distances in the configured
        min/max interval and reached within target_tolerance. No episode is
        started until reset; use positive physical scales for normalization."""
        super().__init__()
        if n_actions not in {3, 4}:
            raise ValueError("WindyTargetNavigationEnv supports 3 or 4 actions.")
        if max_episode_steps < 1 or n_waypoints < 1:
            raise ValueError("max_episode_steps and n_waypoints must be positive.")
        if not 0.0 <= drag <= 1.0:
            raise ValueError("drag must lie in [0, 1].")
        if not 0.0 <= wind_persistence <= 1.0:
            raise ValueError("wind_persistence must lie in [0, 1].")
        if not 0.0 < target_min_distance <= target_max_distance:
            raise ValueError(
                "Require 0 < target_min_distance <= target_max_distance."
            )

        self.n_actions = int(n_actions)
        self.max_episode_steps = int(max_episode_steps)
        self.n_waypoints = int(n_waypoints)
        self.dt = float(dt)
        self.thrust = float(thrust)
        self.drag = float(drag)
        self.max_speed = float(max_speed)
        self.max_wind = float(max_wind)
        self.wind_persistence = float(wind_persistence)
        self.wind_noise = float(wind_noise)
        self.target_min_distance = float(target_min_distance)
        self.target_max_distance = float(target_max_distance)
        self.target_tolerance = float(target_tolerance)

        angles = np.linspace(0.0, 2.0 * math.pi, self.n_actions, endpoint=False)
        self.action_vectors = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        self.action_vectors = self.action_vectors.astype(np.float64)

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self.position = np.zeros(2, dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.wind = np.zeros(2, dtype=np.float64)
        self.target = np.zeros(2, dtype=np.float64)
        self.step_count = 0
        self.waypoints_reached = 0

    @staticmethod
    def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
        """Rescale a vector only if its Euclidean norm exceeds maximum."""
        norm = float(np.linalg.norm(vector))
        if norm > maximum:
            return vector * (maximum / norm)
        return vector

    def _sample_target(self) -> None:
        """Choose a random direction/distance and store a new target relative to current position."""
        angle = self.np_random.uniform(0.0, 2.0 * math.pi)
        distance = self.np_random.uniform(
            self.target_min_distance, self.target_max_distance
        )
        offset = distance * np.array([math.cos(angle), math.sin(angle)])
        self.target = self.position + offset

    def _observation(self) -> np.ndarray:
        """Return float32 [relative target, velocity, wind], each two-dimensional and bounded in [-1, 1]."""
        relative_target = np.tanh(
            (self.target - self.position) / self.target_max_distance
        )
        normalized_velocity = np.clip(
            self.velocity / self.max_speed, -1.0, 1.0
        )
        normalized_wind = np.clip(self.wind / self.max_wind, -1.0, 1.0)
        return np.concatenate(
            (relative_target, normalized_velocity, normalized_wind)
        ).astype(np.float32)

    def _info(self) -> dict:
        """Report target distance, reached-waypoint count and complete-task success."""
        return {
            "distance": float(np.linalg.norm(self.target - self.position)),
            "waypoints_reached": self.waypoints_reached,
            "success": self.waypoints_reached == self.n_waypoints,
        }

    def reset(self, *, seed=None, options=None):
        """Seed the Gymnasium RNG, reset motion/counters, sample wind and target, and return (observation, info)."""
        super().reset(seed=seed)
        del options
        self.position = np.zeros(2, dtype=np.float64)
        self.velocity = self.np_random.normal(0.0, 0.08, size=2)
        self.wind = self.np_random.uniform(
            -0.35 * self.max_wind, 0.35 * self.max_wind, size=2
        )
        self.step_count = 0
        self.waypoints_reached = 0
        self._sample_target()
        return self._observation(), self._info()

    def step(self, action):
        """Apply one thrust action and return the Gymnasium five-tuple.

        Reward is 4 * distance progress - 0.02 - 0.01 * speed squared, plus 3
        per waypoint and 7 on full completion. Termination marks task completion;
        truncation marks the step limit otherwise. Invalid actions raise ValueError.
        State, wind and possibly the next target are updated in place."""
        if not self.action_space.contains(action):
            raise ValueError(
                f"Action {action!r} is invalid for Discrete({self.n_actions})."
            )

        action = int(action)
        self.step_count += 1
        distance_before = float(np.linalg.norm(self.target - self.position))

        acceleration = self.thrust * self.action_vectors[action] + self.wind
        self.velocity = self.drag * self.velocity + self.dt * acceleration
        self.velocity = self._clip_norm(self.velocity, self.max_speed)
        self.position = self.position + self.dt * self.velocity

        self.wind = (
            self.wind_persistence * self.wind
            + self.np_random.normal(0.0, self.wind_noise, size=2)
        )
        self.wind = np.clip(self.wind, -self.max_wind, self.max_wind)

        distance_after = float(np.linalg.norm(self.target - self.position))
        progress = distance_before - distance_after
        reward = (
            4.0 * progress
            - 0.02
            - 0.01 * float(np.dot(self.velocity, self.velocity))
        )

        terminated = False
        if distance_after <= self.target_tolerance:
            self.waypoints_reached += 1
            reward += 3.0
            if self.waypoints_reached == self.n_waypoints:
                reward += 7.0
                terminated = True
            else:
                self._sample_target()

        truncated = self.step_count >= self.max_episode_steps and not terminated
        return self._observation(), float(reward), terminated, truncated, self._info()

