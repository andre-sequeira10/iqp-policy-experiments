"""Same training protocol; omit discarded gradient work after updates freeze."""
import torch
from majority_iqp_model_search import StrictWindowREINFORCE


class OptimizedStrictWindowREINFORCE(StrictWindowREINFORCE):
    def _run_episode(self):
        if getattr(self, 'freeze_episode', None) is None:
            return super()._run_episode()
        # The parent still updates its baseline, logging, and episode counter.
        # Only the now-unused batched policy evaluation is omitted. Rollout
        # operations and random draws match BatchedREINFORCE exactly.
        from torch.distributions import Categorical
        state, _ = self.env.reset()
        states, actions, rewards = [], [], []
        with torch.no_grad():
            for _ in range(self.max_steps):
                x = self._preprocess_state(torch.as_tensor(state, dtype=torch.float32))
                out = self.policy(x)
                dist = Categorical(logits=out) if self.logits else Categorical(probs=out)
                action = dist.sample()
                states.append(x)
                actions.append(action)
                state, reward, done, truncated, _ = self.env.step(action.item())
                rewards.append(float(reward))
                if done or truncated:
                    break
        unused = states[0].new_zeros(len(states))
        return states, unused, rewards, unused, sum(rewards)
