"""Shared training helpers for the paper (legacy module name retained for imports)."""
import numpy as np
import torch
import gymnasium as gym
from torch.distributions import Categorical
from qasp.rl.torchrl import REINFORCE
SCALE=torch.tensor([2.4,3.,.2095,3.5])
torch.set_num_threads(1)

class MLP(torch.nn.Module):
    def __init__(self,seed,hidden=9):
        super().__init__(); torch.manual_seed(seed)
        self.network=torch.nn.Sequential(torch.nn.Linear(4,hidden),torch.nn.Tanh(),torch.nn.Linear(hidden,2))
    def forward(self,x): return self.network(x)

class BatchedREINFORCE(REINFORCE):
    """Same QASP algorithm, with trajectory log-probs recomputed in one batch.

    Parameters do not change within a trajectory. This preserves the loss and
    derivatives, while avoiding one autograd graph per environment step.
    """
    def _run_episode(self):
        state,_=self.env.reset(); states=[]; actions=[]; rewards=[]
        with torch.no_grad():
            for _ in range(self.max_steps):
                x=self._preprocess_state(torch.as_tensor(state,dtype=torch.float32))
                out=self.policy(x)
                dist=Categorical(logits=out) if self.logits else Categorical(probs=out)
                action=dist.sample(); states.append(x); actions.append(action)
                state,reward,done,truncated,_=self.env.step(action.item())
                rewards.append(float(reward))
                if done or truncated: break
        out=self.policy(torch.stack(states))
        dist=Categorical(logits=out) if self.logits else Categorical(probs=out)
        return states,dist.log_prob(torch.stack(actions)),rewards,dist.entropy(),sum(rewards)

class StrictWindowREINFORCE(BatchedREINFORCE):
    """QASP's CartPole/EMA objective, with a full-window freezing safeguard.

    This experiment-specific implementation supports the EMA baseline used
    throughout the comparison. Optimizer, rollout, loss and batch semantics
    are unchanged. Only the prerequisite of 100 completed episodes is new.
    """
    def train(self,num_episodes=500,log_every=50):
        assert self.baseline_type=='ema' and self._is_cartpole()
        history=[]; logps=[]; advantages=[]; entropies=[]
        self.freeze_episode=None; self.optimizer_updates=0

        def update():
            logp=torch.cat(logps); adv=torch.cat(advantages); entropy=torch.cat(entropies)
            if self.normalize_returns and len(adv)>1:
                adv=(adv-adv.mean())/(adv.std()+1e-8)
            loss=-(adv.detach()*logp).mean()
            if self.entropy_coef>0: loss=loss-self.entropy_coef*entropy.mean()
            if self.freeze_episode is None:
                if len(history)>=100 and np.mean(history[-100:])>=195:
                    self.freeze_episode=len(history)
                    print(f'Full rolling-100 criterion reached at episode {len(history)}; freezing updates.',flush=True)
                else:
                    self._optimise_loss(loss); self.optimizer_updates+=1
            logps.clear(); advantages.clear(); entropies.clear()

        for episode in range(1,num_episodes+1):
            states,logp,rewards,entropy,total=self._run_episode()
            history.append(total); disc=self._discount(rewards)
            adv=disc-self.baseline_value
            self._update_baseline(disc.mean().item())
            logps.append(logp); advantages.append(adv); entropies.append(entropy)
            if len(logps)==self.batch_episodes: update()
            if episode%log_every==0:
                print(f'Episode {episode:04d} | avg(return_last20) = {np.mean(history[-20:]):.1f} | baseline = ema={self.baseline_value:.1f}',flush=True)
        if logps: update()
        return history

def make_agent(policy,logits,strict=False):
    trainer=StrictWindowREINFORCE if strict else BatchedREINFORCE
    return trainer(policy=policy,env_name='CartPole-v1',gamma=.99,lr=.03,
        max_steps=200,batch_episodes=5,logits=logits,normalize_mode='clip',
        normalize_returns=True,entropy_coef=.005,baseline_type='ema',
        baseline_momentum=.05,max_grad_norm=1.)

def evaluate(policy,seed,logits,count):
    env=gym.make('CartPole-v1'); returns=[]
    with torch.no_grad():
        for ep in range(count):
            state,_=env.reset(seed=1_000_000+10_000*seed+ep)
            gen=torch.Generator().manual_seed(2_000_000+10_000*seed+ep)
            total=0
            for _ in range(200):
                x=(torch.as_tensor(state,dtype=torch.float32)/SCALE).clamp(-3,3)
                out=policy(x); p=torch.softmax(out,-1) if logits else out
                action=torch.multinomial(p,1,generator=gen).item()
                state,r,done,truncated,_=env.step(action); total+=r
                if done or truncated: break
            returns.append(total)
    env.close(); return np.asarray(returns)
