"""Regression checks for the user-approved full-window freezing correction."""
import contextlib
import copy
import io
import json
import types
import numpy as np
import torch
from majority_iqp_model_search import make_agent,make_policy,ROOT

def synthetic_case(early_only):
    policy=torch.nn.Linear(1,2); agent=make_agent(policy,True,strict=True)
    agent.synthetic_episode=0
    def episode(self):
        self.synthetic_episode+=1
        total=200. if not early_only or self.synthetic_episode<=5 else 100.
        dist=torch.distributions.Categorical(logits=self.policy(torch.ones(1)))
        return [],dist.log_prob(torch.tensor(0)).reshape(1),[total],dist.entropy().reshape(1),total
    agent._run_episode=types.MethodType(episode,agent)
    with contextlib.redirect_stdout(io.StringIO()): agent.train(110,110)
    record={'early_good_episodes_only':early_only,'freeze_episode':agent.freeze_episode,
            'optimizer_updates':agent.optimizer_updates}
    agent.env.close()
    if early_only: assert agent.freeze_episode is None and agent.optimizer_updates==22
    else: assert agent.freeze_episode==100 and agent.optimizer_updates==19
    return record

def main():
    records=[synthetic_case(False),synthetic_case(True)]
    for name in ['mlp','iqpblocksbalanced_21_0.15']:
        seed=12347; original=make_policy(name,seed); params=[]; trajectories=[]
        for strict in [False,True]:
            p=copy.deepcopy(original); agent=make_agent(p,name=='mlp',strict=strict)
            torch.manual_seed(seed); np.random.seed(seed)
            agent.env.reset(seed=seed); agent.env.action_space.seed(seed)
            with contextlib.redirect_stdout(io.StringIO()): r=agent.train(5,5)
            trajectories.append(r); params.append(torch.cat([v.detach().flatten() for v in p.parameters()]))
            agent.env.close()
        err=float((params[0]-params[1]).abs().max())
        assert trajectories[0]==trajectories[1] and err==0
        records.append({'model':name,'unchanged_pre_freeze_trajectory':True,'parameter_difference':err})
    (ROOT/'strict_window_regression_checks.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2))

if __name__=='__main__': main()
