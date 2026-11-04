"""Compare one complete optimizer batch with the unmodified QASP trainer."""
import copy
import contextlib
import io
import json
import types
import numpy as np
import torch
from qasp.rl.torchrl import REINFORCE
from majority_iqp_model_search import make_policy,make_agent,ROOT

def main():
    records=[]
    for name in ['iqp2body_7_0.15','mlp','iqpblocksbalanced_21_0.15']:
        seed=12347
        original=make_policy(name,seed)
        final=[]; returns=[]
        for native in [True,False]:
            policy=copy.deepcopy(original); agent=make_agent(policy,name=='mlp')
            if native: agent._run_episode=types.MethodType(REINFORCE._run_episode,agent)
            torch.manual_seed(seed); np.random.seed(seed)
            agent.env.reset(seed=seed); agent.env.action_space.seed(seed)
            with contextlib.redirect_stdout(io.StringIO()):
                ret=agent.train(num_episodes=5,log_every=5)
            final.append(torch.cat([p.detach().flatten() for p in policy.parameters()]))
            returns.append(ret); agent.env.close()
        err=float((final[0]-final[1]).abs().max())
        assert returns[0]==returns[1],(name,returns)
        assert err<1e-5,(name,err)
        records.append({'model':name,'native_and_batched_returns':returns,
                        'max_parameter_difference_after_update':err})
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'training_equivalence_check.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2))

if __name__=='__main__': main()
