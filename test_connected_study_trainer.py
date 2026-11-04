"""Check actual optimizer-batch equivalence for the new connected policy."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/iqp-search-mpl')
import copy
import contextlib
import io
import json
import types
import numpy as np
import torch
from qasp.rl.torchrl import REINFORCE
from connected_iqp_mlp_study import ROOT,specs,policy
from majority_iqp_model_search import make_agent

rows=[]
for name in ['iqp_connected','mlp_h19_lr0p1']:
    spec=specs()[name];initial=policy(spec,12347);records=[]
    for native in [True,False]:
        model=copy.deepcopy(initial);agent=make_agent(model,spec['kind']=='mlp',strict=True)
        for group in agent.opt.param_groups:group['lr']=spec['lr']
        if native:agent._run_episode=types.MethodType(REINFORCE._run_episode,agent)
        torch.manual_seed(12347);np.random.seed(12347)
        agent.env.reset(seed=12347);agent.env.action_space.seed(12347)
        with contextlib.redirect_stdout(io.StringIO()): r=agent.train(5,5)
        records.append((r,torch.cat([p.detach().flatten() for p in model.parameters()])))
        agent.env.close()
    assert records[0][0]==records[1][0]
    error=float((records[0][1]-records[1][1]).abs().max());assert error<1e-5
    rows.append({'model':name,'same_trajectory':True,'max_parameter_error':error})
(ROOT/'training_equivalence.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
