"""Evaluate initial checkpoints without training, separate from selection metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from majority_iqp_model_search import ROOT,make_policy,evaluate

def audit(path,count=100):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    record=saved['record']; name=record['model']['name']; seed=record['seed']
    model=make_policy(name,seed)
    for k,p in model.named_parameters(): p.data.copy_(saved['initial'][k])
    ret=evaluate(model,seed,name in ('mlp','mlp19'),count)
    return {'model':name,'seed':seed,'initial_mean_return':float(ret.mean()),
            'evaluation_episodes':count,'trained_mean_return':record['heldout']}

def main():
    p=argparse.ArgumentParser(); p.add_argument('phase');p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--episodes',type=int,default=100); args=p.parse_args()
    import joblib
    paths=sorted((ROOT/args.phase).glob('*.pt'))
    rows=joblib.Parallel(n_jobs=args.jobs)(joblib.delayed(audit)(p,args.episodes) for p in paths)
    (ROOT/(args.phase+'_initial_policy_audit.json')).write_text(json.dumps(rows,indent=2))
    for name in sorted({r['model'] for r in rows}):
        group=[r for r in rows if r['model']==name]
        print(name,len(group),'initial',sum(r['initial_mean_return'] for r in group)/len(group),
            'trained',sum(r['trained_mean_return'] for r in group)/len(group))

if __name__=='__main__': main()
