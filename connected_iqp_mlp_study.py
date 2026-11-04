"""Prospective connected-IQP and learning-rate-tuned MLP size study."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/iqp-search-mpl')
import argparse
import contextlib
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from majority_iqp_model_search import MLP,make_agent,evaluate
from majority_iqp_block_model import BlockMajorityIQP

ROOT=Path(__file__).resolve().parent/'results'/'connected_iqp_mlp_study'
WIDTHS=[1,2,4,6,9,12,16,19]
LRS=[.003,.01,.03,.1]
DEV=list(range(1001,1009))
TEST=list(range(2001,2021))
torch.set_num_threads(1)


def specs():
    result={}
    for width in WIDTHS:
        for lr in LRS:
            name=f'mlp_h{width:02d}_lr{str(lr).replace(".","p")}'
            result[name]={'name':name,'kind':'mlp','width':width,'lr':lr,'parameters':7*width+2}
    for kind in ['connected','disconnected']:
        name=f'iqp_{kind}'
        result[name]={'name':name,'kind':kind,'n':21,'lr':.03,'parameters':139 if kind=='connected' else 133}
    return result


def policy(spec,seed):
    if spec['kind']=='mlp': return MLP(seed,spec['width'])
    if spec['kind']=='connected':
        from connected_majority_iqp import ConnectedMajorityIQP
        return ConnectedMajorityIQP(seed=seed,n=21,sigma=.5,bridge_sigma=.5)
    return BlockMajorityIQP({'n':21,'sigma':.5,'family':'iqpblocksbalanced'},seed)


def solve(tr):
    s=np.r_[0,np.cumsum(tr)]; hits=np.flatnonzero(s[100:]-s[:-100]>=19500)
    return int(hits[0]+100) if len(hits) else 501


def run(name,seed,phase):
    spec=specs()[name]; folder=ROOT/phase; folder.mkdir(parents=True,exist_ok=True)
    stem=folder/f'{name}__{seed}'; target=stem.with_suffix('.json')
    if target.exists(): return json.loads(target.read_text())
    started=time.time(); model=policy(spec,seed)
    initial={k:v.detach().clone() for k,v in model.named_parameters()}
    agent=make_agent(model,spec['kind']=='mlp',strict=True)
    for group in agent.opt.param_groups: group['lr']=spec['lr']
    agent.lr=spec['lr']
    torch.manual_seed(seed); np.random.seed(seed)
    agent.env.reset(seed=seed); agent.env.action_space.seed(seed)
    with stem.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log):
        returns=np.array(agent.train(500,50))
    heldout=evaluate(model,seed,spec['kind']=='mlp',100 if phase=='test' else 20)
    agent.env.close()
    record={'spec':spec,'seed':seed,'phase':phase,'auc':float(returns.mean()/200),
            'solve_episode':solve(returns),'final50':float(returns[-50:].mean()),
            'heldout':float(heldout.mean()),'freeze_episode':agent.freeze_episode,
            'updates':agent.optimizer_updates,'seconds':time.time()-started,
            'parameters':sum(p.numel() for p in model.parameters()),
            'initial_hash':hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()}
    assert record['parameters']==spec['parameters']
    if spec['kind']=='connected':
        record['initial_bridges']=initial['bridges'].tolist()
        record['final_bridges']=model.bridges.detach().tolist()
        record['initial_cut_purities']=(1-.5*torch.sin(initial['bridges']).square()).tolist()
        record['final_cut_purities']=(1-.5*torch.sin(model.bridges.detach()).square()).tolist()
    np.savez_compressed(stem.with_suffix('.npz'),training_returns=returns,evaluation_returns=heldout)
    torch.save({'initial':initial,'final':model.state_dict(),'record':record},stem.with_suffix('.pt'))
    target.write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)
    return record


def lock():
    target=ROOT/'development_selection.json'
    if target.exists():
        print(target.read_text()); return
    candidates=specs(); means={}
    for name,spec in candidates.items():
        if spec['kind']!='mlp': continue
        rows=[json.loads((ROOT/'development'/f'{name}__{s}.json').read_text()) for s in DEV]
        means[name]={'auc':float(np.mean([r['auc'] for r in rows])),
                     'solve':float(np.mean([r['solve_episode'] for r in rows]))}
    by_width={}
    for width in WIDTHS:
        names=[k for k,v in candidates.items() if v.get('width')==width]
        by_width[str(width)]=min(names,key=lambda k:(-means[k]['auc'],means[k]['solve'],candidates[k]['lr']))
    best_auc=min(by_width.values(),key=lambda k:(-means[k]['auc'],means[k]['solve'],candidates[k]['width']))
    best_solve=min(means,key=lambda k:(means[k]['solve'],-means[k]['auc'],candidates[k]['width'],candidates[k]['lr']))
    data={'selected_by_width':by_width,'best_auc_mlp':best_auc,'best_solve_mlp':best_solve,
          'development_means':means,'test_seeds':TEST,
          'selection_rule':'Within each width choose highest mean development AUC; ties use smaller solve episode, then lower learning rate. Also designate the fastest-convergence configuration across the full development grid, tie-break by higher AUC then smaller width and lower learning rate.'}
    target.write_text(json.dumps(data,indent=2)); print(json.dumps(data,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','lock','status'])
    p.add_argument('--phase',choices=['development','test'],default='development')
    p.add_argument('--kind',choices=['mlp','iqp','all'],default='all')
    p.add_argument('--jobs',type=int,default=4);p.add_argument('--models',nargs='+')
    a=p.parse_args()
    if a.mode=='lock': lock();return
    if a.mode=='status':
        rows=[json.loads(f.read_text()) for f in (ROOT/a.phase).glob('*.json')]
        for name in sorted({r['spec']['name'] for r in rows}):
            group=[r for r in rows if r['spec']['name']==name]
            print(name,len(group),'AUC',round(np.mean([r['auc'] for r in group]),4),
                  'solve',round(np.mean([r['solve_episode'] for r in group]),1))
        return
    if a.phase=='test':
        locked=json.loads((ROOT/'development_selection.json').read_text())
        names=list(dict.fromkeys(list(locked['selected_by_width'].values())+[locked['best_solve_mlp'],'iqp_connected','iqp_disconnected']))
    else: names=list(specs())
    if a.kind=='mlp': names=[n for n in names if n.startswith('mlp')]
    if a.kind=='iqp': names=[n for n in names if n.startswith('iqp')]
    if a.models:
        assert set(a.models)<=set(names)
        names=a.models
    seeds=TEST if a.phase=='test' else DEV
    # Interleave models across seeds; each task still has explicit reset seeds.
    import joblib
    joblib.Parallel(n_jobs=a.jobs)(joblib.delayed(run)(name,seed,a.phase) for seed in seeds for name in names)


if __name__=='__main__': main()
