"""Audit every locked corrected-protocol run against its saved raw returns."""
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import numpy as np
import torch
from majority_iqp_model_search import ROOT, make_policy


def main():
    lock=json.loads((ROOT/'locked_model_round2.json').read_text())
    folder=ROOT/lock['phase']; checked=[]
    for name in [lock['model'],'mlp']:
        for seed in lock['validation_seeds']:
            stem=folder/f'{name.replace(".","p")}__{seed}'
            r=json.loads(stem.with_suffix('.json').read_text())
            arrays=np.load(stem.with_suffix('.npz'))
            train=arrays['training_returns']; test=arrays['evaluation_returns']
            assert train.shape==(500,) and test.shape==(100,)
            assert np.all(np.isfinite(train)) and np.all((train>=1)&(train<=200))
            assert np.all(np.isfinite(test)) and np.all((test>=1)&(test<=200))
            assert abs(r['auc']-train.mean()/200)<1e-12
            assert abs(r['heldout']-test.mean())<1e-12
            assert abs(r['final50']-train[-50:].mean())<1e-12
            cumulative=np.r_[0.,np.cumsum(train)]
            hits=np.flatnonzero(cumulative[100:]-cumulative[:-100]>=19500)
            exact_solve=int(hits[0]+100) if len(hits) else 501
            assert exact_solve==r['restricted_solve_episode']
            freeze=r['optimizer_freeze_episode']
            qualifying=[k for k in range(100,501,5) if train[k-100:k].mean()>=195]
            assert freeze==(qualifying[0] if qualifying else None)
            assert r['optimizer_updates']==(freeze//5-1 if freeze else 100)
            assert r['freeze_protocol']=='full_100_episodes'
            saved=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=False)
            initial=make_policy(name,seed)
            assert r['parameters']==sum(p.numel() for p in initial.parameters())
            for key,value in initial.named_parameters():
                assert torch.equal(value,saved['initial'][key])
            assert r['initial_sha256']==hashlib.sha256(b''.join(
                v.numpy().tobytes() for v in saved['initial'].values())).hexdigest()
            if name!='mlp':
                assert torch.allclose(saved['initial']['bias'],torch.full((21,),np.pi/2))
                weights=saved['initial']['weights']
                assert weights.shape==(21,4) and torch.all((weights>=0)&(weights<2*np.pi))
            checked.append({'model':name,'seed':seed,'freeze_episode':freeze,'passed':True})
    versions={}
    for package in ['numpy','torch','pennylane','gymnasium','qasp','joblib','nbformat','nbclient']:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]='source checkout'
    sources=list(ROOT.parent.parent.glob('*majority*py'))+[ROOT/'locked_model_round2.json']
    qasp_source=Path('./qasp/rl/torchrl.py')
    sources.append(qasp_source)
    files=sources+sorted(folder.glob('*'))
    manifest={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
              'versions':versions,'runs_checked':len(checked),'checks':checked,
              'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}}
    (ROOT/'validation_round2_strict_integrity.json').write_text(json.dumps(manifest,indent=2))
    print(f'PASS: all {len(checked)} runs, raw metrics, full-window freezes, checkpoints, initializations.')


if __name__=='__main__': main()
