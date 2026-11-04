"""Fixed-protocol uniform-angle follow-up; original study remains read-only."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/iqp-search-mpl')
import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import time
import types
from pathlib import Path
import numpy as np
import torch
from connected_majority_iqp import ConnectedMajorityIQP
from connected_iqp_mlp_study import ROOT as BASE, TEST, solve
from paper_training import make_agent, evaluate
from paper_statistics import infer

ROOT = BASE.parent / 'connected_iqp_uniform_initialization'
NAME = 'iqp_connected_uniform'
torch.set_num_threads(1)


def model(seed, n=21):
    m = ConnectedMajorityIQP(seed=seed, n=n, sigma=.5, bridge_sigma=.5)
    with torch.no_grad():
        for p, offset in [(m.bias,1900000), (m.interactions,2000000), (m.bridges,2100000)]:
            p.uniform_(-math.pi, math.pi, generator=torch.Generator().manual_seed(seed+offset))
    return m


def protocol():
    ROOT.mkdir(exist_ok=True)
    target = ROOT / 'protocol.json'
    p = {
        'model': NAME, 'parameters':139, 'qubits':21,
        'changed': 'All 21 trainable biases, 28 intra-triple interaction angles and 6 bridge angles initialized independently Uniform[-pi,pi].',
        'unchanged': 'Exact connected circuit, global majority, common neural weights bitwise identical per seed, preprocessing, REINFORCE objective and optimizer, learning rate .03, 500 episodes with 200-step limit, five-episode batches, corrected full-100-window freezing rule, and evaluation seed mappings.',
        'neural_weights': 'Original Uniform[0,2pi) weights, unchanged, not Uniform[-pi,pi].',
        'initialization_stream_offsets': {'bias':1900000,'interactions':2000000,'bridges':2100000},
        'seeds': TEST, 'evaluation_episodes':100, 'initial_audit_episodes':100,
        'primary_metrics': 'Normalized training AUC=sum(R)/(500*200); first full rolling-100 mean >=195, assigned 501 if unreached.',
        'comparisons': 'Uniform IQP versus original connected IQP and all nine previously locked MLP configurations, pairing seed IDs. Reuse unchanged stored baseline trajectories, no reruns or reselection.',
        'inference': 'Individual paired-bootstrap 95% CIs (50000 resamples), exact two-sided sign-randomization p-values, single Holm correction across 20 follow-up primary comparisons (10 models times 2 metrics). Positive differences favor uniform IQP; solve differences count episodes saved.',
        'scope': 'Requested follow-up on a previously inspected cohort, not a new untouched confirmatory cohort. No tuning, seed selection, or manuscript changes. Preserve all outcomes.'}
    if target.exists(): assert json.loads(target.read_text()) == p
    else: target.write_text(json.dumps(p,indent=2))


def verify():
    protocol()
    checks = []
    for seed in TEST:
        m = model(seed)
        old = torch.load(BASE/'test'/f'iqp_connected__{seed}.pt',map_location='cpu',weights_only=False)
        assert torch.equal(m.weights, old['initial']['weights'])
        assert sum(p.numel() for p in m.parameters()) == 139
        for p in [m.bias,m.interactions,m.bridges]:
            assert torch.all((p>=-math.pi)&(p<=math.pi))
        checks.append({'seed':seed,'neural_weights_identical':True,'parameters':139})
    import pennylane as qml
    m = model(12347,n=9); x = torch.tensor([.2,-.3,.4,-.5])
    @qml.qnode(qml.device('default.qubit',wires=9))
    def reference():
        for i in range(9): qml.Hadamard(i)
        angles=(x@m.weights.T+m.bias).detach().numpy()
        for i,a in enumerate(angles): qml.RZ(float(a),i)
        for a,s in zip(m.interactions.detach().flatten().numpy(),m.supports): qml.PauliRot(float(a),'Z'*len(s),list(s))
        for a,s in zip(m.bridges.detach().numpy(),m.bridge_supports): qml.PauliRot(float(a),'ZZ',list(s))
        for i in range(9): qml.Hadamard(i)
        return qml.probs(wires=range(9))
    full=np.asarray(reference()); weights=np.array([i.bit_count() for i in range(512)])
    expected=np.bincount(weights,weights=full,minlength=10)
    error=float(abs(expected-m.weight_probs(x).detach().numpy()).max()); assert error<3e-6
    from qasp.rl.torchrl import REINFORCE
    records=[]
    for native in [True,False]:
        p=model(12347); agent=make_agent(p,False,strict=True)
        if native: agent._run_episode=types.MethodType(REINFORCE._run_episode,agent)
        torch.manual_seed(12347);np.random.seed(12347)
        agent.env.reset(seed=12347);agent.env.action_space.seed(12347)
        with contextlib.redirect_stdout(io.StringIO()): returns=agent.train(5,5)
        records.append((returns,torch.cat([v.detach().flatten() for v in p.parameters()])))
        agent.env.close()
    assert records[0][0]==records[1][0]
    training_error=float((records[0][1]-records[1][1]).abs().max()); assert training_error<1e-5
    result={'initialization_checks':checks,'n9_max_probability_error':error,
            'same_native_training_trajectories':True,'max_training_parameter_error':training_error}
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


def run(seed):
    folder=ROOT/'test';folder.mkdir(exist_ok=True)
    stem=folder/f'{NAME}__{seed}'
    if stem.with_suffix('.json').exists(): return json.loads(stem.with_suffix('.json').read_text())
    started=time.time();m=model(seed)
    initial={k:p.detach().clone() for k,p in m.named_parameters()}
    initial_returns=evaluate(m,seed,False,100)
    agent=make_agent(m,False,strict=True)
    torch.manual_seed(seed);np.random.seed(seed)
    agent.env.reset(seed=seed);agent.env.action_space.seed(seed)
    with stem.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log):
        returns=np.asarray(agent.train(500,50))
    heldout=evaluate(m,seed,False,100);agent.env.close()
    r={'name':NAME,'seed':seed,'parameters':139,'lr':.03,'auc':float(returns.mean()/200),
       'solve_episode':solve(returns),'final50':float(returns[-50:].mean()),'heldout':float(heldout.mean()),
       'initial_return':float(initial_returns.mean()),'freeze_episode':agent.freeze_episode,
       'updates':agent.optimizer_updates,'seconds':time.time()-started,
       'initial_hash':hashlib.sha256(b''.join(p.numpy().tobytes() for p in initial.values())).hexdigest()}
    np.savez_compressed(stem.with_suffix('.npz'),training_returns=returns,evaluation_returns=heldout,initial_returns=initial_returns)
    torch.save({'initial':initial,'final':m.state_dict(),'record':r},stem.with_suffix('.pt'))
    stem.with_suffix('.json').write_text(json.dumps(r,indent=2))
    print(json.dumps(r),flush=True);return r


def analyze():
    locked=json.loads((BASE/'development_selection.json').read_text())
    comparisons=list(dict.fromkeys(['iqp_connected']+list(locked['selected_by_width'].values())+[locked['best_solve_mlp']]))
    groups={};curves={};summary={}
    for name in [NAME]+comparisons:
        folder=ROOT/'test' if name==NAME else BASE/'test'
        rows=[];traces=[];initial=[]
        for seed in TEST:
            stem=folder/f'{name}__{seed}';r=json.loads(stem.with_suffix('.json').read_text())
            a=np.load(stem.with_suffix('.npz'));tr=a['training_returns'];ev=a['evaluation_returns']
            assert tr.shape==(500,) and ev.shape==(100,) and np.isfinite(tr).all()
            assert np.all((tr>=1)&(tr<=200)) and np.all((ev>=1)&(ev<=200))
            assert abs(r['auc']-tr.mean()/200)<1e-12 and r['solve_episode']==solve(tr)
            assert abs(r['heldout']-ev.mean())<1e-12
            if name==NAME:
                saved=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=False)
                assert all(torch.equal(p,saved['initial'][k]) for k,p in model(seed).named_parameters())
                qualifying=[e for e in range(100,501,5) if tr[e-100:e].mean()>=195]
                assert r['freeze_episode']==(qualifying[0] if qualifying else None)
                assert a['initial_returns'].shape==(100,)
                assert abs(r['initial_return']-a['initial_returns'].mean())<1e-12
                initial.append(float(a['initial_returns'].mean()))
            else: initial.append(json.loads((BASE/'initial_audit'/f'{name}__{seed}.json').read_text())['initial_return'])
            rows.append(r);traces.append(tr)
        groups[name]=rows;curves[name]=np.stack(traces)
        summary[name]={'parameters':rows[0]['parameters'],'solved':sum(r['solve_episode']<=500 for r in rows),
                       'initial_return':float(np.mean(initial))}
        for key in ['auc','solve_episode','heldout','final50']:
            v=np.array([r[key] for r in rows]);rng=np.random.default_rng(871)
            ci=np.quantile(rng.choice(v,size=(50000,len(v)),replace=True).mean(1),[.025,.975])
            summary[name][key]={'mean':float(v.mean()),'sem':float(v.std(ddof=1)/np.sqrt(len(v))),'ci95':ci.tolist()}
    paired={}
    for name in comparisons:
        paired[name]={}
        for key,sign in [('auc',1),('solve_episode',-1)]:
            paired[name][key]=infer([sign*(a[key]-b[key]) for a,b in zip(groups[NAME],groups[name])],20261027)
    tests=sorted([v[k] for v in paired.values() for k in ['auc','solve_episode']],key=lambda r:r['two_sided_exact_sign_p'])
    previous=0
    for i,r in enumerate(tests):
        previous=max(previous,min(1.,(len(tests)-i)*r['two_sided_exact_sign_p']));r['holm_p']=previous
    result={'summary':summary,'uniform_minus_comparator':paired,'holm_comparisons':len(tests),'test_seeds':TEST,
            'scope':'Fixed-initialization follow-up on previously inspected paired seeds. No tuning. Raw baseline results are unchanged.'}
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2))
    sources=[Path(__file__),Path(__file__).parent/'connected_majority_iqp.py',
             Path(__file__).parent/'paper_training.py',Path(__file__).parent/'paper_blocks.py',
             ROOT/'protocol.json',BASE/'development_selection.json']
    sources+=list((ROOT/'test').glob('*.npz'))+list((ROOT/'test').glob('*.json'))
    repository=Path(__file__).resolve().parent
    original=json.loads((repository/'MANIFEST_SHA256.json').read_text())
    unchanged={str(f.relative_to(repository)):hashlib.sha256(f.read_bytes()).hexdigest()==original[str(f.relative_to(repository))]['sha256']
               for f in (BASE/'test').iterdir() if f.suffix in ['.json','.npz']}
    assert all(unchanged.values())
    (ROOT/'integrity.json').write_text(json.dumps({'baseline_files_unchanged':unchanged,
        'sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        'audited_uniform_runs':20},indent=2))
    import matplotlib.pyplot as plt
    shown=[NAME,'iqp_connected',locked['best_auc_mlp'],locked['best_solve_mlp'],locked['selected_by_width']['19']]
    labels=['IQP: uniform biases/interactions (139)','IQP: original initialization (139)',
            'MLP (30; best development AUC)','MLP (86; fastest development)','MLP (135; parameter matched)']
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for name,label,color in zip(shown,labels,['C3','C2','C0','C1','C4']):
        tr=curves[name];smooth=np.stack([np.convolve(r,np.ones(20)/20,'valid') for r in tr])
        for ax,x,y in [(axes[0],np.arange(20,501),smooth),(axes[1],np.arange(1,501),tr.cumsum(1)/100000)]:
            mean=y.mean(0);sem=y.std(0,ddof=1)/np.sqrt(len(y))
            ax.plot(x,mean,label=label,color=color);ax.fill_between(x,mean-sem,mean+sem,color=color,alpha=.12)
        solves=np.array([r['solve_episode'] for r in groups[name]]);x=np.arange(100,501)
        axes[2].step(x,[(solves<=e).mean() for e in x],where='post',color=color,label=label)
    axes[0].set(ylabel='20-episode mean return',ylim=(0,205),title='Training return')
    axes[1].set(ylabel='Accumulated normalized return',ylim=(0,1),title='Endpoint = normalized AUC')
    axes[2].set(ylabel='Fraction reaching criterion',ylim=(0,1.03),title='Rolling-100 mean return >= 195')
    for ax in axes:ax.set_xlabel('Training episode');ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=8,frameon=False)
    fig.tight_layout(rect=(0,.12,1,.96));fig.savefig(ROOT/'initialization_comparison.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    selected=locked['selected_by_width']
    for ax,key,ylabel in zip(axes,['auc','solve_episode'],['Normalized return AUC','Restricted episodes to criterion']):
        rows=[summary[selected[str(w)]] for w in [1,2,4,6,9,12,16,19]]
        x=np.array([r['parameters'] for r in rows]);y=np.array([r[key]['mean'] for r in rows]);ci=np.array([r[key]['ci95'] for r in rows])
        ax.errorbar(x,y,yerr=np.maximum(0,np.stack([y-ci[:,0],ci[:,1]-y])),fmt='o-',capsize=3,color='C0',label='MLP: rate selected on development')
        for name,label,color in zip(shown[:2],labels[:2],['C3','C2']):
            r=summary[name][key];ax.axhline(r['mean'],color=color,label=label);ax.axhspan(*r['ci95'],color=color,alpha=.08)
        ax.set(xlabel='MLP trainable parameters',ylabel=ylabel);ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=1,fontsize=8,frameon=False)
    fig.tight_layout(rect=(0,.14,1,1));fig.savefig(ROOT/'initialization_mlp_size_comparison.png',dpi=200);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','run','analyze','status']);p.add_argument('--jobs',type=int,default=4)
    a=p.parse_args();protocol()
    if a.mode=='verify':verify()
    elif a.mode=='analyze':analyze()
    elif a.mode=='status':
        rows=[json.loads(f.read_text()) for f in (ROOT/'test').glob('*.json')]
        print(json.dumps({'complete':len(rows),'seeds':[r['seed'] for r in rows],
                          'partial_mean_auc':float(np.mean([r['auc'] for r in rows])) if rows else None}))
    else:
        assert (ROOT/'verification.json').exists()
        from joblib import Parallel,delayed
        Parallel(n_jobs=a.jobs)(delayed(run)(seed) for seed in TEST)
