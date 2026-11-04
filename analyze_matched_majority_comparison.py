"""Parameter-matched follow-up; retain and reuse the prior IQP cohort."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/iqp-search-mpl')
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import nbformat as nbf
from analyze_majority_iqp_search import infer
from majority_iqp_model_search import ROOT,make_policy


def main():
    protocol=json.loads((ROOT/'matched_mlp_protocol.json').read_text())
    seeds=protocol['seeds']; qname=protocol['iqp_model']
    groups={}; curves={}; summary={}
    for name in [qname,'mlp','mlp19']:
        phase=protocol['phase'] if name=='mlp19' else 'validation_round2_strict'
        folder=ROOT/phase; groups[name]={}; curves[name]=[]
        for seed in seeds:
            stem=folder/f'{name.replace(".","p")}__{seed}'
            r=json.loads(stem.with_suffix('.json').read_text()); groups[name][seed]=r
            arr=np.load(stem.with_suffix('.npz')); tr=arr['training_returns']; ev=arr['evaluation_returns']
            assert tr.shape==(500,) and ev.shape==(100,)
            assert np.all(np.isfinite(tr)) and np.all(np.isfinite(ev))
            assert np.all((tr>=1)&(tr<=200)) and np.all((ev>=1)&(ev<=200))
            assert abs(r['auc']-tr.mean()/200)<1e-12
            assert abs(r['heldout']-ev.mean())<1e-12
            s=np.r_[0,np.cumsum(tr)]; hits=np.flatnonzero(s[100:]-s[:-100]>=19500)
            assert r['restricted_solve_episode']==(int(hits[0]+100) if len(hits) else 501)
            qualifying=[k for k in range(100,501,5) if tr[k-100:k].mean()>=195]
            assert r['optimizer_freeze_episode']==(qualifying[0] if qualifying else None)
            assert r['freeze_protocol']=='full_100_episodes'
            if name=='mlp19':
                m=make_policy(name,seed)
                assert sum(p.numel() for p in m.parameters())==135
                saved=torch.load(stem.with_suffix('.pt'),weights_only=False)
                assert all(torch.equal(p,saved['initial'][k]) for k,p in m.named_parameters())
            curves[name].append(np.convolve(tr,np.ones(20)/20,'valid'))
        rows=list(groups[name].values()); summary[name]={'seeds':len(rows),'parameters':rows[0]['parameters']}
        for key in ['auc','restricted_solve_episode','final50','heldout']:
            a=np.array([r[key] for r in rows])
            summary[name][key]={'mean':float(a.mean()),'sem':float(a.std(ddof=1)/np.sqrt(len(a)))}
        summary[name]['solved_runs']=sum(r['restricted_solve_episode']<=500 for r in rows)
    stats={}
    for i,key in enumerate(['auc','restricted_solve_episode','final50','heldout']):
        sign=-1 if key=='restricted_solve_episode' else 1
        stats[key]=infer([sign*(groups[qname][s][key]-groups['mlp19'][s][key]) for s in seeds],20261027+i)
    ordered=sorted(['auc','restricted_solve_episode'],key=lambda k:stats[k]['two_sided_exact_sign_p'])
    running=0
    for i,k in enumerate(ordered):
        running=max(running,min(1.,(2-i)*stats[k]['two_sided_exact_sign_p']))
        stats[k]['holm_p']=running
    result={'protocol':protocol,'summary':summary,'iqp_minus_matched_mlp':stats,
            'note':'Positive solve differences mean fewer IQP episodes. Unsolved runs are assigned 501. Follow-up on the existing cohort, not new independent IQP validation. Intervals do not account for the preceding architecture search.'}
    initial=json.loads((ROOT/(protocol['phase']+'_initial_policy_audit.json')).read_text())
    previous=json.loads((ROOT/'validation_round2_strict_initial_policy_audit.json').read_text())
    assert len(initial)==20
    result['initial_return']={name:float(np.mean([r['initial_mean_return'] for r in initial+previous if r['model']==name])) for name in groups}
    (ROOT/'parameter_matched_comparison.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    labels={qname:'Majority IQP: 21 qubits, 133 parameters','mlp':'MLP 4–9–2: 65 parameters','mlp19':'MLP 4–19–2: 135 parameters'}
    for name in ['mlp','mlp19',qname]:
        a=np.array(curves[name]); mean=a.mean(0); se=a.std(0,ddof=1)/np.sqrt(len(a))
        line,=axes[0].plot(np.arange(20,501),mean,label=labels[name])
        axes[0].fill_between(np.arange(20,501),mean-se,mean+se,alpha=.16,color=line.get_color())
        t=np.arange(100,501); sol=np.array([groups[name][s]['restricted_solve_episode'] for s in seeds])
        axes[1].step(t,[(sol<=x).mean() for x in t],where='post',color=line.get_color())
    axes[0].set(xlabel='Training episode',ylabel='Return (20-episode moving mean)',ylim=(0,205),title='Training returns: mean ± SEM')
    axes[1].set(xlabel='Training episode',ylabel='Fraction reaching rolling-100 mean ≥195',ylim=(0,1.02),title='First crossing of the convergence criterion')
    axes[0].legend(fontsize=8,loc='lower right')
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('CartPole: parameter-matched follow-up, 20 matched seeds')
    fig.tight_layout(); fig.savefig(ROOT/'parameter_matched_comparison.png',dpi=200); plt.close(fig)
    nb=nbf.v4.new_notebook()
    nb.metadata={'kernelspec':{'display_name':'torchlane','language':'python','name':'python3'},'language_info':{'name':'python'}}
    nb.cells=[nbf.v4.new_markdown_cell(r'''# Majority IQP versus a parameter-matched MLP

The IQP model is unchanged: 21 qubits in seven disjoint triples $B_j$.
Each triple applies Hadamards, three neural $R_Z(w_i^Ts+b_i)$ rotations,
three pair $R_{ZZ}$ gates, one $R_{ZZZ}$ gate, and final Hadamards.
All 21 output bits enter a single binary majority: action 0 for at most ten 1s.
The trainable parameter count is $21(4+1)+7(3+1)=133$.

**Why simulation is easy.** At fixed input $s$, there are no inter-block gates,
so $|\psi(s)\rangle=\bigotimes_{j=1}^7|\psi_j(s)\rangle$ and the seven
three-bit outcomes are independent. Compute each eight-outcome distribution
and let $q_j(r)=\Pr[|x_{B_j}|=r\mid s]$, $r=0,1,2,3$. Then
$$F(z)=\prod_{j=1}^7\sum_{r=0}^3q_j(r)z^r,\qquad
\pi(0\mid s)=\sum_{k=0}^{10}[z^k]F(z).$$
Successive convolution takes $O(n^2)$ arithmetic operations for $n$ qubits
partitioned into constant-size triples. Full bitstring sampling is also easy:
sample each three-bit block independently. The implementation uses equivalent
FFT convolution. This is exact factorization, up to numerical precision, not
a Fourier truncation. Exponential exact Pauli support does not prevent this
aggregate simulation. This is not the earlier connected random cubic graph.

**Matched baseline.** Linear(4,19) → Tanh → Linear(19,2), with softmax actions:
$4\cdot19+19+19\cdot2+2=135$ trainable parameters. Only the hidden width changes;
the initialization scheme, preprocessing, optimizer, learning rate, loss,
500-episode budget, 200-step cap and full-100-window freezing rule are unchanged.
Each seed receives 100 frozen-policy evaluation episodes.

The 20 IQP and smaller-MLP runs are reused without modification. This is a
requested follow-up on their cohort, not another independent confirmation of
the selected IQP model. No MLP-specific learning-rate tuning was performed.
Matching parameter count does not match expressivity or initialization.
The reported convergence metric is the first rolling-100 mean ≥195; runs that
never reach it within 500 episodes are assigned 501, not omitted.
Paired-bootstrap intervals and exact paired sign-randomization tests compare
IQP with the 135-parameter MLP. Holm correction covers the two primary endpoints.
These follow-up statistics do not remove the earlier architecture-selection bias.
'''),nbf.v4.new_code_cell('''from pathlib import Path
import json
from IPython.display import display, Image, Markdown
ROOT = Path.cwd() / 'results' / 'majority_iqp_model_search'
result = json.loads((ROOT / 'parameter_matched_comparison.json').read_text())
rows = ['| Model | Parameters | AUC | Episodes to criterion | Held-out return | Solved |', '|---|---:|---:|---:|---:|---:|']
for name, r in result['summary'].items():
    rows.append(f"| {name} | {r['parameters']} | {r['auc']['mean']:.4f} | {r['restricted_solve_episode']['mean']:.1f} | {r['heldout']['mean']:.2f} | {r['solved_runs']}/20 |")
display(Markdown('\\n'.join(rows)))'''),
    nbf.v4.new_code_cell("display(Image(filename=str(ROOT / 'parameter_matched_comparison.png')))"),
    nbf.v4.new_code_cell("result['iqp_minus_matched_mlp']"),
    nbf.v4.new_code_cell("result['initial_return']"),
    nbf.v4.new_markdown_cell('''## Reproduction

Run `majority_iqp_model_search.py run --phase validation_matched_strict --models mlp19`
with the 20 seeds in `matched_mlp_protocol.json`; then run
`audit_majority_initial_policy.py validation_matched_strict --jobs 3 --episodes 100`
and `analyze_matched_majority_comparison.py`. All raw returns and checkpoints
are retained. The analysis verifies return metrics, freezing and initialization.
The paper is unchanged.''')]
    nbf.write(nb,ROOT.parent.parent/'majority_iqp_parameter_matched.ipynb')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
