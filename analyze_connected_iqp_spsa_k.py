"""Audit and compare the locked K-only SPSA follow-up."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/iqp-search-mpl')
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from connected_iqp_spsa_k_study import ROOT, BASE, OLD, TEST, TIMING_SEEDS, SHOTS, solve
from analyze_majority_iqp_search import infer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    selection=json.loads((ROOT/'selection.json').read_text());k=selection['selected_k']
    groups={};curves={};summary={};timings={}
    names=['analytical','k008',f'k{k:03d}']
    for name in names:
        rows=[];traces=[]
        for seed in TEST:
            if name=='analytical':stem=BASE/'test'/f'iqp_connected__{seed}'
            elif name=='k008':stem=OLD/'test'/f'iqp_connected_shots4096_spsa8__{seed}'
            else:stem=ROOT/'test'/f'k{k:03d}__{seed}'
            r=json.loads(stem.with_suffix('.json').read_text());a=np.load(stem.with_suffix('.npz'))
            tr=a['training_returns'];ev=a['evaluation_returns']
            assert tr.shape==(500,) and ev.shape==(100,) and np.isfinite(tr).all()
            assert np.all((tr>=1)&(tr<=200)) and np.all((ev>=1)&(ev<=200))
            assert r['solve_episode']==solve(tr) and abs(r['auc']-tr.mean()/200)<1e-12
            qualifying=[e for e in range(100,501,5) if tr[e-100:e].mean()>=195]
            assert r['freeze_episode']==(qualifying[0] if qualifying else None)
            if name!='analytical':
                q=8 if name=='k008' else k
                assert r['circuit_evaluations']['gradient']==2*q*r['optimized_transitions']
                assert r['training_shots']==SHOTS*(sum(tr)+r['circuit_evaluations']['gradient'])
                reference=json.loads((BASE/'test'/f'iqp_connected__{seed}.json').read_text())
                assert r['initial_hash']==reference['initial_hash']
            rows.append(r);traces.append(tr)
        groups[name]=rows;curves[name]=np.stack(traces)
        summary[name]={'solved':sum(r['solve_episode']<=500 for r in rows),'seeds':len(rows)}
        keys=['auc','solve_episode','heldout','final50']+([] if name=='analytical' else ['training_shots'])
        for key in keys:
            v=np.array([r[key] for r in rows]);rng=np.random.default_rng(12361)
            boot=rng.choice(v,size=(50000,len(v)),replace=True).mean(1)
            summary[name][key]={'mean':float(v.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()}
        ts=[json.loads((ROOT/'serial_timing'/f'{name}__{s}.json').read_text()) for s in TIMING_SEEDS]
        assert all(r['trajectory_exactly_reproduced'] for r in ts)
        times=np.array([r['training_seconds'] for r in ts])
        timings[name]={'mean_training_seconds':float(times.mean()),'mean_training_cpu_seconds':float(np.mean([r['training_cpu_seconds'] for r in ts])),
                       'range_seconds':[float(times.min()),float(times.max())],'per_seed':ts}
    selected=names[-1];paired={}
    for reference in names[:2]:
        paired[reference]={key:infer([a[key]-b[key] for a,b in zip(groups[selected],groups[reference])],87109)
                           for key in ['auc','solve_episode']}
        timings[selected]['speed_ratio_vs_'+reference]=timings[selected]['mean_training_seconds']/timings[reference]['mean_training_seconds']
    metrics=sorted([p for pair in paired.values() for p in pair.values()],key=lambda r:r['two_sided_exact_sign_p'])
    previous=0
    for i,r in enumerate(metrics):
        previous=max(previous,min(1,(len(metrics)-i)*r['two_sided_exact_sign_p']));r['holm_p']=previous
    gaps={'auc_loss_vs_analytical':summary['analytical']['auc']['mean']-summary[selected]['auc']['mean'],
          'additional_restricted_episodes':summary[selected]['solve_episode']['mean']-summary['analytical']['solve_episode']['mean']}
    result={'selection':selection,'summary':summary,'paired_selected_minus_reference':paired,'serial_timings':timings,
            'test_mean_targets_met':gaps['auc_loss_vs_analytical']<=.03 and gaps['additional_restricted_episodes']<=25,
            'test_gaps':gaps,'timing_platform':platform.platform(),'timing_threads_per_run':1,
            'timing_note':'Three fixed seeds, serial runs; full 500-episode training excluding setup and held-out evaluation. Not QPU runtime. All timing trajectories reproduced saved test trajectories exactly.',
            'statistical_note':'Individual paired-bootstrap intervals. A mean lying within a tolerance does not establish statistical equivalence. Tuning used eight separate development seeds, but the comparison cohort was inspected in earlier studies.'}
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2))
    old_manifest=json.loads((OLD/'integrity.json').read_text())['sha256']
    unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in old_manifest.items()}
    assert all(unchanged.values())
    sources=[Path(__file__),Path(__file__).with_name('connected_iqp_spsa_k_study.py'),ROOT/'protocol.json',ROOT/'selection.json']
    for folder in ['development','test','serial_timing']:
        sources+=list((ROOT/folder).glob('*.json'))+list((ROOT/folder).glob('*.npz'))
    (ROOT/'integrity.json').write_text(json.dumps({'prior_experiment_files_unchanged':unchanged,
        'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2))

    labels=['Analytical',r'SPSA $K=8$',f'SPSA $K={k}$'];colors=['#1b9e77','#d95f02','#7570b3']
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for name,label,color in zip(names,labels,colors):
        tr=curves[name];smooth=np.stack([np.convolve(t,np.ones(20)/20,'valid') for t in tr])
        for ax,x,v in [(axes[0],np.arange(20,501),smooth),(axes[1],np.arange(1,501),tr.cumsum(1)/100000)]:
            m=v.mean(0);se=v.std(0,ddof=1)/np.sqrt(len(v));ax.plot(x,m,label=label,color=color)
            ax.fill_between(x,m-1.96*se,m+1.96*se,color=color,alpha=.12)
        hits=np.array([r['solve_episode'] for r in groups[name]])
        axes[2].step(np.arange(1,501),[(hits<=ep).mean() for ep in range(1,501)],where='post',color=color,label=label)
    for ax in axes:ax.set_xlabel('Training episode');ax.grid(alpha=.2)
    axes[0].set(ylabel='20-episode mean return',ylim=(0,205),title='Training return')
    axes[1].set(ylabel='Accumulated normalized return',ylim=(0,1.02),title='Endpoint = normalized AUC')
    axes[2].set(ylabel='Fraction reaching criterion',ylim=(0,1.02),title='Rolling-100 mean return ≥195')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Connected 21-qubit IQP: original initialization, 20 matched seeds\nSPSA: 4,096 shots; only K changes. Shading: pointwise 95% intervals.',fontsize=12)
    fig.savefig(ROOT/'training_comparison.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    for i,(name,color) in enumerate(zip(names,colors)):
        vals=np.array([r['training_seconds']/60 for r in timings[name]['per_seed']])
        axes[0].bar(i,vals.mean(),color=color,alpha=.65);axes[0].scatter(i+np.linspace(-.1,.1,len(vals)),vals,color=color,edgecolor='black',s=30)
    axes[0].set(xticks=range(3),xticklabels=labels,ylabel='Full training time (minutes)',title='Serial CPU timing: three fixed seeds')
    for i,(name,color) in enumerate(zip(names[1:],colors[1:])):
        shots=np.array([r['training_shots']/1e9 for r in groups[name]])
        axes[1].bar(i,shots.mean(),color=color,alpha=.65)
        axes[1].scatter(i+np.linspace(-.16,.16,len(shots)),shots,color=color,s=12,edgecolor='black',linewidth=.3)
    axes[1].set(xticks=[0,1],xticklabels=labels[1:],ylabel='Simulated training shots (billions)',title='Shot resources: all 20 seeds')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Training excludes held-out evaluation; dots show individual seeds\nCPU simulation time must not be interpreted as quantum-hardware runtime',fontsize=11)
    fig.savefig(ROOT/'time_and_shot_cost.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    stages=selection['stages'];ks=[r['k'] for r in stages]
    for ax,key,label,tolerance in zip(axes,['auc','solve_episode'],['Mean normalized AUC','Restricted episodes to criterion'],[-.03,25]):
        reference=selection['analytical_development'][key]
        ax.plot(ks,[r[key] for r in stages],'o-',color='#7570b3',label='Finite-shot SPSA')
        ax.axhline(reference,color='#1b9e77',label='Analytical reference')
        ax.axhline(reference+tolerance,color='#666666',linestyle=':',label='Selection threshold')
        ax.set_xscale('log',base=2);ax.set_xticks(ks,ks);ax.set(xlabel='SPSA directions K',ylabel=label);ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',frameon=False,ncol=3)
    fig.suptitle('K selection on eight separate development seeds',fontsize=12)
    fig.savefig(ROOT/'development_k_sweep.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
