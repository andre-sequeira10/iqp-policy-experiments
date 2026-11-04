"""Audit and analyze the locked connected-IQP / tuned MLP size experiment."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/iqp-search-mpl')
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from connected_iqp_mlp_study import ROOT,DEV,TEST,WIDTHS,LRS,specs,policy,solve
from majority_iqp_model_search import evaluate
from analyze_majority_iqp_search import infer


def names():
    lock=json.loads((ROOT/'development_selection.json').read_text())
    mlps=list(dict.fromkeys(list(lock['selected_by_width'].values())+[lock['best_solve_mlp']]))
    return lock,mlps+['iqp_connected','iqp_disconnected']


def initial_audit(name,seed):
    folder=ROOT/'initial_audit';folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'{name}__{seed}.json'
    if target.exists(): return json.loads(target.read_text())
    spec=specs()[name]; model=policy(spec,seed)
    result=evaluate(model,seed,spec['kind']=='mlp',100)
    row={'name':name,'seed':seed,'initial_return':float(result.mean()),'episodes':100}
    target.write_text(json.dumps(row,indent=2));return row


def bootstrap(x,level=.95,seed=731,n=50000):
    x=np.asarray(x);rng=np.random.default_rng(seed)
    means=rng.choice(x,size=(n,len(x)),replace=True).mean(1)
    return np.quantile(means,[(1-level)/2,1-(1-level)/2]).tolist()


def holm(comparisons):
    ordered=sorted(comparisons,key=lambda c:c['two_sided_exact_sign_p'])
    previous=0.
    for i,c in enumerate(ordered):
        previous=max(previous,min(1.,(len(ordered)-i)*c['two_sided_exact_sign_p']))
        c['holm_p']=previous


def analyze():
    lock,all_names=names();groups={};arrays={};summary={};audited=[]
    for name in all_names:
        groups[name]=[];arrays[name]=[];initial=[]
        for seed in TEST:
            stem=ROOT/'test'/f'{name}__{seed}'
            r=json.loads(stem.with_suffix('.json').read_text()); groups[name].append(r)
            a=np.load(stem.with_suffix('.npz'));tr=a['training_returns'];ev=a['evaluation_returns']
            assert tr.shape==(500,) and ev.shape==(100,)
            assert np.all((tr>=1)&(tr<=200)) and np.all((ev>=1)&(ev<=200))
            assert abs(r['auc']-tr.mean()/200)<1e-12 and abs(r['heldout']-ev.mean())<1e-12
            assert abs(r['final50']-tr[-50:].mean())<1e-12 and r['solve_episode']==solve(tr)
            qualifying=[k for k in range(100,501,5) if tr[k-100:k].mean()>=195]
            assert r['freeze_episode']==(qualifying[0] if qualifying else None)
            assert r['updates']==(r['freeze_episode']//5-1 if r['freeze_episode'] else 100)
            expected=policy(r['spec'],seed)
            saved=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=False)
            assert r['parameters']==sum(p.numel() for p in expected.parameters())
            assert all(torch.equal(p,saved['initial'][k]) for k,p in expected.named_parameters())
            assert r['initial_hash']==hashlib.sha256(b''.join(v.numpy().tobytes() for v in saved['initial'].values())).hexdigest()
            arrays[name].append(tr)
            initial.append(json.loads((ROOT/'initial_audit'/f'{name}__{seed}.json').read_text())['initial_return'])
            audited.append({'name':name,'seed':seed,'passed':True})
        rows=groups[name];summary[name]={'parameters':rows[0]['parameters'],'lr':rows[0]['spec']['lr'],'n':len(rows),
                                      'solved':sum(r['solve_episode']<=500 for r in rows),
                                      'initial_return':float(np.mean(initial))}
        for key in ['auc','solve_episode','final50','heldout']:
            x=[r[key] for r in rows]
            summary[name][key]={'mean':float(np.mean(x)),'sem':float(np.std(x,ddof=1)/np.sqrt(len(x))),
                                'ci95':bootstrap(x)}
    def compare(first,second):
        result={}
        for i,key in enumerate(['auc','solve_episode','heldout']):
            sign=-1 if key=='solve_episode' else 1
            result[key]=infer([sign*(a[key]-b[key]) for a,b in zip(groups[first],groups[second])],20261027+i)
        return result
    comparisons={name:compare('iqp_connected',name) for name in all_names if name.startswith('mlp')}
    holm([v[k] for v in comparisons.values() for k in ['auc','solve_episode']])
    connection=compare('iqp_connected','iqp_disconnected')
    saturation={};high=lock['selected_by_width']['19']
    for width in [9,12,16]:
        low=lock['selected_by_width'][str(width)];result={}
        for i,(key,margin) in enumerate([('auc',.02),('solve_episode',25)]):
            sign=-1 if key=='solve_episode' else 1
            difference=[sign*(a[key]-b[key]) for a,b in zip(groups[high],groups[low])]
            interval=bootstrap(difference,level=1-.05/3,seed=761+i,n=100000)
            result[key]={'mean_benefit_of_width19':float(np.mean(difference)),
                         'ci98_3333':interval,'equivalence_margin':margin,
                         'equivalence_established':interval[0]>-margin and interval[1]<margin}
        result['equivalent_on_both']=all(result[k]['equivalence_established'] for k in ['auc','solve_episode'])
        saturation[str(width)]=result
    purities={}
    for stage in ['initial','final']:
        values=np.array([r[stage+'_bridges'] for r in groups['iqp_connected']],dtype=float)
        p=1-.5*np.sin(values)**2
        purities[stage]={'min':float(p.min()),'max':float(p.max()),
                         'cut_count':int(p.size),'cuts_with_purity_below_1_minus_1e_8':int((p<1-1e-8).sum())}
    result={'selection':lock,'summary':summary,'connected_minus_mlp':comparisons,
            'connected_minus_disconnected_exploratory':connection,'saturation':saturation,
            'interblock_cut_purities':purities,'audited_test_runs':len(audited),
            'interpretation':'Positive comparisons favor connected IQP, including fewer solve episodes. Unsolved runs are assigned 501. All IQP-vs-MLP AUC/solve tests receive a single Holm correction; held-out and connected-vs-disconnected tests are secondary/exploratory. Bootstrap CIs are not selection-adjusted simultaneous CIs.'}
    (ROOT/'test_analysis.json').write_text(json.dumps(result,indent=2))
    versions={}
    for package in ['numpy','torch','pennylane','gymnasium','joblib','nbformat']:
        versions[package]=importlib.metadata.version(package)
    files=[ROOT/'protocol.json',ROOT/'development_selection.json']
    files+=list(ROOT.parent.parent.glob('*connected*py'))
    files+=[ROOT.parent.parent/'majority_iqp_model_search.py',ROOT.parent.parent/'majority_iqp_block_model.py',
            Path(__file__).resolve().parent/'qasp/rl/torchrl.py']
    files+=list((ROOT/'test').glob('*.npz'))+list((ROOT/'test').glob('*.json'))
    integrity={'python':sys.version,'versions':versions,'audited_runs':audited,
               'sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    (ROOT/'test_integrity.json').write_text(json.dumps(integrity,indent=2))
    plot(result,groups,arrays)
    print(json.dumps({'selected_best_auc':lock['best_auc_mlp'],'selected_fastest':lock['best_solve_mlp'],
                      'summary':summary,'saturation':saturation,'purities':purities},indent=2))


def plot(result,groups,arrays):
    lock=result['selection'];s=result['summary']; selected=lock['selected_by_width']
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.7))
    for ax,key,label in zip(axes,['auc','solve_episode'],['Normalized training-return AUC','Restricted episodes to criterion']):
        x=np.array([s[selected[str(w)]]['parameters'] for w in WIDTHS])
        y=np.array([s[selected[str(w)]][key]['mean'] for w in WIDTHS])
        ci=np.array([s[selected[str(w)]][key]['ci95'] for w in WIDTHS])
        ax.errorbar(x,y,yerr=np.maximum(0,np.stack((y-ci[:,0],ci[:,1]-y))),fmt='o-',capsize=3,label='MLP: learning rate selected on development')
        for name,color,style,label2 in [('iqp_connected','C2','-','Connected IQP (139 parameters)'),('iqp_disconnected','C3',':','Disconnected IQP (133 parameters)')]:
            v=s[name][key];ax.axhline(v['mean'],color=color,ls=style,label=label2)
            if name=='iqp_connected': ax.axhspan(*v['ci95'],color=color,alpha=.10)
        ax.set(xlabel='MLP trainable parameters',ylabel=label);ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,loc='lower center',ncol=3,bbox_to_anchor=(.5,.005))
    fig.suptitle('MLP size study: 20 untouched test seeds; error bars are bootstrap 95% CIs')
    fig.tight_layout(rect=(0,.08,1,.96));fig.savefig(ROOT/'mlp_size_scaling.png',dpi=200);plt.close(fig)
    show=list(dict.fromkeys(['iqp_connected','iqp_disconnected',lock['best_auc_mlp'],lock['best_solve_mlp'],selected['19']]))
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    colors={'iqp_connected':'C2','iqp_disconnected':'C3'}
    for name,color in zip([n for n in show if n.startswith('mlp')],['C0','C1','C4']): colors[name]=color
    for name in show:
        if name.startswith('iqp'): label=name.replace('iqp_','').capitalize()+f" IQP ({s[name]['parameters']} parameters)"
        else: label=f"MLP ({s[name]['parameters']} parameters, lr={s[name]['lr']})"
        curves=np.array([np.convolve(a,np.ones(20)/20,'valid') for a in arrays[name]])
        m=curves.mean(0);se=curves.std(0,ddof=1)/np.sqrt(len(curves))
        line,=axes[0].plot(np.arange(20,501),m,label=label,color=colors[name],ls=':' if name=='iqp_disconnected' else '-')
        axes[0].fill_between(np.arange(20,501),m-se,m+se,color=line.get_color(),alpha=.12)
        hits=np.array([r['solve_episode'] for r in groups[name]]);x=np.arange(100,501)
        axes[1].step(x,[(hits<=t).mean() for t in x],where='post',color=line.get_color(),ls=':' if name=='iqp_disconnected' else '-')
    axes[0].set(xlabel='Training episode',ylabel='Return (20-episode moving mean)',ylim=(0,205),title='Mean ± SEM')
    axes[1].set(xlabel='Training episode',ylabel='Fraction reaching rolling-100 mean ≥195',ylim=(0,1.02),title='First crossing of the convergence criterion')
    axes[0].legend(fontsize=7,loc='lower right')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Connected IQP and development-selected classical baselines: 20 new test seeds')
    fig.tight_layout();fig.savefig(ROOT/'locked_comparison.png',dpi=200);plt.close(fig)
    grid=np.array([[lock['development_means'][f'mlp_h{w:02d}_lr{str(lr).replace(".","p")}']['auc'] for lr in LRS] for w in WIDTHS])
    fig,ax=plt.subplots(figsize=(6.8,5.2));im=ax.imshow(grid,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set_xticks(range(len(LRS)),[str(v) for v in LRS]);ax.set_yticks(range(len(WIDTHS)),[str(7*w+2) for w in WIDTHS])
    ax.set(xlabel='Learning rate',ylabel='MLP trainable parameters',title='Development AUC: eight seeds per configuration')
    for i in range(len(WIDTHS)):
        for j in range(len(LRS)): ax.text(j,i,f'{grid[i,j]:.3f}',ha='center',va='center',color='white' if grid[i,j]<.5 else 'black')
    fig.colorbar(im,ax=ax,label='Normalized AUC');fig.tight_layout();fig.savefig(ROOT/'development_grid.png',dpi=200);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit-initial','analyze']);p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
    if a.mode=='audit-initial':
        import joblib
        _,all_names=names()
        rows=joblib.Parallel(n_jobs=a.jobs)(joblib.delayed(initial_audit)(n,s) for s in TEST for n in all_names)
        print('Initial-policy audits:',len(rows))
    else: analyze()


if __name__=='__main__':main()
