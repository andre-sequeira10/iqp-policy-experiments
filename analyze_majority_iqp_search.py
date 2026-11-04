"""Report all development runs or the single locked fresh-seed comparison."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/iqp-search-mpl')
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'results'/'majority_iqp_model_search'

def infer(d,seed):
    d=np.asarray(d,float); rng=np.random.default_rng(seed)
    boot=rng.choice(d,size=(50000,len(d)),replace=True).mean(1)
    # Exact paired sign randomization is practical for 20 seed pairs.
    count=0; total=2**len(d); observed=abs(d.mean())
    for start in range(0,total,16384):
        ids=np.arange(start,min(start+16384,total),dtype=np.uint64)
        signs=2*((ids[:,None]>>np.arange(len(d),dtype=np.uint64))&1).astype(float)-1
        count+=np.count_nonzero(abs((signs*d).mean(1))>=observed-1e-12)
    return {'mean_difference':float(d.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'two_sided_exact_sign_p':count/total,'positive_pairs':int(sum(d>0))}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('phase'); args=parser.parse_args()
    folder=ROOT/args.phase
    rows=[json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))]
    models=sorted({r['model']['name'] for r in rows})
    summary={}
    for name in models:
        group=sorted([r for r in rows if r['model']['name']==name],key=lambda r:r['seed'])
        summary[name]={'n':len(group),'seeds':[r['seed'] for r in group],'parameters':group[0]['parameters']}
        for key in ['auc','restricted_solve_episode','final50','heldout']:
            values=np.array([r[key] for r in group])
            summary[name][key]={'mean':float(values.mean()),'sem':float(values.std(ddof=1)/np.sqrt(len(values)))}
        summary[name]['solved_fraction']=float(np.mean([r['restricted_solve_episode']<=500 for r in group]))
    result={'phase':args.phase,'summary':summary}
    if args.phase.startswith('validation'):
        round2=args.phase.startswith('validation_round2')
        lockfile='locked_model_round2.json' if round2 else 'locked_model.json'
        lock=json.loads((ROOT/lockfile).read_text()); name=lock['model']
        q={r['seed']:r for r in rows if r['model']['name']==name}
        c={r['seed']:r for r in rows if r['model']['name']=='mlp'}
        assert set(q)==set(c)==set(lock['validation_seeds'])
        seeds=sorted(q); result['comparisons']={}
        for i,key in enumerate(['auc','restricted_solve_episode','final50','heldout']):
            sign=-1 if key=='restricted_solve_episode' else 1
            result['comparisons'][key]=infer([sign*(q[s][key]-c[s][key]) for s in seeds],20261026+i)
        endpoints=['auc','restricted_solve_episode']
        ordered=sorted(endpoints,key=lambda k:result['comparisons'][k]['two_sided_exact_sign_p'])
        running=0
        for i,key in enumerate(ordered):
            running=max(running,min(1.,(2-i)*result['comparisons'][key]['two_sided_exact_sign_p']))
            result['comparisons'][key]['holm_p']=running
        for key in endpoints:
            result['comparisons'][key]['search_adjusted_p']=min(1.,(2 if round2 else 1)*result['comparisons'][key]['holm_p'])
        result['passes_locked_success_rule']=all(result['comparisons'][k]['ci95'][0]>0 and
            result['comparisons'][k]['search_adjusted_p']<.05 for k in endpoints)
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for model,label in [('mlp','Classical MLP (65 parameters)'),(name,f'Majority IQP ({summary[name]["parameters"]} parameters)')]:
            curves=[]; solves=[]
            for seed in seeds:
                stem=f'{model.replace(".","p")}__{seed}'
                arr=np.load(folder/(stem+'.npz'))['training_returns']
                curves.append(np.convolve(arr,np.ones(20)/20,'valid'))
                solves.append((q if model==name else c)[seed]['restricted_solve_episode'])
            curves=np.array(curves); m=curves.mean(0); se=curves.std(0,ddof=1)/np.sqrt(len(seeds))
            line,=axes[0].plot(np.arange(20,501),m,label=label)
            axes[0].fill_between(np.arange(20,501),m-se,m+se,color=line.get_color(),alpha=.2)
            x=np.arange(100,501); axes[1].step(x,[np.mean(np.array(solves)<=v) for v in x],where='post',label=label)
        axes[0].set(xlabel='Training episode',ylabel='Return (20-episode moving mean)',ylim=(0,205))
        axes[1].set(xlabel='Training episode',ylabel='Fraction reaching rolling-100 mean ≥195',ylim=(0,1.02))
        axes[0].set_title('Training returns: mean ± SEM',fontsize=10)
        axes[1].set_title('First crossing of the convergence criterion',fontsize=10)
        for ax in axes: ax.grid(alpha=.2)
        axes[0].legend(fontsize=8); fig.suptitle('CartPole: locked model, 20 fresh matched seeds')
        fig.tight_layout(); fig.savefig(ROOT/(args.phase+'_comparison.png'),dpi=200); plt.close(fig)
    (ROOT/(args.phase+'_summary.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
