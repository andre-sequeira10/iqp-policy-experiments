"""Redraw the manuscript connected-model comparison from saved arrays, no training."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'generated/connected')
    args=parser.parse_args()
    folder=ROOT/'results/connected_iqp_mlp_study'
    selection=json.loads((folder/'development_selection.json').read_text())
    names=list(dict.fromkeys(['iqp_connected',selection['best_auc_mlp'],
                             selection['best_solve_mlp'],selection['selected_by_width']['19']]))
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    summary={}
    for name in names:
        rows=[];solve=[]
        for seed in range(2001,2021):
            stem=folder/'test'/f'{name}__{seed}'
            record=json.loads(stem.with_suffix('.json').read_text())
            with np.load(stem.with_suffix('.npz'),allow_pickle=False) as data:
                returns=data['training_returns'].copy()
            assert returns.shape==(500,) and np.isfinite(returns).all(),str(stem)
            assert np.all((returns>=1)&(returns<=200)),str(stem)
            assert np.isclose(returns.mean()/200,record['auc'],rtol=0,atol=1e-12),str(stem)
            sums=np.r_[0.,returns.cumsum()]
            hits=np.flatnonzero(sums[100:]-sums[:-100]>=19500)
            criterion=int(hits[0]+100) if len(hits) else 501
            assert criterion==record['solve_episode'],str(stem)
            rows.append(returns);solve.append(criterion)
        rows=np.stack(rows);steps=np.arange(1,501)
        label=('Connected IQP' if name=='iqp_connected' else 'MLP')+f" ({record['parameters']} parameters)"
        if name==selection['best_solve_mlp']:label+='; fastest development'
        summary[name]={'seeds':20,'parameters':record['parameters'],
                       'auc_mean':float(rows.mean()/200),'episodes_to_criterion_mean':float(np.mean(solve)),
                       'final_50_mean':float(rows[:,-50:].mean())}
        series=[np.stack([np.convolve(r,np.ones(20)/20,mode='valid') for r in rows]),
                rows.cumsum(axis=1)/steps,rows.cumsum(axis=1)/(200*500)]
        for ax,x,y in zip(axes,[steps[19:],steps,steps],series):
            mean=y.mean(0);sem=y.std(0,ddof=1)/np.sqrt(len(y))
            line,=ax.plot(x,mean,label=label)
            ax.fill_between(x,mean-sem,mean+sem,color=line.get_color(),alpha=.12)
    for ax,title,ylabel in zip(axes,['Training returns','Cumulative average return','Accumulated normalized return'],
                              ['Trailing 20-episode return','Mean return through episode','Return / (200 × 500)']):
        ax.set(title=title,xlabel='Training episode',ylabel=ylabel,xlim=(1,500));ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,loc='lower right')
    output=args.output.resolve()
    # Do not overwrite publication evidence via a command-line path.
    for protected in [ROOT/'results',ROOT/'tests/results']:
        if output==protected or protected in output.parents:
            raise ValueError('Output must not be inside archived results.')
    output.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(output/f'connected_learning_curves.{ext}',dpi=180,bbox_inches='tight')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print(f'Figures saved to {output}')

if __name__=='__main__':main()
