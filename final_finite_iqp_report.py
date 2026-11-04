"""Final 20-seed analysis and separate serial, training-only timing replay."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/iqp-search-mpl')
import argparse
import contextlib
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
import finite_spsa_mlp_comparison as c
from connected_iqp_mlp_study import specs, policy, solve
from connected_iqp_spsa_k_study import KTrainer
from majority_iqp_model_search import make_agent

torch.set_num_threads(1)
ROOT = c.ROOT / 'final'
TIMING_SEEDS = [2001, 2002, 2003]
BEST = c.references()[0]
NAMES = ['iqp_connected', 'k8', 'k32', 'k64', BEST]
LABELS = ['Analytical IQP', 'IQP: K=8', 'IQP: K=32', 'IQP: K=64', 'MLP (135)']
COLORS = ['#009E73', '#D55E00', '#9983BF', '#0072B2', '#222222']


def timing():
    folder = ROOT / 'serial_timing'
    folder.mkdir(parents=True, exist_ok=True)
    spec = {'seeds': TIMING_SEEDS, 'methods': NAMES, 'episodes': 500,
            'scope': 'Serial full training, one torch/BLAS thread, excluding construction and held-out evaluation. Same 500-episode trajectories must replay exactly. Method order rotated across seeds. No parallel experiment workers launched by this study.',
            'interpretation': 'Three fixed seeds for descriptive timing only, separate from the 20-seed learning comparison. Classical simulator CPU/wall time, not quantum hardware latency.',
            'platform': platform.platform(), 'machine': platform.machine(), 'torch': torch.__version__,
            'cpu': 'Apple M1', 'physical_memory_bytes': 17179869184,
            'logical_cpus': 8, 'hardware_source': 'sysctl read on benchmark host'}
    (folder/'protocol.json').write_text(json.dumps(spec, indent=2))
    for i, seed in enumerate(TIMING_SEEDS):
        for name in NAMES[i:] + NAMES[:i]:
            stem = folder / f'{name}__{seed}'
            if stem.with_suffix('.json').exists():
                continue
            m = policy(specs()[BEST if name == BEST else 'iqp_connected'], seed)
            print(f'START timing {name} seed={seed}', flush=True)
            with stem.with_suffix('.log').open('w') as log, contextlib.redirect_stdout(log):
                if name.startswith('k'):
                    agent = KTrainer(m, seed, int(name[1:]))
                    start = time.perf_counter(); cpu_start = time.process_time()
                    tr = agent.train()
                    elapsed = time.perf_counter()-start; cpu_elapsed = time.process_time()-cpu_start
                    stamps = agent.episode_seconds
                    agent.agent.env.close()
                else:
                    agent = make_agent(m, name == BEST, strict=True)
                    torch.manual_seed(seed); np.random.seed(seed)
                    agent.env.reset(seed=seed); agent.env.action_space.seed(seed)
                    stamps = []
                    original_reset = agent.env.reset
                    def timed_reset(*args, **kwargs):
                        stamps.append(time.perf_counter()-start)
                        return original_reset(*args, **kwargs)
                    agent.env.reset = timed_reset
                    start = time.perf_counter(); cpu_start = time.process_time()
                    tr = np.asarray(agent.train(500, 50))
                    elapsed = time.perf_counter()-start; cpu_elapsed = time.process_time()-cpu_start
                    stamps = stamps[1:] + [elapsed]
                    agent.env.close()
            with np.load(c.stem_for(name, seed).with_suffix('.npz')) as f:
                original = f['training_returns']
            assert np.array_equal(tr, original), f'Trajectory mismatch {name} {seed}'
            assert len(stamps) == 500
            hit = solve(tr)
            result = {'model': name, 'seed': seed, 'training_seconds': elapsed,
                      'training_cpu_seconds': cpu_elapsed, 'criterion_episode': hit,
                      'seconds_to_criterion': float(stamps[hit-1]) if hit <= 500 else None,
                      'trajectory_exactly_reproduced': True}
            np.savez_compressed(stem.with_suffix('.npz'), training_returns=tr, episode_seconds=stamps)
            stem.with_suffix('.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)


def report():
    ROOT.mkdir(parents=True, exist_ok=True)
    c.report()
    rows, curves, summary = {}, {}, {}
    ids = np.random.default_rng(20261027).integers(0,20,(50000,20))
    for name in NAMES:
        rows[name], curves[name] = c.load(name)
        summary[name] = {'seeds': 20, 'solved': sum(r['solve_episode'] <= 500 for r in rows[name])}
        for metric in ['auc', 'solve_episode', 'heldout', 'final50']:
            summary[name][metric] = c.ci([r[metric] for r in rows[name]], ids)
        if name.startswith('k'):
            summary[name]['training_shots_mean'] = float(np.mean([r['training_shots'] for r in rows[name]]))
        timings = [ROOT/'serial_timing'/f'{name}__{s}.json' for s in TIMING_SEEDS]
        if all(f.exists() for f in timings):
            rr = [json.loads(f.read_text()) for f in timings]
            v = np.array([r['training_seconds'] for r in rr])
            summary[name]['serial_training_seconds'] = {'mean':float(v.mean()), 'min':float(v.min()), 'max':float(v.max()), 'seeds':TIMING_SEEDS}
    paired = json.loads((c.ROOT/'comparison.json').read_text())['paired_IQP_minus_MLP'][BEST]
    out = {'summary':summary, 'paired_IQP_minus_best_MLP':paired,
           'learning_seeds': c.TEST, 'best_MLP':BEST,
           'caveats': 'Best MLP is selected retrospectively by highest observed cohort AUC. Comparisons exploratory. Timings are separate serial replays on three seeds, not a 20-seed timing benchmark. No hardware noise or hardware runtime included.'}
    (ROOT/'comparison.json').write_text(json.dumps(out, indent=2))
    plot(curves, summary)
    files = [Path(__file__), ROOT/'comparison.json']
    files += [c.stem_for(name,s).with_suffix(ext) for name in NAMES for s in c.TEST for ext in ['.json','.npz']]
    files += sorted((ROOT/'serial_timing').glob('*.json'))
    files += sorted((ROOT/'serial_timing').glob('*.npz'))
    files += sorted(ROOT.glob('connected_iqp_finite_sampling_final.*'))
    (ROOT/'manifest.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, indent=2))
    print(json.dumps(out, indent=2), flush=True)


def plot(curves, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11, 'axes.titlesize':11, 'pdf.fonttype':42, 'ps.fonttype':42})
    ids = np.random.default_rng(71025).integers(0,20,(5000,20))
    weights = np.stack([np.bincount(row,minlength=20)/20 for row in ids])
    fig, axes = plt.subplots(2,2,figsize=(9.2,6.4),layout='constrained')
    for i,(name,label,color) in enumerate(zip(NAMES,LABELS,COLORS)):
        tr = curves[name]
        smooth = np.stack([np.convolve(t,np.ones(20)/20,'valid') for t in tr])
        for ax,x,v in [(axes[0,0],np.arange(20,501),smooth),(axes[0,1],np.arange(1,501),tr.cumsum(1)/100000)]:
            ax.plot(x,v.mean(0),color=color,label=label,lw=1.8)
            lo,hi=np.quantile(weights@v,[.025,.975],axis=0)
            ax.fill_between(x,lo,hi,color=color,alpha=.10)
        a=summary[name]['auc']; m=a['mean']; lo,hi=a['ci95']
        ax=axes[1,0]; ax.bar(i,m,color=color,alpha=.8)
        ax.scatter(i+np.linspace(-.22,.22,20),tr.mean(1)/200,color=color,s=12,edgecolors='white',linewidths=.4,zorder=3)
        ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='none',color='black',capsize=3)
        ax.text(i,max(hi,float((tr.mean(1)/200).max()))+.024,f'{m:.4f}',ha='center',fontsize=9)
        t=summary[name].get('serial_training_seconds')
        if t:
            v=np.array([json.loads((ROOT/'serial_timing'/f'{name}__{s}.json').read_text())['training_seconds'] for s in TIMING_SEEDS])/60
            axes[1,1].bar(i,v.mean(),color=color,alpha=.8)
            axes[1,1].scatter(i+np.linspace(-.15,.15,3),v,color=color,edgecolors='white',s=22,zorder=3)
            axes[1,1].text(i,max(v)*1.12,f'{v.mean():.2f}',ha='center',fontsize=9)
    ticks=['Analytical\nIQP','K=8','K=32','K=64','MLP']
    axes[0,0].set(title='(a) Training returns',xlabel='Training episode',ylabel='20-episode mean return',ylim=(0,205))
    axes[0,1].set(title='(b) Accumulated reward',xlabel='Training episode',ylabel=r'$\sum_{e=1}^{t} R_e/(500\times200)$',ylim=(0,1.02))
    axes[1,0].set(title='(c) Normalized AUC: 20 seeds',ylabel='AUC over all 500 episodes',ylim=(0,1.08),xticks=range(5),xticklabels=ticks)
    axes[1,1].set(title='(d) Serial training time: 3 replay seeds',ylabel='Minutes per 500-episode run (log scale)',yscale='log',xticks=range(5),xticklabels=ticks)
    axes[1,1].margins(y=.3)
    for ax in axes.flat: ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    for ext in ['png','pdf','svg']: fig.savefig(ROOT/f'connected_iqp_finite_sampling_final.{ext}',dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['timing','report']);a=p.parse_args()
    timing() if a.mode=='timing' else report()
