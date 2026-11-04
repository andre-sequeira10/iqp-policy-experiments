"""Fixed K=8,32,64 finite-shot comparison; no model or seed selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/iqp-search-mpl')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import connected_iqp_spsa_k_study as study
from connected_iqp_finite_shot_test import ROOT as OLD, SHOTS
from connected_iqp_mlp_study import ROOT as BASE, TEST, solve
from analyze_majority_iqp_search import infer

ROOT = BASE.parent / 'finite_spsa_mlp_comparison'
KS = [8, 32, 64]


def references():
    lock = json.loads((BASE / 'development_selection.json').read_text())
    names = list(dict.fromkeys(list(lock['selected_by_width'].values()) + [lock['best_solve_mlp']]))
    means = {name: float(np.mean([json.loads((BASE / 'test' / f'{name}__{s}.json').read_text())['auc'] for s in TEST])) for name in names}
    return max(names, key=lambda n: means[n]), lock['best_auc_mlp'], means


def protocol():
    ROOT.mkdir(exist_ok=True)
    best, selected, means = references()
    p = {'ks': KS, 'log2_rule': 'ceil(log2(139))=8; K=8 and logarithmic K are the same condition, not separate replicates.',
         'seeds': TEST, 'episodes': 500, 'max_episode_return': 200, 'shots_per_IQP_evaluation': SHOTS,
         'model': 'Original connected 21-qubit majority IQP, 139 weights, unchanged neural embedding and original initialization.',
         'changes': 'Only K changes relative to the existing finite-shot experiment. Reuse K=8; run all 20 matched seeds for K=32 and K=64. No K selection or early elimination.',
         'sampling': 'Independent Born action-count simulation via Binomial(4096,p), including rollout and perturbed loss evaluations, with Jeffreys smoothing. Not a quantum-hardware experiment.',
         'MLP': 'Reuse standard REINFORCE/autodiff training and sampled environment actions. No artificial quantum measurement noise is imposed on classical probabilities.',
         'main_MLP': best, 'main_MLP_selection': 'Highest observed test AUC among previously locked MLP settings; explicitly retrospective, not an independently selected confirmatory baseline.',
         'secondary_MLP': selected, 'secondary_MLP_selection': 'Highest development AUC from the original MLP tuning study, chosen without these test returns.',
         'MLP_test_auc': means,
         'metrics': 'Normalized AUC=sum(500 training episode returns)/(500*200). First full rolling-100 mean>=195, assigned 501 if unreached. Updates freeze only at qualifying five-episode boundaries; all 500 episodes retained.',
         'uncertainty': 'Seed-bootstrap 95% intervals; paired comparisons use matching seed IDs. Pointwise curve intervals are not simultaneous bands. All seeds retained. Previously inspected comparison cohort, not fresh confirmatory testing.',
         'scope': 'Only K=8,32,64 appear in these reports. Previous high-K raw results are preserved, not deleted. No paper edits. Parallel execution times are not a controlled training-time benchmark.'}
    target = ROOT / 'protocol.json'
    if target.exists():
        assert json.loads(target.read_text()) == p
    else:
        target.write_text(json.dumps(p, indent=2))
    manifest = ROOT / 'reference_integrity_before.json'
    if not manifest.exists():
        files = [BASE / 'development_selection.json', OLD / 'protocol.json']
        for s in TEST:
            for name in set([best, selected, 'iqp_connected']):
                files.extend((BASE / 'test' / f'{name}__{s}').with_suffix(ext) for ext in ['.json', '.npz'])
            files.extend((OLD / 'test' / f'iqp_connected_shots4096_spsa8__{s}').with_suffix(ext) for ext in ['.json', '.npz'])
        manifest.write_text(json.dumps({str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, indent=2))
    return p


def run_one(k, seed):
    # Workers import their own module; set its output root explicitly in each.
    study.ROOT = ROOT
    return study.run(k, seed, 'test')


def run(jobs):
    protocol()
    from joblib import Parallel, delayed
    Parallel(n_jobs=jobs)(delayed(run_one)(k, s) for s in TEST for k in [32, 64])
    report()


def stem_for(name, seed):
    if name == 'k8':
        return OLD / 'test' / f'iqp_connected_shots4096_spsa8__{seed}'
    if name.startswith('k'):
        return ROOT / 'test' / f'k{int(name[1:]):03d}__{seed}'
    return BASE / 'test' / f'{name}__{seed}'


def load(name):
    rows, traces = [], []
    for s in TEST:
        stem = stem_for(name, s)
        r = json.loads(stem.with_suffix('.json').read_text())
        with np.load(stem.with_suffix('.npz')) as f:
            tr, ev = f['training_returns'], f['evaluation_returns']
        assert tr.shape == (500,) and ev.shape == (100,)
        assert np.isfinite(tr).all() and np.all((tr >= 1) & (tr <= 200))
        assert r['seed'] == s and abs(r['auc'] - tr.mean()/200) < 1e-12
        assert r['solve_episode'] == solve(tr)
        qualifying = [e for e in range(100, 501, 5) if tr[e-100:e].mean() >= 195]
        assert r['freeze_episode'] == (qualifying[0] if qualifying else None)
        if name.startswith('k'):
            k = int(name[1:]); calls = r['circuit_evaluations']
            assert calls['gradient'] == 2*k*r['optimized_transitions']
            assert calls['rollout'] == sum(tr) and calls['evaluation'] == sum(ev)
            assert r['training_shots'] == SHOTS*(calls['rollout'] + calls['gradient'])
            ref = json.loads((BASE / 'test' / f'iqp_connected__{s}.json').read_text())
            assert r['initial_hash'] == ref['initial_hash']
        rows.append(r); traces.append(tr)
    return rows, np.stack(traces)


def ci(values, indices):
    values = np.asarray(values)
    return {'mean': float(values.mean()), 'ci95': np.quantile(values[indices].mean(1), [.025, .975]).tolist()}


def report():
    p = protocol(); best, selected = p['main_MLP'], p['secondary_MLP']
    names = ['k8', 'k32', 'k64'] + list(dict.fromkeys([best, selected]))
    rows, curves, summaries = {}, {}, {}
    indices = np.random.default_rng(20261027).integers(0, len(TEST), (50000, len(TEST)))
    for name in names:
        rows[name], curves[name] = load(name)
        summaries[name] = {'seeds': len(TEST), 'solved': sum(r['solve_episode'] <= 500 for r in rows[name])}
        for metric in ['auc', 'solve_episode', 'heldout', 'final50']:
            summaries[name][metric] = ci([r[metric] for r in rows[name]], indices)
        if name.startswith('k'):
            summaries[name]['training_shots'] = ci([r['training_shots'] for r in rows[name]], indices)
            summaries[name]['shifted_evaluations_per_state_per_update'] = 2*int(name[1:])
        else:
            summaries[name]['parameters'] = rows[name][0]['parameters']
    paired = {}
    for reference in dict.fromkeys([best, selected]):
        paired[reference] = {}
        for k in KS:
            paired[reference][f'k{k}'] = {metric: infer([a[metric]-b[metric] for a,b in zip(rows[f'k{k}'], rows[reference])], 7182)
                                          for metric in ['auc', 'solve_episode']}
        # Six contrasts per reference: three K settings times two outcomes.
        tests = sorted([v for row in paired[reference].values() for v in row.values()], key=lambda v: v['two_sided_exact_sign_p'])
        last = 0
        for i, test in enumerate(tests):
            last = max(last, min(1., (len(tests)-i)*test['two_sided_exact_sign_p']))
            test['holm_p_within_reference'] = last
    out = {'protocol': p, 'summary': summaries, 'paired_IQP_minus_MLP': paired,
           'interpretation': 'AUC uses every unsmoothed return. Convergence is a restricted measure with unreached runs assigned 501. Main MLP is retrospectively best on this cohort; statistical contrasts are exploratory. Larger K increases shot budget; episode counts, not quantum-query costs, are matched.'}
    (ROOT / 'comparison.json').write_text(json.dumps(out, indent=2))
    before = json.loads((ROOT / 'reference_integrity_before.json').read_text())
    unchanged = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() == h for path,h in before.items()}
    assert all(unchanged.values())
    (ROOT / 'integrity.json').write_text(json.dumps({'all_reference_files_unchanged': True, 'checks': unchanged,
        'sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__), ROOT/'protocol.json', ROOT/'comparison.json'] + list((ROOT/'test').glob('*.json')) + list((ROOT/'test').glob('*.npz'))}}, indent=2))
    plot(curves, summaries, best, 'training_returns_auc', 'Highest observed comparison-cohort AUC; retrospective MLP reference')
    if selected != best:
        plot(curves, summaries, selected, 'training_returns_auc_selected_mlp', 'MLP chosen by development AUC before this comparison cohort')
    lines = ['Finite-shot connected IQP versus classical MLP', '20 matched seeds; 500 episodes; 200-step cap; IQP 4,096 shots.', 'K=8 is exactly ceil(log2(139)); it is one condition.', '', 'Model | normalized AUC [95% bootstrap CI] | restricted criterion episode | reached /20']
    for name in names:
        r = summaries[name]; a=r['auc']; lo,hi=a['ci95']
        lines.append(f"{name} | {a['mean']:.4f} [{lo:.4f},{hi:.4f}] | {r['solve_episode']['mean']:.1f} | {r['solved']}/20")
    lines += ['', out['interpretation'], 'Finite-shot results simulate Born action counts; no hardware noise is included. MLP uses standard exact probability/backprop access and stochastic environment actions. No manuscript changes.']
    (ROOT/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


def plot(curves, summary, mlp, filename, note, interim=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['k8', 'k32', 'k64', mlp]
    labels = [r'IQP: $K=8=\lceil\log_2 p\rceil$', r'IQP: $K=32$', r'IQP: $K=64$', f"MLP ({summary[mlp]['parameters']} parameters)"]
    colors = ['#d95f02', '#7570b3', '#0072b2', '#222222']
    rng = np.random.default_rng(71025)
    # Bootstrap complete seeds, preserving all time dependence within each run.
    ids = rng.integers(0, len(TEST), (5000, len(TEST)))
    weights = np.stack([np.bincount(row, minlength=len(TEST))/len(TEST) for row in ids])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
    for i,(name,label,color) in enumerate(zip(names,labels,colors)):
        tr = curves[name]
        smooth = np.stack([np.convolve(t, np.ones(20)/20, 'valid') for t in tr])
        for ax,x,v in [(axes[0], np.arange(20,501), smooth), (axes[1], np.arange(1,501), tr.cumsum(1)/100000)]:
            ax.plot(x, v.mean(0), color=color, label=label, linewidth=1.9)
            lo,hi = np.quantile(weights@v, [.025,.975], axis=0)
            ax.fill_between(x,lo,hi,color=color,alpha=.12)
        a = summary[name]['auc']; m=a['mean']; lo,hi=a['ci95']
        axes[2].bar(i,m,color=color,alpha=.7,width=.65)
        axes[2].scatter(i+np.linspace(-.19,.19,len(TEST)), tr.mean(1)/200, color=color, s=14, edgecolors='white', linewidths=.4, zorder=3)
        axes[2].errorbar(i,m,yerr=[[m-lo],[hi-m]],color='black',capsize=4,fmt='none',zorder=4)
        axes[2].text(i, min(1.025, hi+.025), f'{m:.3f}', ha='center',fontsize=10)
    axes[0].set(title='Training returns', xlabel='Training episode', ylabel='20-episode mean return', ylim=(0,205))
    axes[1].set(title='Accumulated reward', xlabel='Training episode', ylabel=r'$\sum_{e=1}^{t} R_e\, /\, (500\times200)$', ylim=(0,1.02))
    axes[2].set(title='Normalized training AUC', ylabel='AUC over all 500 episodes', ylim=(0,1.08), xticks=range(4), xticklabels=['K=8', 'K=32', 'K=64', 'MLP'])
    for ax in axes: ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=4, frameon=False)
    prefix = 'INTERIM: ' if interim else ''
    fig.suptitle(prefix+f'CartPole: connected 21-qubit majority IQP, p=139; 4,096 shots; {len(TEST)} matched seeds\n'+note+'\nBands/bars: seed-bootstrap 95% intervals; dots: individual seeds.',fontsize=11)
    for ext in ['png','svg','pdf']: fig.savefig(ROOT/f'{filename}.{ext}',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('mode',choices=['run','report','status']);parser.add_argument('--jobs',type=int,default=4)
    args = parser.parse_args()
    if args.mode == 'run': run(args.jobs)
    elif args.mode == 'report': report()
    else:
        for k in [32,64]:
            done = [s for s in TEST if (ROOT/'test'/f'k{k:03d}__{s}.json').exists()]
            print(f'K={k}: {len(done)}/20 complete; seeds={done}')
