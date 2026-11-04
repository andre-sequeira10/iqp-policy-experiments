"""Serial, matched full-training benchmark. Existing experiments are read-only."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
import contextlib
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from connected_iqp_mlp_study import policy, specs, solve
from majority_iqp_model_search import make_agent
from optimized_connected_iqp import OptimizedConnectedMajorityIQP
from optimized_iqp_training import OptimizedStrictWindowREINFORCE

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/iqp_simulator_optimization/full_training_inference_only'
SEEDS = [2001, 2002, 2003]
METHODS = ['reference', 'compiled', 'compiled_frozen', 'mlp', 'mlp_frozen']


def run(name, seed):
    stem = OUT/f'{name}__{seed}'
    if stem.with_suffix('.json').exists():
        saved = json.loads(stem.with_suffix('.json').read_text())
        assert all(saved[k] for k in ['returns_bitwise_equal', 'final_state_bitwise_equal',
                                      'freeze_equal', 'updates_equal'])
        return saved
    ref_name = 'mlp_h19_lr0p03' if name.startswith('mlp') else 'iqp_connected'
    ref = ROOT/'results/connected_iqp_mlp_study/test'/f'{ref_name}__{seed}'
    old = torch.load(ref.with_suffix('.pt'), weights_only=False)
    if name.startswith('compiled'):
        model = OptimizedConnectedMajorityIQP(seed=seed, n=21, sigma=.5, bridge_sigma=.5)
        setup = time.perf_counter()
        model.warmup()
        compilation_seconds = time.perf_counter()-setup
    else:
        model = policy(specs()[ref_name], seed)
        compilation_seconds = 0.
    assert all(torch.equal(v, old['initial'][k]) for k, v in model.named_parameters())
    agent = make_agent(model, name.startswith('mlp'), strict=True)
    if name.endswith('_frozen'):
        # Same inherited train method and constructor; only _run_episode changes.
        agent.__class__ = OptimizedStrictWindowREINFORCE
    torch.manual_seed(seed)
    np.random.seed(seed)
    agent.env.reset(seed=seed)
    agent.env.action_space.seed(seed)
    print(f'START {name} seed={seed}', flush=True)
    with stem.with_suffix('.log').open('w') as log, contextlib.redirect_stdout(log):
        t0, c0 = time.perf_counter(), time.process_time()
        tr = np.asarray(agent.train(500, 50))
        wall, cpu = time.perf_counter()-t0, time.process_time()-c0
    agent.env.close()
    with np.load(ref.with_suffix('.npz')) as f:
        returns_equal = np.array_equal(tr, f['training_returns'])
    state_equal = all(torch.equal(v, old['final'][k]) for k, v in model.state_dict().items())
    max_error = max(float((v-old['final'][k]).abs().max()) for k,v in model.state_dict().items())
    result = {'method': name, 'seed': seed, 'training_seconds': wall,
              'cpu_seconds': cpu, 'compilation_seconds': compilation_seconds,
              'returns_bitwise_equal': returns_equal,
              'final_state_bitwise_equal': state_equal, 'max_state_error': max_error,
              'freeze_episode': agent.freeze_episode, 'updates': agent.optimizer_updates,
              'auc': float(tr.mean()/200), 'criterion_episode': solve(tr),
              'freeze_equal': agent.freeze_episode == old['record']['freeze_episode'],
              'updates_equal': agent.optimizer_updates == old['record']['updates']}
    np.savez_compressed(stem.with_suffix('.npz'), training_returns=tr)
    stem.with_suffix('.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    assert all(result[k] for k in ['returns_bitwise_equal', 'final_state_bitwise_equal',
                                   'freeze_equal', 'updates_equal']), result
    return result


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'protocol.json').write_text(json.dumps({
        'seeds': SEEDS, 'methods': METHODS, 'episodes': 500,
        'torch': torch.__version__, 'platform': platform.platform(),
        'threads': 1, 'scope': 'Training only; compile warmup reported separately. First four methods run serially in rotated order across seeds; MLP frozen-work elimination is a separate three-seed follow-up. Shared desktop, not an exclusive-host benchmark. No changes to circuit, decoder, precision, initialization, loss, optimizer, batching, random streams, freeze rule or total episodes.'}, indent=2))
    rows = []
    core = METHODS[:4]
    for i, seed in enumerate(SEEDS):
        for name in core[i:]+core[:i]:
            rows.append(run(name, seed))
    rows.extend(run('mlp_frozen', seed) for seed in SEEDS)
    summary = {name: {'mean_seconds': float(np.mean([r['training_seconds'] for r in rows if r['method']==name])),
                      'per_seed_seconds': [r['training_seconds'] for r in rows if r['method']==name]}
               for name in METHODS}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    files = [ROOT/n for n in ['optimized_connected_iqp.py', 'optimized_iqp_training.py',
                             'benchmark_optimized_iqp_training.py', 'test_optimized_iqp.py']]
    files += sorted(OUT.glob('*.json')) + sorted(OUT.glob('*.npz'))
    files = [p for p in files if p.name != 'manifest.json']
    (OUT/'manifest.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in files}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
