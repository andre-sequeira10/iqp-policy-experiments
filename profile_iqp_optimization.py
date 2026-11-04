"""Profile and audit acceleration without changing scientific settings."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import json
import time
from pathlib import Path
import numpy as np
import torch
from connected_majority_iqp import ConnectedMajorityIQP
from optimized_connected_iqp import OptimizedConnectedMajorityIQP
from majority_iqp_model_search import MLP

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent/'results'/'iqp_simulator_optimization'
ROOT.mkdir(exist_ok=True)


def benchmark():
    reference = ConnectedMajorityIQP(seed=2001)
    start = time.perf_counter()
    fast = OptimizedConnectedMajorityIQP(seed=2001)
    fast.warmup()
    setup = time.perf_counter()-start
    mlp = MLP(seed=2001, hidden=19)
    gen = torch.Generator().manual_seed(442)
    checks, timings = [], []
    for batch in [None, 1, 5, 32, 200, 1000]:
        shape = (4,) if batch is None else (batch,4)
        x = torch.randn(shape, generator=gen)
        outputs = [m(x) for m in [reference, fast]]
        coeff = torch.randn(outputs[0].shape, generator=gen)
        grads = [torch.autograd.grad((y*coeff).sum(),tuple(m.parameters())) for y,m in zip(outputs,[reference,fast])]
        error = float((outputs[0]-outputs[1]).abs().max().detach())
        grad_error = max(float((a-b).abs().max()) for a,b in zip(*grads))
        checks.append({'batch':batch, 'probability_error':error, 'gradient_error':grad_error,
                       'probabilities_bitwise_equal':torch.equal(*outputs),
                       'gradients_bitwise_equal':all(torch.equal(a,b) for a,b in zip(*grads))})
        assert error < 2e-6 and grad_error < 2e-5, checks[-1]
        for label, model in [('reference',reference),('jit',fast),('mlp',mlp)]:
            for backward in [False,True]:
                reps = 100 if batch is None else (10 if batch <= 200 else 3)
                elapsed = []
                for _ in range(3):
                    model.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(backward):
                        y=model(x)
                        if backward:y[...,0].sum().backward()
                for _ in range(reps):
                    model.zero_grad(set_to_none=True)
                    start=time.perf_counter()
                    with torch.set_grad_enabled(backward):
                        y=model(x)
                        if backward:y[...,0].sum().backward()
                    elapsed.append(time.perf_counter()-start)
                timings.append({'model':label,'batch':batch,'backward':backward,'median_ms':1000*float(np.median(elapsed))})
    for batch in [None,200]:
        x=torch.randn((4,) if batch is None else (batch,4),generator=gen)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as prof:
            for _ in range(5):
                reference.zero_grad(set_to_none=True)
                if batch is None:
                    with torch.no_grad():reference(x)
                else: reference(x)[...,0].sum().backward()
        (ROOT/f'profile_batch_{batch}.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=25))
    result={'setup_seconds':setup,'checks':checks,'timings':timings}
    (ROOT/'microbenchmark.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':benchmark()
