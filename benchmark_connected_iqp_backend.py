"""Read-only model benchmark: chain-of-triples contraction versus statevectors.

Each configuration executes in a fresh process. Existing learning runs continue,
so timings are indicative CPU measurements, not an isolated-machine benchmark.
"""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/iqp-search-mpl'
import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent/'results'/'connected_backend_benchmark'


def worker(n, engine, batch):
    import numpy as np
    import torch
    import psutil
    from connected_majority_iqp import ConnectedMajorityIQP
    torch.set_num_threads(1)
    model = ConnectedMajorityIQP(seed=123, n=n)
    x = torch.tensor([.2, -.3, .4, -.5])
    if batch > 1:
        x = x.repeat(batch, 1)
    row = {'n': n, 'engine': engine, 'batch': batch,
           'torch_version': torch.__version__, 'numpy_version': np.__version__,
           'parameters': sum(p.numel() for p in model.parameters()),
           'timing_note': 'One thread per process; other training workers remain active. Setup excluded. No GPU.',
           'largest_transfer_tensor_bytes': batch*(n//3)*((n+1)//2+1)*16*8,
           'dense_state_complex128_bytes': 16*(2**n),
           'dense_state_complex64_bytes': 8*(2**n)}
    if engine == 'custom':
        @torch.no_grad()
        def fn():
            return model.weight_probs(x)
        repetitions = 30 if n <= 99 else 5
        row['numeric_precision'] = 'float32/complex64'
    else:
        import pennylane as qml
        row['pennylane_version'] = qml.__version__
        try:
            device = qml.device('lightning.qubit', wires=n, c_dtype=np.complex64) if engine == 'lightning.qubit.c64' else qml.device(engine, wires=n)
        except Exception as error:
            return {**row, 'unavailable': str(error)}
        angles = (x@model.weights.T+model.bias).detach().numpy()
        interactions = model.interactions.detach().flatten().numpy()
        bridges = model.bridges.detach().numpy()
        # Same rounded float32 gate angles as the original model, cast for PL.
        @qml.qnode(device, interface=None, diff_method=None)
        def circuit():
            for i in range(n): qml.Hadamard(i)
            for i, t in enumerate(angles): qml.RZ(float(t), i)
            for t, support in zip(interactions, model.supports):
                qml.MultiRZ(float(t), wires=support)
            for t, support in zip(bridges, model.bridge_supports):
                qml.MultiRZ(float(t), wires=support)
            for i in range(n): qml.Hadamard(i)
            return qml.state()
        weights = np.fromiter((i.bit_count() for i in range(2**n)), dtype=np.int32)
        def fn():
            state = np.asarray(circuit())
            return np.bincount(weights, weights=np.abs(state)**2, minlength=n+1)
        repetitions = 3 if n >= 21 else 10
        row['numeric_precision'] = ('complex64' if engine == 'lightning.qubit.c64' else 'complex128')+' statevector; float32-rounded input angles'
    process = psutil.Process(os.getpid())
    row['rss_before_evaluation_bytes'] = process.memory_info().rss
    result = fn()
    values = result.detach().numpy() if engine == 'custom' else result
    row['normalization_max_error'] = float(np.max(abs(values.sum(-1)-1)))
    row['min_probability'] = float(values.min())
    assert row['normalization_max_error'] < 1e-3 and row['min_probability'] > -1e-4
    if engine != 'custom':
        with torch.no_grad(): reference = model.weight_probs(x).numpy()
        row['max_abs_error_vs_custom'] = float(np.max(abs(reference-values)))
        assert row['max_abs_error_vs_custom'] < 3e-6
    wall, cpu = [], []
    for _ in range(repetitions):
        start=time.perf_counter(); start_cpu=time.process_time(); fn()
        cpu.append(time.process_time()-start_cpu); wall.append(time.perf_counter()-start)
    row['repetitions'] = repetitions
    row['forward_wall_ms_median'] = float(np.median(wall)*1000)
    row['forward_cpu_ms_median'] = float(np.median(cpu)*1000)
    row['forward_peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if engine == 'custom':
        wall, cpu = [], []
        for _ in range(5 if n <= 201 else 2):
            model.zero_grad(set_to_none=True)
            start=time.perf_counter(); start_cpu=time.process_time()
            value=model(x)[..., 0].mean(); value.backward()
            cpu.append(time.process_time()-start_cpu); wall.append(time.perf_counter()-start)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        row['forward_backward_wall_ms_median'] = float(np.median(wall)*1000)
        row['forward_backward_cpu_ms_median'] = float(np.median(cpu)*1000)
        row['forward_backward_peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        row['finite_gradients'] = True
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true')
    parser.add_argument('--n',type=int);parser.add_argument('--engine');parser.add_argument('--batch',type=int,default=1)
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.n,args.engine,args.batch)));return
    ROOT.mkdir(parents=True,exist_ok=True)
    jobs=[(n,e,1) for n in [9,15,21] for e in ['custom','default.qubit','lightning.qubit']]
    jobs += [(n,'custom',1) for n in [39,99,201,501,999]]
    jobs += [(n,'custom',32) for n in [39,99,201]]
    jobs += [(21,'lightning.qubit.c64',1)]
    rows=[]
    for n,engine,batch in jobs:
        path=ROOT/f'{engine}_n{n}_b{batch}.json'
        if path.exists(): row=json.loads(path.read_text())
        else:
            result=subprocess.run([sys.executable,__file__,'--worker','--n',str(n),'--engine',engine,'--batch',str(batch)],capture_output=True,text=True,timeout=150)
            if result.returncode: raise RuntimeError(result.stderr)
            row=json.loads(result.stdout.strip().splitlines()[-1])
            path.write_text(json.dumps(row,indent=2))
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k in ['n','engine','batch','forward_wall_ms_median','forward_backward_wall_ms_median','max_abs_error_vs_custom','unavailable']}),flush=True)
    (ROOT/'benchmark.json').write_text(json.dumps({'results':rows,'limitations':['Concurrent training on the same host: timings are indicative, not controlled performance claims.','Custom uses complex64 and PennyLane default statevectors use complex128.','Total peak process RSS includes imports and allocator caches; it is not the tensor-network workspace alone.','Large-n tests establish forward/backward execution and normalization checks, not CartPole convergence or a fullstate correctness comparison.']},indent=2))


if __name__=='__main__': main()
