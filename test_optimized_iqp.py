"""Audit execution-only optimizations with changing inputs and parameters."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
import json
from pathlib import Path
import numpy as np
import torch
from connected_majority_iqp import ConnectedMajorityIQP
from optimized_connected_iqp import OptimizedConnectedMajorityIQP

torch.set_num_threads(1)
checks = []
gen = torch.Generator().manual_seed(9271)
for n in [3, 9, 21, 45]:
    reference = ConnectedMajorityIQP(seed=17, n=n)
    fast = OptimizedConnectedMajorityIQP(seed=17, n=n)
    for update in range(2):
        for shape in [(4,), (1,4), (7,4), (2,3,4)]:
            # A strided, non-contiguous input; batch lengths reuse the same trace.
            base = torch.randn((*shape[:-1], 8), generator=gen)
            x1 = base[...,::2].detach().requires_grad_()
            x2 = base[...,::2].detach().requires_grad_()
            a, b = reference(x1), fast(x2)
            coeff = torch.randn(a.shape, generator=gen)
            ga = torch.autograd.grad((a*coeff).sum(), (x1,*reference.parameters()))
            gb = torch.autograd.grad((b*coeff).sum(), (x2,*fast.parameters()))
            rec = {'qubits': n, 'update': update, 'shape': shape,
                   'probabilities_equal': torch.equal(a,b),
                   'input_and_parameter_gradients_equal': all(torch.equal(u,v) for u,v in zip(ga,gb))}
            assert rec['probabilities_equal'] and rec['input_and_parameter_gradients_equal'], rec
            with torch.no_grad():
                assert torch.equal(reference(x1),fast(x2))
            checks.append(rec)
        # Mutate every parameter after caching, then reload the checkpoint.
        with torch.no_grad():
            for param in reference.parameters():
                param.add_(torch.randn(param.shape, generator=gen)*.07)
        fast.load_state_dict(reference.state_dict())
    # Graphs are not serialized as stale parameter copies.
    assert set(fast.state_dict()) == set(reference.state_dict())
target = Path(__file__).resolve().parent/'results/iqp_simulator_optimization/extended_validation.json'
target.write_text(json.dumps(checks, indent=2))
print(f'PASS: {len(checks)} output/input-gradient/parameter-gradient checks, including cached parameter updates and noncontiguous inputs.')
