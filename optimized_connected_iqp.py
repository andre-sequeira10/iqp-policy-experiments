"""Execution-only acceleration for the existing chain-of-triples IQP policy.

Parameter names, initialization, precision, circuit, decoder, and arithmetic
order are retained. The reference implementation remains unchanged.
"""
import math
import torch
from connected_majority_iqp import ConnectedMajorityIQP


class _Contraction(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.blocks = n // 3

    def forward(self, x, weights, bias, interactions, bridges, z, multi, h,
                virtual_signs, count_phases):
        local = x @ weights.T + bias
        phase = local.reshape(*local.shape[:-1], self.blocks, 3) @ z + interactions @ multi
        state = torch.exp(-.5j * phase) / math.sqrt(8)
        right = torch.stack((torch.cos(bridges / 2), -1j * torch.sin(bridges / 2)), -1)
        right = torch.cat((right, torch.tensor([[1., 0.]], dtype=right.dtype, device=x.device)), 0)
        amplitudes = ((state[..., None, None, :] * virtual_signs) @ h).movedim(-1, -3)
        amplitudes = amplitudes * right[:, None, None, :]
        a = amplitudes.flatten(-2)
        outer = a[..., :, :, None] * a.conj()[..., :, None, :]
        transfer = count_phases @ outer.flatten(-2)
        nf = count_phases.shape[0]
        transfer = transfer.reshape(*local.shape[:-1], self.blocks, nf, 2, 2, 2, 2).transpose(-3, -2)
        transfer = transfer.reshape(*local.shape[:-1], self.blocks, nf, 4, 4)
        environment = transfer[..., 0, :, 0, :]
        for j in range(1, self.blocks):
            environment = (environment.unsqueeze(-2) @ transfer[..., j, :, :, :]).squeeze(-2)
        return torch.fft.irfft(environment[..., 0], n=self.n + 1)


class OptimizedConnectedMajorityIQP(ConnectedMajorityIQP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kernels = {}

    def _arguments(self, x):
        return (x, self.weights, self.bias, self.interactions, self.bridges,
                self.z, self.multi, self.h, self.virtual_signs, self.count_phases)

    def weight_probs(self, x):
        # Keep the reference autograd graph: tracing its backward can change
        # roundoff after many optimizer steps, even when isolated tests agree.
        if torch.is_grad_enabled():
            return super().weight_probs(x)
        # Traces retain dynamic batch lengths; rank/device/dtype need distinct graphs.
        key = (x.ndim, x.device.type, x.device.index, x.dtype)
        if key not in self._kernels:
            self._kernels[key] = torch.jit.trace(_Contraction(self.n), self._arguments(x), check_trace=True)
        return self._kernels[key](*self._arguments(x))

    def warmup(self):
        x = self.weights.new_zeros(4)
        with torch.no_grad():
            self.weight_probs(x)
            self.weight_probs(x.repeat(2, 1))
