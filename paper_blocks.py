"""Three-qubit block primitives inherited by the connected policy; no standalone study."""
import math
import numpy as np
import torch

class BlockMajorityIQP(torch.nn.Module):
    def __init__(self,spec,seed):
        super().__init__(); self.n=spec['n']; assert self.n%6==3
        self.blocks=self.n//3; gen=torch.Generator().manual_seed(seed)
        self.weights=torch.nn.Parameter(torch.rand(self.n,4,generator=gen)*(2*math.pi))
        bias_noise=spec['sigma']*torch.randn(self.n,generator=gen)
        if spec.get('family')=='iqpblocksbalanced': bias_noise.zero_()
        self.bias=torch.nn.Parameter(math.pi/2+bias_noise)
        self.interactions=torch.nn.Parameter(spec['sigma']*torch.randn(self.blocks,4,generator=gen))
        bits=((torch.arange(8)[:,None]>>torch.arange(2,-1,-1))&1)
        z=(1-2*bits).float().T
        self.register_buffer('z',z)
        self.register_buffer('multi',torch.stack([z[0]*z[1],z[0]*z[2],z[1]*z[2],z.prod(0)]))
        self.register_buffer('counts',torch.nn.functional.one_hot(bits.sum(1),4).float())
        h=torch.tensor([[1.]])
        for _ in range(3): h=torch.kron(h,torch.tensor([[1.,1.],[1.,-1.]]))/math.sqrt(2)
        self.register_buffer('h',h.to(torch.complex64))
        self.supports=[]
        for start in range(0,self.n,3):
            self.supports.extend([(start,start+1),(start,start+2),(start+1,start+2),(start,start+1,start+2)])

    def weight_probs(self,x):
        local=x@self.weights.T+self.bias
        phase=local.reshape(*local.shape[:-1],self.blocks,3)@self.z+self.interactions@self.multi
        state=(torch.exp(-.5j*phase)/math.sqrt(8))@self.h
        counts=state.abs().square()@self.counts
        # Product of generating polynomials; length n+1 prevents wraparound.
        return torch.fft.irfft(torch.fft.rfft(counts,n=self.n+1).prod(-2),n=self.n+1)

    def forward(self,x):
        p=self.weight_probs(x)[...,:self.n//2+1].sum(-1)
        probs=torch.stack((p,1-p),-1).clamp_min(1e-8)
        return probs/probs.sum(-1,keepdim=True)
