"""Wide IQP policy with independent entangled triples and global majority.

The circuit is simulated exactly by convolving each triple's Hamming-weight
distribution. This preserves the Born majority policy; it is not a surrogate.
"""
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

def verify():
    import numpy as np
    import pennylane as qml
    checks=[]
    for n,family,sigma in [(9,'iqpblocks',.15),(15,'iqpblocks',.15),
                           (9,'iqpblocksbalanced',.15),(15,'iqpblocksbalanced',.15),
                           (21,'iqpblocksbalanced',.5)]:
        p=BlockMajorityIQP({'n':n,'sigma':sigma,'family':family},17)
        x=torch.tensor([.2,-.3,.4,-.5]); dev=qml.device('default.qubit',wires=n)
        @qml.qnode(dev)
        def circuit():
            for i in range(n): qml.Hadamard(i)
            angles=(x@p.weights.T+p.bias).detach().numpy()
            for i in range(n): qml.RZ(float(angles[i]),i)
            for a,s in zip(p.interactions.detach().flatten().numpy(),p.supports):
                qml.MultiRZ(float(a),wires=list(s))
            for i in range(n): qml.Hadamard(i)
            return qml.probs(wires=range(n))
        exact=np.asarray(circuit()); counts=np.zeros(n+1)
        for i,v in enumerate(exact): counts[i.bit_count()]+=v
        err=np.max(abs(counts-p.weight_probs(x).detach().numpy()))
        assert err<3e-6,err
        p.zero_grad(); p(x)[0].backward(); analytic=float(p.interactions.grad[0,0])
        with torch.no_grad():
            orig=p.interactions[0,0].item(); eps=.002
            p.interactions[0,0]=orig+eps; plus=p(x)[0].item()
            p.interactions[0,0]=orig-eps; minus=p(x)[0].item(); p.interactions[0,0]=orig
        assert abs(analytic-(plus-minus)/(2*eps))<3e-4
        print('PASS',family,n,'max weight-distribution error',err,flush=True)
        checks.append({'family':family,'qubits':n,'sigma':sigma,
                       'max_weight_distribution_error':float(err),
                       'interaction_gradient_error':abs(analytic-(plus-minus)/(2*eps))})
    import json
    from pathlib import Path
    target=Path(__file__).resolve().parent/'results'/'majority_iqp_model_search'
    target.mkdir(parents=True,exist_ok=True)
    (target/'block_circuit_checks.json').write_text(json.dumps(checks,indent=2))

if __name__=='__main__':
    torch.set_num_threads(1); verify()
