"""Exact differentiable majority probabilities for a connected chain of triples.

An RZZ bridge has operator Schmidt rank two. The ket bond dimension is two
and the doubled Born-probability bond dimension is four. Count-generating
functions contract these bonds without treating blocks as independent.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/iqp-search-mpl')
import math
import time
import numpy as np
import torch
from majority_iqp_block_model import BlockMajorityIQP


class ConnectedMajorityIQP(BlockMajorityIQP):
    def __init__(self,seed,n=21,sigma=.5,bridge_sigma=.5):
        super().__init__({'n':n,'sigma':sigma,'family':'iqpblocksbalanced'},seed)
        generator=torch.Generator().manual_seed(seed+900000)
        self.bridges=torch.nn.Parameter(bridge_sigma*torch.randn(self.blocks-1,generator=generator))
        self.bridge_supports=[(3*j+2,3*j+3) for j in range(self.blocks-1)]
        signs=torch.stack((torch.ones(8),self.z[0]))[:,None,:]*torch.stack((torch.ones(8),self.z[2]))[None,:,:]
        self.register_buffer('virtual_signs',signs.to(torch.complex64))
        frequencies=torch.arange((n+1)//2+1,dtype=torch.float32)
        count=torch.tensor([v.bit_count() for v in range(8)],dtype=torch.float32)
        self.register_buffer('count_phases',torch.exp(-2j*math.pi*frequencies[:,None]*count[None,:]/(n+1)))

    def weight_probs(self,x):
        local=x@self.weights.T+self.bias
        phase=local.reshape(*local.shape[:-1],self.blocks,3)@self.z+self.interactions@self.multi
        state=torch.exp(-.5j*phase)/math.sqrt(8)
        # RZZ(t)=cos(t/2) I⊗I - i sin(t/2) Z⊗Z. Assign each bridge's
        # coefficient to the outgoing/right bond of the preceding triple.
        right=torch.stack((torch.cos(self.bridges/2),-1j*torch.sin(self.bridges/2)),-1)
        right=torch.cat((right,torch.tensor([[1.,0.]],dtype=right.dtype,device=x.device)),0)
        amplitudes=((state[...,None,None,:]*self.virtual_signs)@self.h).movedim(-1,-3)
        amplitudes=amplitudes*right[:,None,None,:]
        a=amplitudes.flatten(-2)  # physical output, combined (left,right) ket bond
        outer=a[..., :, :,None]*a.conj()[..., :,None,:]
        transfer=self.count_phases@outer.flatten(-2)
        batch_shape=local.shape[:-1]; nf=self.count_phases.shape[0]
        # Reshape (left,right,left*,right*) to (left,left*,right,right*).
        transfer=transfer.reshape(*batch_shape,self.blocks,nf,2,2,2,2).transpose(-3,-2)
        transfer=transfer.reshape(*batch_shape,self.blocks,nf,4,4)
        environment=transfer[...,0,:,0,:]
        for j in range(1,self.blocks):
            environment=(environment.unsqueeze(-2)@transfer[...,j,:,:,:]).squeeze(-2)
        generating_function=environment[...,0]
        return torch.fft.irfft(generating_function,n=self.n+1)


def verify():
    import json
    from pathlib import Path
    import pennylane as qml
    torch.set_num_threads(1); records=[]
    for n in [9,21]:
        model=ConnectedMajorityIQP(seed=123,n=n)
        x=torch.tensor([.2,-.3,.4,-.5])
        device=qml.device('default.qubit',wires=n)
        @qml.qnode(device)
        def reference():
            for i in range(n): qml.Hadamard(i)
            angles=(x@model.weights.T+model.bias).detach().numpy()
            for i,t in enumerate(angles): qml.RZ(float(t),i)
            for t,s in zip(model.interactions.detach().flatten().numpy(),model.supports):
                qml.MultiRZ(float(t),wires=list(s))
            for t,s in zip(model.bridges.detach().numpy(),model.bridge_supports):
                qml.MultiRZ(float(t),wires=list(s))
            for i in range(n): qml.Hadamard(i)
            return qml.state()
        state=np.asarray(reference()); full=np.abs(state)**2
        weights=np.fromiter((v.bit_count() for v in range(2**n)),dtype=np.int32)
        expected=np.bincount(weights,weights=full,minlength=n+1)
        actual=model.weight_probs(x).detach().numpy()
        error=float(np.max(abs(expected-actual))); assert error<3e-6,error
        cut=state.reshape(8,-1); rho=cut@cut.conj().T
        purity=float(np.trace(rho@rho).real); assert purity<1-1e-5,purity
        predicted_purity=1-.5*math.sin(float(model.bridges[0].detach()))**2
        assert abs(purity-predicted_purity)<1e-10,(purity,predicted_purity)
        model.zero_grad(); model(x)[0].backward(); gradient_errors={}
        for label,parameter,index in [('weights',model.weights,(0,0)),('bias',model.bias,(4,)),
                                      ('pair',model.interactions,(0,0)),('triple',model.interactions,(0,3)),
                                      ('bridge',model.bridges,(0,))]:
            derivative=parameter.grad[index].item()
            with torch.no_grad():
                original=parameter[index].item(); eps=.002
                parameter[index]=original+eps; plus=model(x)[0].item()
                parameter[index]=original-eps; minus=model(x)[0].item(); parameter[index]=original
            err=abs(derivative-(plus-minus)/(2*eps)); assert err<4e-4,(label,err)
            gradient_errors[label]=err
        with torch.no_grad():
            old=model.bridges.clone(); model.bridges.zero_()
            base=BlockMajorityIQP({'n':n,'sigma':.5,'family':'iqpblocksbalanced'},123)
            zero_bridge_error=float((model.weight_probs(x)-base.weight_probs(x)).abs().max())
            model.bridges.copy_(old)
        assert zero_bridge_error<3e-6
        xs=torch.stack((x,-x)); acts=torch.tensor([0,1])
        vector=-torch.distributions.Categorical(probs=model(xs)).log_prob(acts).mean()
        gv=torch.autograd.grad(vector,tuple(model.parameters()))
        scalar=-torch.stack([torch.distributions.Categorical(probs=model(s)).log_prob(a) for s,a in zip(xs,acts)]).mean()
        gs=torch.autograd.grad(scalar,tuple(model.parameters()))
        batch_error=max(float((v-s).abs().max()) for v,s in zip(gv,gs)); assert batch_error<1e-5
        started=time.perf_counter()
        with torch.no_grad():
            for _ in range(100): model(x)
        record={'qubits':n,'parameters':sum(p.numel() for p in model.parameters()),
                'max_probability_error':error,'first_block_reduced_purity':purity,
                'predicted_first_cut_purity':predicted_purity,
                'finite_difference_errors':gradient_errors,'zero_bridge_error':zero_bridge_error,
                'batch_gradient_error':batch_error,'milliseconds_per_inference':(time.perf_counter()-started)*10}
        records.append(record); print(json.dumps(record),flush=True)
    target=Path(__file__).resolve().parent/'results'/'connected_iqp_mlp_study'
    target.mkdir(parents=True,exist_ok=True)
    (target/'circuit_verification.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__': verify()
