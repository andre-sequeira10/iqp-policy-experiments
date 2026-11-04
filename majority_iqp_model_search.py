"""Prospective CartPole architecture search; no manuscript or old result edits.

Neural features remain independent Uniform[0, 2pi) trainable affine weights.
Only commuting circuit block, its initialization, and qubit count vary.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/iqp-search-mpl')
import argparse
import contextlib
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from torch.distributions import Categorical
from qasp.rl.torchrl import REINFORCE

ROOT = Path(__file__).resolve().parent / 'results' / 'majority_iqp_model_search'
SCALE = torch.tensor([2.4, 3., .2095, 3.5])
DEVELOPMENT = [211, 223, 227, 229]
VALIDATION = [401,409,419,421,431,433,439,443,449,457,
              461,463,467,479,487,491,499,503,509,521]
torch.set_num_threads(1)

def config(name):
    if name in ('mlp', 'mlp19'):
        return {'name': name, 'hidden': 9 if name=='mlp' else 19}
    family, n, scale = name.split('_')
    return {'name': name, 'n': int(n), 'sigma': float(scale), 'family': family}

class MajorityIQP(torch.nn.Module):
    def __init__(self, spec, seed):
        super().__init__()
        n = spec['n']; self.n = n
        self.edges = sorted({tuple(sorted((i,(i+1)%n))) for i in range(n)})
        self.triples = sorted({tuple(sorted((i,(i+1)%n,(i+2)%n))) for i in range(n)})
        if spec.get('family') == 'iqp3body': self.edges=[]
        if spec.get('family') == 'iqp2body': self.triples=[]
        gen = torch.Generator().manual_seed(seed)
        self.weights = torch.nn.Parameter(torch.rand(n,4,generator=gen)*(2*math.pi))
        # A balanced, feature-agnostic operating point for the local rotations.
        self.bias = torch.nn.Parameter(math.pi/2 + spec['sigma']*torch.randn(n,generator=gen))
        self.interactions = torch.nn.Parameter(spec['sigma']*torch.randn(
            len(self.edges)+len(self.triples),generator=gen))
        bits = ((torch.arange(2**n)[:,None] >> torch.arange(n-1,-1,-1)) & 1)
        z = (1-2*bits).float().T
        self.register_buffer('z',z)
        supports = self.edges+self.triples
        self.register_buffer('multi',torch.stack([z[list(t)].prod(0) for t in supports]))
        self.register_buffer('majority0',bits.sum(1)<=n//2)
        h = torch.tensor([[1.]])
        for _ in range(n): h = torch.kron(h,torch.tensor([[1.,1.],[1.,-1.]]))/math.sqrt(2)
        self.register_buffer('h',h.to(torch.complex64))

    def basis_probs(self,x):
        phase = (x @ self.weights.T+self.bias) @ self.z + self.interactions @ self.multi
        state = torch.exp(-.5j*phase)/math.sqrt(2**self.n)
        return (state @ self.h).abs().square()

    def forward(self,x):
        p = self.basis_probs(x)[...,self.majority0].sum(-1)
        probs = torch.stack((p,1-p),-1).clamp_min(1e-8)
        return probs/probs.sum(-1,keepdim=True)

class MLP(torch.nn.Module):
    def __init__(self,seed,hidden=9):
        super().__init__(); torch.manual_seed(seed)
        self.network=torch.nn.Sequential(torch.nn.Linear(4,hidden),torch.nn.Tanh(),torch.nn.Linear(hidden,2))
    def forward(self,x): return self.network(x)

def make_policy(name,seed):
    if name in ('mlp','mlp19'): return MLP(seed,config(name)['hidden'])
    spec=config(name)
    if spec['family'].startswith('iqpblocks'):
        from majority_iqp_block_model import BlockMajorityIQP
        return BlockMajorityIQP(spec,seed)
    return MajorityIQP(spec,seed)

class BatchedREINFORCE(REINFORCE):
    """Same QASP algorithm, with trajectory log-probs recomputed in one batch.

    Parameters do not change within a trajectory. This preserves the loss and
    derivatives, while avoiding one autograd graph per environment step.
    """
    def _run_episode(self):
        state,_=self.env.reset(); states=[]; actions=[]; rewards=[]
        with torch.no_grad():
            for _ in range(self.max_steps):
                x=self._preprocess_state(torch.as_tensor(state,dtype=torch.float32))
                out=self.policy(x)
                dist=Categorical(logits=out) if self.logits else Categorical(probs=out)
                action=dist.sample(); states.append(x); actions.append(action)
                state,reward,done,truncated,_=self.env.step(action.item())
                rewards.append(float(reward))
                if done or truncated: break
        out=self.policy(torch.stack(states))
        dist=Categorical(logits=out) if self.logits else Categorical(probs=out)
        return states,dist.log_prob(torch.stack(actions)),rewards,dist.entropy(),sum(rewards)

class StrictWindowREINFORCE(BatchedREINFORCE):
    """QASP's CartPole/EMA objective, with a full-window freezing safeguard.

    This experiment-specific implementation supports the EMA baseline used
    throughout the comparison. Optimizer, rollout, loss and batch semantics
    are unchanged. Only the prerequisite of 100 completed episodes is new.
    """
    def train(self,num_episodes=500,log_every=50):
        assert self.baseline_type=='ema' and self._is_cartpole()
        history=[]; logps=[]; advantages=[]; entropies=[]
        self.freeze_episode=None; self.optimizer_updates=0

        def update():
            logp=torch.cat(logps); adv=torch.cat(advantages); entropy=torch.cat(entropies)
            if self.normalize_returns and len(adv)>1:
                adv=(adv-adv.mean())/(adv.std()+1e-8)
            loss=-(adv.detach()*logp).mean()
            if self.entropy_coef>0: loss=loss-self.entropy_coef*entropy.mean()
            if self.freeze_episode is None:
                if len(history)>=100 and np.mean(history[-100:])>=195:
                    self.freeze_episode=len(history)
                    print(f'Full rolling-100 criterion reached at episode {len(history)}; freezing updates.',flush=True)
                else:
                    self._optimise_loss(loss); self.optimizer_updates+=1
            logps.clear(); advantages.clear(); entropies.clear()

        for episode in range(1,num_episodes+1):
            states,logp,rewards,entropy,total=self._run_episode()
            history.append(total); disc=self._discount(rewards)
            adv=disc-self.baseline_value
            self._update_baseline(disc.mean().item())
            logps.append(logp); advantages.append(adv); entropies.append(entropy)
            if len(logps)==self.batch_episodes: update()
            if episode%log_every==0:
                print(f'Episode {episode:04d} | avg(return_last20) = {np.mean(history[-20:]):.1f} | baseline = ema={self.baseline_value:.1f}',flush=True)
        if logps: update()
        return history

def make_agent(policy,logits,strict=False):
    trainer=StrictWindowREINFORCE if strict else BatchedREINFORCE
    return trainer(policy=policy,env_name='CartPole-v1',gamma=.99,lr=.03,
        max_steps=200,batch_episodes=5,logits=logits,normalize_mode='clip',
        normalize_returns=True,entropy_coef=.005,baseline_type='ema',
        baseline_momentum=.05,max_grad_norm=1.)

def solve_episode(returns):
    if len(returns)<100: return 501
    hits=np.flatnonzero(np.convolve(returns,np.ones(100)/100,mode='valid')>=195)
    return int(hits[0]+100) if len(hits) else 501

def evaluate(policy,seed,logits,count):
    env=gym.make('CartPole-v1'); returns=[]
    with torch.no_grad():
        for ep in range(count):
            state,_=env.reset(seed=1_000_000+10_000*seed+ep)
            gen=torch.Generator().manual_seed(2_000_000+10_000*seed+ep)
            total=0
            for _ in range(200):
                x=(torch.as_tensor(state,dtype=torch.float32)/SCALE).clamp(-3,3)
                out=policy(x); p=torch.softmax(out,-1) if logits else out
                action=torch.multinomial(p,1,generator=gen).item()
                state,r,done,truncated,_=env.step(action); total+=r
                if done or truncated: break
            returns.append(total)
    env.close(); return np.asarray(returns)

def run(name,seed,phase,episodes=500):
    folder=ROOT/phase; folder.mkdir(parents=True,exist_ok=True)
    stem=folder/f'{name.replace(".","p")}__{seed}'; metadata=stem.with_suffix('.json')
    if metadata.exists(): return json.loads(metadata.read_text())
    started=time.time(); spec=config(name)
    policy=make_policy(name,seed)
    initial={k:v.detach().clone() for k,v in policy.named_parameters()}
    strict='strict' in phase
    agent=make_agent(policy,name in ('mlp','mlp19'),strict=strict)
    torch.manual_seed(seed); np.random.seed(seed)
    agent.env.reset(seed=seed); agent.env.action_space.seed(seed)
    with stem.with_suffix('.log').open('w') as f,contextlib.redirect_stdout(f):
        returns=np.asarray(agent.train(num_episodes=episodes,log_every=50))
    heldout=evaluate(policy,seed,name in ('mlp','mlp19'),100 if phase.startswith('validation') else 20)
    agent.env.close()
    record={'model':spec,'seed':seed,'phase':phase,'episodes':episodes,
        'auc':float(returns.mean()/200),'restricted_solve_episode':solve_episode(returns),
        'final50':float(returns[-50:].mean()),'heldout':float(heldout.mean()),
        'parameters':sum(p.numel() for p in policy.parameters()),'seconds':time.time()-started,
        'freeze_protocol':'full_100_episodes' if strict else 'legacy_available_window',
        'optimizer_freeze_episode':getattr(agent,'freeze_episode',None),
        'optimizer_updates':getattr(agent,'optimizer_updates',None),
        'initial_sha256':hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()}
    np.savez_compressed(stem.with_suffix('.npz'),training_returns=returns,evaluation_returns=heldout)
    torch.save({'initial':initial,'final':policy.state_dict(),'record':record},stem.with_suffix('.pt'))
    metadata.write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True); return record

def preflight():
    import pennylane as qml
    for name in ['iqp_3_0.15','iqp_5_0.15','iqp_7_0.15','iqp_9_0.15',
                 'iqp3body_7_0.15','iqp2body_7_0.15']:
        p=MajorityIQP(config(name),17); n=p.n
        x=torch.tensor([.2,-.3,.4,-.5])
        dev=qml.device('default.qubit',wires=n)
        @qml.qnode(dev)
        def circuit():
            for i in range(n): qml.Hadamard(i)
            angles=(x@p.weights.T+p.bias).detach().numpy()
            for i in range(n): qml.RZ(float(angles[i]),i)
            for a,s in zip(p.interactions.detach().numpy(),p.edges+p.triples):
                qml.MultiRZ(float(a),wires=list(s))
            for i in range(n): qml.Hadamard(i)
            return qml.probs(wires=range(n))
        err=float(np.max(np.abs(p.basis_probs(x).detach().numpy()-circuit())))
        assert err<2e-6,err
        p.zero_grad(); p(x)[0].backward(); analytic=float(p.interactions.grad[0])
        with torch.no_grad():
            orig=p.interactions[0].item(); eps=.002
            p.interactions[0]=orig+eps; plus=p(x)[0].item()
            p.interactions[0]=orig-eps; minus=p(x)[0].item(); p.interactions[0]=orig
        assert abs(analytic-(plus-minus)/(2*eps))<2e-4
        states=torch.randn(20,4); acts=torch.arange(20)%2
        loss1=-Categorical(probs=p(states)).log_prob(acts).mean()
        g1=torch.autograd.grad(loss1,tuple(p.parameters()))
        loss2=-torch.stack([Categorical(probs=p(s)).log_prob(a) for s,a in zip(states,acts)]).mean()
        g2=torch.autograd.grad(loss2,tuple(p.parameters()))
        assert all(torch.allclose(a,b,atol=2e-5,rtol=2e-4) for a,b in zip(g1,g2))
        print('PASS',name,'PennyLane max probability error',err,flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['preflight','run','summary'])
    parser.add_argument('--models',nargs='+',default=['iqp_3_0.15','iqp_5_0.15','iqp_7_0.15'])
    parser.add_argument('--seeds',nargs='+',type=int,default=DEVELOPMENT[:2])
    parser.add_argument('--phase',default='development'); parser.add_argument('--jobs',type=int,default=3)
    parser.add_argument('--episodes',type=int,default=500); args=parser.parse_args()
    if args.mode=='preflight': preflight()
    elif args.mode=='run':
        import joblib
        joblib.Parallel(n_jobs=args.jobs)(joblib.delayed(run)(name,seed,args.phase,args.episodes)
            for name in args.models for seed in args.seeds)
    else:
        rows=[json.loads(p.read_text()) for p in sorted((ROOT/args.phase).glob('*.json'))]
        for name in sorted({r['model']['name'] for r in rows}):
            group=[r for r in rows if r['model']['name']==name]
            print(name,len(group),{k:round(float(np.mean([r[k] for r in group])),4)
                for k in ['auc','restricted_solve_episode','final50','heldout','seconds']})

if __name__=='__main__': main()
