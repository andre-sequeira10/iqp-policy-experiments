"""Fixed-protocol finite-shot SPSA follow-up; analytical results stay untouched."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
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
from connected_majority_iqp import ConnectedMajorityIQP
from connected_iqp_mlp_study import ROOT as BASE, TEST, solve
from majority_iqp_model_search import make_agent, SCALE
from analyze_majority_iqp_search import infer

ROOT = BASE.parent / 'connected_iqp_finite_shots'
NAME = 'iqp_connected_shots4096_spsa8'
SHOTS = 4096
K = 8
torch.set_num_threads(1)


def model(seed):
    return ConnectedMajorityIQP(seed=seed, n=21, sigma=.5, bridge_sigma=.5)


def protocol():
    ROOT.mkdir(exist_ok=True)
    p = {
        'name': NAME, 'qubits':21, 'parameters':139, 'shots_per_evaluation':SHOTS,
        'initialization':'Original: identical saved neural weights, bias pi/2, Normal(0,.5^2) internal and bridge angles.',
        'gradient':'Two-sided SPSA, K=ceil(log2(139))=8 independent Rademacher directions per update; independent shot samples on both sides.',
        'perturbation':'c_k=.1*(k+1)^(-.101), k starts at zero and counts optimizer updates; all 139 parameters perturbed.',
        'sampling':'Independent Binomial(4096, p_action0) counts, exactly the action-count law of independent Born shots followed by binary majority. Jeffreys smoothing (N_a+.5)/(4096+1). Sampling in rollout, both perturbed losses, and held-out evaluation. Exact probabilities used only inside the classical shot simulator; no analytic gradients.',
        'training':{'episodes':500,'max_steps':200,'batch_episodes':5,'lr':.03,'optimizer':'Adam AMSGrad','gamma':.99,'entropy_coef':.005,'baseline':'EMA','baseline_momentum':.05,'normalize_returns':True,'max_grad_norm':1.,'freeze':'Full rolling-100 mean >=195, checked every five episodes; continue evaluation rollouts through episode 500.'},
        'seeds':TEST, 'heldout_episodes':100,
        'rng_offsets':{'rollout_shots':3100000,'gradient_shots':3200000,'directions':3300000,'evaluation_shots':3400000},
        'primary_metrics':'Normalized training AUC=sum(rewards)/(500*200); first full rolling-100 mean>=195, assigned 501 if not reached.',
        'comparators':'Reuse original analytical connected IQP and all nine locked MLP settings on the same 20 seed IDs. No retuning or seed selection.',
        'inference':'Paired bootstrap 95% CIs (50000 resamples), exact two-sided sign-randomization p-values; Holm correction across ten comparators times two metrics. Follow-up on an already inspected cohort, not untouched confirmatory evidence.',
        'scope':'Finite-shot simulation, not hardware or device-noise experiment. Joint change of probability access and gradient estimator; not an isolated measurement-noise test. No manuscript edits.'}
    path=ROOT/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())==p
    else: path.write_text(json.dumps(p,indent=2))


class ShotOracle:
    def __init__(self, policy, seed):
        self.policy=policy
        self.rng={k:np.random.default_rng(seed+offset) for k,offset in
                  [('rollout',3100000),('gradient',3200000),('evaluation',3400000)]}
        self.calls={k:0 for k in self.rng}

    @torch.no_grad()
    def __call__(self, x, kind):
        # Binary action counts are sufficient: explicitly drawing full bitstrings
        # would produce exactly the same Binomial action-count distribution.
        p=self.policy(x)[...,0].cpu().numpy().astype(np.float64)
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        counts=self.rng[kind].binomial(SHOTS,p)
        self.calls[kind]+=int(np.size(p))
        q=(counts+.5)/(SHOTS+1)
        return np.stack((q,1-q),axis=-1)


def loss(probs, actions, advantages, entropy_coef=.005):
    selected=probs[np.arange(len(actions)),actions]
    entropy=-(probs*np.log(probs)).sum(axis=-1)
    return float(-(advantages*np.log(selected)).mean()-entropy_coef*entropy.mean())


def assign(policy, flat):
    with torch.no_grad():
        offset=0
        for p in policy.parameters():
            p.copy_(flat[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
        assert offset==len(flat)


class FiniteTrainer:
    def __init__(self, policy, seed):
        self.policy=policy;self.seed=seed
        self.agent=make_agent(policy,False,strict=True)
        self.oracle=ShotOracle(policy,seed)
        self.rng=np.random.default_rng(seed+3300000)
        self.updates=0;self.freeze_episode=None;self.optimized_transitions=0
        self.diagnostics=[]
        assert sum(p.numel() for p in policy.parameters())==139
        assert K==math.ceil(math.log2(139))
        assert self.agent.opt.param_groups[0]['amsgrad']

    @torch.no_grad()
    def optimize(self, states, actions, advantages):
        x=torch.stack(states);actions=np.asarray(actions)
        adv=advantages.detach().numpy().astype(np.float64)
        original=torch.cat([p.detach().flatten() for p in self.policy.parameters()]).clone()
        estimate=np.zeros(len(original),dtype=np.float64)
        c=.1*(self.updates+1)**(-.101)
        differences=[]
        try:
            for _ in range(K):
                delta=self.rng.choice([-1.,1.],size=len(original))
                perturb=torch.as_tensor(c*delta,dtype=original.dtype)
                assign(self.policy,original+perturb)
                plus=loss(self.oracle(x,'gradient'),actions,adv)
                assign(self.policy,original-perturb)
                minus=loss(self.oracle(x,'gradient'),actions,adv)
                differences.append(plus-minus)
                estimate+=(plus-minus)/(2*c)*delta
        finally:
            assign(self.policy,original)
        estimate/=K
        assert np.isfinite(estimate).all()
        self.agent.opt.zero_grad(set_to_none=True)
        offset=0
        for p in self.policy.parameters():
            p.grad=torch.as_tensor(estimate[offset:offset+p.numel()],dtype=p.dtype).reshape_as(p).clone()
            offset+=p.numel()
        norm=torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()),self.agent.max_grad_norm)
        self.agent.opt.step()
        self.diagnostics.append({'update':self.updates,'states':len(states),'c':c,
                                 'gradient_norm_before_clip':float(norm),'loss_differences':differences})
        self.updates+=1;self.optimized_transitions+=len(states)

    def train(self, episodes=500):
        torch.manual_seed(self.seed);np.random.seed(self.seed)
        a=self.agent;a.env.reset(seed=self.seed);a.env.action_space.seed(self.seed)
        history=[];states=[];actions=[];advantages=[]
        for episode in range(1,episodes+1):
            state,_=a.env.reset();rewards=[]
            for _ in range(a.max_steps):
                x=a._preprocess_state(torch.as_tensor(state,dtype=torch.float32))
                probs=torch.as_tensor(self.oracle(x,'rollout'),dtype=torch.float32)
                action=int(torch.distributions.Categorical(probs=probs).sample())
                states.append(x);actions.append(action)
                state,reward,done,truncated,_=a.env.step(action);rewards.append(float(reward))
                if done or truncated:break
            history.append(sum(rewards))
            disc=a._discount(rewards);advantages.append(disc-a.baseline_value)
            a._update_baseline(float(disc.mean()))
            if episode%a.batch_episodes==0 or episode==episodes:
                adv=torch.cat(advantages)
                if a.normalize_returns and len(adv)>1:adv=(adv-adv.mean())/(adv.std()+1e-8)
                if self.freeze_episode is None:
                    if episode>=100 and np.mean(history[-100:])>=195:self.freeze_episode=episode
                    else:self.optimize(states,actions,adv)
                states.clear();actions.clear();advantages.clear()
            if episode%25==0:
                print(f'episode={episode} last20={np.mean(history[-20:]):.2f} updates={self.updates} freeze={self.freeze_episode}',flush=True)
        return np.asarray(history)

    def evaluate(self, count=100):
        env=gym.make('CartPole-v1');returns=[]
        for ep in range(count):
            state,_=env.reset(seed=1000000+10000*self.seed+ep)
            gen=torch.Generator().manual_seed(2000000+10000*self.seed+ep);total=0
            for _ in range(200):
                x=(torch.as_tensor(state,dtype=torch.float32)/SCALE).clamp(-3,3)
                p=torch.as_tensor(self.oracle(x,'evaluation'),dtype=torch.float32)
                action=torch.multinomial(p,1,generator=gen).item()
                state,r,done,truncated,_=env.step(action);total+=r
                if done or truncated:break
            returns.append(total)
        env.close();return np.asarray(returns)


def verify():
    protocol();checks=[]
    for seed in TEST:
        m=model(seed);old=torch.load(BASE/'test'/f'iqp_connected__{seed}.pt',weights_only=False,map_location='cpu')
        assert all(torch.equal(p,old['initial'][k]) for k,p in m.named_parameters())
        checks.append(seed)
    # Validate the estimator normalization with an exactly balanced direction set.
    from scipy.linalg import hadamard
    directions=hadamard(4);theta=np.array([.3,-.2,.4,.1]);gradient=np.array([1.,-2.,3.,.5]);c=.1
    estimate=np.mean([((gradient@(theta+c*d)-gradient@(theta-c*d))/(2*c))*d for d in directions],axis=0)
    assert np.allclose(estimate,gradient,atol=1e-12)
    m=model(90017);o=ShotOracle(m,90017);x=torch.tensor([.2,-.3,.4,-.5])
    batch=x.repeat(4000,1);sample=o(batch,'gradient')
    p=float(m(x)[0].detach());expected=(SHOTS*p+.5)/(SHOTS+1)
    variance=SHOTS*p*(1-p)/(SHOTS+1)**2
    assert abs(sample[:,0].mean()-expected)<5*math.sqrt(variance/4000)
    assert abs(sample[:,0].var()/variance-1)<.1
    assert np.allclose(sample.sum(-1),1) and sample.min()>0
    actions=np.arange(4000)%2;adv=np.linspace(-1,1,4000)
    q=torch.as_tensor(sample,dtype=torch.float64);dist=torch.distributions.Categorical(probs=q)
    reference=-(torch.as_tensor(adv)*dist.log_prob(torch.as_tensor(actions))).mean()-.005*dist.entropy().mean()
    assert abs(loss(sample,actions,adv)-float(reference))<1e-12
    trainer=FiniteTrainer(m,90017);start=time.time()
    # Disable backward globally during the smoke test: SPSA must need no autograd.
    original_backward=torch.Tensor.backward
    def forbidden(*args,**kwargs):raise AssertionError('Analytical backward used in finite-shot training')
    torch.Tensor.backward=forbidden
    try:tr=trainer.train(5)
    finally:torch.Tensor.backward=original_backward
    assert trainer.updates==1 and trainer.oracle.calls['gradient']==16*sum(tr)
    assert trainer.oracle.calls['rollout']==sum(tr)
    assert all(torch.isfinite(p).all() for p in m.parameters())
    trainer.agent.env.close()
    result={'original_initializations_bitwise_identical':checks,'balanced_spsa_linear_test':True,
            'binomial_empirical_mean':float(sample[:,0].mean()),'binomial_expected_mean':expected,
            'binomial_variance_ratio':float(sample[:,0].var()/variance),'same_REINFORCE_loss':True,
            'no_autograd_smoke_test':True,'five_episode_returns':tr.tolist(),
            'smoke_seconds':time.time()-start,'calls':trainer.oracle.calls}
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


def run(seed):
    folder=ROOT/'test';folder.mkdir(exist_ok=True);stem=folder/f'{NAME}__{seed}'
    if stem.with_suffix('.json').exists():return json.loads(stem.with_suffix('.json').read_text())
    start=time.time();m=model(seed);initial={k:p.detach().clone() for k,p in m.named_parameters()}
    trainer=FiniteTrainer(m,seed)
    with stem.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log):tr=trainer.train()
    ev=trainer.evaluate();trainer.agent.env.close()
    calls=trainer.oracle.calls
    assert calls['rollout']==sum(tr) and calls['evaluation']==sum(ev)
    assert calls['gradient']==2*K*trainer.optimized_transitions
    r={'name':NAME,'seed':seed,'parameters':139,'shots':SHOTS,'K':K,'auc':float(tr.mean()/200),
       'solve_episode':solve(tr),'final50':float(tr[-50:].mean()),'heldout':float(ev.mean()),
       'freeze_episode':trainer.freeze_episode,'updates':trainer.updates,'seconds':time.time()-start,
       'optimized_transitions':trainer.optimized_transitions,'circuit_evaluations':calls,
       'training_shots':SHOTS*(calls['rollout']+calls['gradient']),
       'evaluation_shots':SHOTS*calls['evaluation'],
       'initial_hash':hashlib.sha256(b''.join(p.numpy().tobytes() for p in initial.values())).hexdigest()}
    np.savez_compressed(stem.with_suffix('.npz'),training_returns=tr,evaluation_returns=ev)
    torch.save({'initial':initial,'final':m.state_dict(),'record':r},stem.with_suffix('.pt'))
    stem.with_suffix('.diagnostics.json').write_text(json.dumps(trainer.diagnostics,indent=2))
    stem.with_suffix('.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
    return r


def analyze():
    locked=json.loads((BASE/'development_selection.json').read_text())
    comparisons=list(dict.fromkeys(['iqp_connected']+list(locked['selected_by_width'].values())+[locked['best_solve_mlp']]))
    groups={};curves={};summary={}
    for name in [NAME]+comparisons:
        folder=ROOT/'test' if name==NAME else BASE/'test';rows=[];traces=[]
        for seed in TEST:
            stem=folder/f'{name}__{seed}';r=json.loads(stem.with_suffix('.json').read_text())
            a=np.load(stem.with_suffix('.npz'));tr=a['training_returns'];ev=a['evaluation_returns']
            assert tr.shape==(500,) and ev.shape==(100,) and np.isfinite(tr).all()
            assert np.all((tr>=1)&(tr<=200)) and np.all((ev>=1)&(ev<=200))
            assert abs(r['auc']-tr.mean()/200)<1e-12 and r['solve_episode']==solve(tr)
            if name==NAME:
                old=json.loads((BASE/'test'/f'iqp_connected__{seed}.json').read_text())
                assert r['initial_hash']==old['initial_hash']
                qualifying=[e for e in range(100,501,5) if tr[e-100:e].mean()>=195]
                assert r['freeze_episode']==(qualifying[0] if qualifying else None)
                assert r['circuit_evaluations']['rollout']==sum(tr)
                assert r['circuit_evaluations']['gradient']==16*r['optimized_transitions']
            rows.append(r);traces.append(tr)
        groups[name]=rows;curves[name]=np.stack(traces)
        summary[name]={'parameters':rows[0]['parameters'],'solved':sum(r['solve_episode']<=500 for r in rows)}
        for key in ['auc','solve_episode','heldout','final50']:
            v=np.array([r[key] for r in rows]);rng=np.random.default_rng(871)
            ci=np.quantile(rng.choice(v,size=(50000,len(v)),replace=True).mean(1),[.025,.975])
            summary[name][key]={'mean':float(v.mean()),'ci95':ci.tolist()}
    paired={}
    for name in comparisons:
        paired[name]={}
        for key,sign in [('auc',1),('solve_episode',-1)]:
            paired[name][key]=infer([sign*(a[key]-b[key]) for a,b in zip(groups[NAME],groups[name])],20261027)
    tests=sorted([v[k] for v in paired.values() for k in ['auc','solve_episode']],key=lambda r:r['two_sided_exact_sign_p'])
    previous=0
    for i,r in enumerate(tests):
        previous=max(previous,min(1.,(len(tests)-i)*r['two_sided_exact_sign_p']));r['holm_p']=previous
    data={'summary':summary,'paired_finite_shots_minus_comparator':paired,
          'resources':{'training_shots_total':sum(r['training_shots'] for r in groups[NAME]),
                       'training_shots_mean':float(np.mean([r['training_shots'] for r in groups[NAME]])),
                       'evaluation_shots_total':sum(r['evaluation_shots'] for r in groups[NAME])}}
    (ROOT/'comparison.json').write_text(json.dumps(data,indent=2))
    original=json.loads((BASE/'test_integrity.json').read_text())
    unchanged={k:hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in original['sha256'].items()
               if Path(k).name in ['connected_majority_iqp.py','majority_iqp_model_search.py','majority_iqp_block_model.py','torchrl.py']
               or (Path(k).parent==BASE/'test' and Path(k).suffix in ['.json','.npz'])}
    assert all(unchanged.values())
    sources=[Path(__file__),ROOT/'protocol.json',ROOT/'verification.json']
    sources+=list((ROOT/'test').glob('*.npz'))+list((ROOT/'test').glob('*.json'))
    (ROOT/'integrity.json').write_text(json.dumps({'baseline_files_unchanged':unchanged,
        'sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        'audited_finite_shot_runs':20},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    names=[NAME,'iqp_connected',locked['selected_by_width']['19'],locked['best_auc_mlp']]
    labels=['Connected IQP: 4096 shots + SPSA (139)','Connected IQP: analytical (139)','MLP: parameter-matched (135)','MLP: development-selected AUC (30)']
    colors=['#d95f02','#1b9e77','#7570b3','#666666']
    for name,label,color in zip(names,labels,colors):
        tr=curves[name];window=20
        smooth=np.stack([np.convolve(t,np.ones(window)/window,'valid') for t in tr])
        for ax,v,x in [(axes[0],smooth,np.arange(window,501)),(axes[1],np.cumsum(tr,axis=1)/100000,np.arange(1,501))]:
            mean=v.mean(0);se=v.std(0,ddof=1)/np.sqrt(len(v))
            ax.plot(x,mean,label=label,color=color);ax.fill_between(x,mean-1.96*se,mean+1.96*se,color=color,alpha=.12)
        times=np.array([r['solve_episode'] for r in groups[name]])
        axes[2].step(np.arange(1,501),[(times<=e).mean() for e in range(1,501)],color=color,label=label)
    for ax in axes:ax.set_xlabel('Training episode');ax.grid(alpha=.2)
    axes[0].set_ylabel('20-episode mean return');axes[0].set_ylim(0,205)
    axes[0].set_title('Training return',fontsize=11)
    axes[1].set_ylabel('Accumulated normalized return');axes[1].set_ylim(0,1.02)
    axes[1].set_title('Endpoint = normalized AUC',fontsize=11)
    axes[2].set_ylabel('Fraction reaching criterion');axes[2].set_ylim(0,1.02)
    axes[2].set_title('Rolling-100 mean return ≥195',fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Connected 21-qubit majority policy: original initialization, 20 matched seeds\nShading: pointwise 95% normal-approximation intervals across seeds',fontsize=12)
    fig.savefig(ROOT/'finite_shot_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
    selected=locked['selected_by_width']
    for ax,key,label in zip(axes,['auc','solve_episode'],['Normalized training AUC','Restricted episodes to criterion']):
        records=[summary[selected[str(w)]] for w in [1,2,4,6,9,12,16,19]]
        x=np.array([r['parameters'] for r in records]);y=np.array([r[key]['mean'] for r in records]);ci=np.array([r[key]['ci95'] for r in records])
        ax.errorbar(x,y,yerr=np.maximum(0,np.stack([y-ci[:,0],ci[:,1]-y])),fmt='o-',capsize=3,color='#7570b3',label='MLPs: learning rate selected on development')
        for name,title,color in zip(names[:2],labels[:2],colors[:2]):
            r=summary[name][key];ax.axhline(r['mean'],color=color,label=title);ax.axhspan(*r['ci95'],color=color,alpha=.12)
        ax.set(xlabel='MLP trainable parameters',ylabel=label);ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',frameon=False,ncol=1)
    fig.suptitle('Means and individual 95% bootstrap intervals across 20 seeds\nRuns not reaching the criterion within 500 episodes are assigned 501',fontsize=11)
    fig.savefig(ROOT/'finite_shot_mlp_size_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(data,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','run','analyze','status']);p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args();protocol()
    if args.mode=='verify':verify()
    elif args.mode=='run':
        from joblib import Parallel,delayed
        Parallel(n_jobs=args.jobs)(delayed(run)(seed) for seed in TEST)
    elif args.mode=='analyze':analyze()
    else:
        files=list((ROOT/'test').glob(f'{NAME}__*.json'));files=[f for f in files if not f.name.endswith('.diagnostics.json')]
        print('Completed',len(files),'of',len(TEST))
