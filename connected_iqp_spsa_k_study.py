"""K-only SPSA tuning, locked paired evaluation, and serial timing comparison."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/iqp-search-mpl')
import argparse
import contextlib
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from connected_iqp_finite_shot_test import FiniteTrainer, model, loss, assign, SHOTS, ROOT as OLD
from connected_iqp_mlp_study import ROOT as BASE, DEV, TEST, solve
from majority_iqp_model_search import make_agent, evaluate
from analyze_majority_iqp_search import infer

ROOT=BASE.parent/'connected_iqp_spsa_k_study'
GRID=[32,64,128,256,512]
TIMING_SEEDS=[2001,2002,2003]
torch.set_num_threads(1)


def protocol():
    ROOT.mkdir(exist_ok=True)
    data={'change':'Only the number of independent SPSA directions K changes. All 139 parameters, original initialization, 4096 shots, perturbation schedule, optimizer, seeds, loss, preprocessing and freeze rule unchanged.',
          'grid':GRID,'development_seeds':DEV,'test_seeds':TEST,
          'selection':'Run ascending K on all eight development seeds. Stop at the smallest K with mean AUC >= analytical development AUC-.03, restricted mean convergence episode <= analytical+25, and at least analytical number solved minus one. If none qualifies through 512, select the smallest maximum normalized shortfall in AUC/.03 and convergence/25 (ties smaller K) and report target not met. No test-set retuning.',
          'metrics':'AUC=sum(training returns)/(500*200); first full rolling-100 return>=195, 501 if unreached. All 500 episodes retained. Held-out evaluation 100 episodes on test; 20 on development, not used in selection.',
          'timing':'After all parallel experiments finish, serial single-worker CPU timing on seeds 2001,2002,2003. Full 500-episode training only, excluding model setup and held-out evaluation; same one-thread policy for analytical, K=8 and selected K. Cyclic method order across seeds. Record total training time, episode timestamps and time to first convergence criterion. This is classical simulator wall time, not quantum hardware runtime.',
          'inference':'Paired bootstrap 95% CIs over 20 test seeds. These test seeds were inspected in earlier studies; K selected on disjoint development seeds. Meeting a mean target alone is not statistical equivalence.',
          'scope':'No changes to manuscript or earlier experiments. Report simulated quantum-shot cost separately from CPU time.'}
    p=ROOT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==data
    else:p.write_text(json.dumps(data,indent=2))


class KTrainer(FiniteTrainer):
    def __init__(self,policy,seed,k):
        super().__init__(policy,seed);self.k=k
        self.optimize_seconds=0.;self.episode_seconds=[];self.training_start=None
        reset=self.agent.env.reset
        def timed_reset(*args,**kwargs):
            if self.training_start is not None:self.episode_seconds.append(time.perf_counter()-self.training_start)
            return reset(*args,**kwargs)
        self.agent.env.reset=timed_reset

    @torch.no_grad()
    def optimize(self,states,actions,advantages):
        start=time.perf_counter();x=torch.stack(states);actions=np.asarray(actions)
        adv=advantages.detach().numpy().astype(np.float64)
        original=torch.cat([p.detach().flatten() for p in self.policy.parameters()]).clone()
        estimate=np.zeros(len(original),dtype=np.float64);c=.1*(self.updates+1)**(-.101)
        differences=[]
        try:
            for _ in range(self.k):
                delta=self.rng.choice([-1.,1.],size=len(original))
                perturb=torch.as_tensor(c*delta,dtype=original.dtype)
                assign(self.policy,original+perturb)
                plus=loss(self.oracle(x,'gradient'),actions,adv)
                assign(self.policy,original-perturb)
                minus=loss(self.oracle(x,'gradient'),actions,adv)
                differences.append(plus-minus);estimate+=(plus-minus)/(2*c)*delta
        finally:assign(self.policy,original)
        estimate/=self.k;assert np.isfinite(estimate).all()
        self.agent.opt.zero_grad(set_to_none=True);offset=0
        for p in self.policy.parameters():
            p.grad=torch.as_tensor(estimate[offset:offset+p.numel()],dtype=p.dtype).reshape_as(p).clone();offset+=p.numel()
        norm=torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()),self.agent.max_grad_norm)
        self.agent.opt.step()
        self.diagnostics.append({'update':self.updates,'states':len(states),'c':c,'gradient_norm_before_clip':float(norm),'loss_differences':differences})
        self.updates+=1;self.optimized_transitions+=len(states);self.optimize_seconds+=time.perf_counter()-start

    def train(self,episodes=500):
        self.training_start=time.perf_counter()
        result=super().train(episodes)
        self.training_seconds=time.perf_counter()-self.training_start
        # Parent train first resets for seeding, then resets at each episode start.
        self.episode_seconds=self.episode_seconds[2:]+[self.training_seconds]
        assert len(self.episode_seconds)==episodes
        return result


def run(k,seed,phase):
    folder=ROOT/phase;folder.mkdir(exist_ok=True);stem=folder/f'k{k:03d}__{seed}'
    if stem.with_suffix('.json').exists():return json.loads(stem.with_suffix('.json').read_text())
    m=model(seed);initial={name:p.detach().clone() for name,p in m.named_parameters()}
    original=torch.load(BASE/('development' if phase=='development' else 'test')/f'iqp_connected__{seed}.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v,original['initial'][name]) for name,v in initial.items())
    trainer=KTrainer(m,seed,k)
    with stem.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log):tr=trainer.train()
    eval_start=time.perf_counter();ev=trainer.evaluate(20 if phase=='development' else 100)
    eval_seconds=time.perf_counter()-eval_start;trainer.agent.env.close()
    calls=trainer.oracle.calls
    assert calls['rollout']==sum(tr) and calls['evaluation']==sum(ev)
    assert calls['gradient']==2*k*trainer.optimized_transitions
    result={'k':k,'seed':seed,'phase':phase,'auc':float(tr.mean()/200),'solve_episode':solve(tr),
            'heldout':float(ev.mean()),'final50':float(tr[-50:].mean()),'freeze_episode':trainer.freeze_episode,
            'updates':trainer.updates,'training_seconds':trainer.training_seconds,'optimization_seconds':trainer.optimize_seconds,
            'evaluation_seconds':eval_seconds,'circuit_evaluations':calls,'optimized_transitions':trainer.optimized_transitions,
            'training_shots':SHOTS*(calls['rollout']+calls['gradient']),
            'initial_hash':hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()}
    np.savez_compressed(stem.with_suffix('.npz'),training_returns=tr,evaluation_returns=ev,episode_seconds=trainer.episode_seconds)
    torch.save({'initial':initial,'final':m.state_dict(),'record':result},stem.with_suffix('.pt'))
    stem.with_suffix('.diagnostics.json').write_text(json.dumps(trainer.diagnostics))
    stem.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result


def select(jobs):
    target=ROOT/'selection.json'
    if target.exists():print(target.read_text());return
    from joblib import Parallel,delayed
    reference=[json.loads((BASE/'development'/f'iqp_connected__{s}.json').read_text()) for s in DEV]
    base_auc=np.mean([r['auc'] for r in reference]);base_time=np.mean([r['solve_episode'] for r in reference]);base_solved=sum(r['solve_episode']<=500 for r in reference)
    stages=[];chosen=None
    for k in GRID:
        records=Parallel(n_jobs=jobs)(delayed(run)(k,s,'development') for s in DEV)
        auc=float(np.mean([r['auc'] for r in records]));conv=float(np.mean([r['solve_episode'] for r in records]));solved=sum(r['solve_episode']<=500 for r in records)
        passed=auc>=base_auc-.03 and conv<=base_time+25 and solved>=base_solved-1
        row={'k':k,'auc':auc,'solve_episode':conv,'solved':solved,'target_met':bool(passed),
             'score':float(max((base_auc-auc)/.03,(conv-base_time)/25))}
        stages.append(row);(ROOT/'development_summary.json').write_text(json.dumps(stages,indent=2))
        print('STAGE '+json.dumps(row),flush=True)
        if passed:chosen=k;break
    if chosen is None:chosen=min(stages,key=lambda r:(r['score'],r['k']))['k']
    result={'selected_k':chosen,'development_target_met':next(r['target_met'] for r in stages if r['k']==chosen),
            'analytical_development':{'auc':float(base_auc),'solve_episode':float(base_time),'solved':base_solved},'stages':stages}
    target.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


def verify():
    import io
    outcomes=[]
    for cls in [FiniteTrainer,lambda m,s:KTrainer(m,s,8)]:
        m=model(90017);trainer=cls(m,90017)
        with contextlib.redirect_stdout(io.StringIO()):tr=trainer.train(5)
        outcomes.append((tr,torch.cat([p.detach().flatten() for p in m.parameters()]),trainer.oracle.calls))
        trainer.agent.env.close()
    assert np.array_equal(outcomes[0][0],outcomes[1][0]) and torch.equal(outcomes[0][1],outcomes[1][1])
    assert outcomes[0][2]==outcomes[1][2]
    result={'K8_bitwise_same_parameters_after_update':True,'K8_identical_trajectories_and_shot_counts':True}
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2));print(result)


def timing():
    k=json.loads((ROOT/'selection.json').read_text())['selected_k'];folder=ROOT/'serial_timing';folder.mkdir(exist_ok=True)
    names=['analytical','k008',f'k{k:03d}']
    for i,seed in enumerate(TIMING_SEEDS):
        for name in names[i:]+names[:i]:
            stem=folder/f'{name}__{seed}'
            if stem.with_suffix('.json').exists():continue
            m=model(seed)
            with stem.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log):
                if name=='analytical':
                    a=make_agent(m,False,strict=True);torch.manual_seed(seed);np.random.seed(seed)
                    a.env.reset(seed=seed);a.env.action_space.seed(seed)
                    reset=a.env.reset;stamps=[]
                    def timed_reset(*args,**kwargs):
                        stamps.append(time.perf_counter()-start);return reset(*args,**kwargs)
                    a.env.reset=timed_reset;start=time.perf_counter();cpu_start=time.process_time()
                    tr=np.asarray(a.train(500,50));elapsed=time.perf_counter()-start;cpu_elapsed=time.process_time()-cpu_start
                    stamps=stamps[1:]+[elapsed];a.env.close()
                    original=np.load(BASE/'test'/f'iqp_connected__{seed}.npz')['training_returns']
                else:
                    a=KTrainer(m,seed,int(name[1:]));cpu_start=time.process_time()
                    tr=a.train();cpu_elapsed=time.process_time()-cpu_start;elapsed=a.training_seconds;stamps=a.episode_seconds;a.agent.env.close()
                    path=OLD/'test'/f'iqp_connected_shots4096_spsa8__{seed}.npz' if name=='k008' else ROOT/'test'/f'k{k:03d}__{seed}.npz'
                    original=np.load(path)['training_returns']
            assert np.array_equal(tr,original),f'Trajectory replay mismatch: {name} seed {seed}'
            hit=solve(tr);r={'name':name,'seed':seed,'training_seconds':elapsed,'training_cpu_seconds':cpu_elapsed,'criterion_episode':hit,
                           'seconds_to_criterion':float(stamps[hit-1]) if hit<=500 else None,'trajectory_exactly_reproduced':True}
            np.savez_compressed(stem.with_suffix('.npz'),training_returns=tr,episode_seconds=stamps)
            stem.with_suffix('.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','select','test','timing']);p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args();protocol()
    if args.mode=='verify':verify()
    elif args.mode=='select':select(args.jobs)
    elif args.mode=='timing':timing()
    else:
        from joblib import Parallel,delayed
        k=json.loads((ROOT/'selection.json').read_text())['selected_k']
        Parallel(n_jobs=args.jobs)(delayed(run)(k,s,'test') for s in TEST)
