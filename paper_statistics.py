"""Paired inference helper used by the paper; no model-search experiment."""
import numpy as np

def infer(d,seed):
    d=np.asarray(d,float); rng=np.random.default_rng(seed)
    boot=rng.choice(d,size=(50000,len(d)),replace=True).mean(1)
    # Exact paired sign randomization is practical for 20 seed pairs.
    count=0; total=2**len(d); observed=abs(d.mean())
    for start in range(0,total,16384):
        ids=np.arange(start,min(start+16384,total),dtype=np.uint64)
        signs=2*((ids[:,None]>>np.arange(len(d),dtype=np.uint64))&1).astype(float)-1
        count+=np.count_nonzero(abs((signs*d).mean(1))>=observed-1e-12)
    return {'mean_difference':float(d.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'two_sided_exact_sign_p':count/total,'positive_pairs':int(sum(d>0))}
