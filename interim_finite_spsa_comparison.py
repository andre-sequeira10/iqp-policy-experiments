"""Show only the common completed seed cohort; do not change running jobs."""
import json
import numpy as np
import finite_spsa_mlp_comparison as c

root = c.ROOT
all_seeds = list(c.TEST)
names = ['k8', 'k32', 'k64']
best, selected, _ = c.references()
names += list(dict.fromkeys([best, selected]))
seeds = [s for s in all_seeds if all(c.stem_for(name,s).with_suffix('.json').exists() for name in names)]
assert len(seeds) >= 2
c.TEST = seeds
curves, summary, records = {}, {}, {}
indices = np.random.default_rng(731).integers(0,len(seeds),(50000,len(seeds)))
for name in names:
    records[name], curves[name] = c.load(name)
    summary[name] = {'seeds':len(seeds),'solved':sum(r['solve_episode']<=500 for r in records[name])}
    for key in ['auc','solve_episode','heldout','final50']:
        summary[name][key] = c.ci([r[key] for r in records[name]],indices)
    if not name.startswith('k'):
        summary[name]['parameters'] = records[name][0]['parameters']
c.ROOT = root/'interim'
c.ROOT.mkdir(exist_ok=True)
note = 'Highest observed full-cohort MLP AUC; retrospective reference. Completed-seed subset only.'
c.plot(curves,summary,best,'training_returns_auc',note,interim=True)
c.plot(curves,summary,selected,'training_returns_auc_selected_mlp','MLP selected by original development AUC; completed-seed subset only.',interim=True)
data={'status':'interim','completed_common_seeds':seeds,'planned_seeds':all_seeds,'main_mlp':best,
      'summary':summary,'warning':'Completion-based subset can favor fast-converging runs. No final inferential claim; await all 20 seeds. Every curve uses the same listed seeds.'}
(c.ROOT/'comparison.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2))
