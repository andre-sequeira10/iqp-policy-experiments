"""Publication assets from the locked test cohort; never reruns training."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/iqp-search-mpl')
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from qasp.circuits.tikz import tape_to_tikz

ROOT = Path(__file__).resolve().parent / 'results/connected_iqp_mlp_study'
OUT = ROOT / 'paper_assets'
OUT.mkdir(exist_ok=True)
data = json.loads((ROOT / 'test_analysis.json').read_text())
selection = data['selection']
summary = data['summary']
names = ['iqp_connected', selection['best_auc_mlp'],
         selection['best_solve_mlp'], selection['selected_by_width']['19']]
labels = ['Connected IQP (139)',
          'MLP (30; best development AUC)', 'MLP (86; fastest development)',
          'MLP (135; parameter matched)']
colors = ['C2', 'C0', 'C1', 'C4']
plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))
episodes = np.arange(1, 501)
for name, label, color in zip(names, labels, colors):
    returns = np.stack([np.load(ROOT / 'test' / f'{name}__{seed}.npz')['training_returns']
                        for seed in range(2001, 2021)])
    cumulative = returns.cumsum(axis=1)
    smooth = np.stack([np.convolve(r, np.ones(20)/20, mode='valid') for r in returns])
    metrics = [smooth, cumulative / episodes, cumulative / (200 * 500)]
    assert np.isclose(metrics[-1][:, -1].mean(), summary[name]['auc']['mean'])
    for ax, x, y in zip(axes, [episodes[19:], episodes, episodes], metrics):
        mean = y.mean(axis=0)
        sem = y.std(axis=0, ddof=1) / np.sqrt(len(y))
        ax.plot(x, mean, label=label, color=color, lw=1.6)
        ax.fill_between(x, mean-sem, mean+sem, color=color, alpha=.10, linewidth=0)
for ax, title, ylabel in zip(axes,
        ['(a) Training returns', '(b) Cumulative average return', '(c) Accumulated normalized return'],
        ['20-episode mean return', r'$t^{-1}\sum_{e=1}^{t} R_e$', r'$(200\times500)^{-1}\sum_{e=1}^{t} R_e$']):
    ax.set(title=title, xlabel='Training episode', ylabel=ylabel, xlim=(1,500))
    ax.grid(alpha=.18)
axes[0].set_ylim(0,205); axes[1].set_ylim(0,205); axes[2].set_ylim(0,1)
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
           frameon=False, fontsize=9, bbox_to_anchor=(.5,0))
fig.tight_layout(rect=(0,.15,1,1))
for extension in ['pdf','png']:
    fig.savefig(OUT / f'connected_iqp_training_curves.{extension}', dpi=220, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(9.2,3.8))
selected = selection['selected_by_width']
for ax, key, ylabel in zip(axes, ['auc','solve_episode'],
                          ['Normalized return AUC', 'Restricted episodes to criterion']):
    rows = [summary[selected[str(w)]] for w in [1,2,4,6,9,12,16,19]]
    x = np.array([r['parameters'] for r in rows]); y = np.array([r[key]['mean'] for r in rows])
    ci = np.array([r[key]['ci95'] for r in rows])
    ax.errorbar(x,y,yerr=np.maximum(0,np.stack([y-ci[:,0],ci[:,1]-y])),
                fmt='o-',color='C0',capsize=3,label='MLP: development-selected learning rate')
    r = summary['iqp_connected'][key]
    ax.axhline(r['mean'],color='C2',label='Connected IQP (139)')
    ax.axhspan(*r['ci95'],color='C2',alpha=.09)
    ax.set(xlabel='MLP trainable parameters',ylabel=ylabel);ax.grid(alpha=.18)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=1,
           frameon=False,fontsize=9,bbox_to_anchor=(.5,-.01))
fig.tight_layout(rect=(0,.19,1,1))
for extension in ['pdf','png']:
    fig.savefig(OUT / f'connected_iqp_mlp_size.{extension}',dpi=220,bbox_inches='tight')
plt.close(fig)

# The full 21-wire circuit, not a disconnected surrogate or illustrative subset.
# Operations are exactly those used by ConnectedMajorityIQP; disjoint commuting
# gates are scheduled in parallel by the QASP exporter.
parameter_labels = []
with qml.tape.QuantumTape() as tape:
    for i in range(21): qml.Hadamard(i)
    for i in range(21):
        qml.RZ(.1, i); parameter_labels.append(rf'\phi_{{{i+1}}}(s)')
    for j in range(7):
        start = 3*j
        for k,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
            qml.PauliRot(.1,'ZZ',[start+a,start+b])
            parameter_labels.append(rf'\alpha_{{{j+1},{k+1}}}')
        qml.PauliRot(.1,'ZZZ',list(range(start,start+3)))
        parameter_labels.append(rf'\gamma_{{{j+1}}}')
    for j in range(6):
        qml.PauliRot(.1,'ZZ',[3*j+2,3*j+3])
        parameter_labels.append(rf'\beta_{{{j+1}}}')
    for i in range(21): qml.Hadamard(i)
    qml.probs(wires=range(21))
assert len(tape.operations)==97 and len(parameter_labels)==55
source = tape_to_tikz(tape, wire_order=list(range(21)),
    wire_labels={i:rf'\ket{{0}}_{{{i+1}}}' for i in range(21)},
    parameter_labels=parameter_labels, row_sep='0.10cm',column_sep='0.10cm',
    gate_colors={'single':'DCEAF7','two':'F3D7B5','multi':'E6D6F2','measurement':'D9EAD3'})
(OUT/'connected_iqp_full_circuit.tex').write_text(source)
(OUT/'asset_metadata.json').write_text(json.dumps({
    'qubits':21,'gate_operations':97,'encoded_angles':21,'pair_gates':21,
    'triple_gates':7,'bridges':6,'trainable_parameters':139,
    'export_api':'qasp.circuits.tikz.tape_to_tikz',
    'test_seeds':list(range(2001,2021)),
    'curve_bands':'one standard error across independent training seeds',
    'auc':'sum(training_returns)/(200*500)'},indent=2))
print(OUT)
