"""Integrity and structural checks only: never import experiments or checkpoints."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'MANIFEST_SHA256.json'
EXCLUDE = {'.git', '.venv', '__pycache__', '.ipynb_checkpoints', 'generated', '.pytest_cache'}

def inventory():
    result = {}
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(p in EXCLUDE or p.endswith('.egg-info') for p in rel.parts):
            continue
        if path == MANIFEST or path.name == '.DS_Store' or path.suffix in {'.pyc', '.log'}:
            continue
        if path.is_symlink():
            raise ValueError(f'Symlink not allowed in release: {rel}')
        result[str(rel)] = {'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true',
                        help='Maintainer operation: record the current release contents.')
    args=parser.parse_args()
    current=inventory()
    errors=[]
    for name, info in current.items():
        if info['bytes'] >= 100 * 1024**2:
            errors.append(f'File exceeds GitHub regular-file size limit: {name}')
        if name.endswith('.py'):
            try: ast.parse((ROOT/name).read_text(), filename=name)
            except SyntaxError as e: errors.append(str(e))
        if name.endswith('.ipynb'):
            nb=json.loads((ROOT/name).read_text())
            if nb.get('nbformat') != 4 or not isinstance(nb.get('cells'),list):
                errors.append(f'Invalid notebook structure: {name}')
    if any(n.startswith('qasp/') for n in current):
        errors.append('QASP source must not be bundled: install the external dependency.')
    for required in ['README.md','requirements.txt','docs/EXPERIMENTS.md','REPOSITORY_PROVENANCE.json',
                     'connected_majority_iqp.py',
                     'results/connected_iqp_mlp_study/development_selection.json']:
        if required not in current: errors.append(f'Missing {required}')
    if errors:
        print('\n'.join(errors),file=sys.stderr); return 1
    if args.write_manifest:
        MANIFEST.write_text(json.dumps(current,indent=2)+'\n')
        print(f'Wrote manifest for {len(current)} files.')
    else:
        expected=json.loads(MANIFEST.read_text())
        problems=[n for n in expected if current.get(n)!=expected[n]]
        if problems:
            print('Missing or changed release files:\n'+'\n'.join(problems),file=sys.stderr);return 1
        extras=set(current)-set(expected)
        print(f'PASS: {len(expected)} release files verified; {len(extras)} additional untracked files.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
