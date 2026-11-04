"""Maintainer formatting utility: normalize cell IDs to notebook format 4.5."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    files=list(ROOT.glob('*.ipynb'))+list((ROOT/'tests').glob('*.ipynb'))
    for path in files:
        nb=json.loads(path.read_text())
        nb['nbformat_minor']=5
        seen=set()
        for i,cell in enumerate(nb['cells']):
            ident=cell.get('id',f'cell-{i:04d}')
            if ident in seen:ident=f'cell-{i:04d}'
            cell['id']=ident;seen.add(ident)
        path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n')
    print(f'Normalized {len(files)} notebooks without changing code or outputs.')
