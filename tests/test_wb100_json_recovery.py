import importlib.util,json
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/wb100_aggregation_recovery.py'
spec=importlib.util.spec_from_file_location('recovery',path)

def test_lossless_counts_and_scientific_values(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(path.parent))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    value={'counts':{'misses':np.int64(0),'resolved':np.int64(17781)},'budget':np.float64(1.234e-9),'nested':[np.bool_(True),{'gate':'PASS'}]}
    normalized=module.json_scalars(value)
    assert json.loads(json.dumps(normalized))=={'counts':{'misses':0,'resolved':17781},'budget':1.234e-9,'nested':[True,{'gate':'PASS'}]}
    assert normalized['budget']==value['budget']
