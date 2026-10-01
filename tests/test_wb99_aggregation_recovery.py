"""Regression on one immutable already-seen exported step, not a new physics call."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import wb99_recover_aggregation as recovery
import wb99_contract as contract
import wb97_contract as local
from alignment.wb90_measurement_contract import read_public
def test_patch_changes_only_two_export_keys():
    before=recovery.SOURCE.read_text();after=recovery.corrected_source()
    assert len([1 for a,b in zip(before.splitlines(),after.splitlines()) if a!=b])==1
    assert after.count("'accepted' if name=='accepted_branch'")==2
def test_real_export_reader_and_all_decisions_on_one_frozen_step(tmp_path):
    parent=recovery.PARENT;(tmp_path/'event').mkdir()
    with (parent/'event/acts.json.doubling.ndjson').open() as f:fidelity=json.loads(next(f));step=json.loads(next(f))
    assert 'accepted_position_closure_mm' in step and 'accepted_branch_position_closure_mm' not in step
    ref=tmp_path/'ref';ref.mkdir()
    with (contract.WB97/'steps.ndjson').open() as f:controls=next(f);source=next(f)
    (ref/'steps.ndjson').write_text(controls+source+json.dumps({'record':'terminal','steps':1,'traces':1})+'\n')
    (tmp_path/'event/acts.json.doubling.ndjson').write_text('\n'.join(json.dumps(x) for x in (fidelity,step,{'record':'terminal','steps':1,'traces':1,'replay_max_T':0}))+'\n')
    (tmp_path/'controls.json').symlink_to(parent/'controls.json')
    p=read_public(parent/'protocol.json');p.update(expected_steps=1,expected_traces=1)
    inventory=read_public(parent/'inventory.json');inventory['traces']=inventory['traces'][:1];inventory['traces'][0]['steps']=1
    nodes=np.fromfile(local.NODES,dtype='<i2').reshape(81,81,861,3)
    s=recovery.load_corrected()(tmp_path,inventory,p,ref,read_public(local.OLD),nodes)
    assert s['execution_contract']=='PASS' and s['counts']['steps']==1
    assert s['hypothesis']=='NOT_SUPPORTED' # Known independent control alias survives reader repair.
    assert s['counts']['source_unknown']==s['counts']['reference_unknown']==0
