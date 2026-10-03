"""Shared-seed arithmetic controls and mutation checks on saved seen inputs."""
import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_wb112_shared_seed import quadratic,references,analyze,inputs,BASE,HISTORICAL

def test_same_direction_shared_seed_matches_analytic_profile():
    q=quadratic(np.ones(8),np.vstack([np.eye(4)]*2),np.eye(4),[np.eye(4)]*2,np.ones(4))
    assert q['Q_fixed_seed']==pytest.approx(8.)
    assert q['Q_shared_seed']==pytest.approx(8./3.)
    assert q['Q_ignoring_cross_block']==pytest.approx(4.)
    assert q['Q_profile_identity']==pytest.approx(8./3.)
    assert q['linear_seed_delta']==pytest.approx([-2./3.]*4)
    assert q['p_value']==q['statistical_decision']=='NOT_EVALUATED'

def test_opposite_contrast_cannot_be_absorbed_by_shared_seed():
    q=quadratic(np.r_[np.ones(4),-np.ones(4)],np.vstack([np.eye(4)]*2),np.eye(4),[np.eye(4)]*2,np.ones(4))
    assert q['Q_shared_seed']==pytest.approx(8.) and q['Q_ignoring_cross_block']==pytest.approx(4.)
    assert q['linear_seed_delta']==pytest.approx([0.]*4)

def test_mixed_unit_conditioning_preserves_quadratic_and_seed_delta():
    r=np.arange(8,dtype=float);j=np.vstack([np.eye(4),2*np.eye(4)]);c=np.diag([1.,4.,.002**2,.003**2]);cs=[c,2*c]
    a=quadratic(r,j,c,cs,np.ones(4));b=quadratic(r,j,c,cs,np.array([1.,1.,.001,.001]))
    for k in ('Q_fixed_seed','Q_shared_seed','Q_profile_identity','Q_ignoring_cross_block'):
        assert a[k]==pytest.approx(b[k],rel=1e-12)
    assert a['linear_seed_delta']==pytest.approx(b['linear_seed_delta'],rel=1e-12)

@pytest.mark.parametrize('bad',['negative','singular','asymmetric','NaN','infinite_r','zero_scale'])
def test_invalid_covariance_or_response_rejected_without_floor(bad):
    r=np.ones(8);j=np.vstack([np.eye(4)]*2);c=np.eye(4);s=np.ones(4)
    if bad=='negative':c[0,0]=-1.
    elif bad=='singular':c[0,0]=0.
    elif bad=='asymmetric':c[0,1]=.1
    elif bad=='NaN':j[0,0]=np.nan
    elif bad=='infinite_r':r[0]=np.inf
    elif bad=='zero_scale':s[0]=0.
    with pytest.raises((ValueError,np.linalg.LinAlgError)):quadratic(r,j,c,[np.eye(4)]*2,s)

@pytest.fixture(scope='module')
def saved():
    _,f,p=inputs();repair=[json.loads(x) for x in p.read_text().splitlines()]
    trace=[json.loads(x) for x in (HISTORICAL/'event/acts.json.calls.ndjson').read_text().splitlines()]
    control=json.loads((BASE/'control.json').read_text());return f,repair,trace,control

def test_reference_lineage_is_a_local_contract_not_association(saved):
    f,repair,_,_=saved;ev,pairs=references(f,repair)
    assert len(ev)==4 and all(not row['explicit_association_keys'] for row in ev)
    assert all(not row['shared_cluster_ids'] for row in pairs)
    assert all(row['qop_sigma_over_mean']>200. for row in ev)

@pytest.mark.parametrize('bad',['header','cluster','native_qop','covariance','mean','frame_map','missing_reference',
  'seed','missing_perturbation','wrong_perturbation','wrong_target','duplicate_output'])
def test_saved_identity_covariance_and_response_mutations_rejected(saved,bad):
    f,repair,trace,c=copy.deepcopy(saved);ref=next(x for x in repair if x['state_index']==0 and x['station']==0)
    if bad=='header':ref['event']+=1
    elif bad=='cluster':ref['clusters'][0]+=1
    elif bad=='native_qop':ref['native_parameters'][4][0]*=1000.
    elif bad=='covariance':f['references'][0]['fixed_z_covariance'][0][0]*=2.
    elif bad=='mean':f['references'][0]['fixed_z_state'][0][0]+=1.
    elif bad=='frame_map':ref['raw_to_native_jacobian'][0][0]+=1.
    elif bad=='missing_reference':repair.remove(ref)
    elif bad=='seed':c['seed'][0][0]+=1.
    elif bad=='missing_perturbation':
        trace.remove(next(x for x in trace if x['record']=='input' and x['station']==1 and x['label']=='xi_plus' and x['axis']==0))
    elif bad=='wrong_perturbation':
        x=next(x for x in trace if x['record']=='input' and x['station']==1 and x['label']=='xi_plus' and x['axis']==0);x['seed'][4][0]*=1000.
    elif bad=='wrong_target':c['targets'][0]['frame'][2][3]+=1.
    elif bad=='duplicate_output':trace.append(copy.deepcopy(next(x for x in trace if x['record']=='output')))
    with pytest.raises((ValueError,np.linalg.LinAlgError)):analyze(f,repair,trace,c)

def test_analyzer_preserves_unknown_station3_and_never_invokes_backend():
    source=(Path(__file__).resolve().parents[1]/'scripts/audit_wb112_shared_seed.py').read_text()
    assert 'UNKNOWN_MISSING_JACOBIAN' in source and 'DESCRIPTIVE_ONLY_ASSUMPTIONS_UNVERIFIED' in source
    assert 'calypso_payload_command' not in source and 'condor_submit' not in source
    assert 'np.linalg.pinv' not in source and 'scipy.stats' not in source
