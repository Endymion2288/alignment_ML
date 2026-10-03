#!/usr/bin/env python3
"""Saved-value covariance/lineage audit. No propagation or calibrated test."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from audit_wb106_trace import check,digest
from wb111_contract import OUT as BASE,HISTORICAL
from wb107_contract import verify

OUT=ROOT/'outputs/mc24_four_station_wb112_shared_seed_audit_v1'
WORKBOOK=ROOT/'workbook/2026-10-03_112_四站共享种子不确定度与分段来源只读审计.md'
WB92=ROOT/'outputs/mc24_four_station_wb92_common_seed_acts_v3'

def matrix(a,shape):
    a=np.asarray(a,dtype=float);check(a.shape==shape and np.isfinite(a).all(),'matrix shape/finite');return a

def spd(a):
    check(np.isfinite(a).all() and np.allclose(a,a.T,rtol=1e-12,atol=1e-12),'covariance symmetry/finite')
    np.linalg.cholesky(a);return a

def quadratic(r,js,c0,cs,scales):
    """Two independent scaled covariance/normal solves; no rank truncation."""
    scales=matrix(scales,(4,));check((scales>0).all(),'positive coordinate scales')
    r=matrix(r,(8,));j=matrix(js,(8,4));c0=matrix(c0,(4,4))
    cs=[matrix(c,(4,4)) for c in cs];check(len(cs)==2,'two targets')
    scale8=np.tile(scales,2);r=r/scale8;j=j/scale8[:,None]*scales[None,:]
    c0=spd(c0/scales[:,None]/scales[None,:]);cs=[spd(c/scales[:,None]/scales[None,:]) for c in cs]
    d=np.zeros((8,8));d[:4,:4]=cs[0];d[4:,4:]=cs[1]
    shared=spd(d+j@c0@j.T);naive=shared.copy();naive[:4,4:]=0.;naive[4:,:4]=0.
    qfixed=float(r@np.linalg.solve(d,r));qshared=float(r@np.linalg.solve(shared,r));qnaive=float(r@np.linalg.solve(naive,r))
    prior_precision=np.linalg.solve(c0,np.eye(4));normal=spd(prior_precision+j.T@np.linalg.solve(d,j))
    delta=-np.linalg.solve(normal,j.T@np.linalg.solve(d,r));remaining=r+j@delta
    qprior=float(delta@prior_precision@delta);qdata=float(remaining@np.linalg.solve(d,remaining));qprofile=qprior+qdata
    err=abs(qshared-qprofile)/max(1.,abs(qshared),abs(qprofile))
    check(err<=1e-8,'Woodbury/profile arithmetic disagreement')
    check(qshared<=qfixed+1e-8*max(1.,qfixed),'shared covariance monotonicity')
    return {'Q_fixed_seed':qfixed,'Q_shared_seed':qshared,'Q_ignoring_cross_block':qnaive,
      'Q_profile_identity':qprofile,'Q_profile_prior_part':qprior,'Q_profile_data_part':qdata,
      'woodbury_relative_difference':err,'linear_seed_delta':(delta*scales).tolist(),
      'seed_delta_in_marginal_sigma':(delta/np.sqrt(np.diag(c0))).tolist(),
      'linear_residual_after_delta':(remaining*scale8).tolist(),
      'marginal_residual_in_sigma':(r/np.sqrt(np.diag(shared))).tolist(),
      'shared_covariance_scaled':shared.tolist(),'cross_block_scaled':shared[:4,4:].tolist(),
      'condition_number_shared_scaled':float(np.linalg.cond(shared)),
      'condition_number_normal_scaled':float(np.linalg.cond(normal)),
      'p_value':'NOT_EVALUATED','statistical_decision':'NOT_EVALUATED'}

def references(f,repair):
    refs=sorted([r for r in repair if r['state_index']==0],key=lambda r:r['station'])
    check([r['station'] for r in refs]==[0,1,2,3],'reference multiplicity/stations')
    evidence=[];cluster_sets=[];scales=np.asarray(f['protocol']['output_scales'])
    for station,(raw,ref) in enumerate(zip(refs,f['references'])):
        expected=dict(raw);expected['clusters']=sorted(expected['clusters']);expected['q_over_p_per_MeV']=raw['native_parameters'][4][0]
        check(expected==ref,'WB91->fixture reference mismatch')
        check(raw['run']==f['actual_run'] and raw['event']==f['actual_event'] and raw['input_event_header_verified'] is True,'reference event identity')
        ids=raw['clusters'];check(bool(ids) and len(ids)==len(set(ids)) and all(type(i) is int and i>0 for i in ids),'cluster integer identity')
        cluster_sets.append(set(ids));c=matrix(ref['fixed_z_covariance'],(4,4));cr=matrix(raw['raw_covariance'],(4,4))
        spd(c/scales[:,None]/scales[None,:]);spd(cr/scales[:,None]/scales[None,:])
        a=matrix(raw['raw_to_native_jacobian'],(5,4));b=matrix(raw['native_to_fixed_z_jacobian'],(4,5))
        native=matrix(raw['native_covariance'],(5,5));q=raw['native_parameters'][4][0]
        check(q==1e-5 and np.isclose(native[4,4],50000*q*q,rtol=1e-15,atol=0.),'dummy qop source/value')
        rebuilt=b@native@b.T;direct=(b@a)@cr@(b@a).T
        denom=max(1.,float(np.max(np.abs(c/scales[:,None]/scales[None,:]))))
        err=max(float(np.max(np.abs((rebuilt-c)/scales[:,None]/scales[None,:]))),float(np.max(np.abs((direct-c)/scales[:,None]/scales[None,:]))))/denom
        check(err<=1e-6,'covariance frame reconstruction')
        fit=matrix(raw['raw_fit'],(4,1)).reshape(4);z=raw['z_state_mm'];dz=z-raw['z_center_mm'];fixed=fit.copy();fixed[:2]+=dz*fit[2:]
        check(np.max(np.abs(matrix(ref['fixed_z_state'],(4,1)).reshape(4)-fixed))<=1e-9,'mean/frame/lever arm')
        n=matrix(raw['raw_normal'],(4,4));scaled_normal=scales[:,None]*n*scales[None,:]
        check(np.max(np.abs((cr/scales[:,None]/scales[None,:])@scaled_normal-np.eye(4)))<=1e-6,'raw covariance/normal')
        keys=[k for k in ('truth_particle_id','truth_barcode','association_id','common_track_id','route_id') if k in raw]
        evidence.append({'station':station,'z_mm':z,'h':fixed.tolist(),'marginal_sigma':np.sqrt(np.diag(c)).tolist(),
          'covariance_frame_relative_difference':err,'cluster_count':len(ids),'explicit_association_keys':keys,
          'qop_mean_per_MeV':q,'qop_variance_per_MeV2':native[4,4],'qop_sigma_over_mean':float(np.sqrt(native[4,4])/abs(q))})
    pairs=[{'stations':[i,j],'shared_cluster_ids':sorted(cluster_sets[i]&cluster_sets[j])} for i in range(4) for j in range(i+1,4)]
    return evidence,pairs

def analyze(f,repair,trace,control):
    ev,pairs=references(f,repair);seed=np.r_[np.asarray(f['references'][0]['fixed_z_state']).reshape(4),f['references'][0]['q_over_p_per_MeV']]
    check(np.array_equal(seed,np.asarray(control['seed']).reshape(5)) and control['seed_z_mm']==f['references'][0]['z_state_mm'],'common seed identity')
    inputs=[r for r in trace if r['record']=='input'];byid={r['call_id']:r for r in inputs}
    check(len(byid)==len(inputs),'duplicate input IDs');outputs={r['call_id']:r for r in trace if r['record']=='output'}
    check(len(outputs)==len([r for r in trace if r['record']=='output']),'duplicate output IDs')
    matrices={'full':[],'half':[]};residual=[];details=[];steps=np.asarray(f['protocol']['seed_steps']);scales=np.asarray(f['protocol']['output_scales'])
    for station in (1,2):
        t=control['targets'][station-1];nominal=byid[t['call_id']]
        check(nominal['station']==station and nominal['label']=='nominal' and nominal['frame']==t['frame'] and
          np.array_equal(np.asarray(nominal['seed']).reshape(5),seed),'nominal identity')
        check(outputs[t['call_id']]['h']==t['official_h'],'nominal output guard')
        r=np.asarray(t['official_h']).reshape(4)-np.asarray(f['references'][station]['fixed_z_state']).reshape(4);residual.append(r)
        used=[]
        for name,factor,labels in (('full',1.,('xi_plus','xi_minus')),('half',.5,('xi_half_plus','xi_half_minus'))):
            j=np.zeros((4,4))
            for axis in range(4):
                values=[]
                for sign,label in zip((1.,-1.),labels):
                    found=[x for x in inputs if (x['station'],x['axis'],x['label'])==(station,axis,label)]
                    check(len(found)==1,'missing/duplicate perturbation');inp=found[0];cid=inp['call_id'];xs=seed.copy();xs[axis]+=sign*factor*steps[axis]
                    check(np.array_equal(np.asarray(inp['seed']).reshape(5),xs) and inp['seed_z_mm']==control['seed_z_mm'] and inp['frame']==t['frame'],'perturbation input/frame/qop')
                    check(cid in outputs,'missing successful perturbation');values.append(matrix(outputs[cid]['h'],(4,1)).reshape(4));used.append(cid)
                j[:,axis]=(values[0]-values[1])/(2*factor*steps[axis])
            matrices[name].append(j)
        delta=(matrices['full'][-1]-matrices['half'][-1])*steps[:4][None,:]/scales[:,None]
        details.append({'station':station,'official_h':t['official_h'],'r':r.tolist(),'perturbation_call_ids':used,
          'Hxi_4D_full':matrices['full'][-1].tolist(),'Hxi_4D_half':matrices['half'][-1].tolist(),
          'halving_scaled_effect_difference_max':float(np.max(np.abs(delta)))})
    check(control['targets'][2]['official_has_value'] is False,'historical station3 failure')
    c0=np.asarray(f['references'][0]['fixed_z_covariance']);cs=[np.asarray(f['references'][s]['fixed_z_covariance']) for s in (1,2)]
    quantities={k:quadratic(np.concatenate(residual),np.vstack(js),c0,cs,scales) for k,js in matrices.items()}
    return {'schema':'wb112_shared_seed_summary_v1','integrity_gate':'PASS','classification':'DESCRIPTIVE_ONLY_ASSUMPTIONS_UNVERIFIED',
      'identity':{k:f[k] for k in ('actual_run','actual_event','input_xaod','ordinal')},'references':ev,'cluster_overlap':pairs,
      'cross_station_association':'UNKNOWN','independent_station_noise':'UNVERIFIED','physical_qop_or_prior':'UNKNOWN',
      'local_linearization_over_seed_uncertainty':'UNVERIFIED','derivative_details':details,'quadratic_diagnostics':quantities,
      'station3_uncertainty_and_profile':'UNKNOWN_MISSING_JACOBIAN','station3_fixed_mean_difference':
        read_public(BASE/'summary.json')['rows'][2]['measurement_difference'],
      'new_propagation_calls':0,'new_reconstruction_calls':0,'root_or_truth_access':False,'held_out_access':False,
      'qualification':'NOT_EVALUATED','physics_screening':'NOT_EVALUATED','final_oracle':'NOT_EVALUATED'}

def inputs():
    original=read_public(WB92/'fixtures.json')[12];f=read_public(BASE/'fixture.json')
    return original,f,Path(original['wb91_repair_path'])

def freeze():
    check(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='4station','branch')
    check(not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip(),'tracked changes')
    hashes=verify(BASE)['hashes'].copy();m=read_public(ROOT/'docs/wb111_navigation_reachability_result_manifest.json')
    check((m['execution_contract'],m['classification'])==('PASS','TARGET_REACHABLE_WITHOUT_GEOMETRY_NAVIGATION'),'historical result')
    for p,h in m['artifacts'].items():check(digest(ROOT/p)==h,'historical artifact');hashes[str(ROOT/p)]=h
    original,f,repair=inputs();check(digest(repair)==original['wb91_repair_sha256'],'WB91 source')
    paths=[Path(__file__),ROOT/'tests/test_wb112_shared_seed.py',ROOT/'docs/wb111_navigation_reachability_result_manifest.json',
      WB92/'fixtures.json',repair,repair.parent/'validation.json',repair.parent/'input_manifest.json',ROOT/'scripts/setup_environment.sh',ROOT/'research/wb91/Audit.h',
      ROOT/'research/wb91/CovarianceContract.h',ROOT/'scripts/prepare_wb91_calypso_extension.py',ROOT/'scripts/wb91_reconstruct.py',
      ROOT.parent/'calypso/Tracker/TrackerRecAlgs/TrackerSegmentFit/src/SegmentFitAlg.cxx',
      ROOT.parent/'calypso/MagneticField/MagFieldElements/MagFieldElements/FaserFieldCache.h']
    for p in paths:hashes[str(p)]=digest(p)
    check((f['index'],f['actual_run'],f['actual_event'],f['ordinal'])==(12,100044,2268,2268),'allowlist')
    OUT.mkdir(exist_ok=False)
    with (OUT/'contract_workbook.md').open('x') as w:w.write(WORKBOOK.read_text())
    hashes[str(OUT/'contract_workbook.md')]=digest(OUT/'contract_workbook.md')
    write_new(OUT/'freeze.json',{'schema':'wb112_shared_seed_freeze_v1','hashes':hashes,'branch':'4station',
      'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'population':1,
      'new_propagation_calls':0,'root_or_truth_access':False})

def run():
    frozen=verify(OUT);(OUT/'execution_lock').mkdir(exist_ok=False);original,f,repair=inputs()
    check(digest(repair)==original['wb91_repair_sha256'],'repair input');rows=[json.loads(x) for x in repair.read_text().splitlines()]
    trace=[json.loads(x) for x in (HISTORICAL/'event/acts.json.calls.ndjson').read_text().splitlines()]
    c=read_public(BASE/'control.json');s=analyze(f,rows,trace,c)
    manifest=read_public(repair.parent/'input_manifest.json')
    check(manifest['row']['input_xaod']==f['input_xaod'] and manifest['row']['xaod_entry_index']==f['ordinal'],'reconstruction source identity')
    check((manifest['geometry'],manifest['global_tag'])==('FASERNU-04','OFLCOND-FASER-06'),'reconstruction conditions')
    s['reconstruction_manifest']={k:manifest[k] for k in ('geometry','global_tag','acts','ghostbusters','truth_association')}
    write_new(OUT/'summary.json',s)
    write_new(OUT/'execution_receipt.json',{'exit_code':0,'frozen_identities':len(frozen['hashes']),'population':1,
      'numerical_reference_states':4,'propagated_covariance_targets':[1,2],'new_propagation_calls':0,'root_or_truth_access':False,
      'command':'source scripts/setup_environment.sh ml; python scripts/audit_wb112_shared_seed.py run',
      'fixture_sha256':digest(BASE/'fixture.json'),'repair_sha256':digest(repair),'trace_sha256':digest(HISTORICAL/'event/acts.json.calls.ndjson')})
    verify(OUT)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','run','verify'));a=p.parse_args()
    if a.action=='verify':verify(OUT)
    else:globals()[a.action]()
