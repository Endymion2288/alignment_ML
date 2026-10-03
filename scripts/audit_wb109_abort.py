"""Saved-data audit of world-aborter provenance and terminal geometry."""
import json
from pathlib import Path
import numpy as np
import audit_wb107_bound as previous
from audit_wb106_trace import check

ORIGINAL_VALIDATE=previous.validate
ORIGINAL_AUDIT=previous.audit

def without_abort(rows):return [r for r in rows if r['record']!='end_world_check']
def clean(rows):return [{k:v for k,v in r.items() if k!='loaded_libraries'} for r in rows]

def vector(value):
    a=np.asarray(value,dtype=float)
    check(a.shape==(3,1) and np.isfinite(a).all(),'nonfinite or invalid vector')
    return a.reshape(3)

def frame(value):
    a=np.asarray(value,dtype=float)
    check(a.shape==(4,4) and np.isfinite(a).all(),'nonfinite or invalid frame')
    check(np.array_equal(a[3],[0,0,0,1]) and np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),rtol=0,atol=1e-12),'invalid rigid transform')
    return a

def boundary_geometry(r):
    """Derive a face from actual world bounds, using the existing tolerance."""
    world=r['world']
    if world is None or world['bounds_type']!=1:
        return {'classification':'UNKNOWN','reason':'no cuboid world receipt'}
    tf=frame(world['transform']);limits=np.asarray(world['bounds_values'],dtype=float)
    check(limits.shape==(3,) and np.isfinite(limits).all() and (limits>0).all(),'world bounds invalid')
    local=(np.linalg.inv(tf)@np.r_[vector(r['position']),1.])[:3]
    direction=tf[:3,:3].T@vector(r['direction'])
    tol=r['options']['surfaceTolerance_mm']
    candidates=np.flatnonzero(np.abs(np.abs(local)-limits)<=tol)
    matches=[]
    # Every recorded world boundary must itself lie on a world cuboid face.
    # This guards against a self-consistent but changed bounds/frame receipt.
    bsurfs=r['world_boundary_surfaces'];check(len(bsurfs)==6,'world cuboid boundary count')
    faces=[]
    for b in bsurfs:
        bt=frame(b['frame']);centre=(np.linalg.inv(tf)@np.r_[bt[:3,3],1.])[:3]
        axes=np.flatnonzero(np.abs(np.abs(centre)-limits)<=tol)
        check(len(axes)==1,'world boundary inconsistent with bounds')
        axis=int(axes[0]);others=[i for i in range(3) if i!=axis]
        normal=tf[:3,:3].T@bt[:3,2]
        check(np.all(np.abs(centre[others])<=tol) and abs(abs(normal[axis])-1.)<=1e-12 and
          np.all(np.abs(normal[others])<=1e-12),'world boundary transform not a cuboid face')
        faces.append((axis,1 if centre[axis]>0 else -1))
        plane_local=(np.linalg.inv(bt)@np.r_[vector(r['position']),1.])[:3]
        if abs(plane_local[2])<=tol:matches.append({'geometry_id':b['geometry_id'],'local_z_mm':float(plane_local[2]),
          'current_surface_pointer_match':b['current_surface_pointer_match']})
    check(set(faces)=={(a,s) for a in range(3) for s in (-1,1)},'world boundary face coverage')
    valid=len(candidates)==1 and bool(np.all(np.abs(local)<=limits+tol))
    axis=int(candidates[0]) if len(candidates)==1 else None
    outward=axis is not None and direction[axis]*np.sign(local[axis])>0
    current=r['current_surface'];nav=r['navigation_boundary_surface']
    same_boundary=current is not None and nav is not None and current==nav
    plane_distance=None
    if current is not None:
        plane_distance=float((np.linalg.inv(frame(current['frame']))@np.r_[vector(r['position']),1.])[2])
    supported=valid and outward and bool(matches) and same_boundary and plane_distance is not None and abs(plane_distance)<=tol
    return {'classification':'WORLD_BOUNDARY_GEOMETRY_SUPPORTED' if supported else 'UNSUPPORTED',
       'world_name':world['name'],'world_bounds_half_lengths_mm':limits.tolist(),
       'world_local_position_mm':local.tolist(),'world_local_direction':direction.tolist(),
       'face_axis':axis,'face_sign':int(np.sign(local[axis])) if axis is not None else None,
       'face_gap_mm':float(abs(local[axis])-limits[axis]) if axis is not None else None,
       'direction_outward':bool(outward),'current_matches_navigation_boundary':same_boundary,
       'current_surface_local_z_mm':plane_distance,'world_boundary_plane_matches':matches}

def validate(rows,baseline108,baseline106,fixture):
    filtered=without_abort(rows)
    bound=ORIGINAL_VALIDATE(filtered,baseline106,fixture)
    check(clean(filtered)==clean(baseline108),'bound/control trace differs from sealed WB108')
    calls=[r['call_id'] for r in filtered if r['record']=='before_official']
    new=[r for r in rows if r['record']=='end_world_check']
    check(bool(new) and all(r.get('call_id') in calls for r in new),'missing/foreign abort calls')
    last=None
    for cid in calls:
        seq=[r for r in rows if r.get('call_id')==cid]
        checks=[r for r in seq if r['record']=='end_world_check']
        check(bool(checks),'missing per-call abort receipt')
        check(checks[0]['stage']=='prePropagation','abort initial stage')
        before=next(r for r in seq if r['record']=='before_diagnostic')
        bind=next(r for r in seq if r['record']=='bound_before')
        check(seq.index(before)<seq.index(checks[0]) and seq.index(checks[-1])<seq.index(bind),'abort receipt order')
        inp=next(r for r in seq if r['record']=='input')
        initial=next(r for r in seq if r['record']=='before_official')
        check(checks[0]['position']==initial['start_position'] and checks[0]['direction']==initial['start_direction'] and checks[0]['path_mm']==0,'abort initial state')
        options=checks[0]['options']
        check(options['maxSteps']==10000 and options['maxStepSize_mm']==10000. and options['surfaceTolerance_mm']==1e-4,'physical option identity')
        seed=np.asarray(inp['seed']).reshape(5)
        momentum=1/abs(seed[4]/.001)
        transverse=momentum*np.linalg.norm(np.asarray(initial['start_direction']).reshape(3)[:2])
        check(options['loopProtection']==(transverse<.3) and options['forward']==initial['forward'],'loop/direction identity')
        check(options['pathLimit']==float(np.finfo(float).max),'path limit identity')
        check(options['stepTolerance']==1e-4,'ACTS stepTolerance identity')
        check(options==new[0]['options'],'physical options differ across calls')
        true_rows=[]
        for k,r in enumerate(checks):
            for name in ('returned','end_of_world_before','current_volume_null_before','current_volume_null_after',
              'target_reached_before','target_reached_after','navigation_break_before','navigation_break_after',
              'current_surface_is_target','navigator_target_matches_requested'):
                check(isinstance(r[name],bool),'non-boolean flag '+name)
            vector(r['position']);vector(r['direction'])
            check(np.isfinite(r['path_mm']) and isinstance(r['step_index'],int) and r['step_index']>=0,'path/step invalid')
            check(r['stage'] in ('prePropagation','postStep'),'abort stage')
            check(r['options']==options,'options changed within call')
            check(r['end_of_world_before']==r['current_volume_null_before']==r['returned'],'abort decision changed')
            check(r['target_reached_after']==r['returned'],'original targetReached semantics')
            check(r['current_volume_null_before']==r['current_volume_null_after'],'abort mutated current volume')
            check(r['navigation_break_before']==r['navigation_break_after'],'abort mutated navigation break')
            check((r['current_volume_name'] is None)==r['current_volume_null_before'],'volume name/null mismatch')
            check(r['navigator_target_matches_requested'],'navigator target identity')
            if k:
                check(r['step_index']>=checks[k-1]['step_index'] and r['path_mm']>=checks[k-1]['path_mm'],'abort order/path')
            if r['returned']:true_rows.append(r);check(k==len(checks)-1,'continued after world abort')
        if cid==124:
            check(checks[-1]['position']==bind['position'] and checks[-1]['direction']==bind['direction'] and checks[-1]['path_mm']==bind['path_mm'],'abort/bound state mismatch')
            if len(true_rows)==1:
                last=true_rows[0]
                check(np.array_equal(last['target_surface']['frame'],bind['surface_frame']) and last['target_surface']['geometry_id']==0,'abort target frame/id')
                check(last['stage']=='postStep' and not last['current_surface_is_target'],'world termination stage or target')
                tf=frame(last['target_surface']['frame'])
                local=(np.linalg.inv(tf)@np.r_[vector(last['position']),1.])[:3]
                check(abs(local[2])>options['surfaceTolerance_mm'],'world termination actually at target')
        else:check(not true_rows,'world-abort on successful call')
    if last is None:
        return {'classification':'OTHER_OR_UNRESOLVED_TERMINATION','deeper_acceptance_or_seed_mechanism':'UNKNOWN','bound_classification':bound['classification']}
    return {'classification':'END_OF_WORLD_BEFORE_TARGET','call_id':124,'end_of_world_evaluations':len(new),
      'failed_call_evaluations':len([r for r in new if r['call_id']==124]),
      'target_reached_before':last['target_reached_before'],'target_reached_after':last['target_reached_after'],
      'current_volume_null':last['current_volume_null_before'],'navigation_break':last['navigation_break_before'],
      'position_mm':vector(last['position']).tolist(),'target_distance_mm':bound['absolute_plane_distance_mm'],
      'world_boundary_geometry':boundary_geometry(last),'deeper_acceptance_or_seed_mechanism':'UNKNOWN'}

def audit(out):
    from wb109_contract import BASE
    from wb107_contract import BASE as wb106
    load=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
    rows=load(out/'event/acts.json.calls.ndjson')
    result=validate(rows,load(BASE/'event/acts.json.calls.ndjson'),load(wb106/'event/acts.json.calls.ndjson'),json.loads((out/'fixture.json').read_text()))
    # Run unchanged WB107 file/library/conditions gates on the original saved
    # artifacts, filtering only the new passive records for its old validator.
    saved=previous.validate
    try:
        previous.validate=lambda actual,b,f:ORIGINAL_VALIDATE(without_abort(actual),b,f)
        old=ORIGINAL_AUDIT(out)
    finally:previous.validate=saved
    return {**old,'schema':'wb109_world_abort_summary_v1','bound_classification':old['classification'],**result}
