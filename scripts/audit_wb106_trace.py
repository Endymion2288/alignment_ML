#!/usr/bin/env python3
"""Independent audit for the WB106 instrumented saved-call trace."""
from __future__ import annotations
import json
from pathlib import Path
import hashlib
import re
import numpy as np
from alignment.common_track_geometry import se3_exp


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def check(condition, message):
    if not condition: raise ValueError(message)


def validate_trace(lines, fixture):
    events=[r for r in lines if r['record']=='event']
    check(len(events)==1,'event receipt multiplicity')
    for key in ('input_xaod','ordinal','actual_run','actual_event'):
        check(events[0][key]==fixture[key],'identity '+key)
    p=fixture['protocol'];refs=fixture['references']
    seed=np.r_[np.asarray(refs[0]['fixed_z_state']).reshape(4),refs[0]['q_over_p_per_MeV']]
    steps=np.asarray(p['seed_steps']);gs=np.asarray(p['geometry_steps'])
    z=refs[0]['z_state_mm']
    expected=[]
    for station in range(4):
        expected += [(station,-1,k) for k in ('nominal','repeat')]
        expected += [(station,axis,'xi_'+k) for axis in range(5) for k in ('plus','minus','half_plus','half_minus')]
        expected += [(station,-1,'xi_mixed_'+k) for k in ('full','half')]
        if station:
            expected += [(station,axis,'theta_'+k) for axis in range(6) for k in ('plus','minus','half_plus','half_minus')]
            expected += [(station,-1,'theta_mixed_'+k) for k in ('full','half')]
    inputs=[r for r in lines if r['record']=='input']
    check(bool(inputs) and len(inputs)<=len(expected),'input count')
    terminals=[r for r in lines if r['record']=='terminal']
    check(len(terminals)==1 and lines[-1] is terminals[0],'terminal receipt')
    check(terminals[0]['calls']==len(inputs),'terminal calls')
    factors={'plus':1.,'minus':-1.,'half_plus':.5,'half_minus':-.5,'full':1.,'half':.5}
    for cid,r in enumerate(inputs):
        station,axis,label=expected[cid]
        check((r['call_id'],r['station'],r['axis'],r['label'])==(cid,station,axis,label),'call order')
        base=np.eye(4);base[2,3]=refs[station]['z_state_mm']
        frame=base.copy();xs=seed.copy()
        if label.startswith('xi_'):
            delta=np.zeros(5)
            if 'mixed' in label: delta=steps*np.array([1,-1,1,-1,1])*.25;key=label.split('mixed_')[1]
            else: delta[axis]=steps[axis];key=label[len('xi_'):]
            xs += delta*factors[key]
        elif label.startswith('theta_'):
            delta=np.zeros(6)
            if 'mixed' in label: delta=gs*np.array([1,-1,1,-1,1,-1])*.25;key=label.split('mixed_')[1]
            else: delta[axis]=gs[axis];key=label[len('theta_'):]
            frame=se3_exp(delta*factors[key])@base
        check(np.array_equal(np.asarray(r['seed']).reshape(5),xs),'seed perturbation/qop')
        check(r['seed_z_mm']==z,'seed z')
        check(np.allclose(r['frame'],frame,rtol=0,atol=1e-10),'frame perturbation')
        rows=[x for x in lines if x.get('call_id')==cid]
        kinds=[x['record'] for x in rows]
        if 'boundary' in kinds:
            check(kinds==['input','boundary'] and station==0,'boundary ordering')
        else:
            check(kinds in (['input','before_official','after_official'],['input','before_official','after_official','output']),'official ordering')
            b=rows[1];a=rows[2]
            direction=np.r_[xs[2:4],1.];direction/=np.linalg.norm(direction)
            pos=np.r_[xs[:2],z];distance=float((frame[:3,3]-pos)@direction)
            check(np.allclose(b['start_position'],pos[:,None],rtol=0,atol=1e-9),'start position')
            check(np.allclose(b['start_direction'],direction[:,None],rtol=0,atol=1e-12),'start direction')
            check(abs(b['distance']-distance)<1e-8 and b['forward']==(distance>=0),'direction/distance')
            check(np.allclose(b['target_frame'],frame,rtol=0,atol=1e-10) and b['target_geometry_id']==0,'synthetic plane')
            if not a['has_value']:
                check(cid==len(inputs)-1 and kinds[-1]=='after_official','fail-closed no later calls')
            else:
                check(kinds[-1]=='output','missing successful output')
                o=rows[-1];gp=np.asarray(o['global_position']).reshape(3)
                local=(np.linalg.inv(frame)@np.r_[gp,1.])[:3]
                check(np.allclose(np.asarray(o['local']).reshape(3),local,rtol=0,atol=1e-8),'output local transform')
    sensors=[r for r in lines if r['record']=='sensors']
    check([r['station'] for r in sensors]==list(range(inputs[-1]['station']+1)),'sensor prefix')
    for row in sensors:
        ref=refs[row['station']]
        check([r['strip'] for r in row['sensors']]==ref['clusters'],'sensor cluster identity')
        check(len({r['geometry_id'] for r in row['sensors']})==len(row['sensors']),'geometry id multiplicity')
    return inputs


def audit(out: Path) -> dict:
    event = out/'event'
    fixture = json.loads((event/'fixture.json').read_text())
    lines = [json.loads(line) for line in (event/'acts.json.calls.ndjson').read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError('empty diagnostic trace')
    event_rows = [x for x in lines if x.get('record') == 'event']
    sensor_rows = [x for x in lines if x.get('record') == 'sensors']
    inputs = [x for x in lines if x.get('record') == 'input']
    before = [x for x in lines if x.get('record') == 'before_official']
    after = [x for x in lines if x.get('record') == 'after_official']
    outputs = [x for x in lines if x.get('record') == 'output']
    terminals = [x for x in lines if x.get('record') == 'terminal']
    inputs=validate_trace(lines,fixture)
    er = event_rows[0]
    for key in ('input_xaod', 'ordinal', 'actual_run', 'actual_event'):
        if er[key] != fixture[key]:
            raise ValueError('event identity mismatch '+key)
    if any(x['station'] not in (0,1,2,3) or len(x['sensors']) == 0 for x in sensor_rows):
        raise ValueError('sensor receipt incomplete')
    if not before or not after or len(after) != len(before):
        raise ValueError('official call receipt incomplete')
    failure = any(not x.get('has_value', True) for x in after)
    log = (event/'athena.log').read_text(errors='replace')
    required = {
        'event_header': 'start processing event #2268, run #100044' in log,
        'sqlite': str(out/'identity_payload/tracker_alignment.sqlite') in log,
        'field_service': 'Using FASER magnetic field service' in log,
        'field_map': 'Initialized the field map from ' in log,
        'surface_error': 'SurfaceError:1. Returning empty parameters.' in log,
        'fail_closed': 'WB92 fail-closed: official ACTS propagation returned no state' in log,
    }
    identity_ok = all(required[k] for k in ('event_header','sqlite','field_service','field_map'))
    frozen=json.loads((out/'freeze.json').read_text())
    check(digest(event/'fixture.json')==frozen['hashes'][str(out/'fixture.json')],'frozen fixture')
    loaded=er['loaded_libraries']
    binary=json.loads((out/'binary_manifest.json').read_text())
    check(binary['binary'] in loaded and digest(binary['binary'])==binary['sha256'],'diagnostic binary')
    for name,h in json.loads((out/'generated_source_manifest.json').read_text()).items():
        check(digest(out/'isolated_source'/name)==h,'generated source')
    official=[path for path in loaded if any(k in path for k in ('FaserActs','ActsCore','MagField'))]
    check(bool(official),'official libraries absent')
    for path in official:
        check(path in frozen['hashes'] and digest(path)==frozen['hashes'][path],'official library hash')
    maps=re.findall(r'Initialized the field map from\s+([^\n]+)',log)
    check(bool(maps),'field map absent')
    for path in maps:
        path=path.strip();check(path in frozen['hashes'] and digest(path)==frozen['hashes'][path],'field map hash')
    check(identity_ok,'conditions evidence')
    # A trace proves the official call returned no end parameters. It does not
    # prove whether an intermediate navigator surface or the requested plane
    # was the first failing surface.
    classification = 'UNKNOWN'
    reason = 'missing target/intermediate surface error attribution'
    if failure and identity_ok and required['surface_error'] and required['fail_closed']:
        reason = 'official target propagation returned empty parameters with valid conditions receipt; failure surface not exposed'
    elif not identity_ok:
        reason = 'conditions/event identity receipt incomplete'
    return {
        'schema': 'wb106_surface_trace_summary_v1',
        'identity': {k: fixture[k] for k in ('input_xaod','ordinal','actual_run','actual_event')},
        'trace_counts': {'lines':len(lines),'inputs':len(inputs),'before_official':len(before),
                         'after_official':len(after),'outputs':len(outputs),'terminals':len(terminals)},
        'conditions_evidence': required,
        'failure_reproduced': bool(failure),
        'integrity_gate': 'PASS',
        'classification': classification,
        'reason': reason,
        'new_physical_calls': len(before),
        'boundary_calls': len(inputs)-len(before),
        'failed_call': next((inputs[r['call_id']] for r in after if not r['has_value']),None),
        'held_out_access': False,
        'next_required_hook': 'Acts propagation error category plus target/intermediate surface attribution',
    }


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--output-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.write_text(json.dumps(audit(a.output_root.resolve()), sort_keys=True, indent=2)+'\n')
