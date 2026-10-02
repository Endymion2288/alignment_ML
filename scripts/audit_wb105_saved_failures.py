#!/usr/bin/env python3
"""Independent offline audit of saved WB104/WB92 failure artifacts."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def arr(x, shape):
    a = np.asarray(x, dtype=float).reshape(shape)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite')
    return a


def peak(x):
    return float(np.max(np.abs(x)))


def derivative(samples, steps, scales, protocol):
    full = np.asarray([(arr(s['plus'], 4)-arr(s['minus'], 4))/2 for s in samples]).T / scales[:, None]
    half = np.asarray([arr(s['half_plus'], 4)-arr(s['half_minus'], 4) for s in samples]).T / scales[:, None]
    error = peak(full-half)
    tolerance = protocol['step_halving_absolute_floor'] + protocol['step_halving_relative_tolerance']*peak(half)
    return {'error': error, 'tolerance': tolerance, 'pass': error <= tolerance,
            'scaled_effect': half.tolist(), 'jacobian': (half*scales[:, None]/np.asarray(steps)[None, :]).tolist()}


def taylor(sample, nominal, jacobian, scales, protocol):
    delta = arr(sample['delta'], jacobian.shape[1]); effect = jacobian @ delta
    stored = arr(sample['linear_effect'], 4)
    if not np.allclose(effect, stored, rtol=1e-10, atol=1e-12):
        raise ValueError('stored linear effect mismatch')
    full = peak((arr(sample['full'], 4)-nominal-effect)/scales)
    half = peak((arr(sample['half'], 4)-nominal-.5*effect)/scales)
    tolerance = protocol['taylor_absolute_floor'] + protocol['taylor_relative_tolerance']*peak(effect/scales)
    contraction = half/full if full > protocol['taylor_absolute_floor'] else None
    return {'full_error': full, 'half_error': half, 'tolerance': tolerance,
            'contraction': contraction,
            'pass': full <= tolerance and (contraction is None or contraction <= protocol['taylor_contraction_maximum'])}


def audit_event(root: Path, index: int):
    d = root/'events'/f'{index:02d}'
    fixture = json.loads((d/'fixture.json').read_text())
    result = {'index': index, 'identity': {k: fixture[k] for k in ('input_xaod','ordinal','actual_run','actual_event')}}
    acts_path = d/'acts.json'
    if not acts_path.exists():
        result.update({'gate':'EXECUTION_FAIL','reason':'acts.json absent','log_sha256':hashlib.sha256((d/'athena.log').read_bytes()).hexdigest()})
        return result
    acts = json.loads(acts_path.read_text()); protocol = fixture['protocol']; scales=np.asarray(protocol['output_scales'])
    for k in ('input_xaod','ordinal','actual_run','actual_event'):
        if acts[k] != fixture[k]: raise ValueError(f'identity {index} {k}')
    recomputed=[]; stored=[]
    for target in acts['targets']:
        xi=derivative(target['xi'], protocol['seed_steps'], scales, protocol)
        tx=taylor(target['xi_direction'], arr(target['h'],4), np.asarray(xi['jacobian']), scales, protocol)
        recomputed += [{'name':'Hxi_step_halving','station':target['station'],**xi},
                       {'name':'Hxi_Taylor','station':target['station'],**tx}]
        v=json.loads((d/'validation.json').read_text())
        stored += [c for c in v['checks'] if c['name'] in ('Hxi_step_halving','Hxi_Taylor') and c['station']==target['station']]
    diffs=[]
    for r,s in zip(recomputed,stored):
        for k in ('error','tolerance','full_error','half_error','contraction'):
            if k in r and k in s:
                if r[k] is not None and s[k] is not None:
                    diffs.append(abs(float(r[k])-float(s[k])))
        if r['pass'] != s['pass']: raise ValueError(f'decision mismatch {index} {r["name"]} {r["station"]}')
    result.update({'gate':'PASS','validation_gate':v['gate'],'checks_recomputed':len(recomputed),
                   'max_scalar_difference':max(diffs) if diffs else 0.,'recomputed':recomputed,
                   'stored_failure_count':sum(not c['pass'] for c in stored),
                   'acts_sha256':hashlib.sha256(acts_path.read_bytes()).hexdigest(),
                   'validation_sha256':hashlib.sha256((d/'validation.json').read_bytes()).hexdigest()})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();results=[audit_event(root,i) for i in (1,4,8,12,16,20)]
    out={'schema':'wb105_saved_failure_audit_v1','root':str(root),'events':results,
         'gate':'PASS' if all(r['gate']=='PASS' for r in results) else 'FAIL',
         'new_physical_calls':0,'held_out_access':False}
    a.output.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')


if __name__=='__main__': main()
