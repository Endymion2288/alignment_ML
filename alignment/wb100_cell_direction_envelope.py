"""Frozen coverage and usefulness screens, distinct from formal qualification."""
import json
from collections import Counter
import numpy as np


def decision(complete, unknown, misses, information):
    if misses:
        return 'NOT_SUPPORTED'
    if not complete or unknown:
        return 'UNKNOWN'
    return 'SUPPORTED_BUT_LIMITED' if information else 'NOT_SUPPORTED'


def information(ratios, p):
    if not ratios or any(x is None or not np.isfinite(x) or x < 0 for x in ratios):
        return {'gate': 'UNKNOWN', 'n': len(ratios)}
    a = np.asarray(ratios)
    median = float(np.median(a))
    fraction = float(np.mean(a <= p['informativeness_ratio_max']))
    return {'gate': 'PASS' if median <= p['informativeness_median_ratio_max'] and fraction >= p['informativeness_fraction_required'] else 'FAIL',
            'n': len(a), 'median': median, 'fraction_le_0_5': fraction,
            'quantiles': {str(q): float(np.quantile(a, q)) for q in (0, .1, .5, .9, .99, 1)}}


def audit_row(r, p):
    e, v = r['envelope'], r['evaluation']
    expected = e['E'] + p['uncertainty_factor'] * v['uncertainty']
    if abs(v['budget'] - expected) > 1e-15 * max(1, abs(expected)):
        raise ValueError('budget arithmetic')
    resolved = v['gate'] == 'PASS' and v['defect'] > p['resolved_factor'] * v['uncertainty']
    if resolved != v['resolved'] or (resolved and v['defect'] > expected) != v['false_negative']:
        raise ValueError('direction decision')
    if abs(sum(np.diff(e['cuts_mm'])) - r['h_mm']) > 1e-12 * max(1, r['h_mm']):
        raise ValueError('face lengths')
    if 'full_direction' in r and abs(e['E'] - (max(abs(x-y) for x,y in zip(r['full_direction'], e['predictor_direction'])) + e['Rcomm'] + e['Rbend_lipschitz'] + e['Rbend_jump'])) > 1e-12:
        raise ValueError('control envelope arithmetic')
    s = v['slope']
    if s['gate'] == 'PASS':
        sr = s['defect'] > p['resolved_factor'] * s['uncertainty']
        if sr != s['resolved'] or (sr and s['defect'] > s['budget']) != s['false_negative']:
            raise ValueError('slope decision')


def aggregate(out, p, inventory):
    counts, classes, traces = Counter(), {}, {}
    ratios, components, examples = [], [], []
    with (out/'metrics.ndjson').open() as stream:
        for line in stream:
            r = json.loads(line); audit_row(r,p)
            t,i=r['trace'],r['index']
            if i != traces.get(t,0): raise ValueError('metric identity order')
            meta=inventory['traces'][t]
            if any(r[k]!=meta[k] for k in ('tolerance','cap_m','path','station','sample')): raise ValueError('inventory stratum')
            traces[t]=i+1
            c=classes.setdefault(r['classification'],Counter()); v=r['evaluation'];e=r['envelope']
            for key,value in {'steps':1,'unknown':v['gate']!='PASS' or v['slope']['gate']!='PASS',
                              'resolved':v['resolved'],'misses':v['false_negative'],
                              'slope_misses':v['slope'].get('false_negative',False),
                              'proved_constant_outside':e['proved_constant_outside']}.items():
                c[key]+=int(value);counts[key]+=int(value)
            if v['resolved'] and r['classification']!='domain':
                ratios.append(e['information_ratio'])
                components.append([e[k] for k in ('Rcomm','Rbend_lipschitz','Rbend_jump','E','baseline_C')])
            if v['resolved'] and len(examples)<12:examples.append(r)
    terminal=json.loads((out/'terminal.json').read_text())
    complete=(counts['steps']==p['expected_steps'] and len(traces)==p['expected_traces'] and not terminal['smoke'] and
              terminal['steps']==counts['steps'] and all(traces.get(i)==v['steps'] for i,v in enumerate(inventory['traces'])))
    actual_info=information(ratios,p)
    controls=json.loads((out/'controls.json').read_text());cc=Counter();cr=[];shrink=[];alias=[]
    for r in controls['rows']:
        audit_row(r,p);v=r['evaluation'];e=r['envelope']
        cc['rows']+=1;cc['unknown']+=v['gate']!='PASS' or v['slope']['gate']!='PASS'
        cc['resolved']+=v['resolved'];cc['misses']+=v['false_negative'];cc['slope_misses']+=v['slope'].get('false_negative',False)
        cc[r['origin']]+=1
        if v['resolved']:cr.append(e['information_ratio'])
        if r['WB99_alias']:alias.append({'kind':r['kind'],'h_mm':r['h_mm'],'q':r['q_over_p_Acts'],'E':e['E'],'defect':v['defect'],'covered':not v['false_negative']})
        if r['origin']=='fixed_shrink':shrink.append({k:r[k] for k in ('kind','h_mm','q_over_p_Acts')}|{'E':e['E'],'Rcomm':e['Rcomm'],'Rbend_lipschitz':e['Rbend_lipschitz'],'Rbend_jump':e['Rbend_jump'],'defect':v['defect']})
    shrink_zero=[r for r in shrink if r['h_mm']==min(p['shrink_h_mm'])]
    control_info={'gate':'PASS' if cr and all(x is not None and x<=p['informativeness_ratio_max'] for x in cr) else 'FAIL',
                  'n':len(cr),'maximum_ratio':max(cr) if cr else None}
    prereq=(complete and cc['WB99']==p['expected_old_controls'] and cc['fixed_shrink']==72 and len(alias)==18 and
            all(x['E']>0 and x['covered'] for x in alias) and len(shrink_zero)==9 and
            all(x['E']<=p['absolute_direction_floor'] for x in shrink_zero))
    unknown=counts['unknown']+cc['unknown'];misses=counts['misses']+counts['slope_misses']+cc['misses']+cc['slope_misses']
    usable=actual_info['gate']=='PASS' and control_info['gate']=='PASS'
    comp=np.asarray(components)
    return {'schema':'wb100_summary_v1','execution_contract':'PASS' if prereq else 'FAIL','complete':complete,
            'coverage':'NOT_SUPPORTED' if misses else ('UNKNOWN' if unknown or not prereq else 'PASS'),
            'hypothesis':decision(prereq,unknown,misses,usable),'qualification':'NOT_EVALUATED','held_out_access':False,'new_wrapper_queries':0,
            'counts':dict(counts),'classes':{k:dict(v) for k,v in classes.items()},'traces':len(traces),'controls':dict(cc),
            'actual_information':actual_info,'control_information':control_info,'aliases':alias,'shrink':shrink,'examples':examples,
            'resolved_non_domain_component_medians':{k:float(np.median(comp[:,i])) for i,k in enumerate(('Rcomm','Rbend_lipschitz','Rbend_jump','E','baseline_C'))} if len(comp) else {},
            'historical':{'WB86':'FAIL','WB92':'FAIL','WB94':'UNKNOWN','WB95':'SUPPORTED_BUT_LIMITED','WB96':'NOT_SUPPORTED','WB97':'NOT_SUPPORTED','WB98':'SUPPORTED_BUT_LIMITED','WB99':'NOT_SUPPORTED'}}
