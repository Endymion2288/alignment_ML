"""Streaming all-step audit; unknown coverage cannot become a supported hypothesis."""
import json
import math
from collections import Counter


def decision(complete, closure_failures, reference_unknown, controls_pass, counts, p):
    if not complete or closure_failures or reference_unknown or not controls_pass:
        return 'UNKNOWN'
    crossing = sum(counts.get(k, {}).get('count', 0) for k in ('domain', 'mesh'))
    significant_crossing = sum(counts.get(k, {}).get('significant', 0) for k in ('domain', 'mesh'))
    total_significant = sum(x.get('significant', 0) for x in counts.values())
    interior = counts.get('interior', {})
    cr = significant_crossing / crossing if crossing else 0
    ir = interior.get('significant', 0) / interior['count'] if interior.get('count', 0) else None
    if ir is None:
        return 'UNKNOWN'  # No cell-interior negative-control population.
    concentrated = total_significant > 0 and significant_crossing / total_significant >= p['crossing_significant_fraction_required']
    enriched = cr > 0 and cr >= p['crossing_rate_enrichment_required'] * ir
    return 'SUPPORTED_BUT_LIMITED' if concentrated and enriched else 'NOT_SUPPORTED'


def aggregate(path, inventory, p):
    counts = {}; cells = {}; reasons = Counter(); traces = Counter(); controls = None; terminal = None
    closure_failures = reference_unknown = total = 0
    maxima = {x: 0. for x in ('position_closure_mm', 'direction_closure', 'stage_closure_mm', 'estimate_closure_mm',
                             'fine_reference_scaled', 'position_L1_mm', 'direction_max')}
    largest = []; unknown_examples = []; domain_examples = []
    expected = inventory['traces']
    for line in path.open():
        row = json.loads(line)
        if row['record'] == 'controls':
            if controls is not None or total: raise ValueError('control record order')
            controls = row; continue
        if row['record'] == 'terminal':
            if terminal is not None: raise ValueError('duplicate terminal')
            terminal = row; continue
        if row['record'] != 'step' or terminal is not None: raise ValueError('step record order')
        if controls is None: raise ValueError('missing controls')
        t = row['trace']; source = expected[t]
        if row['index'] != traces[t] or any(row[k] != source[k] for k in ('tolerance', 'cap_m', 'path', 'station', 'sample')):
            raise ValueError('step identity/order changed')
        traces[t] += 1; total += 1
        if traces[t] > source['steps']: raise ValueError('extra source step')
        if traces[t] == source['steps'] and row['saved_step']['position_mm'] != source['last_position_mm']:
            raise ValueError('terminal free state changed')
        for key in ('position_L1_mm', 'direction_max', 'reference_fine_scaled'):
            if key in row and (not math.isfinite(row[key]) or row[key] < 0): raise ValueError('nonfinite/negative metric')
        closure = row.get('closure', {})
        for source_key, target_key in (('position_max_mm','position_closure_mm'), ('direction_max','direction_closure'),
                                       ('stage_max_mm','stage_closure_mm'), ('estimate_difference_mm','estimate_closure_mm')):
            maxima[target_key] = max(maxima[target_key], closure.get(source_key, 0))
        # Recompute numeric gates from raw vectors, independently of the C++ booleans.
        if row['closure_gate'] == 'PASS':
            estimate = row['saved_step']['accepted_error_estimate']
            if (closure['position_max_mm'] > p['closure_position_mm'] or closure['stage_max_mm'] > p['closure_position_mm']
                or closure['direction_max'] > p['closure_direction']
                or closure['estimate_difference_mm'] > max(p['closure_estimate_absolute_mm'], p['closure_estimate_relative']*estimate)):
                raise ValueError('closure gate arithmetic disagrees')
        else: closure_failures += 1
        if 'reference_ladder' in row:
            ladders = row['reference_ladder']
            if [x['dz_mm'] for x in ladders] != p['reference_dz_mm']: raise ValueError('ladder changed')
            differences = []
            for a,b in zip(ladders,ladders[1:]):
                differences.append(max(max(abs(x-y) for x,y in zip(a['position_mm'],b['position_mm'])),
                                       max(abs(x-y) for x,y in zip(a['direction'],b['direction']))/p['direction_scale']))
            coarse,fine = differences
            valid = fine <= p['reference_scaled_tolerance'] and (fine <= p['reference_contraction_ratio']*coarse
                    or max(coarse,fine) <= p['reference_roundoff_scaled'])
            if valid != (row['reference_gate']=='PASS'): raise ValueError('reference gate arithmetic disagrees')
            dp = [x-y for x,y in zip(row['saved_step']['position_mm'],ladders[-1]['position_mm'])]
            du = [x-y for x,y in zip(row['saved_step']['direction'],ladders[-1]['direction'])]
            if dp != row['position_defect_mm'] or du != row['direction_defect']: raise ValueError('defect vector arithmetic disagrees')
            uncertainty = sum(abs(x-y) for x,y in zip(ladders[-1]['position_mm'],ladders[-2]['position_mm']))
            threshold = max(p['significance_factor']*uncertainty, p['significance_factor']*closure['position_L1_mm'],
                            p['significance_factor']*controls['controls']['calibration_C']*row['saved_step']['accepted_error_estimate'],
                            p['significance_absolute_position_mm'])
            if bool(valid and sum(abs(x) for x in dp)>threshold) != row['significant_underestimate']:
                raise ValueError('significance arithmetic disagrees')
        if row['reference_gate'] != 'PASS':
            reference_unknown += 1; reasons[row.get('unknown_reason','reference contraction/precision')]+=1
            if len(unknown_examples)<30: unknown_examples.append(row)
        category = row.get('classification',{}).get('class','unclassified')
        significant = int(row.get('significant_underestimate',False))
        for group,key in ((counts,category),(cells,str((row['tolerance'],row['cap_m'],row['path'],row['station'],row['sample'])))):
            cell = group.setdefault(key, {'count':0,'significant':0,'reference_unknown':0,'closure_failures':0,
                                        'max_position_L1_mm':0.,'max_direction':0.})
            cell['count']+=1;cell['significant']+=significant;cell['reference_unknown']+=row['reference_gate']!='PASS';cell['closure_failures']+=row['closure_gate']!='PASS'
            cell['max_position_L1_mm']=max(cell['max_position_L1_mm'],row.get('position_L1_mm',0))
            cell['max_direction']=max(cell['max_direction'],row.get('direction_max',0))
        for key in ('position_L1_mm','direction_max'):
            maxima[key]=max(maxima[key],row.get(key,0))
        maxima['fine_reference_scaled']=max(maxima['fine_reference_scaled'],row.get('reference_fine_scaled',0))
        if category=='domain' and len(domain_examples)<30: domain_examples.append(row)
        if 'position_L1_mm' in row:
            largest.append(row);largest.sort(key=lambda x:x['position_L1_mm'],reverse=True);del largest[20:]
    complete = (terminal is not None and len(traces)==p['expected_traces'] and total==inventory['steps']
                and terminal['steps']==total and terminal['traces']==len(traces)
                and all(traces[i]==x['steps'] for i,x in enumerate(expected)))
    controls_pass = controls is not None and controls['controls']['gate']=='PASS' and controls['probe_max_T']<=p['field_probe_T_tolerance']
    return {'schema':'wb97_local_defect_summary_v1','coverage_gate':'PASS' if complete else 'FAIL',
            'execution_contract':'PASS' if complete and controls_pass and not closure_failures else 'FAIL',
            'reference_contract':'PASS' if complete and not reference_unknown else 'UNKNOWN',
            'hypothesis':decision(complete,closure_failures,reference_unknown,controls_pass,counts,p),
            'traces':len(traces),'steps':total,'closure_failures':closure_failures,'reference_unknown':reference_unknown,
            'unknown_reasons':dict(reasons),'controls':controls,'maxima':maxima,'classification':counts,'cells':cells,
            'largest_defects':largest,'domain_examples':domain_examples,'unknown_examples':unknown_examples,
            'qualification':'NOT_EVALUATED','cumulative_attribution':'UNKNOWN','physical_field_accuracy':'UNKNOWN',
            'WB86':'FAIL','WB92':'FAIL','WB94':'UNKNOWN','WB95':'SUPPORTED_BUT_LIMITED','WB96':'NOT_SUPPORTED'}
