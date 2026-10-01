"""Independent all-step arithmetic checks and all-trace vector/mixed gates."""
import json
import numpy as np

SCALE=np.array([1.,1.,.001,.001])
def vector(x):
    a=np.asarray(x,dtype=float)
    if a.shape!=(4,) or not np.isfinite(a).all():raise ValueError('invalid four-state vector')
    return a
def norm(x):return float(np.max(np.abs(vector(x)/SCALE)))
def decision(complete, prerequisites, closures, mixed):
    if not complete or not prerequisites or not closures or not mixed:return 'UNKNOWN'
    if any(x=='UNKNOWN' for x in closures+mixed):return 'UNKNOWN'
    return 'SUPPORTED_BUT_LIMITED' if all(x=='PASS' for x in closures+mixed) else 'NOT_SUPPORTED'

def mixed_checks(traces):
    groups={}
    for row in traces:
        if row['path']=='direct':groups.setdefault((row['tolerance'],row['cap_m'],row['station']),{})[row['sample']]=row
    checks=[]
    for key,rows in sorted(groups.items()):
        base={'tolerance':key[0],'cap_m':key[1],'station':key[2]}
        if set(rows)!={0,23,24} or any(x['closure_gate']=='UNKNOWN' for x in rows.values()):
            checks.append({**base,'gate':'UNKNOWN','reason':'missing or numerically unknown triple'});continue
        vectors={i:vector(row['endpoint_ladders'][-1]['closure_residual']) for i,row in rows.items()}
        budgets={i:row['closure_budget_scaled'] for i,row in rows.items()}
        values=[]
        for name,weights in (('full_minus_nominal',{23:1,0:-1}),('half_minus_nominal',{24:1,0:-1}),('full_minus_two_half_plus_nominal',{23:1,24:-2,0:1})):
            residual=sum(w*vectors[i] for i,w in weights.items());budget=sum(abs(w)*budgets[i] for i,w in weights.items())
            values.append({'name':name,'residual':residual.tolist(),'scaled':norm(residual),'budget_scaled':budget,'gate':'PASS' if norm(residual)<=budget else 'FAIL'})
        checks.append({**base,'gate':'PASS' if all(x['gate']=='PASS' for x in values) else 'FAIL','checks':values})
    return checks

def aggregate(out,inventory,p):
    raw=json.loads((out/'traces.json').read_text());traces=raw['traces'];source=inventory['traces'];seen={};unknown=0;fdunknown=0;junknown=0
    previous={};maximum={'recurrence_arithmetic_scaled':0.,'split_arithmetic_scaled':0.,'local_reference_fine_scaled':0.,'local_jacobian_fine_scaled':0.,'fd_scaled':0.}
    fd_count=0;examples=[]
    for line in (out/'steps.ndjson').open():
        row=json.loads(line);t=row['trace'];i=row['index']
        if not 0<=t<len(source) or i!=seen.get(t,0):raise ValueError('step identity/order')
        seen[t]=i+1;previous.setdefault(t,[np.zeros(4) for _ in range(3)])
        ladders=row['ladders']
        if len(ladders)==3:
            for level,l in enumerate(ladders):
                a=np.asarray(l['A'],dtype=float)
                if a.shape!=(4,4) or not np.isfinite(a).all():raise ValueError('invalid Jacobian')
                delta=vector(row['end_X'])-vector(l['reference_X'])
                if norm(delta-vector(l['delta']))>1e-12:raise ValueError('local delta arithmetic')
                computed=a@previous[t][level]+delta
                discrepancy=norm(computed-vector(l['prediction']))
                maximum['recurrence_arithmetic_scaled']=max(maximum['recurrence_arithmetic_scaled'],discrepancy)
                if discrepancy>1e-10:raise ValueError('transport recurrence arithmetic')
                maximum['split_arithmetic_scaled']=max(maximum['split_arithmetic_scaled'],norm(vector(l['position_contribution'])+vector(l['slope_contribution'])-vector(l['prediction'])))
                previous[t][level]=vector(l['prediction'])
            differences=[norm(vector(b['reference_X'])-vector(a['reference_X'])) for a,b in zip(ladders,ladders[1:])]
            coarse,fine=differences
            valid=fine<=p['reference_scaled_tolerance'] and (fine<=p['reference_contraction_ratio']*coarse or max(coarse,fine)<=p['reference_roundoff_scaled'])
            if valid!=(row['reference_gate']=='PASS'):raise ValueError('reference gate arithmetic')
            maximum['local_reference_fine_scaled']=max(maximum['local_reference_fine_scaled'],fine)
            a0,a1,a2=(np.asarray(l['A']) for l in ladders)
            scaled=lambda a:float(np.max(np.abs(a*SCALE[None,:]/SCALE[:,None])))
            jfine=scaled(a2-a1);jcoarse=scaled(a1-a0)
            if (max(jfine,jcoarse)<=p['jacobian_scaled_tolerance'])!=(row['jacobian_gate']=='PASS'):raise ValueError('Jacobian gate arithmetic')
            maximum['local_jacobian_fine_scaled']=max(maximum['local_jacobian_fine_scaled'],jfine)
        unknown+=row['reference_gate']!='PASS';junknown+=row['jacobian_gate']!='PASS';fdunknown+=row['fd_gate']=='UNKNOWN'
        if 'fd' in row:
            fd_count+=1
            maximum['fd_scaled']=max(maximum['fd_scaled'],row['fd']['fine_coarse_scaled'],*(x['versus_variational_scaled'] for x in row['fd']['rows']))
        if row['reference_gate']!='PASS' or row['jacobian_gate']!='PASS' or row['fd_gate']=='UNKNOWN':
            if len(examples)<20:examples.append(row)
    complete=len(traces)==p['expected_traces'] and sum(seen.values())==p['expected_steps'] and raw['steps']==p['expected_steps'] and all(seen.get(i)==row['steps'] for i,row in enumerate(source))
    for i,row in enumerate(traces):
        if row['trace']!=i or row['source_steps']!=source[i]['steps'] or any(row[k]!=source[i][k] for k in ('tolerance','cap_m','path','station','sample')):raise ValueError('trace identity')
        if row['closure_gate']!='UNKNOWN':
            last=row['endpoint_ladders'][-1]
            residual=vector(last['prediction'])-vector(last['actual_same_z_defect'])
            if norm(residual-vector(last['closure_residual']))>1e-12:raise ValueError('global residual arithmetic')
            budget=max(p['closure_budget_floor_scaled'],p['closure_budget_factor']*row['global_fine_scaled'],
                p['closure_budget_factor']*row['transport_fine_scaled'],p['closure_budget_factor']*raw['controls']['max_closure_scaled'])
            if budget!=row['closure_budget_scaled'] or (norm(residual)<=budget)!=(row['closure_gate']=='PASS'):raise ValueError('closure budget/gate arithmetic')
    mixed=mixed_checks(traces);complete=complete and len(mixed)==36
    prerequisites=(raw['controls']['gate']=='PASS' and raw['field_probe_max_T']<=p['field_probe_T_tolerance'] and unknown==0 and junknown==0 and fdunknown==0 and all(t['global_gate']=='PASS' for t in traces))
    return {'schema':'wb98_defect_transport_summary_v1','coverage_gate':'PASS' if complete else 'FAIL','execution_contract':'PASS' if complete and raw['controls']['gate']=='PASS' else 'FAIL',
        'numerical_prerequisites':'PASS' if prerequisites else 'UNKNOWN','hypothesis':decision(complete,prerequisites,[t['closure_gate'] for t in traces],[m['gate'] for m in mixed]),
        'traces':len(traces),'steps':sum(seen.values()),'local_reference_unknown':unknown,'local_jacobian_unknown':junknown,'fd_unknown':fdunknown,'fd_checkpoints':fd_count,
        'closure_pass':sum(t['closure_gate']=='PASS' for t in traces),'closure_fail':sum(t['closure_gate']=='FAIL' for t in traces),'closure_unknown':sum(t['closure_gate']=='UNKNOWN' for t in traces),
        'mixed':mixed,'maximum_checks':maximum,'unknown_examples':examples,'controls':raw['controls'],
        'field_probe_count':raw['field_probe_count'],'field_probe_max_T':raw['field_probe_max_T'],'traces_detail':traces,
        'qualification':'NOT_EVALUATED','physical_field_accuracy':'UNKNOWN','float_global_reference_accuracy':'UNKNOWN',
        'WB86':'FAIL','WB92':'FAIL','WB94':'UNKNOWN','WB95':'SUPPORTED_BUT_LIMITED','WB96':'NOT_SUPPORTED','WB97':'NOT_SUPPORTED'}
