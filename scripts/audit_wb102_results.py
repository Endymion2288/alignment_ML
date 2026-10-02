"""Independent streaming and scalar audits for WB102 outputs."""
import json
from pathlib import Path
import numpy as np
from alignment.wb90_measurement_contract import read_public


def require(value, message):
    if not value:
        raise ValueError(message)


def audit_trial_stream(out, lam, raw):
    directory = Path(out)/'event'/f'lambda_{lam:g}'
    from alignment.wb90_measurement_contract import ROOT, digest
    source_path = ROOT/'scripts/audit_wb101_trials.py'
    source = source_path.read_text()
    # Keep the full independent RKN/envelope arithmetic; adapt only output layout
    # and the fact that new disabled trajectories have no historical traces.
    from wb101_sources import replace_one
    source = replace_one(source, "if setting['direction_threshold']==0:historical[a['call_id']]=b",
                         "if historical_enabled and setting['direction_threshold']==0:historical[a['call_id']]=b")
    source = replace_one(source, "if cid not in historical:require(stepIndex==len(row['accepted_trace']),'accepted trial trace coverage')",
                         "if call['direction_control']['threshold']:require(stepIndex==len(row['accepted_trace']),'accepted trial trace coverage')")
    source = source.replace("out/'event/", "out/'")
    source = replace_one(source, "'disabled_complete_trace_fidelity':'PASS'",
                         "'disabled_complete_trace_fidelity':'PASS' if historical_enabled else 'NOT_APPLICABLE'")
    namespace = {'historical_enabled':lam == 1, '__name__':'wb102_frozen_saved_trial_audit'}
    exec(compile(source, str(source_path), 'exec'), namespace)
    old = read_public(ROOT/'outputs/mc24_four_station_wb96_acts_tolerance_v2/event/acts.json')
    p = read_public(directory/'fixture.json')['wb101_protocol']
    result = namespace['audit'](directory, raw, old, p)
    if lam == 1:
        parent = ROOT/'outputs/mc24_four_station_wb101_direction_acceptance_v2/event'
        for name in ('acts.json.trials.ndjson', 'acts.json.traces.ndjson'):
            require(digest(directory/name) == digest(parent/name), 'lambda=1 complete saved stream reproduction')
    result.update(parent_audit_source_sha256=digest(source_path),
                  audit_transformation='output layout; historical disabled comparison only at lambda=1; enabled observer coverage only',
                  lambda_1_full_stream_reproduction='PASS' if lam == 1 else 'NOT_APPLICABLE')
    return result


def scalar_audit(summary, raw, refs, p):
    max_diff = 0.0; cells = 0; computed = {}; refcomputed = {}
    def flat(row):
        require(len(row) == 4, 'scalar output shape')
        return [float(x[0]) if isinstance(x, list) and len(x) == 1 else float(x) for x in row]
    def scalar_response(a, lam):
        a = [flat(x) for x in a]
        k = [[(a[1+4*j][i]-a[2+4*j][i])/(lam*p['output_scales'][i]) for j in range(5)] for i in range(4)]
        j = [[k[i][c]*p['output_scales'][i]/p['seed_steps'][c] for c in range(5)] for i in range(4)]
        m = [(a[23][i]-a[0][i])/(.25*lam*p['output_scales'][i])-sum((-1)**c*k[i][c] for c in range(5)) for i in range(4)]
        return np.array(j), np.array(k), np.array(m)
    def compare(stored, actual):
        nonlocal max_diff
        error = float(np.max(np.abs(np.asarray(stored)-actual)))
        max_diff = max(max_diff, error)
        require(error <= 1e-12*max(1., float(np.max(np.abs(actual)))), 'independent scalar arithmetic')
    for cell in summary['cells']:
        lam, mode, station, tau, cap = (cell[k] for k in ('lambda','mode','station','threshold','cap_m'))
        setting = next(s for s in raw[lam]['settings'] if s['direction_threshold'] == tau and s['cap_m'] == cap)
        values = [r['state']['h'] for r in setting['targets'][station-1]['samples']]
        reference = next(m for m in refs[lam]['modes'] if m['name'] == mode)
        ref_values = [[r['targets'][station-1]['h'] for r in ladder['samples']] for ladder in reference['ladders']]
        j, k, mixed = scalar_response(values, lam)
        _, kr, mixed_r = scalar_response(ref_values[-1], lam)
        _, prev_k, prev_m = scalar_response(ref_values[-2], lam)
        uk, um = abs(kr-prev_k), abs(mixed_r-prev_m)
        b = p['jacobian_absolute_budget']+p['jacobian_relative_budget']*abs(kr)
        for key, actual in (('J',j),('K',k),('reference_K',kr),('mixed_closure',mixed),
                            ('reference_mixed_closure',mixed_r),('U_K',uk),('U_M',um),('B',b)):
            compare(cell[key], actual)
        computed[lam,mode,station,tau,cap] = k, mixed
        refcomputed[lam,mode,station] = kr, mixed_r, uk, um, b
        cells += 1
    for check in summary['decision_checks']+summary['reference_checks']:
        metric = check['metric']; mode, station = check['mode'], check['station']
        if 'lambdas' in check:
            a, b = check['lambdas']
            ra, rb = refcomputed[a,mode,station], refcomputed[b,mode,station]
            allowance = np.maximum(ra[4],rb[4])+p['uncertainty_factor']*(ra[2]+rb[2])
            if metric == 'reference_plateau':
                error = abs(ra[0]-rb[0])
            else:
                tau, cap = check['threshold'], check['cap_m']
                error = abs(computed[a,mode,station,tau,cap][0]-computed[b,mode,station,tau,cap][0])
        else:
            lam = check['lambda']; kr, mr, uk, um, b = refcomputed[lam,mode,station]
            if metric == 'reference_U_K':
                error, allowance = uk, p['reference_budget_fraction']*b
            else:
                k, m = computed[lam,mode,station,check['threshold'],check['cap_m']]
                if metric == 'reference_agreement':
                    error, allowance = abs(k-kr), b+p['uncertainty_factor']*uk
                else:
                    require(metric == 'mixed_agreement', 'unknown scalar decision')
                    error, allowance = abs(m-mr), b.sum(axis=1)+p['uncertainty_factor']*um
        compare(check['error'], error); compare(check['allowance'], allowance)
        require(check['pass_matrix'] == (error <= allowance).tolist() and check['pass'] == bool(np.all(error <= allowance)), 'scalar frozen decision')
    def mag(row): return max(abs(x/s) for x,s in zip(row,p['output_scales']))
    def diff(a,b): return [x-y for x,y in zip(a,b)]
    def effects(a):
        first = [[x/4 for x in diff(a[1+4*k],a[2+4*k])] for k in range(5)]
        return first + [[sum((-1)**k*first[k][i] for k in range(5)) for i in range(4)]]
    def taylor(a):
        e = effects(a)
        return [[a[index][i]-a[0][i]-factor*e[k][i] for i in range(4)] for k in range(6)
                for index,factor in ((3+4*k,1),(4+4*k,.5))]
    def quadratic(a): return [[a[3+4*k][i]-2*a[4+4*k][i]+a[0][i] for i in range(4)] for k in range(6)]
    for d in summary.get('diagnostics',[]):
        lam,mode,station,tau = (d[k] for k in ('lambda','mode','station','threshold'))
        a = [[flat(r['state']['h']) for r in setting['targets'][station-1]['samples']]
             for setting in raw[lam]['settings'] if setting['direction_threshold']==tau]
        reference = next(m for m in refs[lam]['modes'] if m['name']==mode)
        b = [flat(r['targets'][station-1]['h']) for r in reference['ladders'][-1]['samples']]
        values = {'endpoint':max(mag(diff(x,y)) for cap in a for x,y in zip(cap,b)),
                  'cap_spread':max(mag(diff(x,y)) for cap in a for other in a for x,y in zip(cap,other))}
        for name,fn in (('effect',effects),('taylor_excess',taylor),('full_minus_two_half_excess',quadratic)):
            values[name] = max(mag(diff(x,y)) for cap in a for x,y in zip(fn(cap),fn(b)))
        values.update(T_over_lambda_squared=values['taylor_excess']/lam**2,
                      Q_over_lambda_squared=values['full_minus_two_half_excess']/lam**2)
        for name,value in values.items(): compare(d[name],np.array(value))
    if 'identity' in summary:
        ref_ok = all(i['reference']['modes']['mesh_z_double']['gate']=='NUMERICAL_REFERENCE_SUPPORTED' for i in summary['identity'])
        ref_ok = ref_ok and all(c['pass'] for c in summary['reference_checks'] if c['mode']=='mesh_z_double')
        hypothesis = 'UNKNOWN' if not ref_ok else ('NOT_SUPPORTED' if summary['failures'] or not all(c['pass'] for c in summary['decision_checks']) else 'SUPPORTED_BUT_LIMITED')
        require(summary['hypothesis']==hypothesis and summary['reference_prerequisite']==ref_ok, 'scalar final hypothesis')
    return {'gate':'PASS', 'cells':cells, 'decisions':len(summary['decision_checks']),
            'reference_checks':len(summary['reference_checks']), 'diagnostics':len(summary.get('diagnostics',[])),
            'max_difference':max_diff, 'qualification':'NOT_EVALUATED'}
