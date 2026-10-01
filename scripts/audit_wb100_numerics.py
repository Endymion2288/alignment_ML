"""Independent NumPy quadrature, Rodrigues, bounds and decision audit of all rows."""
import json
import numpy as np
from alignment.wb99_direction_step_doubling import node_field

def audit(out, old, nodes, p):
    bounds=json.loads((out/'bounds.json').read_text())
    mesh=[np.asarray(x) for x in old['mesh_mm']]
    M=max(bounds['M_inside_native'],bounds['M_outside_native']);L=bounds['L_native_per_mm']
    # Independent vectorized scan of every node/adjacent difference.
    scale=old['conditions']['bscale_kT']*old['conditions']['scale']*1000*.000299792458
    inside=float(np.max(np.linalg.norm(nodes.astype(float),axis=3)))*scale
    axes=[]
    for axis in range(3):
        differences=np.diff(nodes.astype(float),axis=axis)*scale
        pitch=np.diff(mesh[axis]).reshape(tuple(len(mesh[axis])-1 if k==axis else 1 for k in range(3)))
        axes.append(float(np.max(np.linalg.norm(differences,axis=3)/pitch)))
    if abs(inside-bounds['M_inside_native'])>1e-16 or np.max(np.abs(np.asarray(axes)-bounds['L_axes_native_per_mm']))>1e-16:
        raise ValueError('independent global field bounds')
    maxes={k:0. for k in ('integral_native_mm','rotation','envelope','reference_defect','uncertainty')}
    counts={'rows':0,'faces':0,'resolved':0,'misses':0}
    batch=[]
    def flush():
        positions=[];weights=[];lengths=[]
        for r in batch:
            e=r['envelope'];u=np.asarray(r['start_direction']);start=np.asarray(r['start_position_mm']);h=r['h_mm']
            expected=[]
            for k in range(3):
                if u[k]==0:continue
                ss=(mesh[k]-start[k])/u[k];expected.extend((k,float(f),float(s)) for f,s in zip(mesh[k],ss) if 0<s<h)
            got=[(f['axis'],f['face_mm'],f['s_mm']) for f in e['face_identities']]
            if len(expected)!=len(got) or any(a[:2]!=b[:2] or abs(a[2]-b[2])>1e-12 for a,b in zip(expected,got)):
                raise ValueError('independent mesh face identity')
            counts['faces']+=len(got)
            cuts=np.asarray(e['cuts_mm']);delta=np.diff(cuts);mid=cuts[:-1]+delta/2;off=delta/(2*np.sqrt(3))
            ss=np.column_stack((mid-off,mid+off)).reshape(-1)
            positions.extend(start+ss[:,None]*u);weights.extend(np.repeat(delta/2,2));lengths.append(len(ss))
        fields=node_field(np.asarray(positions),old,nodes);offset=0
        for r,n in zip(batch,lengths):
            integral=np.sum(fields[offset:offset+n]*np.asarray(weights[offset:offset+n])[:,None],axis=0);offset+=n
            e=r['envelope'];q=r['q_over_p_Acts'];h=r['h_mm'];u=np.asarray(r['start_direction']);full=np.asarray(r['full_direction'])
            err=float(np.max(np.abs(integral-np.asarray(e['integral_native_mm']))));maxes['integral_native_mm']=max(maxes['integral_native_mm'],err)
            w=-q*integral;t=np.linalg.norm(w);sinc=np.sinc(t/np.pi);cosc=.5*np.sinc(t/(2*np.pi))**2
            v=u+sinc*np.cross(w,u)+cosc*np.cross(w,np.cross(w,u))
            rot=float(np.max(np.abs(v-np.asarray(e['predictor_direction']))));maxes['rotation']=max(maxes['rotation'],rot)
            k=abs(q)*M;a=k*h
            # Independently partition the tube with numpy polynomial roots.
            start=np.asarray(r['start_position_mm']);lo=np.asarray(old['conditions']['min_mm']);hi=np.asarray(old['conditions']['max_mm']);tc=[0.,h]
            for axis in range(3):
                for face in (lo[axis],hi[axis]):
                    for sign in (-1,1):
                        coef=[sign*k/2,u[axis],start[axis]-face]
                        roots=np.roots(np.trim_zeros(coef,'f')) if any(coef) else []
                        tc.extend(float(x.real) for x in roots if abs(x.imag)<1e-14 and 0<x.real<h)
            tc=sorted(set(tc));amb=0.;alloutside=True
            for left,right in zip(tc,tc[1:]):
                s=left+(right-left)/2;center=start+s*u;radius=k*s*s/2
                isinside=bool(np.all(center>=lo+radius) and np.all(center<=hi-radius))
                isoutside=bool(np.any(center<lo-radius) or np.any(center>hi+radius))
                alloutside=alloutside and isoutside
                if not isinside and not isoutside:amb+=right-left
            if abs(amb-e['ambiguous_length_mm'])>1e-10*max(1,h) or alloutside!=e['proved_constant_outside']:
                raise ValueError('independent tube partition')
            comm=0 if e['proved_constant_outside'] else a*a*np.exp(a)
            bend=0 if e['proved_constant_outside'] else abs(q)*(L*k*h**3/6+(bounds['M_inside_native']+bounds['M_outside_native'])*e['ambiguous_length_mm'])
            envelope=np.max(np.abs(full-v))+comm+bend
            maxes['envelope']=max(maxes['envelope'],abs(float(envelope)-e['E']))
            if err>1e-13 or rot>1e-13 or abs(envelope-e['E'])>1e-12*max(1,e['E']):raise ValueError('independent envelope arithmetic')
            ev=r['evaluation'];resolved=ev['gate']=='PASS' and ev['defect']>p['resolved_factor']*ev['uncertainty']
            miss=resolved and ev['defect']>envelope+p['uncertainty_factor']*ev['uncertainty']
            if resolved!=ev['resolved'] or miss!=ev['false_negative']:raise ValueError('independent numerical decision')
            counts['rows']+=1;counts['resolved']+=resolved;counts['misses']+=miss
        batch.clear()
    # WB99 fine/source uncertainty was floor-limited; compare the immutable evaluation.
    from pathlib import Path
    prior=out.parent/'mc24_four_station_wb99_aggregation_recovery_v1/metrics.ndjson'
    with (out/'metrics.ndjson').open() as stream,prior.open() as oldstream:
        for line in stream:
            r=json.loads(line);oldr=json.loads(next(oldstream))
            if (r['trace'],r['index'])!=(oldr['trace'],oldr['index']):raise ValueError('independent reference identity')
            d=abs(r['evaluation']['defect']-oldr['defect_max']);U=abs(r['evaluation']['uncertainty']-oldr['uncertainty'])
            maxes['reference_defect']=max(maxes['reference_defect'],d);maxes['uncertainty']=max(maxes['uncertainty'],U)
            if d>1e-15 or U>1e-15:raise ValueError('reference/uncertainty changed')
            batch.append(r)
            if len(batch)==256:flush()
        if any(oldstream):raise ValueError('prior evaluation remaining')
    if batch:flush()
    return {'gate':'PASS','counts':counts,'maxima':maxes,'note':'NumPy audit, not interval arithmetic or physical field oracle'}
