#!/usr/bin/env python3
"""Recompute saved evidence using QR instead of the execution SVD solution.

No ACTS calls or fitting iteration. This is an implementation cross-check, not
independent data validation.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wb131_contract import OUT, PROTOCOL, read, write, verify, require


def main():
    verify()
    protocol=read(PROTOCOL)
    summary=read(OUT/'summary.json')
    events=[]
    calls=0
    for index in protocol['indices']:
        event=OUT/'events'/f'{index:02d}'
        export=read(event/'export.json')
        status=read(event/'status.json')
        result={'index':index,'scientific_verdict':status['verdict']}
        sigma=np.sqrt([r['sigma_sq'] for r in export['rows']])
        if not (event/'derivatives/response.json').exists():
            require('failure' in status,'unrecorded execution failure')
            result['evidence']='EXECUTION_FAILURE'
            events.append(result)
            continue
        response=read(event/'derivatives/response.json')
        calls+=response['official_calls']
        require(response['official_calls']==17*len(export['rows']),'derivative calls')
        requests=read(event/'derivatives/request.json')['arms']
        require([(a['tag'],a['seed']) for a in response['arms']]==[(a['tag'],a['seed']) for a in requests],'arm request')
        # Independent representation: global positions transformed to loc0.
        values={}
        valid=True
        for arm in response['arms']:
            vals=[]
            for original,row in zip(export['rows'],arm['rows']):
                require(row['cluster_id']==original['cluster_id'] and row['wafer_id']==original['wafer_id'],'row association')
                if row['status']!='SUCCESS' or not row['inside_bounds'] or not row['is_on_surface_with_bounds']:
                    valid=False
                    continue
                frame=np.asarray(original['sensor_transform'])
                pos=np.asarray(row['state']['global_position_mm']).reshape(3)
                value=float((frame[:3,:3].T@(pos-frame[:3,3]))[0])
                require(abs(value-row['predicted_loc0_mm'])<1e-9,'global/local prediction')
                vals.append(value)
            values[arm['tag']]=np.array(vals)
        if not valid:
            require(status['verdict']=='UNKNOWN','invalid endpoint conclusion')
            result['evidence']='INVALID_ENDPOINT'
            events.append(result)
            continue
        fit=read(event/'fit.json')
        if fit['gate']=='UNKNOWN_NOMINAL_REPEAT':
            old=read(event/'nominal.json')
            previous=np.array([r['state']['local_position_mm'][0][0] for r in old['rows']])
            require(np.max(np.abs(values['nominal']-previous))>protocol['nominal_repeat_max_mm'],'repeat rejection')
            events.append(result)
            continue
        derivatives=[]
        for factor in protocol['fd_factors']:
            columns=[]
            for axis,step in enumerate(protocol['fd_steps']):
                columns.append((values[f'{axis}:{factor}:1']-values[f'{axis}:{factor}:-1'])/(2*factor*step))
            derivatives.append(np.column_stack(columns))
        for h,key in zip(derivatives,('h_full','h_half')):
            np.testing.assert_allclose(h,fit[key],rtol=1e-8,atol=1e-5)
        if 'full_fit' not in fit:
            require(fit['gate']=='UNKNOWN_NOMINAL_REPEAT','missing fit evidence')
            events.append(result)
            continue
        h=derivatives[0]
        a=h*np.asarray(protocol['parameter_scales'])/sigma[:,None]
        b=(np.array([r['local_position'][0] for r in export['rows']])-values['nominal'])/sigma
        singular=np.linalg.svd(a,compute_uv=False)
        rank=int(np.sum(singular>singular[0]*protocol['rank_relative_cutoff']))
        require(rank==fit['full_fit']['rank'],'rank')
        if rank==4:
            q,r=np.linalg.qr(a,mode='reduced')
            delta=np.linalg.solve(r,q.T@b)
            np.testing.assert_allclose(delta,fit['full_fit']['delta_scaled'],rtol=1e-7,atol=1e-7)
            projected=a@delta
            cost0=float(b@b)
            unexplained=float(np.sum((b-projected)**2))
            require(abs(cost0-fit['full_fit']['nominal_cost'])<=max(1e-8,cost0*1e-9),'nominal cost')
            result.update(qr_delta_scaled=delta.tolist(),orthogonal_cost=unexplained,
                          orthogonal_cost_fraction=unexplained/cost0,
                          linear_station_costs={})
            for station in sorted({x['station'] for x in export['rows']}):
                mask=np.array([x['station']==station for x in export['rows']])
                result['linear_station_costs'][str(station)]=float(np.sum((b-projected)[mask]**2))
        if fit['gate']=='READY':
            replay=read(event/'replay/response.json')
            metrics=read(event/'metrics.json')
            calls+=replay['official_calls']
            actual=np.array([row['predicted_loc0_mm'] for row in replay['arms'][0]['rows']])
            residual=np.array([row['local_position'][0] for row in export['rows']])-actual
            cost=float(np.sum((residual/sigma)**2))
            error=actual-values['nominal']-projected*sigma
            relative=float(np.linalg.norm(error/sigma)/max(np.linalg.norm(projected),protocol['fd_norm_floor']))
            rms=float(np.sqrt(np.mean(error**2)))
            pass_linear=(relative<=protocol['nonlinear_weighted_relative_tolerance'] or rms<=protocol['nonlinear_absolute_rms_mm'])
            expected='PASS_CONDITIONAL_ONE_STEP' if pass_linear and cost/cost0<=protocol['maximum_cost_ratio'] else 'FAIL_ONE_STEP_HYPOTHESIS'
            require(expected==status['verdict']==metrics['verdict'],'scientific verdict')
            require(abs(cost-metrics['actual_cost'])<=max(1e-8,cost*1e-9),'actual cost')
            result.update(cost_ratio=cost/cost0,nonlinear_weighted_relative_error=relative)
        else:
            require(not (event/'replay').exists(),'replay after rejected gate')
            result['numerical_gate']=fit['gate']
        events.append(result)
    attempts=sum(p.read_text(errors='replace').count('WB131_CALL_BEGIN id=') for p in (OUT/'events').glob('*/*/athena.log'))
    require(calls==summary['completed_propagation_calls'] and attempts==summary['attempted_propagation_calls'],'total calls')
    require(attempts<=2646,'budget')
    write(OUT/'independent_audit.json',{'schema':'wb131_independent_saved_audit_v1','artifact_consistency':'PASS',
                                      'events':events,'completed_propagation_calls':calls,'attempted_propagation_calls':attempts,
                                      'new_propagation_calls':0,'new_reconstruction_calls':0,'held_out_access':False})


if __name__=='__main__':
    main()
