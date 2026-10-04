#!/usr/bin/env python3
"""Independent scalar-axis saved evidence audit; no event read or propagation."""
import math
import re
import sys
sys.dont_write_bytecode=True
from wb133_contract import OUT, PROTOCOL, read, write, require, verify


def scalar_local(position, frame):
    d=[position[k]-frame[k][3] for k in range(3)]
    return [sum(frame[k][axis]*d[k] for k in range(3)) for axis in range(3)]


def close(actual, expected, name):
    require(abs(actual-expected)<=max(1e-9,abs(expected)*1e-10),name)


def main():
    verify();p=read(PROTOCOL);summary=read(OUT/'summary.json');events=[];attempts=completed=rows=controls=0
    for index in p['indices']:
        event=OUT/'events'/f'{index:02d}';ex=read(event/'export.json');native=read(event/'native_response.json')
        req=read(event/'request.json');source=req['source'];status=read(event/'status.json')
        require(native['identity']==ex['identity']==req['identity'],'event identity')
        mapping={};accepted=[]
        for s in native['states']:
            m=s['measurement']
            if not m or not m['selected_prd']:continue
            require(m['cluster_id'] not in mapping and m['prd_link_valid'] and m['prd_link_resolved'] and m['selected_prd_pointer_equal'],'exact link')
            require(m['rot_identifier']==m['cluster_id'],'ROT ID');mapping[m['cluster_id']]=s
            mask=s['type_mask'];require(mask==sum(1<<k for k,v in enumerate(s['type_flags']) if v),'type flags/mask')
            if mask&1 and not mask&((1<<5)|(1<<6)):
                require(s['parameter_index'] is not None,'candidate parameters')
                a=native['flat_parameters'][s['parameter_index']]
                accepted.append((a['global_position_native'][2],s['tsos_index'],s,a))
        _,_,s,a=min(accepted,key=lambda t:t[:2])
        require(source['tsos_index']==s['tsos_index'] and source['parameter_index']==s['parameter_index'] and
                source['type_flags']==s['type_flags'] and source['cluster_id']==s['measurement']['cluster_id'] and
                source['position_mm']==a['global_position_native'] and source['native_parameters']==a['native_parameters'] and
                source['qop_per_MeV']==a['native_parameters'][4],'source exact identity')
        norm=math.sqrt(sum(v*v for v in a['global_momentum_native']))
        for v,expected in zip(source['direction'],a['global_momentum_native']):close(v,expected/norm,'source direction')
        log=(event/'athena.log').read_text(errors='replace')
        begins=[int(v) for v in re.findall(r'WB133_CALL_BEGIN id=(\d+)',log)]
        ends=[int(v) for v in re.findall(r'WB133_CALL_END id=(\d+)',log)]
        require(begins==list(range(1,len(begins)+1)) and ends==list(range(1,len(ends)+1)) and len(ends)<=len(begins),'call log ledger')
        require('message limit (500) reached for WB133AcceptedAnchorPrediction' not in log,'saturated logs')
        require(status['attempted_calls']==len(begins) and status['completed_logged_calls']==len(ends),'status call counters')
        attempts+=len(begins);completed+=len(ends)
        if not (event/'audit.json').exists():
            require(status['scientific_verdict']=='UNKNOWN' and 'failure' in status,'unrecorded failure')
            events.append({'index':index,'scientific_verdict':'UNKNOWN','failure':status['failure']});continue
        response=read(event/'response.json');audit=read(event/'audit.json');baseline=read(event/'baseline.json')
        require(response['source']==source and response['identity']==ex['identity'] and response['material_source']=='None' and
                response['field_mode']=='FASER' and response['new_reconstruction_calls']==0 and not response['covariance_transport'],'response model')
        require(response['official_calls']==len(ends)==len(begins)==len(ex['rows'])-1 and response['new_propagation_calls']==len(ends),'call coverage')
        require(len(response['rows'])==len(audit['rows'])==len(baseline['rows'])==len(ex['rows']),'row coverage')
        rows+=len(ex['rows']);controls+=response['zero_path_controls'];require(response['zero_path_controls']==1,'source control count')
        station_values={k:[] for k in range(4)};old_values={k:[] for k in range(4)};total_valid=0;call_ids=[];source_controls=0
        for orig,row,saved,old in zip(ex['rows'],response['rows'],audit['rows'],baseline['rows']):
            require(all(orig[k]==row[k]==saved[k]==old[k] for k in ('cluster_id','wafer_id','station')),'row ID/order')
            f=orig['sensor_transform'];normal=[f[k][2] for k in range(3)];denom=sum(normal[k]*source['direction'][k] for k in range(3))
            require(abs(denom)>=1e-12,'parallel plane')
            path=sum(normal[k]*(f[k][3]-source['position_mm'][k]) for k in range(3))/denom
            close(row['tangent_path_mm'],path,'path sign');anchor=orig['cluster_id']==source['cluster_id']
            if anchor:
                source_controls+=1;require(row['call_id'] is None and row['kind']=='SOURCE_IDENTITY_CONTROL' and row['propagation_direction']=='ZERO_PATH','source control')
            else:
                call_ids.append(row['call_id']);require(row['propagation_direction']==('FORWARD' if path>=0 else 'BACKWARD'),'direction')
            require(old['status']=='SUCCESS','original P endpoint')
            oldpos=[v[0] for v in old['state']['global_position_mm']];oldlocal=scalar_local(oldpos,f)
            close(oldlocal[0],old['state']['local_position_mm'][0][0],'baseline representation')
            oldres=orig['local_position'][0]-oldlocal[0];close(oldres,saved['original_P_residual_mm'],'baseline residual')
            old_values[orig['station']].append((oldres,orig['sigma_sq']))
            linked=mapping.get(orig['cluster_id']);native_pred=None
            if linked:
                param=native['flat_parameters'][linked['parameter_index']]
                native_pred=scalar_local(param['global_position_native'],f)[0]
                close(native_pred,saved['native_prediction_mm'],'native reference')
            else:require(saved['native_prediction_mm'] is None and not saved['exact_native_link'],'missing reference retained')
            valid=False
            if row['status']=='SUCCESS':
                pos=[v[0] for v in row['global_position_mm']];local=scalar_local(pos,f)
                for v,w in zip(local,row['local_position_mm']):close(v,w[0],'local coordinates')
                close(local[0],row['predicted_loc0_mm'],'component');residual=orig['local_position'][0]-local[0]
                close(residual,row['local_residual_mm'],'residual sign');close(residual,saved['residual_mm'],'saved residual')
                close(local[0]-oldlocal[0],saved['new_minus_P_mm'],'P contrast')
                if native_pred is not None:close(local[0]-native_pred,saved['new_minus_native_mm'],'native contrast')
                tol=(max(1e-6,4*2**-23*max(1.,max(abs(v) for v in source['position_mm']),max(abs(f[k][3]) for k in range(3))))
                     if anchor else p['endpoint_plane_tolerance_mm'])
                require(row['plane_tolerance_mm']==tol,'tolerance freeze')
                require(row['bounds_type']==6 and len(row['bounds_values'])==4,'independent rectangle bounds required')
                xmin,ymin,xmax,ymax=row['bounds_values'];inside=xmin<=local[0]<=xmax and ymin<=local[1]<=ymax
                require(inside==row['inside_bounds'],'scalar rectangle bounds')
                valid=inside and abs(local[2])<=tol and (anchor or row['strict_is_on_surface_with_bounds'])
                if anchor:
                    for v,w in zip(pos,source['position_mm']):close(v,w,'unprojected anchor')
                require(not row['covariance_present'],'no transported covariance')
                require(abs(row['qop_per_MeV']-source['qop_per_MeV'])<=max(1e-15,abs(source['qop_per_MeV'])*1e-10),'qop unchanged')
                if valid:station_values[orig['station']].append((residual,orig['sigma_sq']))
            else:require(row['status']=='FAIL_OFFICIAL_NULL' and not anchor,'null propagation')
            require(valid==saved['valid'],'eligibility');total_valid+=int(valid)
        require(source_controls==1 and call_ids==list(range(1,len(ex['rows']))),'response call/control IDs')
        stations={};gross=False
        for station in range(4):
            values=station_values[station];old=old_values[station];metric=audit['stations'][str(station)]
            require(metric['valid_rows']==len(values) and metric['rows']==len(old) and metric['complete']==(len(values)==len(old)),'station coverage')
            if values:
                rms=math.sqrt(sum(v*v for v,_ in values)/len(values));maximum=max(abs(v) for v,_ in values)
                close(rms,metric['rms_mm'],'station RMS');close(maximum,metric['max_abs_mm'],'station max')
                close(sum(v for v,_ in values)/len(values),metric['mean_mm'],'station mean')
                close(sum(v*v/sigma for v,sigma in values),metric['descriptive_cost'],'descriptive cost')
                gross |= rms>p['gross_station_rms_mm'] or maximum>p['gross_max_abs_residual_mm']
                stations[str(station)]={'valid_rows':len(values),'rms_mm':rms,'max_abs_mm':maximum,'original_P_rms_mm':math.sqrt(sum(v*v for v,_ in old)/len(old))}
            for key,value in [('rms_mm',math.sqrt(sum(v*v for v,_ in old)/len(old))),('max_abs_mm',max(abs(v) for v,_ in old))]:
                close(value,audit['original_P_stations'][str(station)][key],'P station aggregation')
        complete=total_valid==len(ex['rows'])
        verdict='UNKNOWN_INCOMPLETE_PREDICTION' if not complete else 'FAIL_GROSS_MODEL' if gross else 'PASS_GROSS_MODEL_ONLY'
        require(verdict==audit['scientific_verdict']==status['scientific_verdict'] and total_valid==audit['valid_rows'] and complete==audit['complete_coverage'],'scientific verdict')
        events.append({'index':index,'scientific_verdict':verdict,'valid_rows':total_valid,'original_rows':len(ex['rows']),
                       'source_tsos_index':source['tsos_index'],'source_qop_per_MeV':source['qop_per_MeV'],'stations':stations})
    verdicts=[e['scientific_verdict'] for e in events]
    overall='PASS_GROSS_MODEL_ONLY' if all(v=='PASS_GROSS_MODEL_ONLY' for v in verdicts) else 'FAIL_GROSS_MODEL' if 'FAIL_GROSS_MODEL' in verdicts else 'UNKNOWN'
    require(overall==summary['scientific_verdict'] and attempts==summary['attempted_propagation_calls'] and completed==summary['completed_propagation_calls'],'overall verdict/counters')
    require(attempts<=141,'budget');verify()
    write(OUT/'independent_audit.json',{'schema':'wb133_independent_saved_audit_v1','artifact_consistency':'PASS',
          'scientific_verdict':overall,'events':events,'evaluated_rows':rows,'zero_path_controls':controls,
          'attempted_original_calls':attempts,'completed_original_calls':completed,'new_propagation_calls':0,
          'new_input_event_executions':0,'new_reconstruction_calls':0,'truth_access':False,'held_out_access':False})
    print('WB133 independent artifact consistency PASS; science',overall)


if __name__=='__main__':main()
