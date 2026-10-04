#!/usr/bin/env python3
"""Saved-only scalar-axis proof for both accepted and excluded linked states."""
import sys
sys.dont_write_bytecode=True
import numpy as np
from wb132_contract import OUT, PROTOCOL, read, write, require
from wb132_typed_recovery import RECOVERY, verify_recovery


def main():
    verify_recovery();protocol=read(PROTOCOL);events=[];totals={'linked':0,'accepted':0,'excluded':0,'unlinked':0}
    for index in protocol['indices']:
        event=OUT/'events'/f'{index:02d}';dest=RECOVERY/'events'/event.name
        raw=read(event/'native_response.json');audit=read(dest/'audit.json');original=read(event/'export.json')
        parent=read(event/'parent_provenance.json')['persisted_tracks'][0]['matching_tracks'][0]
        require(len(raw['flat_parameters'])==len(parent['parameters']),'parameter count')
        for before,after in zip(parent['parameters'],raw['flat_parameters']):require(all(after[k]==v for k,v in before.items()),'original parameter identity')
        links={};accepted={}
        for state in raw['states']:
            require(state['type_mask']==sum(1<<k for k,v in enumerate(state['type_flags']) if v),'mask/vector identity')
            m=state['measurement']
            if not m or not m['selected_prd']:continue
            identifier=m['cluster_id'];require(identifier not in links,'duplicate PRD')
            require(m['prd_link_resolved'] and m['prd_link_valid'] and m['selected_prd_pointer_equal'],'link identity')
            links[identifier]=state
            if state['type_mask']&1 and not state['type_mask']&((1<<5)|(1<<6)):accepted[identifier]=state
        old_ids={r['cluster_id'] for r in parent['cluster_membership'] if r['cluster_id'] in {x['cluster_id'] for x in original['rows']}}
        require(set(accepted)==old_ids,'accepted membership precise closure')
        require(len(accepted)==audit['matched_rows']==protocol['expected_selected_prd_matches'][str(index)],'accepted count')
        require(len(links)==audit['all_exact_linked_rows'],'all-linked count')
        require(len(original['rows'])-len(links)==audit['truly_unlinked_rows'],'unlinked count')
        saved_rows={r['cluster_id']:r for r in audit['rows']+audit['excluded_details']}
        by_station={1:[],2:[],3:[]};unknown=False;max_error=0.
        for measurement in original['rows']:
            state=links.get(measurement['cluster_id'])
            if not state:continue
            m=state['measurement'];saved=saved_rows.get(measurement['cluster_id'])
            if state['parameter_index'] is None or m['native_prediction'] is None:
                if measurement['cluster_id'] in accepted:unknown=True
                continue
            require(saved is not None,'linked prediction retained')
            parameter=raw['flat_parameters'][state['parameter_index']]
            transform=np.asarray(m['sensor_transform']);pos=np.asarray(parameter['global_position_native'])
            d=pos-transform[:3,3]
            local=np.array([sum(transform[k,axis]*d[k] for k in range(3)) for axis in range(3)])
            residual=measurement['local_position'][0]-local[0]
            require(abs(residual-saved['prd_residual_mm'])<1e-9,'residual dot-product')
            require(abs(m['rot_loc1']-local[0]-saved['rot_residual_mm'])<1e-9,'ROT residual')
            require(parameter['native_parameters'][4]==saved['qop_per_MeV'],'qop preserved')
            tol=max(1e-6,4*2**-23*max(1.,max(abs(x) for x in pos),max(abs(x) for x in transform[:3,3])))
            eligible=measurement['cluster_id'] in accepted and abs(local[2])<=tol and m['native_prediction']['inside_bounds']
            require(eligible==saved['eligible'],'typed eligibility')
            if eligible:by_station[measurement['station']].append((residual,measurement['sigma_sq']))
            elif measurement['cluster_id'] in accepted:unknown=True
            max_error=max(max_error,abs(local[0]-m['native_prediction']['loc0_mm']))
        gross=False;stations={}
        for station,values in by_station.items():
            if not values:continue
            rms=float(np.sqrt(sum(v*v for v,_ in values)/len(values)));maximum=max(abs(v) for v,_ in values);cost=sum(v*v/s for v,s in values)
            metric=audit['stations'][str(station)]
            require(abs(rms-metric['prd_rms_mm'])<1e-9 and abs(maximum-metric['max_abs_prd_mm'])<1e-9,'RMS/max')
            require(abs(cost-metric['descriptive_prd_cost'])<max(1e-8,cost*1e-8),'conditional cost')
            gross |= rms>protocol['gross_station_rms_mm'] or maximum>protocol['gross_max_abs_residual_mm']
            stations[str(station)]={'rms_mm':rms,'max_abs_mm':maximum}
        verdict='FAIL_GROSS_NATIVE_RESIDUAL' if gross else 'UNKNOWN' if unknown else 'PASS_GROSS_SCREEN_ONLY'
        require(verdict==audit['native_prediction_hypothesis'],'primary threshold unchanged')
        source_index=read(event/'parent_response.json')['selection']['selected_index']
        source=[s for s in raw['states'] if s['parameter_index']==source_index]
        require(len(source)==1 and source[0]['tsos_index']==audit['source']['tsos_index'],'source mapping')
        require(source[0]['type_flags']==audit['source']['type_flags'],'source flags')
        source_mask=source[0]['type_mask'];anchor=bool(source_mask&1) and not bool(source_mask&((1<<5)|(1<<6))) and source[0]['measurement'] is not None
        require(anchor==audit['source']['measurement_anchor'],'source role')
        for key in ('new_reconstruction_calls','new_propagation_calls','algorithm_field_queries'):require(raw[key]==0,'unexpected work')
        totals['linked']+=len(links);totals['accepted']+=len(accepted);totals['excluded']+=len(links)-len(accepted);totals['unlinked']+=len(original['rows'])-len(links)
        events.append({'index':index,'hypothesis':verdict,'all_links':len(links),'accepted':len(accepted),
                       'source_type':source[0]['type_description'],'source_measurement_anchor':anchor,'stations':stations,
                       'max_prediction_representation_error_mm':max_error})
    require(totals['accepted']==111 and totals['linked']+totals['unlinked']==147,'population')
    write(RECOVERY/'independent_audit.json',{'schema':'wb132_typed_independent_audit_v1','artifact_consistency':'PASS',
             'events':events,'totals':totals,'new_propagation_calls':0,'new_reconstruction_calls':0,
             'new_input_event_executions':0,'held_out_access':False,'truth_access':False})


if __name__=='__main__':main()
