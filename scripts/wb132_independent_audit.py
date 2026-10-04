#!/usr/bin/env python3
"""Saved-only axis-dot-product recheck; no exporter math or fit reused."""
import sys
sys.dont_write_bytecode=True
import numpy as np
from wb132_contract import ROOT, OUT, PROTOCOL, read, write, require, verify


def main():
    verify();protocol=read(PROTOCOL);summary=read(OUT/'summary.json');events=[];matched_total=missing_total=0
    for index in protocol['indices']:
        event=OUT/'events'/f'{index:02d}';status=read(event/'status.json')
        if not (event/'audit.json').exists():
            require('failure' in status,'unrecorded analysis failure')
            events.append({'index':index,'evidence':'UNKNOWN','failure':status['failure']})
            continue
        raw=read(event/'native_response.json');audit=read(event/'audit.json');original=read(event/'export.json')
        parent=read(event/'parent_provenance.json')['persisted_tracks'][0]['matching_tracks'][0]
        for before,after in zip(parent['parameters'],raw['flat_parameters']):
            require(all(after[k]==v for k,v in before.items()),'original parameters')
        require(len(parent['parameters'])==len(raw['flat_parameters']),'parameter count')
        mappings={}
        for state in raw['states']:
            require(state['type_mask']==sum((1<<k) for k,v in enumerate(state['type_flags']) if v),'type mask')
            m=state['measurement']
            if m is not None and m['selected_prd']:
                require(m['cluster_id'] not in mappings,'duplicate TSOS')
                require(m['prd_link_resolved'] and m['prd_link_valid'] and m['selected_prd_pointer_equal'],'exact PRD')
                mappings[m['cluster_id']]=state
        require(len(mappings)==audit['matched_rows']==protocol['expected_selected_prd_matches'][str(index)],'match count')
        matched_total+=len(mappings);missing_total+=len(audit['missing'])
        rows={r['cluster_id']:r for r in audit['rows']};by_station={1:[],2:[],3:[]};unknown=False
        maximum_frame_error=0.
        for measurement in original['rows']:
            state=mappings.get(measurement['cluster_id'])
            if state is None:
                require(any(r['cluster_id']==measurement['cluster_id'] for r in audit['missing']),'missing row preserved')
                continue
            m=state['measurement'];saved=rows[measurement['cluster_id']]
            if state['parameter_index'] is None or m['native_prediction'] is None:
                require(not saved['eligible'],'missing parameter conclusion');unknown=True;continue
            parameter=raw['flat_parameters'][state['parameter_index']]
            transform=np.asarray(m['sensor_transform']);pos=np.asarray(parameter['global_position_native'])
            displacement=pos-transform[:3,3]
            # Independent scalar projection onto each stored sensor axis.
            local=np.array([sum(transform[k,axis]*displacement[k] for k in range(3)) for axis in range(3)])
            residual=measurement['local_position'][0]-local[0]
            require(abs(residual-saved['prd_residual_mm'])<1e-9,'scalar residual')
            require(abs(m['rot_loc1']-local[0]-saved['rot_residual_mm'])<1e-9,'ROT residual')
            require(parameter['native_parameters'][4]==saved['qop_per_MeV'],'qop identity')
            tolerance=max(1e-6,4*(2**-23)*max(1.,max(abs(x) for x in pos),max(abs(x) for x in transform[:3,3])))
            mask=state['type_mask']
            eligible=bool(mask&1) and not bool(mask&((1<<5)|(1<<6))) and abs(local[2])<=tolerance and m['native_prediction']['inside_bounds']
            require(eligible==saved['eligible'],'eligibility')
            if eligible:by_station[measurement['station']].append((residual,measurement['sigma_sq']))
            else:unknown=True
            maximum_frame_error=max(maximum_frame_error,abs(local[0]-m['native_prediction']['loc0_mm']))
        gross=False;stations={}
        for station,values in by_station.items():
            if not values:continue
            rms=float(np.sqrt(sum(v*v for v,_ in values)/len(values)))
            maximum=max(abs(v) for v,_ in values);cost=sum(v*v/s for v,s in values)
            metric=audit['stations'][str(station)]
            require(abs(rms-metric['prd_rms_mm'])<1e-9 and abs(maximum-metric['max_abs_prd_mm'])<1e-9,'station RMS/max')
            require(abs(cost-metric['descriptive_prd_cost'])<max(1e-8,cost*1e-8),'conditional cost')
            gross |= rms>protocol['gross_station_rms_mm'] or maximum>protocol['gross_max_abs_residual_mm']
            stations[str(station)]={'rms_mm':rms,'max_abs_mm':maximum}
        verdict='FAIL_GROSS_NATIVE_RESIDUAL' if gross else 'UNKNOWN' if unknown else 'PASS_GROSS_SCREEN_ONLY'
        require(verdict==audit['native_prediction_hypothesis']==status['native_prediction_hypothesis'],'hypothesis verdict')
        old_response=read(event/'parent_response.json');source_index=old_response['selection']['selected_index']
        source=[s for s in raw['states'] if s['parameter_index']==source_index]
        require(len(source)==1 and source[0]['tsos_index']==audit['source']['tsos_index'],'source exact TSOS')
        require(source[0]['type_flags']==audit['source']['type_flags'],'source flags')
        for key in ('new_reconstruction_calls','new_propagation_calls','algorithm_field_queries'):require(raw[key]==0,'unexpected API work')
        events.append({'index':index,'hypothesis':verdict,'matched':len(mappings),'missing':len(audit['missing']),
                       'max_prediction_representation_error_mm':maximum_frame_error,'stations':stations,
                       'source_type':source[0]['type_description']})
    if summary['interface']=='PASS':require(matched_total==111 and missing_total==36,'total 147-row accounting')
    write(OUT/'independent_audit.json',{'schema':'wb132_independent_saved_audit_v1','artifact_consistency':'PASS',
                                      'matched_rows':matched_total,'unmatched_rows':missing_total,'events':events,
                                      'new_reconstruction_calls':0,'new_propagation_calls':0,'algorithm_field_queries':0,
                                      'truth_access':False,'held_out_access':False})


if __name__=='__main__':main()
