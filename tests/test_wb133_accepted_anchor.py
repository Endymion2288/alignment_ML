"""Selection, tilted-plane direction and gross-screen negative controls."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from alignment.wb133_accepted_anchor import source_request, tangent_path, audit_event
P=json.loads((Path(__file__).resolve().parents[1]/'configs/research_review/wp133_accepted_anchor_contract.json').read_text())


def fixture():
    export={'identity':{'actual_run':1,'actual_event':2},'rows':[]}
    native={'identity':export['identity'],'states':[],'flat_parameters':[]}
    for k,z in enumerate((-100.,0.,100.,200.)):
        frame=np.eye(4);frame[2,3]=z;identifier=k+10
        row={'cluster_id':identifier,'wafer_id':identifier+100,'station':k,'sensor_transform':frame.tolist(),
             'local_position':[.02,0.],'sigma_sq':.001}
        export['rows'].append(row)
        p={'persistent_index':k,'global_position_native':[0.,0.,z],'global_momentum_native':[0.,0.,100000.],
           'native_parameters':[0.,0.,0.,0.,1e-5],'charge_e':1.,'position_unit':'mm','momentum_unit':'MeV','qop_unit':'MeV^-1'}
        native['flat_parameters'].append(p)
        flags=[False]*12;flags[5 if k==0 else 0]=True
        native['states'].append({'tsos_index':k,'parameter_index':k,'type_flags':flags,
           'type_description':'Outlier' if k==0 else 'Measurement','measurement':{
            'selected_prd':True,'cluster_id':identifier,'rot_identifier':identifier,'wafer_id':row['wafer_id'],'station':k,
            'prd_link_valid':True,'prd_link_resolved':True,'selected_prd_pointer_equal':True,
            'native_prediction':{'inside_bounds':True,'loc0_mm':0.}}})
    req=source_request(export,native)
    response={'schema':'wb133_accepted_anchor_response_v1','identity':export['identity'],'source':req['source'],
      'source_actual_position_mm':[[0.],[0.],[0.]],'source_actual_direction':[[0.],[0.],[1.]],
      'source_actual_qop_per_MeV':1e-5,'source_position_roundtrip_mm':0.,'source_direction_roundtrip':0.,
      'new_reconstruction_calls':0,'material_source':'None','field_mode':'FASER','covariance_transport':False,
      'official_calls':3,'new_propagation_calls':3,'zero_path_controls':1,'rows':[]}
    baseline={'identity':export['identity'],'rows':[]};call=0
    for original in export['rows']:
        z=original['sensor_transform'][2][3];anchor=original['cluster_id']==req['source']['cluster_id']
        if not anchor:call+=1
        response['rows'].append({**{k:original[k] for k in ('cluster_id','wafer_id','station')},'status':'SUCCESS',
          'kind':'SOURCE_IDENTITY_CONTROL' if anchor else 'PROPAGATED','call_id':None if anchor else call,
          'propagation_direction':'ZERO_PATH' if anchor else 'FORWARD' if z>=0 else 'BACKWARD','tangent_path_mm':z,
          'global_position_mm':[[0.],[0.],[z]],'global_direction':[[0.],[0.],[1.]],'local_position_mm':[[0.],[0.],[0.]],
          'qop_per_MeV':1e-5,'covariance_present':False,'inside_bounds':True,'strict_is_on_surface_with_bounds':True,
          'frame_roundtrip_mm':0.,'predicted_loc0_mm':0.,'local_residual_mm':.02,
          'plane_tolerance_mm':req['source']['persistence_plane_tolerance_mm'] if anchor else 1e-5})
        baseline['rows'].append({**{k:original[k] for k in ('cluster_id','wafer_id','station')},'status':'SUCCESS',
                                'state':{'local_position_mm':[[0.],[0.],[0.]]}})
    return export,native,req,response,baseline


class TestWB133(unittest.TestCase):
    def test_outlier_not_selected_and_tie_by_tsos(self):
        e,n,*_=fixture()
        self.assertEqual(source_request(e,n)['source']['parameter_index'],1)
        # Same z for two accepted parameters; preserve exact plane by moving frame too.
        n['flat_parameters'][2]['global_position_native'][2]=0.
        e['rows'][2]['sensor_transform'][2][3]=0.
        n['states'][1]['tsos_index']=20;n['states'][2]['tsos_index']=10
        self.assertEqual(source_request(e,n)['source']['parameter_index'],2)

    def test_bad_prd_link_and_duplicate_rejected(self):
        e,n,*_=fixture();n['states'][1]['measurement']['prd_link_resolved']=False
        with self.assertRaises(ValueError):source_request(e,n)
        e,n,*_=fixture();n['states'].append(copy.deepcopy(n['states'][1]))
        with self.assertRaises(ValueError):source_request(e,n)

    def test_qop_units_sign_and_missing_candidate_rejected(self):
        for corrupt in ('unit','sign','missing'):
            e,n,*_=fixture()
            if corrupt=='unit':n['flat_parameters'][1]['native_parameters'][4]*=1000
            elif corrupt=='sign':n['flat_parameters'][1]['native_parameters'][4]*=-1
            else:n['states'][2]['parameter_index']=None
            with self.assertRaises(ValueError):source_request(e,n)

    def test_tilted_plane_direction_uses_normal_not_center(self):
        angle=.3;c,s=np.cos(angle),np.sin(angle)
        frame=np.eye(4);frame[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]];frame[:3,3]=[-100,0,1]
        source={'position_mm':[0.,0.,0.],'direction':[0.,0.,1.]}
        self.assertGreater(np.dot(frame[:3,3],source['direction']),0)
        self.assertLess(tangent_path(source,frame),0)
        endpoint=np.array(source['position_mm'])+tangent_path(source,frame)*np.array(source['direction'])
        self.assertAlmostEqual(float(frame[:3,2]@(endpoint-frame[:3,3])),0.,places=12)

    def test_parallel_plane_rejected(self):
        with self.assertRaises(ValueError):tangent_path({'position_mm':[0,0,0],'direction':[1,0,0]},np.eye(4))

    def test_positive_screen_and_failure_without_tuning(self):
        args=fixture();self.assertEqual(audit_event(*args,P)['scientific_verdict'],'PASS_GROSS_MODEL_ONLY')
        e,n,req,res,base=args;e['rows'][0]['local_position'][0]=.2;res['rows'][0]['local_residual_mm']=.2
        self.assertEqual(audit_event(e,n,req,res,base,P)['scientific_verdict'],'FAIL_GROSS_MODEL')

    def test_null_bounds_and_missing_row_cannot_pass(self):
        for kind in ('null','bounds','missing'):
            args=fixture();res=args[3]
            if kind=='null':res['rows'][0]['status']='FAIL_OFFICIAL_NULL'
            elif kind=='bounds':res['rows'][0]['inside_bounds']=False
            else:res['rows'].pop()
            if kind=='missing':
                with self.assertRaises(ValueError):audit_event(*args,P)
            else:self.assertEqual(audit_event(*args,P)['scientific_verdict'],'UNKNOWN_INCOMPLETE_PREDICTION')

    def test_rotated_readout_and_residual_sign_guard(self):
        e,n,req,res,base=fixture();angle=.4;c,s=np.cos(angle),np.sin(angle)
        frame=np.array(e['rows'][0]['sensor_transform']);frame[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
        e['rows'][0]['sensor_transform']=frame.tolist();pos=frame[:3,3]+frame[:3,:3]@np.array([.01,.04,0])
        res['rows'][0].update(global_position_mm=pos[:,None].tolist(),local_position_mm=[[.01],[.04],[0.]],predicted_loc0_mm=.01,local_residual_mm=.01)
        audit=audit_event(e,n,req,res,base,P);self.assertAlmostEqual(audit['rows'][0]['prediction_mm'],.01)
        res['rows'][0]['local_residual_mm']=-.01
        with self.assertRaises(ValueError):audit_event(e,n,req,res,base,P)


if __name__=='__main__':unittest.main()
