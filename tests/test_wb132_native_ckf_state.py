"""Geometry and identity negative controls for native data, not a fitter test."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from alignment.wb132_native_ckf_state import native_residual, exact_state_map, parameter_identity, plane_tolerance

P=json.loads((Path(__file__).resolve().parents[1]/'configs/research_review/wp132_native_ckf_state_contract.json').read_text())


class TestNativeState(unittest.TestCase):
    def setUp(self):
        # A rotated sensor and a distinct curvilinear parameter origin.
        angle=.31;c,s=np.cos(angle),np.sin(angle)
        frame=np.array([[c,-s,0,10],[s,c,0,-5],[0,0,1,2000],[0,0,0,1.]])
        local=np.array([.12,2.,0.]);position=frame[:3,:3]@local+frame[:3,3]
        self.parameter={'native_parameters':[0.,0.,.02,.001,1e-6],'global_position_native':position.tolist()}
        self.original={'cluster_id':10469004361915695104,'wafer_id':17,'station':2,'rdo_ids':[11,12],
                       'local_position':[.125,0.],'sigma_sq':.0001,'sensor_transform':frame.tolist()}
        prediction={'persistence_plane_tolerance_mm':plane_tolerance(position,frame),
                    'sensor_local_position_mm':local.tolist(),'sensor_frame_roundtrip_mm':0.,'loc0_mm':.12,
                    'prd_loc0_residual_mm':.005,'rot_loc1_residual_mm':.006,
                    'inside_bounds':True,'strict_is_on_surface':True}
        measurement={'wafer_id':17,'station':2,'rdo_ids':[11,12],'prd_local_position':[.125,0.],
                     'prd_local_covariance':[[.0001,0],[0,1]],'sensor_transform':frame.tolist(),
                     'measurement_surface_type':4,'native_prediction':prediction,'rot_loc1':.126,
                     'rot_covariance':[[.0001,0],[0,1]],'cluster_id':self.original['cluster_id'],
                     'rot_identifier':self.original['cluster_id'],'selected_prd':True,
                     'prd_link_resolved':True,'prd_link_valid':True,'selected_prd_pointer_equal':True}
        self.state={'type_flags':[True]+[False]*11,'measurement':measurement,'tsos_index':8,'parameter_index':7,
                    'type_description':'Measurement','fit_chi2':.1,'fit_dof':1}

    def test_rotated_sensor_uses_global_position_not_curvilinear_loc0(self):
        result=native_residual(self.original,self.parameter,self.state,P)
        self.assertTrue(result['eligible']);self.assertAlmostEqual(result['prd_residual_mm'],.005,12)
        self.assertNotAlmostEqual(result['prd_residual_mm'],self.original['local_position'][0]-self.parameter['native_parameters'][0],5)

    def test_wrong_residual_sign_rejected(self):
        self.state['measurement']['native_prediction']['prd_loc0_residual_mm']=-.005
        with self.assertRaisesRegex(ValueError,'residual sign'):native_residual(self.original,self.parameter,self.state,P)

    def test_wrong_wafer_and_nonplane_rejected(self):
        self.state['measurement']['wafer_id']=18
        with self.assertRaisesRegex(ValueError,'wafer'):native_residual(self.original,self.parameter,self.state,P)
        self.state['measurement']['wafer_id']=17;self.state['measurement']['measurement_surface_type']=0
        with self.assertRaisesRegex(ValueError,'Trk plane'):native_residual(self.original,self.parameter,self.state,P)

    def test_duplicate_and_unresolved_links_rejected(self):
        self.assertEqual(len(exact_state_map([self.state])),1)
        with self.assertRaisesRegex(ValueError,'duplicate'):exact_state_map([self.state,copy.deepcopy(self.state)])
        self.state['measurement']['prd_link_resolved']=False
        with self.assertRaisesRegex(ValueError,'unresolved'):exact_state_map([self.state])

    def test_no_measurement_cannot_be_matched_by_nearest_position(self):
        hole=copy.deepcopy(self.state);hole['measurement']=None
        self.assertEqual(exact_state_map([hole]),{})

    def test_original_parameter_one_ulp_change_rejected(self):
        old=[{'persistent_index':0,'native_parameters':[0.,0.,.1,.2,1e-6]}];new=copy.deepcopy(old)
        parameter_identity(old,new);new[0]['native_parameters'][4]=float(np.nextafter(1e-6,np.inf))
        with self.assertRaisesRegex(ValueError,'parameter identity'):parameter_identity(old,new)

    def test_float_rounding_allowance_does_not_accept_gross_off_plane(self):
        for offset,eligible in [(2e-5,True),(.01,False)]:
            parameter=copy.deepcopy(self.parameter);state=copy.deepcopy(self.state)
            parameter['global_position_native'][2]+=offset
            state['measurement']['native_prediction']['sensor_local_position_mm'][2]=offset
            state['measurement']['native_prediction']['persistence_plane_tolerance_mm']=plane_tolerance(
                np.asarray(parameter['global_position_native']),np.asarray(self.original['sensor_transform']))
            result=native_residual(self.original,parameter,state,P)
            self.assertEqual(result['eligible'],eligible)

    def test_outlier_or_hole_is_not_native_fitted_measurement_evidence(self):
        self.state['type_flags'][6]=True
        result=native_residual(self.original,self.parameter,self.state,P)
        self.assertFalse(result['eligible'])


if __name__=='__main__':unittest.main()
