import unittest
import copy
import numpy as np
from alignment.wb134_material_inventory import thin_theta,toy_process_q,classify_inventory

class TestWB134(unittest.TestCase):
    def test_zero_material_and_unit_negative_control(self):
        self.assertEqual(thin_theta(0,93.7,100000,105.6583755),0)
        value=thin_theta(.3,93.7,100000,105.6583755)
        self.assertTrue(5e-6<value<7e-6)
        self.assertGreater(thin_theta(.3,93.7,100,105.6583755)/value,1000)
        with self.assertRaises(ValueError):thin_theta(-.3,93.7,100000,105.6583755)

    def test_noise_psd_rank_rotation_and_backward_lever(self):
        d=np.array([1.,2.,3.]);d/=np.linalg.norm(d)
        q=toy_process_q(6e-6,1000,d)
        self.assertGreaterEqual(np.linalg.eigvalsh(q).min(),-1e-18);self.assertEqual(np.linalg.matrix_rank(q,tol=1e-15),2)
        a=np.block([[np.eye(3),np.zeros((3,3))],[np.zeros((3,3)),-np.eye(3)]])
        np.testing.assert_allclose(toy_process_q(6e-6,-1000,d),a@q@a.T)
        axis=np.cross(d,[0,0,1]);axis/=np.linalg.norm(axis)
        self.assertAlmostEqual(float(axis@q[:3,:3]@axis),(1000*6e-6)**2)
        np.testing.assert_array_equal(toy_process_q(0,1000,d),np.zeros((6,6)))

    def test_proto_is_not_a_physical_slab(self):
        r={'root_volume_index':0,'sensitive_visit_count':1,'volumes':[{'index':0,'children':[],'surface_indices':[0],'material_kind':'NONE'}],
           'surfaces':[{'index':0,'material_kind':'PROTO_VACUUM','memberships':[{'role':'SENSITIVE_ARRAY','volume_index':0}],
                        'center_sample':{'valid':False,'thickness_mm':0,'thickness_in_X0':0,'thickness_in_L0':0}}]}
        self.assertEqual(classify_inventory(r)['surface_counts']['PROTO_VACUUM'],1)
        corrupt=copy.deepcopy(r);corrupt['surfaces'][0]['center_sample']['valid']=True
        with self.assertRaises(ValueError):classify_inventory(corrupt)
        corrupt=copy.deepcopy(r);corrupt['volumes'][0]['children']=[0]
        with self.assertRaises(ValueError):classify_inventory(corrupt)
        corrupt=copy.deepcopy(r);corrupt['sensitive_visit_count']=2
        with self.assertRaises(ValueError):classify_inventory(corrupt)

if __name__=='__main__':unittest.main()
