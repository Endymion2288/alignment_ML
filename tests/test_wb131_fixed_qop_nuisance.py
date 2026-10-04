"""Independent analytic-plane oracle, sign and identifiability controls."""
import json
from pathlib import Path
import unittest

import numpy as np
from alignment.wb131_fixed_qop_nuisance import conditional_fit, stable_columns

P = json.loads((Path(__file__).resolve().parents[1]/'configs/research_review/wp131_fixed_qop_nuisance_contract.json').read_text())


def plane_prediction(seed, centers, axes, normals):
    origin = np.array([seed[0], seed[1], 0.])
    direction = np.array([seed[2], seed[3], 1.])
    distance = np.sum(normals*(centers-origin), axis=1)/(normals@direction)
    endpoints = origin + distance[:, None]*direction
    return np.sum(axes*(endpoints-centers), axis=1)


def analytic_h(seed, centers, axes, normals):
    origin = np.array([seed[0], seed[1], 0.])
    direction = np.array([seed[2], seed[3], 1.])
    denominator = normals@direction
    distance = np.sum(normals*(centers-origin), axis=1)/denominator
    columns = []
    for axis in range(4):
        ds, dv = np.zeros(3), np.zeros(3)
        if axis<2:
            ds[axis] = 1.
        else:
            dv[axis-2] = 1.
        dl = -(normals@ds + distance*(normals@dv))/denominator
        dx = ds + dl[:,None]*direction + distance[:,None]*dv
        columns.append(np.sum(axes*dx, axis=1))
    return np.column_stack(columns)


class TestWB131(unittest.TestCase):
    def setUp(self):
        # Four stations, alternating stereo, physically orthogonal plane axes.
        self.centers = np.array([[0.,0.,z] for z in (0.,1000.,2000.,3000.) for _ in range(6)])
        angles = np.tile(np.array([-.02,.02,-.02,.02,-.02,.02]),4)
        self.axes = np.column_stack([np.cos(angles),np.sin(angles),np.zeros(24)])
        self.normals = np.column_stack([-.002*np.sin(angles),.002*np.cos(angles),np.ones(24)])
        self.normals /= np.linalg.norm(self.normals,axis=1)[:,None]
        self.seed = np.array([3.,4.,.001,-.002])
        self.sigma = np.linspace(.01,.04,24)
        self.h = analytic_h(self.seed,self.centers,self.axes,self.normals)

    def test_tilted_plane_fd_agrees_with_analytic_derivative(self):
        matrices=[]
        for factor in P['fd_factors']:
            cols=[]
            for axis,step in enumerate(P['fd_steps']):
                dp=np.zeros(4);dp[axis]=step*factor
                cols.append((plane_prediction(self.seed+dp,self.centers,self.axes,self.normals)-
                             plane_prediction(self.seed-dp,self.centers,self.axes,self.normals))/(2*step*factor))
            matrices.append(np.column_stack(cols))
        np.testing.assert_allclose(matrices[0],self.h,rtol=1e-8,atol=1e-8)
        self.assertTrue(stable_columns(*matrices,self.sigma,P)[1])

    def test_physical_seed_recovery_and_opposite_sign_negative_control(self):
        known = np.array([.005,-.003,1e-6,-2e-6])
        measured=plane_prediction(self.seed+known,self.centers,self.axes,self.normals)
        nominal=plane_prediction(self.seed,self.centers,self.axes,self.normals)
        fit=conditional_fit(self.h,measured-nominal,self.sigma,P)
        self.assertEqual(fit['gate'],'READY')
        np.testing.assert_allclose(fit['delta'],known,rtol=1e-5,atol=1e-9)
        delta=np.asarray(fit['delta'])
        costs=[np.sum(((measured-plane_prediction(self.seed+sign*delta,self.centers,self.axes,self.normals))/self.sigma)**2) for sign in (1,-1)]
        self.assertLess(costs[0],fit['nominal_cost']*1e-8)
        self.assertGreater(costs[1],fit['nominal_cost']*3.9)

    def test_no_stereo_is_rank_deficient(self):
        axes=np.tile([1.,0.,0.],(24,1));normals=np.tile([0.,0.,1.],(24,1))
        h=analytic_h(self.seed,self.centers,axes,normals)
        fit=conditional_fit(h,np.ones(24),self.sigma,P)
        self.assertEqual(fit['rank'],2)
        self.assertEqual(fit['gate'],'UNKNOWN_RANK')

    def test_corrupted_half_step_rejected(self):
        corrupted=self.h.copy();corrupted[:,2]*=1.01
        self.assertFalse(stable_columns(self.h,corrupted,self.sigma,P)[1])

    def test_large_update_not_damped(self):
        fit=conditional_fit(self.h,self.h@np.array([20.,0.,0.,0.]),self.sigma,P)
        self.assertEqual(fit['gate'],'UNKNOWN_UPDATE_CAP')
        self.assertAlmostEqual(fit['delta'][0],20.)


if __name__=='__main__':
    unittest.main()
