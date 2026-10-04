"""Analytic controls independent of the six saved development normals."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from wb126_information import audit_cell, spectrum

P = json.loads((ROOT/'configs/research_review/wp126_tracklet_information.json').read_text())


def reference(dz=20.):
    # Independent analytic hit design with four independently constrained parameters.
    design = np.array([[1., 0, 0, 0], [0, 2., 0, 0], [1., 0, 30., 0], [0, 1., 0, 50.],
                       [1., 1., -20., -20.]])
    n = design.T@design
    q, r = np.linalg.qr(design, mode='reduced')
    ri = np.linalg.solve(r, np.eye(4)); c = ri@ri.T
    raw = np.array([2., -3., .01, -.02])
    t = np.eye(4); t[0, 2] = t[1, 3] = dz
    a = np.vstack([np.eye(4), np.zeros(4)])
    b = np.column_stack([t, np.zeros(4)])
    return {'state_index': 0, 'input_event_header_verified': True,
            'raw_normal': n.tolist(), 'raw_covariance': c.tolist(), 'raw_fit': raw.tolist(),
            'fixed_z_state': (t@raw).tolist(), 'fixed_z_covariance': (t@c@t.T).tolist(),
            'raw_to_native_jacobian': a.tolist(), 'native_to_fixed_z_jacobian': b.tolist(),
            'z_state_mm': dz, 'z_center_mm': 0.}


def test_qr_hit_design_and_nonzero_lever_arm():
    ref = reference(); h = np.array(ref['fixed_z_state']) + [.3, -.2, .001, -.002]
    cell = audit_cell(ref, h, P)
    assert cell['spectrum']['numerical_rank'] == 4
    assert cell['Q_raw'] == pytest.approx(cell['spectrum']['Q'])
    assert cell['Q_covariance_solve'] == pytest.approx(cell['spectrum']['Q'])


def test_exact_rank_deficiency_retained_without_damping():
    s = spectrum(np.diag([1., 4., 0., 0.]), [0, 0, 3, 4], P)
    assert s['numerical_rank'] == 2
    assert s['weak_dimension'] == 2
    assert s['weak_scaled_residual_fraction'] == 1
    assert s['Q'] == 0


def test_rotated_degenerate_weak_subspace():
    basis, _ = np.linalg.qr(np.array([[1., 2, 3, 4], [2., -1, 1, 3], [4., 3, -2, 1], [3., 2, 1, -4]]))
    n = basis@np.diag([1.e-5, 1.e-5, 1., 2.])@basis.T
    v = 3*basis[:, 0]+4*basis[:, 1]
    s = spectrum(n, v, P)
    assert s['weak_scaled_residual_fraction'] == pytest.approx(1)
    assert np.allclose(s['weak_projector'], basis[:, :2]@basis[:, :2].T)


def test_consistent_unit_conversion_invariance():
    ref = reference(); h = np.array(ref['fixed_z_state']) + [.3, -.2, .001, -.002]
    original = audit_cell(ref, h, P)
    # Convert mm -> m, retaining slopes; inverse normal and covariance units together.
    u = np.diag([.001, .001, 1., 1.]); ui = np.linalg.inv(u)
    changed = copy.deepcopy(ref); changed['z_state_mm'] *= .001
    for k in ('raw_fit', 'fixed_z_state'): changed[k] = (u@ref[k]).tolist()
    for k in ('raw_covariance', 'fixed_z_covariance'): changed[k] = (u@np.array(ref[k])@u.T).tolist()
    changed['raw_normal'] = (ui.T@np.array(ref['raw_normal'])@ui).tolist()
    changed['raw_to_native_jacobian'] = (np.array(ref['raw_to_native_jacobian'])@ui).tolist()
    changed['native_to_fixed_z_jacobian'] = (u@np.array(ref['native_to_fixed_z_jacobian'])).tolist()
    p = copy.deepcopy(P); p['scales'] = (u@P['scales']).tolist()
    converted = audit_cell(changed, u@h, p)
    assert np.allclose(converted['spectrum']['eigenvalues'], original['spectrum']['eigenvalues'])
    assert converted['spectrum']['Q'] == pytest.approx(original['spectrum']['Q'])


@pytest.mark.parametrize('corruption', ['normal', 'covariance', 'frame', 'angle', 'header', 'nonfinite', 'indefinite', 'units'])
def test_fail_closed_controls(corruption):
    ref = reference(); h = np.array(ref['fixed_z_state']) + [.3, -.2, .001, -.002]
    if corruption == 'normal': ref['raw_normal'][0][0] *= 2
    if corruption == 'covariance': ref['fixed_z_covariance'][2][2] *= 2
    if corruption == 'frame': ref['z_state_mm'] = 0
    if corruption == 'angle': ref['native_to_fixed_z_jacobian'][2][2] *= -1
    if corruption == 'header': ref['input_event_header_verified'] = False
    if corruption == 'nonfinite': ref['raw_normal'][0][0] = float('nan')
    if corruption == 'indefinite': ref['raw_normal'][0][0] = -1.e6
    if corruption == 'units': ref['raw_normal'] = (np.array(ref['raw_normal'])*1.e6).tolist()
    with pytest.raises(ValueError): audit_cell(ref, h, P)


def test_exact_cut_boundary_is_unknown():
    p = copy.deepcopy(P); p['scales'] = [1., 1., 1., 1.]
    ref = reference(0.); n = np.diag([.001, .01, 1., 1.])
    ref['raw_normal'] = n.tolist(); ref['raw_covariance'] = np.linalg.inv(n).tolist()
    ref['fixed_z_covariance'] = ref['raw_covariance']
    assert audit_cell(ref, np.array(ref['fixed_z_state'])+[1, 0, 0, 0], p)['information_status'] == 'UNKNOWN'


def test_non_spd_real_cell_is_not_pseudoinverted():
    ref = reference(); ref['raw_normal'] = np.diag([1., 1., 0., 1.]).tolist()
    with pytest.raises(ValueError): audit_cell(ref, ref['fixed_z_state'], P)
