"""Conditional four-coordinate saved WLS information; no propagation or fit."""
import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def array(value, shape):
    result = np.asarray(value, dtype=float)
    if len(shape) == 1:
        result = result.reshape(-1)
    require(result.shape == shape and np.isfinite(result).all(), 'shape/nonfinite')
    return result


def relative(a, b):
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(b), np.finfo(float).tiny))


def spectrum(normal, residual, protocol):
    """Data rank and weak projector, without altering any eigenvalue."""
    n = array(normal, (4, 4))
    v = array(residual, (4,))
    require(relative(n, n.T) <= protocol['symmetry_relative_tolerance'], 'normal symmetry')
    # Symmetrization removes only roundoff asymmetry; original N is retained upstream.
    symmetric = .5 * (n+n.T)
    lam, vectors = np.linalg.eigh(symmetric)
    require(lam[-1] > 0 and lam[0] >= 0, 'non-PSD information')
    ratios = lam/lam[-1]
    cutoff = protocol['weak_relative_information_cut']
    boundary = bool(np.any(abs(ratios-cutoff) <= protocol['weak_cut_boundary_absolute_relative_margin']))
    weak = ratios <= cutoff
    projector = vectors[:, weak] @ vectors[:, weak].T
    amplitudes = vectors.T @ v
    energy = amplitudes**2
    modal_q = lam*energy
    norm2 = float(v@v)
    weak_v = projector@v
    strong_v = v-weak_v
    rank = int(sum(lam > 4*np.finfo(float).eps*lam[-1]))
    singular, right = np.linalg.svd(symmetric)[1:]
    svweak = singular/singular[0] <= cutoff
    svprojector = right[svweak].T @ right[svweak]
    algebra = protocol['algebra_relative_tolerance']
    require(relative(singular[::-1], lam) <= algebra, 'eigh/SVD spectrum')
    require(np.linalg.norm(svprojector-projector) <= algebra, 'eigh/SVD weak projector')
    require(relative(symmetric@vectors, vectors*lam) <= algebra, 'eigen residual')
    require(relative(vectors.T@vectors, np.eye(4)) <= algebra, 'orthogonality')
    q = float(v@symmetric@v)
    q_modal = float(sum(modal_q))
    require(abs(q-q_modal) <= algebra*max(abs(q), abs(q_modal), np.finfo(float).tiny), 'modal Q')
    return {'eigenvalues': lam.tolist(), 'eigenvectors_columns': vectors.tolist(),
            'relative_eigenvalues': ratios.tolist(), 'numerical_rank': rank,
            'condition': float(lam[-1]/lam[0]) if lam[0] > 0 else None,
            'weak_dimension': int(sum(weak)), 'weak_projector': projector.tolist(),
            'weak_scaled_residual': weak_v.tolist(), 'strong_scaled_residual': strong_v.tolist(),
            'scaled_residual_norm2': norm2,
            'weak_scaled_residual_fraction': float(sum(energy[weak])/norm2) if norm2 else None,
            'angular_scaled_residual_fraction': float(v[2:]@v[2:]/norm2) if norm2 else None,
            'modal_scaled_residual_squared': energy.tolist(), 'modal_Q': modal_q.tolist(),
            'Q': q, 'Q_weak': float(sum(modal_q[weak])), 'Q_strong': float(sum(modal_q[~weak])),
            'Q_weak_fraction': float(sum(modal_q[weak])/q) if q else None,
            'weak_cut_boundary': boundary, 'svd_projector_difference': float(np.linalg.norm(svprojector-projector)),
            'eigen_residual': relative(symmetric@vectors, vectors*lam),
            'orthogonality_error': relative(vectors.T@vectors, np.eye(4))}


def audit_cell(ref, prediction, protocol):
    require(ref['state_index'] == 0 and ref['input_event_header_verified'], 'reference/header')
    nraw = array(ref['raw_normal'], (4, 4))
    craw = array(ref['raw_covariance'], (4, 4))
    fixed = array(ref['fixed_z_state'], (4,))
    raw = array(ref['raw_fit'], (4,))
    cfixed = array(ref['fixed_z_covariance'], (4, 4))
    a = array(ref['raw_to_native_jacobian'], (5, 4))
    b = array(ref['native_to_fixed_z_jacobian'], (4, 5))
    h = array(prediction, (4,))
    scales = array(protocol['scales'], (4,))
    require(np.all(scales > 0), 'units/scales')
    dz = float(ref['z_state_mm']) - float(ref['z_center_mm'])
    require(np.isfinite(dz), 'lever arm')
    t = np.eye(4); t[0, 2] = t[1, 3] = dz
    ti = np.eye(4); ti[0, 2] = ti[1, 3] = -dz
    state_error = float(np.max(abs(fixed-t@raw)))
    jacobian_error = relative((b@a)*scales[None, :]/scales[:, None],
                              t*scales[None, :]/scales[:, None])
    require(state_error <= protocol['state_absolute_tolerance'], 'raw/fixed state')
    require(jacobian_error <= protocol['roundtrip_relative_tolerance'], 'angle/frame export')
    expected = t@craw@t.T
    covariance_error = relative(cfixed/scales[:, None]/scales[None, :],
                                expected/scales[:, None]/scales[None, :])
    require(covariance_error <= protocol['roundtrip_relative_tolerance'], 'covariance frame')
    nfixed = ti.T@nraw@ti
    ns = scales[:, None]*nfixed*scales[None, :]
    cs = cfixed/scales[:, None]/scales[None, :]
    require(relative(cs, cs.T) <= protocol['symmetry_relative_tolerance'], 'covariance symmetry')
    require(np.linalg.eigvalsh(.5*(cs+cs.T))[0] > 0, 'non-SPD covariance')
    inverse_error = relative(ns@cs, np.eye(4))
    require(inverse_error <= protocol['roundtrip_relative_tolerance'], 'normal/covariance inverse')
    r = fixed-h
    v = r/scales
    info = spectrum(ns, v, protocol)
    require(info['eigenvalues'][0] > 0, 'non-SPD normal')
    cholesky = np.linalg.cholesky(.5*(ns+ns.T))
    q_cholesky = float(np.linalg.norm(cholesky.T@v)**2)
    q_covariance = float(v@np.linalg.solve(cs, v))
    raw_residual = ti@r
    q_raw = float(raw_residual@nraw@raw_residual)
    for q in (q_raw, q_cholesky, q_covariance):
        require(abs(q-info['Q']) <= protocol['algebra_relative_tolerance']*
                max(abs(q), abs(info['Q']), np.finfo(float).tiny), 'quadratic closure')
    fraction = info['weak_scaled_residual_fraction']
    status = 'UNKNOWN' if info['weak_cut_boundary'] or fraction is None else 'KNOWN'
    return {'information_status': status, 'raw_normal': nraw.tolist(), 'fixed_normal': nfixed.tolist(),
            'scaled_normal': ns.tolist(), 'fixed_covariance': cfixed.tolist(), 'T': t.tolist(),
            'residual': r.tolist(), 'scaled_residual': v.tolist(), 'prediction': h.tolist(),
            'weak_residual_original_units': (scales*np.array(info['weak_scaled_residual'])).tolist(),
            'strong_residual_original_units': (scales*np.array(info['strong_scaled_residual'])).tolist(),
            'state_error': state_error, 'jacobian_error': jacobian_error,
            'covariance_error': covariance_error, 'normal_inverse_error': inverse_error,
            'Q_raw': q_raw, 'Q_cholesky': q_cholesky, 'Q_covariance_solve': q_covariance,
            'spectrum': info, 'hit_normal_independent_rebuild': 'UNKNOWN_NOT_SAVED',
            'covariance_calibration': 'NOT_EVALUATED', 'physical_likelihood': 'NOT_EVALUATED'}


def hypothesis(cells, protocol):
    primary = [next(c for c in cells if [c['index'], c['station']] == key)
               for key in protocol['primary_cells']]
    known = [c for c in primary if c.get('information_status') == 'KNOWN']
    if any(c['spectrum']['weak_scaled_residual_fraction'] < protocol['primary_min_weak_scaled_residual_fraction'] for c in known):
        return 'NOT_SUPPORTED'
    if len(known) != len(primary):
        return 'UNKNOWN'
    return 'SUPPORTED_BUT_LIMITED'
