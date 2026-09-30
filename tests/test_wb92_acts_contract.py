import copy
import json
from pathlib import Path

import numpy as np
import pytest

from alignment.wb92_acts_contract import derivative_check, taylor_check, verify_identity

P = json.loads((Path(__file__).resolve().parents[1]/'configs/research_review/wp92_common_seed_acts_contract.json').read_text())


def test_scaled_derivative_rejects_step_sensitive_response_and_nonfinite():
    s = {'plus': [1,0,0,0], 'minus': [-1,0,0,0], 'half_plus': [.5,0,0,0], 'half_minus': [-.5,0,0,0]}
    assert derivative_check([s], [.01], np.array(P['output_scales']), P)['pass']
    bad = copy.deepcopy(s); bad['half_plus'][0] = .51
    assert not derivative_check([bad], [.01], np.array(P['output_scales']), P)['pass']
    bad['plus'][0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        derivative_check([bad], [.01], np.array(P['output_scales']), P)


def test_taylor_rejects_wrong_sign_and_noncontracting_error():
    s = {'delta': [1.], 'linear_effect': [1,0,0,0], 'full': [1,0,0,0], 'half': [.5,0,0,0]}
    J = np.array([[1.],[0.],[0.],[0.]])
    assert taylor_check(s, np.zeros(4), J, np.ones(4), P)['pass']
    with pytest.raises(ValueError, match='independent FD'):
        taylor_check(s, np.zeros(4), -J, np.ones(4), P)
    s['full'][0] += .001; s['half'][0] += .001
    assert not taylor_check(s, np.zeros(4), J, np.ones(4), P)['pass']


def test_identity_is_source_aware_and_y_immutable():
    f = {'input_xaod': '/allowed/source.root', 'ordinal': 2200, 'actual_run': 100044,
         'actual_event': 2200, 'variant': 'baseline',
         'references': [{'station': i, 'fixed_z_state': [[1],[2],[.01],[.02]],
                         'q_over_p_per_MeV': 1e-5, 'z_state_mm': i*100., 'clusters': [i]} for i in range(4)]}
    r = {k:f[k] for k in ('input_xaod','ordinal','actual_run','actual_event','variant')}
    r.update(seed=[[1],[2],[.01],[.02],[1e-5]], seed_z_mm=0.,
             targets=[{'station': i, 'y': copy.deepcopy(f['references'][i]['fixed_z_state']),
                       'sensors': [{'strip': i}]} for i in range(4)])
    verify_identity(r, f)
    bad = copy.deepcopy(r); bad['input_xaod'] = '/allowed/other.root'
    with pytest.raises(ValueError, match='identity mismatch'):
        verify_identity(bad, f)
    bad = copy.deepcopy(r); bad['targets'][1]['y'][0][0] += 1
    with pytest.raises(ValueError, match='measurement y changed'):
        verify_identity(bad, f)
