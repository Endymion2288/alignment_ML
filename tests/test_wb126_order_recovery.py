"""Unordered identity recovery must not waive any numeric or membership change."""
import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from wb126_order_recovery import compare_reference


def test_only_permutation_is_allowed():
    a = {'clusters': [5, 3, 4], 'raw_normal': [[1.]], 'run': 123}
    b = copy.deepcopy(a); b['clusters'].sort()
    assert compare_reference(a, b)['ordering_differs']


@pytest.mark.parametrize('change', ['duplicate', 'membership', 'normal', 'covariance', 'run', 'missing'])
def test_recovery_fails_closed(change):
    a = {'clusters': [5, 3, 4], 'raw_normal': [[1.]], 'raw_covariance': [[1.]], 'run': 123}
    b = copy.deepcopy(a)
    if change == 'duplicate': b['clusters'] = [5, 5, 3, 4]
    if change == 'membership': b['clusters'] = [5, 3, 6]
    if change == 'normal': b['raw_normal'][0][0] += 1e-10
    if change == 'covariance': b['raw_covariance'][0][0] += 1e-10
    if change == 'run': b['run'] += 1
    if change == 'missing': del b['raw_normal']
    with pytest.raises((ValueError, KeyError)): compare_reference(a, b)
