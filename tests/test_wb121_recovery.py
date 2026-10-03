"""Exact syntax adapter and unchanged saved stream recovery guards."""
import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from wb121_recovery import compatible_source, load_analyzer, recover, SOURCE
# The test runner installs this exact generated reader before collecting WB120 tests.
from test_wb120_response import synthetic


def inputs():
    data = list(synthetic()); stream = []
    for row in data[0]['rows']:
        stream.extend([{'record': 'before_official', 'input': copy.deepcopy(row['input'])},
                       {'record': 'after_official', 'row': copy.deepcopy(row)}])
    stream.append({'record': 'terminal', 'status': 'COMPLETED', 'official_calls': 36})
    return data + [stream]


def test_only_two_tuple_key_expressions_change_and_generated_reader_imports():
    source = SOURCE.read_text(); new = compatible_source(source)
    assert new.replace('traces[(station, *a)]', 'traces[station, *a]').replace('traces[(station, *b)]', 'traces[station, *b]') == source
    assert load_analyzer(new).ORDER == [(0., 0)] + [(m, s) for m in (.25, .5, 1., 2.) for s in (1, -1)]


def test_exact_saved_recovery_does_not_mutate_runtime_or_stream():
    data = inputs(); before = copy.deepcopy(data)
    s = recover(*data)
    assert data == before
    assert s['official_calls'] == 36 and len(s['comparisons']) == 36
    assert s['classification'] == 'DIAGNOSTIC_ONLY_OFFICIAL_STEPPING_SNAPSHOTS'
    assert s['qualification'] == 'NOT_EVALUATED'


@pytest.mark.parametrize('bad', ['missing_before', 'changed_before', 'changed_after', 'changed_terminal', 'duplicate_after', 'schema', 'call_count', 'control'])
def test_stream_and_runtime_mutations_rejected_without_new_physical_work(bad):
    data = inputs(); r,c,f,h,old,stream = data
    if bad == 'missing_before': stream.pop(2)
    elif bad == 'changed_before': stream[2]['input']['station'] = 99
    elif bad == 'changed_after': stream[3]['row']['snapshots'][0]['snapshot_index'] = 999
    elif bad == 'changed_terminal': stream[-1]['official_calls'] = 18
    elif bad == 'duplicate_after': stream[3] = copy.deepcopy(stream[1])
    elif bad == 'schema': r['schema'] = 'wb119_bounded_response_runtime_v1'
    elif bad == 'call_count': r['official_calls'] = 18
    elif bad == 'control': c['samples'].reverse()
    with pytest.raises(ValueError): recover(*data)


@pytest.mark.parametrize('bad', ['missing_key', 'duplicate_key', 'already_recovered'])
def test_adapter_rejects_unexpected_original_source(bad):
    s = SOURCE.read_text()
    if bad == 'missing_key': s = s.replace('traces[station, *a]', 'unexpected')
    elif bad == 'duplicate_key': s += '\n# traces[station, *a]\n'
    elif bad == 'already_recovered': s = compatible_source(s)
    with pytest.raises(ValueError): compatible_source(s)
