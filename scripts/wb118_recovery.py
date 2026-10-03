#!/usr/bin/env python3
"""Saved WB118 aggregation recovery. Historical code and bytes remain unchanged."""
import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT, read_public, write_new
from wb100_contract import digest, verify
from wb118_contract import OUT as ORIGINAL
from audit_wb118_response import analyze, decompose, check

OUT = ROOT/'outputs/mc24_four_station_wb118_saved_aggregation_recovery_v1'
WORKBOOK = ROOT/'workbook/2026-10-03_118_四站五维更新的四维与动量分离响应前瞻控制.md'


def recover(runtime, control, fixture, historical, stream):
    check(runtime['schema'] == 'wb117_bounded_response_runtime_v1', 'exact original producer schema')
    check(runtime['official_calls'] == len(runtime['rows']) == 24 and runtime['control_used'] == control, 'actual WB118 producer matrix')
    check(len(stream) == 49 and stream[-1] == {'record': 'terminal', 'status': 'COMPLETED', 'official_calls': 24}, 'complete 24-call stream')
    for i, row in enumerate(runtime['rows']):
        check(stream[2*i] == {'record': 'before_official', 'input': row['input']} and
              stream[2*i+1] == {'record': 'after_official', 'row': row}, 'raw stream identity')
    normalized = copy.deepcopy(runtime)
    normalized['schema'] = 'wb118_bounded_response_runtime_v1'
    summary = decompose(analyze(normalized, control, fixture, historical), control, fixture, historical)
    return summary


def freeze():
    check(subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == '4station', 'branch')
    check(not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).strip(), 'tracked changes')
    frozen = verify(ORIGINAL); hashes = frozen['hashes'].copy()
    manifest = ROOT/'docs/wb118_split_response_failure_manifest.json'; failed = read_public(manifest)
    check(failed['original_execution'] == 'FAIL_AGGREGATION_SCHEMA_INTERFACE' and failed['worker_exit'] == 1 and failed['athena_exit'] == 0, 'immutable failure')
    for p, h in failed['artifacts'].items():
        check(digest(ROOT/p) == h, 'saved failure artifact '+p); hashes[str(ROOT/p)] = h
    # Check against the frozen actual generated producer; do not edit it.
    source = ORIGINAL/'isolated_source/WB118Diagnostic/BoundedResponse.cxx'
    text = source.read_text()
    check(text.count('"wb117_bounded_response_runtime_v1"') == 1 and 'if(count!=24)' in text and
          'namespace WB118' in text, 'producer schema/namespace/call origin')
    for p in (Path(__file__), ROOT/'tests/test_wb118_recovery.py', manifest):
        hashes[str(p)] = digest(p)
    OUT.mkdir(exist_ok=False); shutil.copyfile(WORKBOOK, OUT/'contract_workbook.md')
    hashes[str(OUT/'contract_workbook.md')] = digest(OUT/'contract_workbook.md')
    write_new(OUT/'freeze.json', {'schema': 'wb118_saved_aggregation_recovery_freeze_v1', 'hashes': hashes,
        'branch': '4station', 'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'original_execution': 'FAIL_AGGREGATION_SCHEMA_INTERFACE', 'new_physical_calls': 0, 'root_access': False,
        'only_runtime_in_memory_change': {'key': 'schema', 'from': 'wb117_bounded_response_runtime_v1', 'to': 'wb118_bounded_response_runtime_v1'}})


def rebuild():
    runtime = read_public(ORIGINAL/'event/response.json')
    stream = [json.loads(line) for line in (ORIGINAL/'event/response.json.calls.ndjson').read_text().splitlines()]
    return recover(runtime, read_public(ORIGINAL/'control.json'), read_public(ORIGINAL/'fixture.json'),
                   read_public(ORIGINAL/'historical_runtime.json'), stream)


def run():
    frozen = verify(OUT); (OUT/'execution_lock').mkdir(exist_ok=False)
    summary = rebuild(); write_new(OUT/'summary.json', summary); verify(OUT)
    write_new(OUT/'recovery_receipt.json', {'schema': 'wb118_saved_aggregation_recovery_receipt_v1', 'exit_code': 0,
        'identities_verified': len(frozen['hashes']), 'original_worker_exit': 1, 'original_athena_exit': 0,
        'scope': 'in-memory exact schema normalization; unchanged frozen analyze/decompose rules',
        'old_artifacts_unchanged': True, 'new_physical_calls': 0, 'new_event_root_access': False,
        'qualification': 'NOT_EVALUATED', 'physics_screening': 'NOT_EVALUATED', 'final_oracle': 'NOT_EVALUATED'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze', 'verify', 'run'))
    action = parser.parse_args().action
    if action == 'verify':
        verify(OUT)
    else:
        globals()[action]()
