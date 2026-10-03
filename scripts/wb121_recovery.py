#!/usr/bin/env python3
"""Only tuple-key syntax recovery of the frozen WB120 analyzer; saved JSON only."""
import argparse
import ast
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT, read_public, write_new
from wb100_contract import digest, verify
from wb120_contract import OUT as ORIGINAL

OUT = ROOT/'outputs/mc24_four_station_wb121_saved_stepping_recovery_v1'
WORKBOOK = ROOT/'workbook/2026-10-04_121_四站官方步进保存汇总的双环境兼容恢复与诊断.md'
SOURCE = ROOT/'scripts/audit_wb120_response.py'
REPLACEMENTS = (('traces[station, *a]', 'traces[(station, *a)]'),
                ('traces[station, *b]', 'traces[(station, *b)]'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compatible_source(original=None):
    original = SOURCE.read_text() if original is None else original
    require(original.count(REPLACEMENTS[0][0]) == original.count(REPLACEMENTS[1][0]) == 1, 'exact two frozen subscript occurrences')
    compatible = original
    for old, new in REPLACEMENTS:
        compatible = compatible.replace(old, new)
    changed_lines = [(i+1, x, y) for i, (x, y) in enumerate(zip(original.splitlines(), compatible.splitlines())) if x != y]
    require(len(original.splitlines()) == len(compatible.splitlines()) and len(changed_lines) == 1 and changed_lines[0][0] == 157,
            'exact one-line syntax-only diff')
    compile(compatible, str(OUT/'compatible_audit.py'), 'exec')
    return compatible


def load_analyzer(source=None):
    source = compatible_source() if source is None else source
    module = types.ModuleType('wb121_compatible_analyzer')
    module.__file__ = str(OUT/'compatible_audit.py')
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def ml_ast_proof():
    original = SOURCE.read_text(); compatible = compatible_source(original)
    # The original subscript syntax needs Python >=3.11 to parse.
    require(sys.version_info >= (3, 11), 'ML Python >=3.11 for original AST proof')
    old_ast = ast.dump(ast.parse(original), include_attributes=False)
    new_ast = ast.dump(ast.parse(compatible), include_attributes=False)
    require(old_ast == new_ast, 'exact AST equivalence')
    return {'old_new_ast_exact': True, 'ast_sha256': hashlib.sha256(old_ast.encode()).hexdigest(),
            'original_source_sha256': digest(SOURCE), 'compatible_source_sha256': hashlib.sha256(compatible.encode()).hexdigest(),
            'python': sys.version, 'changed_lines': [157], 'expression_replacements': [list(v) for v in REPLACEMENTS]}


def recover(runtime, control, fixture, historical, old_rows, stream, analyzer=None):
    require(runtime['schema'] == 'wb120_stepping_runtime_v1' and runtime['official_calls'] == len(runtime['rows']) == 36, 'frozen runtime schema/matrix')
    require(len(stream) == 73 and stream[-1] == {'record': 'terminal', 'status': 'COMPLETED', 'official_calls': 36}, 'saved complete stream')
    for i, row in enumerate(runtime['rows']):
        require(stream[2*i] == {'record': 'before_official', 'input': row['input']} and
                stream[2*i+1] == {'record': 'after_official', 'row': row}, 'unchanged stream/runtime identity')
    return (load_analyzer() if analyzer is None else analyzer).analyze(runtime, control, fixture, historical, old_rows)


def freeze():
    require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == '4station', 'branch')
    require(not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).strip(), 'tracked changes')
    hashes = verify(ORIGINAL)['hashes'].copy()
    failure = ROOT/'docs/wb120_official_stepping_failure_manifest.json'; previous = read_public(failure)
    require(previous['original_execution'] == 'FAIL_AGGREGATION_PYTHON_VERSION_INTERFACE' and previous['scheduler_exit'] == previous['worker_exit'] == 1 and previous['athena_exit'] == 0, 'immutable original failure')
    for p, h in previous['artifacts'].items():
        require(digest(ROOT/p) == h, 'failure artifact '+p); hashes[str(ROOT/p)] = h
    OUT.mkdir(exist_ok=False); shutil.copyfile(WORKBOOK, OUT/'contract_workbook.md')
    source = compatible_source()
    with (OUT/'compatible_audit.py').open('x') as f:
        f.write(source)
    diff = ''.join(difflib.unified_diff(SOURCE.read_text().splitlines(True), source.splitlines(True), fromfile='frozen_wb120_original', tofile='wb121_tuple_key_compatible'))
    with (OUT/'syntax_only.diff').open('x') as f:
        f.write(diff)
    proofs = ROOT/'outputs/mc24_four_station_wb121_recovery_preflight_v1'
    for p in (Path(__file__), ROOT/'tests/test_wb121_recovery.py', failure, SOURCE, OUT/'contract_workbook.md',
              OUT/'compatible_audit.py', OUT/'syntax_only.diff', *[p for p in proofs.iterdir() if p.is_file()]):
        hashes[str(p)] = digest(p)
    ml = read_public(proofs/'ml_ast_and_test_proof.json'); cal = read_public(proofs/'calypso_import_and_test_proof.json')
    require(ml['old_new_ast_exact'] is True and ml['pytest_exit'] == cal['pytest_exit'] == 0 and cal['compatible_import'] is True,
            'both environment proofs')
    require(ml['compatible_source_sha256'] == cal['compatible_source_sha256'] == digest(OUT/'compatible_audit.py'), 'tested exact generated source')
    require(cal['original_source_parse_rejected'] is True and cal['python_version_info'][:2] == [3, 9], 'actual Calypso Python3.9 regression')
    write_new(OUT/'freeze.json', {'schema': 'wb121_saved_stepping_recovery_freeze_v1', 'branch': '4station',
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'hashes': hashes,
              'original_execution': 'FAIL_AGGREGATION_PYTHON_VERSION_INTERFACE', 'new_physical_calls': 0, 'root_access': False,
              'changed_lines': [157], 'expression_replacements': [list(v) for v in REPLACEMENTS], 'qualification': 'NOT_EVALUATED'})


def rebuild():
    verify(OUT)
    analyzer = load_analyzer((OUT/'compatible_audit.py').read_text())
    runtime = read_public(ORIGINAL/'event/response.json')
    stream = [json.loads(line) for line in (ORIGINAL/'event/response.json.calls.ndjson').read_text().splitlines()]
    return recover(runtime, read_public(ORIGINAL/'control.json'), read_public(ORIGINAL/'fixture.json'),
                   read_public(ORIGINAL/'historical_runtime.json'), read_public(ORIGINAL/'historical_responses.json'), stream, analyzer)


def run():
    frozen = verify(OUT); (OUT/'execution_lock').mkdir(exist_ok=False)
    write_new(OUT/'environment.json', {'python': sys.version, 'executable': sys.executable,
              'scope': 'saved JSON only; no Athena/ROOT/reconstruction/propagation'})
    summary = rebuild(); write_new(OUT/'summary.json', summary); verify(OUT); verify(ORIGINAL)
    write_new(OUT/'recovery_receipt.json', {'schema': 'wb121_saved_stepping_recovery_receipt_v1', 'exit_code': 0,
              'identities_verified': len(frozen['hashes']), 'original_worker_exit': 1, 'original_athena_exit': 0,
              'scope': 'one-line two-expression tuple-key syntax recovery; unchanged AST/scientific rules',
              'old_artifacts_unchanged': True, 'new_physical_calls': 0, 'new_event_root_access': False,
              'qualification': 'NOT_EVALUATED', 'physics_screening': 'NOT_EVALUATED', 'final_oracle': 'NOT_EVALUATED'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze', 'verify', 'run', 'ast'))
    action = parser.parse_args().action
    if action == 'verify': verify(OUT)
    elif action == 'ast': print(json.dumps(ml_ast_proof(), indent=2))
    else: globals()[action]()
