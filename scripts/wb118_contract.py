#!/usr/bin/env python3
"""One frozen seen-event Condor diagnostic with an immutable 24-call budget."""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT, read_public, write_new
from wb100_contract import digest, verify
from wb92_contract import ACTS, EXTERNAL, run as command_run
from wb111_contract import OUT as HISTORICAL
from wb118_sources import files, athena_source

BASE = ROOT/'outputs/mc24_four_station_wb117_bounded_nonlinear_response_v1'
OUT = ROOT/'outputs/mc24_four_station_wb118_split_nonlinear_response_v1'
PREFLIGHT = ROOT/'outputs/mc24_four_station_wb118_compile_preflight_v1'
WORKBOOK = ROOT/'workbook/2026-10-03_118_四站五维更新的四维与动量分离响应前瞻控制.md'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def control():
    old = read_public(BASE/'control.json')
    result = json.loads(json.dumps(old)); result['directions'] = []
    for direction in old['directions']:
        for split in ('four_d_only', 'qop_only'):
            row = json.loads(json.dumps(direction)); row['name'] += '_'+split
            row['parent_direction'] = direction['name']; row['split'] = split
            row['delta'] = direction['delta'][:4]+[0.] if split == 'four_d_only' else [0.]*4+[direction['delta'][4]]
            result['directions'].append(row)
    result['factors'] = [0., .2, 1.]; result['max_official_calls'] = 24
    for row in result['directions']:
        for factor in result['factors']:
            q = result['seed'][4][0]+factor*row['delta'][4]
            check(q < 0 if row['split'] == 'qop_only' and factor == 1. else q > 0, 'split charge sign')
    return result


def compile_check():
    PREFLIGHT.mkdir(exist_ok=False); generated = files()
    for name, content in generated.items():
        p = PREFLIGHT/name; p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('x') as stream:
            stream.write(content)
    flags = HISTORICAL/'isolated_build/WB111Diagnostic/CMakeFiles/WB111Diagnostic.dir/flags.make'
    lines = flags.read_text().splitlines()
    parse = lambda key: shlex.split(next(x.split('=', 1)[1] for x in lines if x.startswith(key+' =')))
    includes = [('-I'+str(PREFLIGHT/'WB118Diagnostic')) if x == '-I'+str(HISTORICAL/'isolated_source/WB111Diagnostic') else x for x in parse('CXX_INCLUDES')]
    defines = [x.replace('WB111Diagnostic', 'WB118Diagnostic') for x in parse('CXX_DEFINES')]
    cmd = ['g++', '-std=c++20', '-fsyntax-only', *defines, *includes, str(PREFLIGHT/'WB118Diagnostic/BoundedResponse.cxx')]
    with (PREFLIGHT/'compile.log').open('x') as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    write_new(PREFLIGHT/'receipt.json', {'command': cmd, 'returncode': r.returncode,
              'scope': 'whole isolated translation unit; no link/event/propagation',
              'source_hashes': {k: digest(PREFLIGHT/k) for k in generated}, 'flags_sha256': digest(flags),
              'compiler': subprocess.check_output(['which', 'g++'], text=True).strip()})
    check(r.returncode == 0, 'compile-only preflight')


def freeze():
    check(subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == '4station', 'branch')
    check(not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).strip(), 'tracked changes')
    hashes = verify(BASE)['hashes'].copy()
    manifest = ROOT/'docs/wb117_bounded_response_result_manifest.json'; prior = read_public(manifest)
    check(prior['execution_contract'] == prior['integrity'] == 'PASS', 'WB117 result')
    for p, h in prior['artifacts'].items():
        check(digest(ROOT/p) == h, 'sealed artifact '+p); hashes[str(ROOT/p)] = h
    pre = read_public(PREFLIGHT/'receipt.json'); check(pre['returncode'] == 0, 'compile preflight')
    generated = files()
    for n, content in generated.items():
        check(hashlib.sha256(content.encode()).hexdigest() == pre['source_hashes'][n], 'compile source identity')
    paths = [Path(__file__), ROOT/'scripts/wb118_sources.py', ROOT/'scripts/audit_wb118_response.py',
             ROOT/'scripts/run_wb118_condor.sh', ROOT/'tests/test_wb118_response.py',
             ROOT/'research/wb118/BoundedResponse.cxx', ROOT/'scripts/setup_environment.sh', manifest,
             BASE/'event/response.json', BASE/'control.json', HISTORICAL/'raw_identity.json',
             ROOT/'scripts/audit_wb117_response.py', ROOT/'research/wb117/BoundedResponse.cxx',
             ROOT/'scripts/wb117_sources.py',
             HISTORICAL/'isolated_build/WB111Diagnostic/CMakeFiles/WB111Diagnostic.dir/flags.make', Path(pre['compiler'])]
    paths += [p for p in PREFLIGHT.rglob('*') if p.is_file()]
    paths += [p for p in (HISTORICAL/'identity_payload').rglob('*') if p.is_file()]
    paths += [EXTERNAL/'Tracking/Acts/FaserActsGeometry/src'/n for n in
              ('FaserActsExtrapolationTool.cxx', 'FaserActsExtrapolationTool.h', 'FaserActsTrackingGeometrySvc.cxx')]
    paths += [EXTERNAL/'Tracking/Acts/FaserActsGeometry/FaserActsGeometry/FASERMagneticFieldWrapper.h',
              ACTS/'include/Acts/EventData/detail/TransformationFreeToBound.hpp',
              ACTS/'include/Acts/Definitions/Units.hpp']
    # Pin the existing actual production chain paths before execution; remaining libraries are observed postrun.
    runtime = read_public(BASE/'event/response.json')
    selected = ('libFaserActsGeometryLib.so', 'libFaserActsGeometry.so', 'libActsCore.so',
                'libMagFieldServices.so', 'libMagFieldConditions.so', 'libMagFieldElements.so',
                'libFaserSCT_GeoModel.so', 'libDipoleGeoModel.so', 'libGeoModelSvc.so')
    pinned = {name: digest(Path(name)) for name in runtime['loaded_libraries'] if Path(name).name in selected}
    check(len(pinned) == len(selected), 'runtime library anchor set')
    paths += [Path(p) for p in pinned]
    for p in paths:
        hashes[str(p)] = digest(p)
    out_fixture = read_public(HISTORICAL/'fixture.json')
    check((out_fixture['index'], out_fixture['ordinal'], out_fixture['actual_run'], out_fixture['actual_event']) == (12, 2268, 100044, 2268), 'allowlist')
    OUT.mkdir(exist_ok=False); shutil.copyfile(WORKBOOK, OUT/'contract_workbook.md')
    write_new(OUT/'fixture.json', out_fixture); write_new(OUT/'control.json', control())
    write_new(OUT/'historical_runtime.json', runtime); write_new(OUT/'pinned_runtime_libraries.json', pinned)
    write_new(OUT/'generation_expectation.json', {'files': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in generated.items()},
              'athena_sha256': hashlib.sha256(athena_source().encode()).hexdigest()})
    for n in ('contract_workbook.md', 'fixture.json', 'control.json', 'historical_runtime.json', 'generation_expectation.json', 'pinned_runtime_libraries.json'):
        hashes[str(OUT/n)] = digest(OUT/n)
    write_new(OUT/'freeze.json', {'schema': 'wb118_bounded_response_freeze_v1', 'hashes': hashes, 'branch': '4station',
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'population': 1, 'max_official_calls': 24, 'new_reconstruction_calls': 0,
              'held_out_access': False, 'truth_access': False, 'qualification': 'NOT_EVALUATED'})


def raw_identity():
    fixture = read_public(OUT/'fixture.json'); raw = Path(fixture['input_xaod']); before = raw.stat()
    h = digest(raw); after = raw.stat(); previous = read_public(HISTORICAL/'raw_identity.json')
    check((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'raw mutation during hashing')
    check(h == previous['sha256'] and fixture['source_stat'] == {'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns}, 'source identity')
    return {'path': str(raw), 'sha256': h, 'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns}


def build():
    source = OUT/'isolated_source'; source.mkdir(exist_ok=False); expected = read_public(OUT/'generation_expectation.json')
    for name, content in files().items():
        path = source/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(content)
        check(digest(path) == expected['files'][name], 'generated source identity')
    write_new(OUT/'generated_source_manifest.json', {k: digest(source/k) for k in expected['files']})
    command_run(['cmake', '-S', str(source), '-B', str(OUT/'isolated_build'), '-DCalypso_DIR='+str(EXTERNAL/'run/cmake')], OUT, OUT/'configure.log')
    command_run(['cmake', '--build', str(OUT/'isolated_build'), '-j', '1'], OUT, OUT/'build.log')
    binary = OUT/'isolated_build/x86_64-el9-gcc13-opt/lib/libWB118Diagnostic.so'
    write_new(OUT/'binary_manifest.json', {'binary': str(binary), 'sha256': digest(binary)})
    return binary


def run():
    verify(OUT); (OUT/'execution_lock').mkdir(exist_ok=False)
    before = raw_identity(); write_new(OUT/'raw_identity_before.json', before)
    binary = build(); shutil.copytree(HISTORICAL/'identity_payload', OUT/'identity_payload')
    copied = {str(p.relative_to(OUT)): digest(p) for p in (OUT/'identity_payload').rglob('*') if p.is_file()}
    for name, h in copied.items():
        check(h == digest(HISTORICAL/name), 'copied alignment payload identity '+name)
    write_new(OUT/'alignment_copy_identity_before.json', copied)
    event = OUT/'event'; event.mkdir(exist_ok=False)
    for n in ('fixture.json', 'control.json'):
        shutil.copyfile(OUT/n, event/n)
    with (event/'athena.py').open('x') as stream:
        stream.write(athena_source())
    check(digest(event/'athena.py') == read_public(OUT/'generation_expectation.json')['athena_sha256'], 'runner source identity')
    platform = binary.parent.parent
    cmd = '\n'.join(['set -eo pipefail', 'export PYTHONPATH='+shlex.quote(str(ROOT))+':"${PYTHONPATH:-}"',
                     'source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso',
                     'source '+shlex.quote(str(platform/'setup.sh')),
                     'export LD_LIBRARY_PATH='+shlex.quote(str(platform/'lib'))+':"$LD_LIBRARY_PATH"',
                     'python '+shlex.quote(str(event/'athena.py'))+' --work-dir '+shlex.quote(str(event))+' --sqlite '+shlex.quote(str(OUT/'identity_payload/tracker_alignment.sqlite'))])
    write_new(event/'command.json', {'script': cmd, 'athena_sha256': digest(event/'athena.py')})
    with (event/'athena.log').open('x') as log:
        r = subprocess.run(['bash', '-c', cmd], cwd=event, stdout=log, stderr=subprocess.STDOUT)
    write_new(OUT/'athena_exit.json', {'exit_code': r.returncode, 'host': os.uname().nodename})
    for name, h in copied.items():
        check(digest(OUT/name) == h, 'alignment payload copy mutation '+name)
    write_new(OUT/'alignment_copy_identity_after.json', copied)
    after = raw_identity(); write_new(OUT/'raw_identity_after.json', after); check(before == after, 'source before/after identity')
    verify(OUT); check(r.returncode == 0, 'Athena execution failure; preserve artifacts')
    runtime = read_public(event/'response.json'); pinned = read_public(OUT/'pinned_runtime_libraries.json')
    for name, h in pinned.items():
        check(name in runtime['loaded_libraries'] and digest(Path(name)) == h, 'pinned runtime library '+name)
    observed = {p: digest(Path(p)) for p in runtime['loaded_libraries'] if any(x in Path(p).name for x in ('Acts', 'Field', 'GeoModel', 'WB118'))}
    write_new(OUT/'runtime_library_observation.json', {'pinned_matches': pinned, 'postrun_observed': observed,
              'all_runtime_pre_pinned': False, 'scope': 'all relevant Acts/Field/GeoModel/diagnostic loaded paths; other Athena libraries not fully pinned'})
    from audit_wb118_response import audit
    write_new(OUT/'summary.json', audit(OUT)); verify(OUT)
    write_new(OUT/'execution_receipt.json', {'exit_code': 0, 'mode': 'single_condor_seen_event_official_bounded_response',
              'official_calls': 24, 'max_events': 1, 'new_reconstruction_calls': 0, 'truth_access': False,
              'held_out_access': False, 'source_before_after_identical': True, 'qualification': 'NOT_EVALUATED'})


def submit():
    verify(OUT)
    setup = 'source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null'
    q = subprocess.check_output(['bash', '-c', setup+' && condor_q -name bigbird24.cern.ch -json'], text=True, timeout=55)
    slots = subprocess.check_output(['bash', '-c', setup+' && condor_status -constraint \'State=="Unclaimed" && Activity=="Idle"\' -af Name'], text=True, timeout=55)
    write_new(OUT/'scheduler_preflight.json', {'queue': json.loads(q) if q.strip() else [], 'idle_slots': len(slots.splitlines()), 'schedd': 'bigbird24.cern.ch'})
    check(bool(slots.strip()), 'no idle slots; no submission')
    sub = (HISTORICAL/'wb111.sub').read_text().replace(str(HISTORICAL), str(OUT)).replace('run_wb111_condor.sh', 'run_wb118_condor.sh')
    with (OUT/'wb118.sub').open('x') as stream:
        stream.write(sub)
    r = subprocess.run(['bash', '-c', setup+' && condor_submit '+shlex.quote(str(OUT/'wb118.sub'))], capture_output=True, text=True, timeout=55)
    write_new(OUT/'submission.json', {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr, 'submit_sha256': digest(OUT/'wb118.sub')})
    print(r.stdout, r.stderr); check(r.returncode == 0, 'submission failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('compile', 'freeze', 'verify', 'run', 'submit'))
    action = parser.parse_args().action
    if action == 'verify':
        verify(OUT)
    elif action == 'compile':
        compile_check()
    else:
        globals()[action]()
