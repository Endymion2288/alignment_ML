#!/usr/bin/env python3
"""Frozen passive complete RK/cache diagnostic; exactly 54 calls."""
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
from wb122_sources import files, athena_source

BASE = ROOT/'outputs/mc24_four_station_wb121_saved_stepping_recovery_v1'
JOINT = ROOT/'outputs/mc24_four_station_wb120_official_qop_stepping_trace_v1'
OUT = ROOT/'outputs/mc24_four_station_wb122_complete_rk_cache_trace_v1'
PREFLIGHT = ROOT/'outputs/mc24_four_station_wb122_compile_preflight_v6'
WORKBOOK = ROOT/'workbook/2026-10-04_122_四站同事件完整RK试探与实际场缓存被动记录诊断.md'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def control():
    old = read_public(JOINT/'control.json')
    check(old['qop_step_per_MeV'] == 1e-8 and old['seed'][4][0] == 1e-5, 'original qop/step')
    result = {k: old[k] for k in ('seed','seed_z_mm','targets','samples','qop_step_per_MeV','output_scales')}
    result.update(max_official_calls=54, calls_per_arm=18, observer_budget_per_call=200000,
                  observer_budget_total=3600000, truth_momentum_used=False, dummy_qop_variance_used=False)
    return result


def freeze():
    check(subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == '4station', 'branch')
    check(not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).strip(), 'tracked changes')
    hashes = verify(BASE)['hashes'].copy()
    manifest = ROOT/'docs/wb121_saved_stepping_result_manifest.json'; prior = read_public(manifest)
    check(prior['recovery_execution'] == prior['integrity'] == 'PASS', 'WB121 sealed result')
    for p, h in prior['artifacts'].items():
        check(digest(ROOT/p) == h, 'sealed artifact '+p); hashes[str(ROOT/p)] = h
    pre = read_public(PREFLIGHT/'receipt.json'); check(pre['returncodes'] == [0,0,0,0] and read_public(PREFLIGHT/'cache_selftest.json')['status']=='PASS', 'compile/cache preflight')
    for env in ('calypso','ml'):
        proof=read_public(ROOT/('outputs/wb122_'+env+'_test_preflight_v4.json'))
        check(proof['pytest_exit']==0 and proof['imports'] is True and proof['analyzer_sha256']==digest(ROOT/'scripts/audit_wb122_response.py') and proof['source_generator_sha256']==digest(ROOT/'scripts/wb122_sources.py'), 'both environment tested source')
        check(proof['generated_source_hashes']==pre['source_hashes'] and proof['generated_athena_compile'] is True, 'tested generated sources')
        hashes[str(ROOT/('outputs/wb122_'+env+'_test_preflight_v4.json'))]=digest(ROOT/('outputs/wb122_'+env+'_test_preflight_v4.json'))
    generated = files()
    for n, content in generated.items():
        check(hashlib.sha256(content.encode()).hexdigest() == pre['source_hashes'][n], 'compile source identity')
    paths = [Path(__file__), ROOT/'scripts/wb122_sources.py', ROOT/'scripts/audit_wb122_response.py',
             ROOT/'scripts/run_wb122_condor.sh', ROOT/'tests/test_wb122_response.py',
             ROOT/'research/wb122/BoundedResponse.cxx', ROOT/'scripts/setup_environment.sh', manifest,
             JOINT/'event/response.json', JOINT/'control.json', HISTORICAL/'raw_identity.json',
             ROOT/'scripts/audit_wb117_response.py', ROOT/'scripts/audit_wb119_response.py',
             ROOT/'research/wb119/BoundedResponse.cxx', ROOT/'research/wb122/Trace.h', ROOT/'research/wb122/CacheSelftest.cxx', ROOT/'scripts/wb122_preflight.py', ROOT/'scripts/wb122_test_env.py', ROOT/'scripts/wb121_recovery.py',
             HISTORICAL/'isolated_build/WB111Diagnostic/CMakeFiles/WB111Diagnostic.dir/flags.make', Path(pre['compiler'])]
    paths += [p for p in PREFLIGHT.rglob('*') if p.is_file()]
    paths += [p for p in (HISTORICAL/'identity_payload').rglob('*') if p.is_file()]
    paths += [EXTERNAL/'Tracking/Acts/FaserActsGeometry/src'/n for n in
              ('FaserActsExtrapolationTool.cxx', 'FaserActsExtrapolationTool.h', 'FaserActsTrackingGeometrySvc.cxx')]
    paths += [EXTERNAL/'Tracking/Acts/FaserActsGeometry/FaserActsGeometry/FASERMagneticFieldWrapper.h',
              ACTS/'include/Acts/EventData/detail/TransformationFreeToBound.hpp',
              ACTS/'include/Acts/Definitions/Units.hpp',
              EXTERNAL/'Tracking/Acts/FaserActsGeometryInterfaces/FaserActsGeometryInterfaces/IFaserActsExtrapolationTool.h']
    paths += [ACTS/'include/Acts'/n for n in ('Propagator/detail/SteppingLogger.hpp','Propagator/ConstrainedStep.hpp',
              'Propagator/EigenStepper.hpp','Propagator/EigenStepper.ipp','Propagator/Propagator.hpp',
              'Propagator/Propagator.ipp','Definitions/Direction.hpp')]
    paths += [EXTERNAL/'MagneticField/MagFieldElements/MagFieldElements'/n for n in
              ('FaserFieldCache.h','BFieldCache.h','BFieldMesh.h','BFieldZone.h','FaserFieldMap.h','BFieldVector.h')]
    paths += [EXTERNAL/'MagneticField/MagFieldElements/src/FaserFieldCache.cxx',
              EXTERNAL/'MagneticField/MagFieldConditions/MagFieldConditions/FaserFieldCacheCondObj.h',
              ACTS/'include/Acts/Propagator/detail/GenericDefaultExtension.hpp',
              ACTS/'include/Acts/Propagator/StandardAborters.hpp']
    # Pin the existing actual production chain paths before execution; remaining libraries are observed postrun.
    runtime = read_public(JOINT/'event/response.json')
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
    write_new(OUT/'step_source_restoration_proof.json', read_public(PREFLIGHT/'step_source_restoration_proof.json'))
    write_new(OUT/'generation_expectation.json', {'files': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in generated.items()},
              'athena_sha256': hashlib.sha256(athena_source().encode()).hexdigest()})
    for n in ('contract_workbook.md', 'fixture.json', 'control.json', 'historical_runtime.json', 'generation_expectation.json', 'pinned_runtime_libraries.json', 'step_source_restoration_proof.json'):
        hashes[str(OUT/n)] = digest(OUT/n)
    write_new(OUT/'freeze.json', {'schema': 'wb122_complete_rk_cache_freeze_v1', 'hashes': hashes, 'branch': '4station',
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'population': 1, 'max_official_calls': 54, 'new_reconstruction_calls': 0,
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
    binary = OUT/'isolated_build/x86_64-el9-gcc13-opt/lib/libWB122Diagnostic.so'
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
    observed = {p: digest(Path(p)) for p in runtime['loaded_libraries'] if any(x in Path(p).name for x in ('Acts', 'Field', 'GeoModel', 'WB122'))}
    write_new(OUT/'runtime_library_observation.json', {'pinned_matches': pinned, 'postrun_observed': observed,
              'all_runtime_pre_pinned': False, 'scope': 'all relevant Acts/Field/GeoModel/diagnostic loaded paths; other Athena libraries not fully pinned'})
    from audit_wb122_response import audit
    write_new(OUT/'summary.json', audit(OUT)); verify(OUT)
    write_new(OUT/'execution_receipt.json', {'exit_code': 0, 'mode': 'single_condor_seen_event_passive_complete_rk_cache_diagnostic',
              'official_calls': 54, 'calls_per_arm':18, 'observer_enabled_calls':18, 'max_events': 1, 'new_reconstruction_calls': 0, 'truth_access': False,
              'held_out_access': False, 'source_before_after_identical': True, 'qualification': 'NOT_EVALUATED'})


def submit():
    verify(OUT)
    setup = 'source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null'
    q = subprocess.check_output(['bash', '-c', setup+' && condor_q -name bigbird24.cern.ch -json'], text=True, timeout=55)
    slots = subprocess.check_output(['bash', '-c', setup+' && condor_status -constraint \'State=="Unclaimed" && Activity=="Idle"\' -af Name'], text=True, timeout=55)
    write_new(OUT/'scheduler_preflight.json', {'queue': json.loads(q) if q.strip() else [], 'idle_slots': len(slots.splitlines()), 'schedd': 'bigbird24.cern.ch'})
    check(bool(slots.strip()), 'no idle slots; no submission')
    sub = (HISTORICAL/'wb111.sub').read_text().replace(str(HISTORICAL), str(OUT)).replace('run_wb111_condor.sh', 'run_wb122_condor.sh')
    with (OUT/'wb122.sub').open('x') as stream:
        stream.write(sub)
    r = subprocess.run(['bash', '-c', setup+' && condor_submit '+shlex.quote(str(OUT/'wb122.sub'))], capture_output=True, text=True, timeout=55)
    write_new(OUT/'submission.json', {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr, 'submit_sha256': digest(OUT/'wb122.sub')})
    print(r.stdout, r.stderr); check(r.returncode == 0, 'submission failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze', 'verify', 'run', 'submit'))
    action = parser.parse_args().action
    if action == 'verify':
        verify(OUT)
    else:
        globals()[action]()
