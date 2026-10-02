#!/usr/bin/env python3
"""WB102 frozen same-pilot Jacobian scale diagnostic."""
from __future__ import annotations
import argparse, copy, json, os, re, shlex, shutil, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT, read_public, write_new, digest, FORBIDDEN
from alignment.wb102_jacobian_scale import analyze, physical_identity, reference_identity
from wb101_contract import artifact_hashes
from wb101_contract import build_controls
from wb101_sources import files as wb101_files
from wb102_sources import files as wb102_files
from wb95_contract import generated_source as wb95_generated_source
from wb92_contract import ACTS, EXTERNAL, run as command_run

PROTOCOL = ROOT/'configs/research_review/wp102_jacobian_scale_contract.json'
P = read_public(PROTOCOL)
WB101 = ROOT/'outputs/mc24_four_station_wb101_direction_acceptance_v2'
WB95 = ROOT/'outputs/mc24_four_station_wb95_field_precision_v1'
OUT = ROOT/'outputs/mc24_four_station_wb102_jacobian_scale_v1'
SOURCES = [PROTOCOL, ROOT/'scripts/wb102_contract.py', ROOT/'scripts/wb102_sources.py', ROOT/'scripts/wb102_athena.py',
           ROOT/'scripts/run_wb102_condor.sh', ROOT/'alignment/wb102_jacobian_scale.py', ROOT/'scripts/setup_environment.sh']


def verify(out):
    f = read_public(out/'freeze.json')
    for path, h in f['hashes'].items():
        if digest(Path(path)) != h:
            raise ValueError('frozen identity changed: '+path)
    if read_public(out/'protocol.json') != P:
        raise ValueError('protocol changed')
    return f


def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'], cwd=ROOT, text=True).strip() != '4station':
        raise ValueError('4station required')
    parent = read_public(WB101/'freeze.json')
    if read_public(WB101/'summary.json')['hypothesis'] != 'NOT_SUPPORTED':
        raise ValueError('WB101 state changed')
    if read_public(WB101/'worker_exit.json')['exit_code'] != 0:
        raise ValueError('WB101 execution incomplete')
    hashes = {}
    for path, h in parent['hashes'].items():
        if digest(Path(path)) != h: raise ValueError('WB101 dependency changed')
        hashes[path] = h
    for name, directory in [('wb101_direction_acceptance', WB101), ('wb95_field_precision', WB95)]:
        index = read_public(ROOT/f'docs/{name}_result_manifest.json')
        if digest(directory/'result_integrity.json') != index['result_integrity_sha256']:
            raise ValueError('historical result index changed')
        for path, h in artifact_hashes(index).items():
            if digest(ROOT/path) != h: raise ValueError('historical artifact changed: '+path)
            hashes[str(ROOT/path)] = h
    for path in (WB101/'event/acts.json', WB101/'event/acts.json.trials.ndjson', WB101/'event/acts.json.traces.ndjson',
                 WB101/'event/fixture.json', WB101/'summary.json', WB101/'freeze.json', WB95/'event/acts.json', WB95/'fixture.json',
                 WB95/'summary.json', WB95/'freeze.json'):
        hashes[str(path)] = digest(path)
    f = copy.deepcopy(read_public(WB101/'fixture.json'))
    if any(x in f['input_xaod'].lower() for x in FORBIDDEN): raise ValueError('forbidden population')
    if (f['ordinal'], f['actual_run'], f['actual_event']) != (2270, 100043, 2270): raise ValueError('pilot identity')
    f['wb102_protocol'] = P
    f['wb95_protocol']['modes'] = P['reference_modes']
    f['wb95_protocol']['mesh_rk4_dz_mm'] = P['reference_dz_mm']
    f['wb101_historical_summary'] = str(WB101/'event/acts.json')
    f['wb95_reference_source'] = str(WB95/'event/acts.json')
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'protocol.json', P)
    write_new(out/'fixture_base.json', f)
    for lam in P['lambdas']:
        lf = copy.deepcopy(f); lf['wb102_lambda'] = lam
        write_new(out/f'fixture_lambda_{lam:g}.json', lf)
    source_paths = SOURCES + [ROOT/'docs/wb101_direction_acceptance_result_manifest.json',
                              ROOT/'docs/wb95_field_precision_result_manifest.json',
                              ROOT/'scripts/audit_wb102_results.py', ROOT/'scripts/finalize_wb102_results.py',
                              ROOT/'tests/test_wb102_jacobian_scale.py', ROOT/'scripts/audit_wb101_trials.py',
                              ROOT/'scripts/wb101_contract.py', ROOT/'scripts/wb101_sources.py', ROOT/'scripts/wb95_contract.py',
                              ROOT/'alignment/wb101_direction_acceptance.py', ROOT/'alignment/wb95_field_precision.py',
                              ROOT/'alignment/wb94_field_boundary.py', ROOT/'alignment/wb96_acts_tolerance.py',
                              ROOT/'research/wb101/Acceptance.h', ROOT/'research/wb101/NodeModel.h', ROOT/'research/wb100/Envelope.h',
                              ROOT/'research/wb99/DirectionDoubling.h', ROOT/'research/wb96/ObservedPropagation.h',
                              ROOT/'research/wb94/ReferenceRK4.h', ROOT/'research/wb95/PrecisionControl.h',
                              ROOT/'research/wb95/CompensatedRK4.h', ROOT/'research/wb95/AuditBody.inc']
    source_paths += [x for x in (WB101/'isolated_source').rglob('*') if x.is_file()]
    source_paths += [WB101/'binary_manifest.json', WB101/'generated_source_manifest.json']
    source_paths += [x for x in (WB95/'isolated_source').rglob('*') if x.is_file()]
    source_paths += [WB95/'binary_manifest.json', WB95/'generated_source_manifest.json']
    source_paths += [Path(x) for x in read_public(WB101/'summary.json')['loaded_scientific_libraries']]
    source_paths += [Path(x) for x in read_public(WB95/'summary.json')['loaded_scientific_libraries']]
    for path in source_paths:
        if not Path(path).is_file(): raise ValueError('missing source '+str(path))
        hashes[str(path)] = digest(path)
    for path in out.glob('*.json'):
        hashes[str(path)] = digest(path)
    write_new(out/'freeze.json', {'schema':'wb102_freeze_v1', 'commit':subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'branch':'4station', 'hashes':hashes, 'population':1, 'held_out_access':False,
        'qualification':'NOT_EVALUATED', 'production_backend_change':False, 'lambdas':P['lambdas']})


def build(out):
    source = out/'isolated_source'; source.mkdir(exist_ok=False)
    for name, content in wb102_files().items():
        path = source/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream: stream.write(content)
    write_new(out/'generated_source_manifest.json', {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()})
    command_run(['cmake','-S',str(source),'-B',str(out/'isolated_build'),'-DCalypso_DIR='+str(EXTERNAL/'run/cmake')], out, out/'configure.log')
    command_run(['cmake','--build',str(out/'isolated_build'),'-j','1'], out, out/'build.log')
    binaries = list((out/'isolated_build').rglob('libWB102Diagnostic.so'))
    if len(binaries) != 1: raise ValueError('WB102 binary missing')
    write_new(out/'binary_manifest.json', {'binary':str(binaries[0]), 'sha256':digest(binaries[0]),
        'generated_manifest_sha256':digest(out/'generated_source_manifest.json')})


def run_one(out, lam):
    work = out/'event'/f'lambda_{lam:g}'; work.mkdir(parents=True, exist_ok=False)
    fixture = out/f'fixture_lambda_{lam:g}.json'; shutil.copyfile(fixture, work/'fixture.json')
    platform = out/'isolated_build/x86_64-el9-gcc13-opt'
    script = '\n'.join(['source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso',
        'source '+shlex.quote(str(platform/'setup.sh')),
        'export LD_LIBRARY_PATH='+shlex.quote(str(platform/'lib'))+':"$LD_LIBRARY_PATH"',
        'python '+shlex.quote(str(ROOT/'scripts/wb102_athena.py'))+' --work-dir '+shlex.quote(str(work))+
        ' --sqlite '+shlex.quote(str(out/'identity_payload/tracker_alignment.sqlite'))])
    write_new(work/'command.json', {'script':script, 'lambda':lam})
    command_run(['bash','-c',script], work, work/'athena.log')


def run(out):
    verify(out)
    build_controls(out)
    build(out)
    shutil.copytree(WB95/'identity_payload', out/'identity_payload')
    write_new(out/'environment.json', {'host':os.uname().nodename, 'python':sys.version, 'condor_ad':os.environ.get('_CONDOR_JOB_AD'),
                                      'protocol':P, 'backend':'scripts/setup_environment.sh calypso'})
    for lam in P['lambdas']:
        run_one(out, lam)
        if lam == 1:
            d = out/'event/lambda_1'; f = read_public(d/'fixture.json')
            physical_identity(read_public(d/'acts.json'), f, read_public(WB101/'event/acts.json'), P, lam)
            reference_identity(read_public(d/'reference.json'), f, read_public(WB95/'event/acts.json'), P, lam)
            write_new(out/'lambda_1_preflight.json', {'gate':'PASS', 'complete_physical_and_reference_reproduction':True})
    write_new(out/'worker_manifest.json', {'lambdas':P['lambdas'], 'calls_expected':P['expected_physical_calls'],
                                           'reference_samples_expected':P['expected_reference_samples']})
    aggregate(out)


def aggregate(out):
    verify(out)
    raw = {}; refs = {}; fixtures = {}
    for lam in P['lambdas']:
        d = out/'event'/f'lambda_{lam:g}'
        raw[lam] = read_public(d/'acts.json'); refs[lam] = read_public(d/'reference.json')
        fixtures[lam] = read_public(d/'fixture.json')
        if digest(d/'fixture.json') != digest(out/f'fixture_lambda_{lam:g}.json'):
            raise ValueError('worker fixture changed')
        if not (d/'acts.json.trials.ndjson').exists() or not (d/'acts.json.traces.ndjson').exists():
            raise ValueError('trial streams missing')
    historic = read_public(WB101/'event/acts.json'); oldref = read_public(WB95/'event/acts.json')
    summary = analyze(raw, refs, fixtures, historic, oldref, P)
    binary = read_public(out/'binary_manifest.json')
    if digest(binary['binary']) != binary['sha256'] or digest(out/'generated_source_manifest.json') != binary['generated_manifest_sha256']:
        raise ValueError('binary or source manifest changed')
    for name, h in read_public(out/'generated_source_manifest.json').items():
        if digest(out/'isolated_source'/name) != h: raise ValueError('generated source changed')
    from audit_wb102_results import audit_trial_stream, scalar_audit
    trial_audits = []
    for lam in P['lambdas']:
        d = out/'event'/f'lambda_{lam:g}'
        if digest(d/'reference.json.nodes_le_i16.bin') != fixtures[lam]['wb95_protocol']['expected_node_interleaved_le_i16_sha256']:
            raise ValueError('reference node payload changed')
        log = (d/'athena.log').read_text(errors='replace')
        if str(out/'identity_payload/tracker_alignment.sqlite') not in log or 'Using FASER magnetic field service' not in log:
            raise ValueError('physical conditions path')
        maps = [Path(x.strip()) for x in re.findall(r'Initialized the field map from\s+([^\n]+)', log)]
        expected_maps = read_public(WB95/'summary.json')['field_map_hashes']
        if not maps or any(str(x) not in expected_maps or digest(x) != expected_maps[str(x)] for x in maps):
            raise ValueError('field map identity')
        expected_libs = read_public(WB101/'summary.json')['loaded_scientific_libraries']
        for r in (raw[lam], refs[lam]):
            if binary['binary'] not in r['loaded_libraries']: raise ValueError('isolated physical library missing')
            for x in r['loaded_libraries']:
                if x in expected_libs and digest(x) != expected_libs[x]: raise ValueError('official library changed')
        trial_audits.append({'lambda':lam, **audit_trial_stream(out, lam, raw[lam])})
    write_new(out/'trial_integrity_audit.json', {'gate':'PASS', 'lambdas':trial_audits})
    write_new(out/'scalar_summary_audit.json', scalar_audit(summary, raw, refs, P))
    summary['binary'] = read_public(out/'binary_manifest.json')
    summary['generated_source_sha256'] = digest(out/'generated_source_manifest.json')
    write_new(out/'summary.json', summary)
    return summary


def submit(out):
    verify(out)
    sub = out/'wb102.sub'
    with sub.open('x') as stream: stream.write('\n'.join(['universe = vanilla', f'executable = {ROOT}/scripts/run_wb102_condor.sh', f'arguments = {ROOT} {out}',
        f'output = {out}/condor.$(ClusterId).out', f'error = {out}/condor.$(ClusterId).err', f'log = {out}/condor.$(ClusterId).log',
        'request_cpus = 3', 'request_memory = 12000', 'request_disk = 40000000', 'requirements = (Arch == "X86_64")',
        '+JobFlavour = "tomorrow"', 'getenv = False', 'should_transfer_files = NO', 'on_exit_remove = True',
        'periodic_remove = (NumJobStarts > 1)', 'queue 1', '']))
    cmd = 'source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(sub))
    result = subprocess.run(['bash','-c',cmd], capture_output=True, text=True)
    write_new(out/'submission.json', {'command':cmd, 'returncode':result.returncode, 'stdout':result.stdout, 'stderr':result.stderr})
    print(result.stdout); print(result.stderr, file=sys.stderr)
    if result.returncode: raise RuntimeError('submission failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze','build','build_smoke','run','aggregate','submit','verify')); parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(); out = args.output_root.resolve()
    if args.action == 'build_smoke':
        if out.parent != ROOT/'outputs' or not out.name.startswith('mc24_four_station_wb102_build_smoke_v'):
            raise ValueError('exclusive build smoke output')
        out.mkdir(exist_ok=False); build_controls(out); build(out)
    elif out != OUT: raise ValueError('exclusive WB102 output')
    elif args.action == 'verify': verify(out)
    else: globals()[args.action](out)
