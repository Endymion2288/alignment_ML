#!/usr/bin/env python3
"""Versioned WB92 freeze, isolated build, bounded Condor execution and checks."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT, digest, read_public, write_new, FORBIDDEN
from alignment.wb92_acts_contract import validate, conditions_comparison
from alignment.physical_common_track_execution import calypso_payload_command, identity_payload

PROTOCOL = ROOT/'configs/research_review/wp92_common_seed_acts_contract.json'
WB91 = ROOT/'outputs/mc24_four_station_wb91_covariance_repair_v7'
EXTERNAL = ROOT.parent/'calypso'
ACTS = Path('/cvmfs/atlas.cern.ch/repo/sw/software/24.0/AthenaExternals/24.0.41/InstallArea/x86_64-el9-gcc13-opt')
SOURCES = [PROTOCOL, ROOT/'research/wb92/CommonSeedAudit.cxx', ROOT/'alignment/wb92_acts_contract.py',
           Path(__file__).resolve(), ROOT/'scripts/wb92_athena.py', ROOT/'scripts/run_wb92_condor.sh',
           ROOT/'scripts/setup_environment.sh', ROOT/'alignment/common_track_geometry.py',
           ROOT/'alignment/wb90_measurement_contract.py', ROOT/'alignment/physical_common_track_execution.py',
           ROOT/'scripts/write_station_alignment_payload.py']
EXTERNAL_SOURCES = [
    'Tracking/Acts/FaserActsGeometry/src/FaserActsExtrapolationTool.cxx',
    'Tracking/Acts/FaserActsGeometry/src/FaserActsExtrapolationTool.h',
    'Tracking/Acts/FaserActsGeometry/src/FaserActsTrackingGeometryTool.cxx',
    'Tracking/Acts/FaserActsGeometry/src/FaserActsAlignmentCondAlg.cxx',
    'Tracking/Acts/FaserActsGeometry/src/FaserActsDetectorElement.cxx',
    'Tracking/Acts/FaserActsGeometry/FaserActsGeometry/FASERMagneticFieldWrapper.h',
    'Tracking/Acts/FaserActsGeometry/python/ActsGeometryConfig.py',
    'DetectorDescription/GeoModel/FaserGeoModel/python/FaserGeoModelConfig.py',
    'MagneticField/MagFieldServices/python/MagFieldServicesConfig.py',
    'MagneticField/MagFieldServices/src/FaserFieldMapCondAlg.cxx',
    'Tracker/TrackerAlignTools/TrackerAlignGenTools/src/TrackerAlignDBTool.cxx',
    'PhysicsAnalysis/NtupleDumper/src/NtupleDumperAlg.cxx',
]
HEADERS = ['include/Acts/EventData/GenericBoundTrackParameters.hpp',
           'include/Acts/EventData/detail/TransformationFreeToBound.hpp',
           'include/Acts/Surfaces/PlaneSurface.hpp', 'include/Acts/Definitions/Units.hpp',
           'lib/cmake/Acts/ActsConfigVersion.cmake']


def run(command, cwd, log):
    with log.open('x') as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'command returned {result.returncode}; {log}')


def verify(out):
    f = read_public(out/'freeze.json')
    if read_public(out/'protocol.json') != read_public(PROTOCOL):
        raise ValueError('protocol changed since freeze')
    for path, expected in f['hashes'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'frozen source/artifact changed: {path}')
    if digest(out/'fixtures.json') != f['fixtures_sha256']:
        raise ValueError('fixtures changed')
    return f


def freeze(out):
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != '4station':
        raise ValueError('4station required')
    inputs = read_public(WB91/'array_inputs.json')
    summary = read_public(WB91/'physical_summary.json')
    if summary['gate'] != 'PASS' or summary['n_events'] != 24:
        raise ValueError('WB91 prerequisite incomplete')
    selection = read_public(WB91/'selection.json')['events']
    paths = SOURCES + [EXTERNAL/p for p in EXTERNAL_SOURCES] + [ACTS/p for p in HEADERS]
    paths += [WB91/'physical_summary.json', WB91/'array_inputs.json', WB91/'selection.json']
    fixtures = []
    for index, row in enumerate(selection):
        if row['role'] != 'development' or any(x in row['input_xaod'].lower() for x in FORBIDDEN):
            raise ValueError('unauthorized input')
        directory = (Path(inputs['pilot_root']) if index == 0 else WB91)/'physical'/f'{index:02d}'
        path = directory/'repair.jsonl'; check = read_public(directory/'validation.json')
        if check['gate'] != 'PASS' or digest(path) != check['repair_sha256']:
            raise ValueError('WB91 input changed')
        references = [r for line in path.read_text().splitlines() if (r := json.loads(line))['state_index'] == 0]
        references.sort(key=lambda r: r['station'])
        if [r['station'] for r in references] != [0, 1, 2, 3]:
            raise ValueError('reference multiplicity unsupported; no replacement')
        if len({(r['run'], r['event']) for r in references}) != 1 or not all(r['input_event_header_verified'] for r in references):
            raise ValueError('actual event identity invalid')
        for r in references:
            r['q_over_p_per_MeV'] = r['native_parameters'][4][0]
            r['clusters'] = sorted(r['clusters'])
        stat = Path(row['input_xaod']).stat()
        fixtures.append({'index': index, 'input_xaod': row['input_xaod'], 'ordinal': int(row['xaod_entry_index']),
                         'actual_run': references[0]['run'], 'actual_event': references[0]['event'],
                         'row': row, 'references': references, 'source_stat': {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns},
                         'wb91_repair_path': str(path), 'wb91_repair_sha256': digest(path)})
        paths.extend([path, directory/'validation.json'])
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'protocol.json', read_public(PROTOCOL)); write_new(out/'fixtures.json', fixtures)
    write_new(out/'freeze.json', {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                                 'branch': '4station', 'hashes': {str(p): digest(p) for p in paths},
                                 'fixtures_sha256': digest(out/'fixtures.json'), 'population': 24,
                                 'held_out_access': False, 'qualification': 'NOT_EVALUATED'})


def prepare_build(out):
    verify(out)
    source = out/'isolated_source'; source.mkdir(exist_ok=False)
    package = source/'WB92Diagnostic'; package.mkdir()
    relocation = '''
set(bad_external "${Calypso_INSTALL_DIR}/../../../../AthenaExternals/${AthenaExternals_VERSION}/InstallArea/${AthenaExternals_PLATFORM}")
foreach(name IN LISTS Calypso_TARGET_NAMES)
  if(TARGET Calypso::${name})
    foreach(property INTERFACE_INCLUDE_DIRECTORIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES INTERFACE_LINK_LIBRARIES)
      get_target_property(value Calypso::${name} ${property})
      if(value)
        string(REPLACE "${bad_external}" "${AthenaExternals_INSTALL_DIR}" fixed "${value}")
        set_target_properties(Calypso::${name} PROPERTIES ${property} "${fixed}")
      endif()
    endforeach()
  endif()
endforeach()
'''
    files = {'CMakeLists.txt': 'cmake_minimum_required(VERSION 3.11)\nproject(WB92 VERSION 1.0.0 LANGUAGES C CXX)\nfind_package(Calypso REQUIRED)\natlas_project(USE Calypso ${Calypso_VERSION})\n',
             'WB92Diagnostic/CMakeLists.txt': 'atlas_subdir(WB92Diagnostic)\n'+relocation+
             'find_package(nlohmann_json REQUIRED)\natlas_add_component(WB92Diagnostic CommonSeedAudit.cxx LINK_LIBRARIES AthenaBaseComps StoreGateLib xAODFaserEventInfo TrackerIdentifier FaserActsGeometryLib FaserActsGeometryInterfacesLib ActsCore PRIVATE_LINK_LIBRARIES nlohmann_json::nlohmann_json)\n',
             'WB92Diagnostic/CommonSeedAudit.cxx': (ROOT/'research/wb92/CommonSeedAudit.cxx').read_text()}
    for name, content in files.items():
        with (source/name).open('x') as stream:
            stream.write(content)
    write_new(out/'generated_source_manifest.json', {name: digest(source/name) for name in files})
    run(['cmake', '-S', str(source), '-B', str(out/'isolated_build'), '-DCalypso_DIR='+str(EXTERNAL/'run/cmake')], out, out/'configure.log')
    run(['cmake', '--build', str(out/'isolated_build'), '-j', '1'], out, out/'build.log')
    binary = out/'isolated_build/x86_64-el9-gcc13-opt/lib/libWB92Diagnostic.so'
    if not binary.is_file():
        raise ValueError('isolated binary absent')
    write_new(out/'binary_manifest.json', {'binary': str(binary), 'sha256': digest(binary),
                                          'generated_source_manifest_sha256': digest(out/'generated_source_manifest.json')})


def negative_controls(result, fixture):
    controls = []
    for name in ('wrong_source', 'wrong_header', 'moving_y', 'stale_geometry', 'fixed_prediction', 'wrong_derivative_sign'):
        r, f = copy.deepcopy(result), copy.deepcopy(fixture)
        if name == 'wrong_source':
            r['input_xaod'] += '.wrong'
        elif name == 'wrong_header':
            r['actual_event'] += 1
        elif name == 'moving_y':
            r['targets'][1]['y'][0][0] += 1.
        elif name == 'stale_geometry':
            f['perturbed_station'] = 1; f['twist'][0] = .01
        elif name == 'fixed_prediction':
            for t in r['targets']:
                for s in t['xi']:
                    for key in s:
                        s[key] = t['h']
        else:
            for t in r['targets']:
                for s in t['xi']:
                    s['plus'], s['minus'] = s['minus'], s['plus']
                    s['half_plus'], s['half_minus'] = s['half_minus'], s['half_plus']
        try:
            rejected = validate(r, f)['gate'] == 'FAIL'
            error = None
        except ValueError as e:
            rejected = True; error = str(e)
        controls.append({'name': name, 'detected': rejected, 'error': error})
    return {'gate': 'PASS' if all(c['detected'] for c in controls) else 'FAIL', 'controls': controls}


def event(out, build_root, index, axis=None, multiplier=None):
    verify(out); verify(build_root)
    if index not in range(24):
        raise ValueError('unfrozen development index')
    binary = read_public(build_root/'binary_manifest.json')
    if digest(build_root/'generated_source_manifest.json') != binary['generated_source_manifest_sha256']:
        raise ValueError('generated source manifest changed')
    for path, expected in read_public(build_root/'generated_source_manifest.json').items():
        if digest(build_root/'isolated_source'/path) != expected:
            raise ValueError('generated build source changed')
    if digest(Path(binary['binary'])) != binary['sha256']:
        raise ValueError('binary changed')
    name = f'{index:02d}' if axis is None else f'geometry_axis{axis}_mult{multiplier:+g}'
    work = out/'events'/name; work.mkdir(parents=True, exist_ok=False)
    fixture = copy.deepcopy(read_public(out/'fixtures.json')[index]); fixture['protocol'] = read_public(out/'protocol.json')
    fixture['variant'] = 'baseline' if axis is None else name
    fixture['perturbed_station'] = -1 if axis is None else fixture['protocol']['sqlite_pilot_station']
    fixture['twist'] = [0.]*6
    if axis is None:
        payload = out/'identity_payload'
        for path, expected in read_public(out/'identity_payload_manifest.json')['hashes'].items():
            if digest(payload/path) != expected:
                raise ValueError('identity payload changed')
    else:
        if index != 0 or axis not in fixture['protocol']['sqlite_pilot_axes'] or multiplier not in fixture['protocol']['sqlite_multipliers']:
            raise ValueError('unfrozen geometry job')
        pilot = read_public(out/'events/00/validation.json')
        if pilot['gate'] != 'PASS' or read_public(out/'events/00/negative_controls.json')['gate'] != 'PASS':
            raise ValueError('pilot prerequisite failed')
        actual = read_public(out/'events/00/acts.json')
        for ref, target in zip(fixture['references'], actual['targets']):
            ref['baseline_sensors'] = target['sensors']
        fixture['twist'][axis] = fixture['protocol']['geometry_steps'][axis]*multiplier
        payload = work/'payload'
        values = identity_payload(); values[fixture['perturbed_station']] = tuple(fixture['twist'])
        command = 'unset Calypso_SET_UP Calypso_EXTONLY_SET_UP Calypso_RELONLY_SET_UP WB92_SET_UP\n'+calypso_payload_command(output_dir=payload, payload=values)
        write_new(work/'payload_command.json', {'script': command})
        run(['bash', '-c', command], work, work/'payload.log')
    stat = Path(fixture['input_xaod']).stat()
    if fixture['source_stat'] != {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
        raise ValueError('ROOT source stat changed')
    write_new(work/'fixture.json', fixture)
    platform_dir = Path(binary['binary']).parent.parent
    command = '\n'.join(['source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso',
                         'source '+shlex.quote(str(platform_dir/'setup.sh')),
                         'export LD_LIBRARY_PATH='+shlex.quote(str(platform_dir/'lib'))+':"$LD_LIBRARY_PATH"',
                         'python '+shlex.quote(str(ROOT/'scripts/wb92_athena.py'))+' --work-dir '+shlex.quote(str(work))+
                         ' --sqlite '+shlex.quote(str(payload/'tracker_alignment.sqlite'))])
    write_new(work/'command.json', {'script': command, 'binary_sha256': binary['sha256'],
                                  'condor_job': os.environ.get('_CONDOR_JOB_AD')})
    run(['bash', '-c', command], work, work/'athena.log')
    result = read_public(work/'acts.json')
    summary = validate(result, fixture)
    text = (work/'athena.log').read_text(errors='replace')
    if str(payload/'tracker_alignment.sqlite') not in text or 'Reading folder /Tracker/Align from sqlite' not in text or 'Using FASER magnetic field service' not in text:
        raise ValueError('official sqlite/field path not demonstrated')
    field_maps = re.findall(r'Initialized the field map from\s+([^\n]+)', text)
    if not field_maps:
        raise ValueError('actual conditions field map absent')
    field_paths = [Path(s.strip().strip('"')) for s in field_maps]
    library_paths = [Path(s) for s in result['loaded_libraries'] if any(t in s for t in ('WB92Diagnostic', 'FaserActs', 'ActsCore', 'MagField', 'TrackerAlign'))]
    if str(Path(binary['binary'])) not in [str(p) for p in library_paths]:
        raise ValueError('isolated binary not loaded')
    summary.update({'acts_sha256': digest(work/'acts.json'), 'fixture_sha256': digest(work/'fixture.json'),
                    'athena_log_sha256': digest(work/'athena.log'), 'field_map_hashes': {str(p): digest(p) for p in field_paths},
                    'loaded_scientific_libraries': {str(p): digest(p) for p in library_paths},
                    'source_identity': {k: fixture[k] for k in ('input_xaod', 'ordinal', 'actual_run', 'actual_event')}})
    write_new(work/'validation.json', summary)
    if axis is None:
        controls = negative_controls(result, fixture); write_new(work/'negative_controls.json', controls)
        if controls['gate'] != 'PASS':
            raise RuntimeError('negative-control gate FAIL')
    if summary['gate'] != 'PASS':
        raise RuntimeError('scientific contract FAIL; expansion stopped')


def pilot(out, reuse=None):
    verify(out)
    write_new(out/'environment.json', {'host': platform.node(), 'environment': {k: os.environ.get(k) for k in
              ('CMAKE_PREFIX_PATH', 'LD_LIBRARY_PATH', 'AtlasVersion', 'AtlasProject', 'CMTCONFIG')}})
    build_root = reuse or out
    if reuse is None:
        prepare_build(out)
    else:
        verify(reuse)
        for path in SOURCES:
            if read_public(out/'freeze.json')['hashes'][str(path)] != read_public(reuse/'freeze.json')['hashes'][str(path)]:
                raise ValueError('reuse source differs')
        write_new(out/'binary_manifest.json', read_public(reuse/'binary_manifest.json'))
    original = Path(read_public(WB91/'array_inputs.json')['payload_source'])
    expected = read_public(WB91/'array_inputs.json')['payload_hashes']
    for name, h in expected.items():
        if digest(original/name) != h:
            raise ValueError('WB91 payload changed')
    shutil.copytree(original, out/'identity_payload')
    write_new(out/'identity_payload_manifest.json', {'original': str(original), 'hashes': expected,
              'catalog_PFN_remains_immutable_WB91': True})
    event(out, build_root, 0)


def submit(out, action, build_root=None):
    verify(out)
    if action == 'pilot':
        jobs = [('pilot', 0, -1, 0.)]
    else:
        if read_public(out/'events/00/validation.json')['gate'] != 'PASS' or read_public(out/'events/00/negative_controls.json')['gate'] != 'PASS':
            raise ValueError('pilot failed; no expansion')
        p = read_public(out/'protocol.json')
        jobs = [('event', i, -1, 0.) for i in range(1,24)] + [('event', 0, axis, mult) for axis in p['sqlite_pilot_axes'] for mult in p['sqlite_multipliers']]
    worker = ROOT/'scripts/run_wb92_condor.sh'
    lines = ['universe = vanilla', f'executable = {worker}',
             f'arguments = {ROOT} {out} $(action) $(index) $(axis) $(multiplier) {build_root or out}',
             f'output = {out}/'+action+'.$(ClusterId).$(ProcId).out', f'error = {out}/'+action+'.$(ClusterId).$(ProcId).err',
             f'log = {out}/'+action+'.$(ClusterId).log', 'request_cpus = 1', 'request_memory = 8000',
             'request_disk = 8000000', 'requirements = (Arch == "X86_64")', '+JobFlavour = "workday"',
             'getenv = False', 'should_transfer_files = NO', 'queue action,index,axis,multiplier from (']
    lines += [' '.join(map(str, j)) for j in jobs] + [')', '']
    sub = out/(action+'.sub')
    with sub.open('x') as stream:
        stream.write('\n'.join(lines))
    command = 'source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(sub))
    r = subprocess.run(['bash', '-c', command], text=True, capture_output=True)
    write_new(out/(action+'_submission.json'), {'command': command, 'stdout': r.stdout, 'stderr': r.stderr,
                                              'returncode': r.returncode, 'submit_sha256': digest(sub)})
    print(r.stdout); print(r.stderr, file=sys.stderr)
    if r.returncode:
        raise RuntimeError('Condor submit failed')


def aggregate(out):
    verify(out); results = []; variants = {}; missing = []; failed = []
    p = read_public(out/'protocol.json')
    names = [f'{i:02d}' for i in range(24)] + [f'geometry_axis{a}_mult{m:+g}' for a in p['sqlite_pilot_axes'] for m in p['sqlite_multipliers']]
    for name in names:
        work = out/'events'/name
        if not (work/'validation.json').is_file():
            missing.append(name); continue
        summary = read_public(work/'validation.json')
        if digest(work/'acts.json') != summary['acts_sha256'] or digest(work/'fixture.json') != summary['fixture_sha256'] or digest(work/'athena.log') != summary['athena_log_sha256']:
            raise ValueError('result/fixture/log modified')
        for path, h in {**summary['field_map_hashes'], **summary['loaded_scientific_libraries']}.items():
            if digest(Path(path)) != h:
                raise ValueError('runtime identity changed')
        if summary['gate'] != 'PASS':
            failed.append(name)
        result = read_public(work/'acts.json')
        if name.startswith('geometry'):
            fixture = read_public(work/'fixture.json')
            axis = next(i for i,v in enumerate(fixture['twist']) if v)
            variants[(axis, fixture['twist'][axis]/p['geometry_steps'][axis])] = result
        else:
            if read_public(work/'negative_controls.json')['gate'] != 'PASS':
                failed.append(name+'_negative_controls')
            results.append({'name': name, 'summary': summary})
    geometry = conditions_comparison(read_public(out/'events/00/acts.json'), variants,
                                    read_public(out/'events/00/fixture.json')) if len(variants) == 12 else {'gate': 'INCOMPLETE'}
    identity = [tuple(r['summary']['source_identity'][k] for k in ('input_xaod', 'ordinal', 'actual_run', 'actual_event')) for r in results]
    if len(set(identity)) != len(identity):
        raise ValueError('duplicate source-aware identity')
    gate = 'FAIL' if failed or geometry['gate'] == 'FAIL' else 'INCOMPLETE' if missing else 'PASS'
    write_new(out/'summary.json', {'gate': gate, 'baseline_count': len(results), 'geometry_count': len(variants),
              'missing': missing, 'failed': failed, 'baseline_results': results, 'geometry_comparison': geometry,
              'qualification': 'NOT_EVALUATED', 'association': 'NOT_EVALUATED', 'covariance': 'NOT_EVALUATED',
              'all_mode_conditions_Htheta': 'UNKNOWN'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'pilot', 'event', 'submit-pilot', 'submit-array', 'aggregate'])
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--build-root', type=Path)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--axis', type=int, default=-1)
    parser.add_argument('--multiplier', type=float, default=0.)
    args = parser.parse_args(); out = args.output_root.resolve()
    if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb92_'):
        parser.error('new WB92 output required')
    build_root = args.build_root.resolve() if args.build_root else None
    if build_root and (not build_root.is_relative_to(ROOT/'outputs') or not build_root.name.startswith('mc24_four_station_wb92_')):
        parser.error('WB92 isolated build required')
    try:
        if args.action == 'freeze': freeze(out)
        elif args.action == 'pilot': pilot(out, build_root if build_root != out else None)
        elif args.action == 'event': event(out, build_root or out, args.index, None if args.axis == -1 else args.axis, args.multiplier)
        elif args.action.startswith('submit-'): submit(out, args.action.split('-')[1], build_root)
        else: aggregate(out)
    except Exception as e:
        if args.action in ('pilot', 'event'):
            write_new(out/(f'error_{args.action}_{args.index}_{args.axis}_{args.multiplier}.json'),
                      {'error': repr(e), 'qualification': 'NOT_EVALUATED'})
        raise
