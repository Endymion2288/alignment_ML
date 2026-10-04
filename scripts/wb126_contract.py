#!/usr/bin/env python3
"""Exclusive WB126 saved-only freeze / audit / seal. No event or ACTS imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from wb126_information import audit_cell, hypothesis, require

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'outputs/mc24_four_station_wb125_physical_seed_v1'
OUT = ROOT/'outputs/mc24_four_station_wb126_tracklet_information_v1'
PROTOCOL = ROOT/'configs/research_review/wp126_tracklet_information.json'
WORKBOOK = ROOT/'workbook/2026-10-04_126_四站保存tracklet信息与弱方向审计.md'
SOURCES = [Path(__file__), ROOT/'scripts/wb126_information.py', ROOT/'tests/test_wb126_information.py', PROTOCOL]
SOURCES.append(ROOT/'scripts/wb126_saved_check.py')
ORIGINAL = ROOT.parent/'calypso/Tracker/TrackerRecAlgs/TrackerSegmentFit/src'
HISTORICAL = ROOT/'outputs/mc24_four_station_wb91_covariance_repair_v5'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write('\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def fragment(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))]


def freeze():
    require(git('branch', '--show-current') == '4station', 'branch')
    require(not git('diff', 'HEAD', '--name-only'), 'tracked sources must be committed')
    protocol = read(PROTOCOL)
    parent_manifest_path = ROOT/'docs/wb125_physical_seed_result_manifest.json'
    parent = read(parent_manifest_path)
    paths = SOURCES + [parent_manifest_path, ROOT/'workbook/2026-10-04_125_四站物理共同轨迹种子六事件对照.md',
                       ROOT/'docs/wb91_covariance_contract_result_manifest.json',
                       HISTORICAL/'extension_manifest.json', HISTORICAL/'freeze.json',
                       ROOT/'scripts/prepare_wb91_calypso_extension.py',
                       ROOT/'research/wb91/CovarianceContract.h', ROOT/'research/wb91/Audit.h',
                       ROOT/'outputs/wb126_test_preflight_default.json', ROOT/'outputs/wb126_test_preflight_calypso.json',
                       ROOT/'outputs/wb126_test_preflight_ml.json']
    # The complete WB125 seal is checked, but only the six authorized saved events
    # are analyzed. This seal contains no source xAOD or held-out event bytes.
    hashes = {}
    for section in ('sources', 'artifacts'):
        for name, h in parent[section].items():
            p = ROOT/name
            require(digest(p) == h, 'parent sealed identity: '+name)
            hashes[str(p)] = h
    extension = read(HISTORICAL/'extension_manifest.json')
    old_freeze = read(HISTORICAL/'freeze.json')
    binary = read(HISTORICAL/'binary_manifest.json')
    require(digest(binary['binary']) == binary['sha256'], 'historical isolated binary')
    require(binary['source_manifest_sha256'] == digest(HISTORICAL/'extension_manifest.json'), 'binary/source receipt')
    paths.extend([HISTORICAL/'binary_manifest.json', Path(binary['binary'])])
    for name in ('SegmentFitAlg.cxx', 'SegmentFitAlg.h'):
        p = ORIGINAL/name
        require(digest(p) == old_freeze['source_hashes'][str(p)], 'historical original fitter')
        paths.append(p)
    generated = HISTORICAL/'isolated_source'
    for name, h in extension['generated_source_hashes'].items():
        p = generated/name
        require(digest(p) == h, 'historical generated source')
        paths.append(p)
    current_h = (ORIGINAL/'SegmentFitAlg.h').read_text()
    historical_h = (generated/'WB91SegmentFit/src/WB91SegmentFitAlg.h').read_text()
    current_c = (ORIGINAL/'SegmentFitAlg.cxx').read_text()
    historical_c = (generated/'WB91SegmentFit/src/WB91SegmentFitAlg.cxx').read_text()
    require(fragment(current_h, 'double sinA = clusters[n]->sinAlpha;', 'if (fitChi2 != 0 || hasCovariance)') ==
            fragment(historical_h, 'double sinA = clusters[n]->sinAlpha;', 'if (fitChi2 != 0 || hasCovariance)'), 'normal sums implementation')
    require(fragment(current_c, 'Eigen::Matrix< double, 5, 5 > s;', 'return;') ==
            fragment(historical_c, 'Eigen::Matrix< double, 5, 5 > s;', 'return;'), 'unregularized fit implementation')
    require('theFit->fitCovariance,normal,fitInfo::zCenter' in historical_c, 'actual audit normal')
    proof = {'source_definition': 'a=(sinA,cosA,z*sinA,z*cosA); N=sum(a*a^T/sigmaSq)',
             'normal_and_fit_fragments_exact_original_generated': True,
             'raw_covariance': 's4.computeInverseWithCheck; not used to construct saved normal',
             'regularization_in_normal': False, 'hit_design_rhs_independent_rebuild': 'UNKNOWN_NOT_SAVED',
             'scope': 'conditional fixed-selected-cluster straight-line WLS, not calibrated curved-track likelihood'}
    for idx in protocol['indices']:
        directory = PARENT/'events'/f'{idx:02d}'
        fixture = read(directory/'fixture.json')
        require(fixture['index'] == idx and fixture['row']['role'] == 'development', 'development selection')
        p = Path(fixture['wb91_repair_path'])
        require(p.is_relative_to(ROOT/'outputs/mc24_four_station_wb91_covariance_repair_v7/physical'), 'six repair source path')
        require(digest(p) == fixture['wb91_repair_sha256'], 'historical fixture repair')
        paths.append(p)
    for name in ('default', 'calypso', 'ml'):
        receipt = read(ROOT/'outputs'/f'wb126_test_preflight_{name}.json')
        require(receipt['pytest_exit'] == 0 and receipt['test_sha256'] == digest(SOURCES[2]) and
                receipt['analysis_sha256'] == digest(SOURCES[1]), 'synthetic preflight identity')
    for p in paths:
        hashes[str(p)] = digest(p)
    OUT.mkdir(exist_ok=False)
    (OUT/'events').mkdir()
    write(OUT/'protocol.json', protocol)
    with (OUT/'contract_workbook.md').open('x') as stream:
        stream.write(WORKBOOK.read_text())
    write(OUT/'source_contract.json', proof)
    for p in (OUT/'protocol.json', OUT/'contract_workbook.md', OUT/'source_contract.json'):
        hashes[str(p)] = digest(p)
    write(OUT/'freeze.json', {'hashes': hashes, 'implementation_commit': git('rev-parse', 'HEAD'),
                            'preregistration_commit': 'cd8c388', 'parent_result_commit': protocol['parent_head'],
                            'raw_root_reads': 0, 'new_propagation_calls': 0, 'held_out_access': False,
                            'python': sys.version, 'numpy': np.__version__, 'host': os.uname().nodename})
    verify()
    print('FREEZE_COMPLETE', len(hashes), flush=True)


def verify():
    frozen = read(OUT/'freeze.json')
    for path, h in frozen['hashes'].items():
        require(digest(path) == h, 'frozen identity: '+path)
    return frozen


def run():
    verify()
    (OUT/'execution_lock').mkdir(exist_ok=False)
    protocol = read(OUT/'protocol.json')
    parent_audit = read(PARENT/'saved_corresponding_boundary_audit.json')
    cells = []
    for idx in protocol['indices']:
        directory = PARENT/'events'/f'{idx:02d}'
        fixture, response = read(directory/'fixture.json'), read(directory/'response.json')
        identity = {k: fixture[k] for k in ('input_xaod', 'ordinal', 'actual_run', 'actual_event')}
        require(response['identity'] == identity, 'event identity')
        original_rows = [json.loads(line) for line in Path(fixture['wb91_repair_path']).read_text().splitlines()]
        require(all([r['run'], r['event']] == [fixture['actual_run'], fixture['actual_event']] for r in original_rows), 'repair event identity')
        association = next(e for e in parent_audit['events'] if e['index'] == idx)
        event_cells = []
        for station in protocol['stations']:
            cell = {'index': idx, 'station': station, 'identity': identity,
                    'association': association['association'], 'physical_fixture_supported': association['physical_fixture_supported'],
                    'conditional_interpretation': True, 'downstream_membership_complete': association['downstream_membership_complete']}
            try:
                ref = fixture['references'][station]
                require(ref['station'] == station, 'station order')
                original = [r for r in original_rows if r['station'] == station and r['state_index'] == 0]
                require(len(original) == 1, 'reference multiplicity')
                require(all(original[0][k] == ref[k] for k in original[0]), 'fixture/original reference')
                rows = [r for r in response['responses'] if
                        (r['arm'], r['kind'], r['station']) == ('P', 'nominal', station)]
                require(len(rows) == 1 and rows[0]['status'] == 'SUCCESS', 'P response')
                require(rows[0]['state']['on_surface'] and not rows[0]['state']['covariance_present'], 'P bound/noC')
                cell.update(audit_cell(ref, rows[0]['state']['h'], protocol))
            except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
                cell.update({'information_status': 'UNKNOWN', 'error': str(error)})
            event_cells.append(cell); cells.append(cell)
        write(OUT/'events'/f'{idx:02d}.json', {'index': idx, 'cells': event_cells})
    require(len(cells) == 24 and len({(c['index'], c['station']) for c in cells}) == 24, 'matrix coverage')
    summary = {'schema': 'wb126_tracklet_information_summary_v1', 'cells': cells,
               'execution_contract': 'PASS' if all(c['information_status'] == 'KNOWN' for c in cells) else 'UNKNOWN',
               'hypothesis': hypothesis(cells, protocol), 'population': 6, 'cell_count': 24,
               'source_information': 'SUPPORTED_CONDITIONAL_SAVED_WLS_NORMAL',
               'independent_hit_rebuild': 'UNKNOWN_NOT_SAVED', 'covariance_calibration': 'NOT_EVALUATED',
               'physical_likelihood': 'NOT_EVALUATED', 'alignment_solve': 'NOT_EVALUATED',
               'qualification': 'NOT_EVALUATED', 'raw_root_reads': 0, 'new_propagation_calls': 0,
               'held_out_access': False, 'next_stage_executed': False, 'freeze_sha256': digest(OUT/'freeze.json')}
    verify()
    write(OUT/'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cells'}, indent=2), flush=True)


def seal():
    frozen = verify(); summary = read(OUT/'summary.json')
    require((OUT/'independent_result_check.json').is_file(), 'independent saved check required')
    require(read(OUT/'independent_result_check.json')['gate'] == 'PASS', 'independent check')
    artifacts = {str(p.relative_to(ROOT)): digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    for name in ('default', 'calypso', 'ml'):
        p = ROOT/'outputs'/f'wb126_test_preflight_{name}.json'; artifacts[str(p.relative_to(ROOT))] = digest(p)
    write(ROOT/'docs/wb126_tracklet_information_result_manifest.json',
          {'schema': 'wb126_tracklet_information_result_manifest_v1',
           'execution_contract': summary['execution_contract'], 'hypothesis': summary['hypothesis'],
           'preregistration_commit': frozen['preregistration_commit'], 'implementation_commit': frozen['implementation_commit'],
           'parent_result_commit': frozen['parent_result_commit'], 'population': 6, 'cell_count': 24,
           'raw_root_reads': 0, 'new_propagation_calls': 0, 'held_out_access': False,
           'covariance_calibration': 'NOT_EVALUATED', 'qualification': 'NOT_EVALUATED',
           'sources': {str(p.relative_to(ROOT)): digest(p) for p in SOURCES}, 'artifacts': artifacts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze', 'verify', 'run', 'seal'))
    globals()[parser.parse_args().action]()
