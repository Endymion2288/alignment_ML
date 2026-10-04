#!/usr/bin/env python3
"""Exclusive, bounded WB131 development execution with immutable input receipts."""
import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment.wb131_fixed_qop_nuisance import derivative_request, fit_event, replay_metrics

PARENT = ROOT / 'outputs/mc24_four_station_wb125_physical_seed_v1'
EXPORTS = ROOT / 'outputs/mc24_four_station_wb127_strip_measurement_v1/recovery_v5'
NOMINAL = ROOT / 'outputs/mc24_four_station_wb129_sensor_surface_curve_v2/recovery_v2'
OUT = ROOT / 'outputs/mc24_four_station_wb131_fixed_qop_nuisance_v1'
PROTOCOL = ROOT / 'configs/research_review/wp131_fixed_qop_nuisance_contract.json'
WORKBOOK = ROOT / 'workbook/2026-10-04_131_固定物理qop四维共同轨迹单步重传播验证.md'
SRC = ROOT / 'research/wb131/FixedQopNuisancePrediction.cxx'
FIELD_MAP = Path('/cvmfs/faser.cern.ch/repo/sw/software/22.0/faser/offline/ReleaseData/v20/MagneticFieldMaps/FaserFieldTable_v2.root')
POOL_PAYLOAD = ROOT/'outputs/mc24_four_station_wb91_covariance_repair_v6/identity_payload/tracker_alignment.pool.root'
SOURCES = [PROTOCOL, SRC, ROOT/'alignment/wb131_fixed_qop_nuisance.py',
           ROOT/'scripts/wb131_athena.py', ROOT/'scripts/wb131_contract.py',
           ROOT/'scripts/wb131_independent_audit.py', ROOT/'tests/test_wb131_fixed_qop_nuisance.py',
           ROOT/'scripts/setup_environment.sh']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path).resolve()
    if any(s in str(path).lower() for s in ('sealed', 'blind', '00350_00399', '00800_00849', '100116', '100117')):
        raise ValueError('forbidden path')
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def build():
    pre = OUT/'build_preflight'
    pre.mkdir(parents=True, exist_ok=False)
    source = pre/'source'
    package = source/'WB131Diagnostic'
    package.mkdir(parents=True)
    old = ROOT/'outputs/mc24_four_station_wb125_build_preflight_v2/source'
    (source/'CMakeLists.txt').write_text((old/'CMakeLists.txt').read_text().replace('WB125', 'WB131'))
    cm = (old/'WB125Diagnostic/CMakeLists.txt').read_text().replace('WB125Diagnostic', 'WB131Diagnostic')
    cm = cm.replace('PersistedProvenance.cxx PhysicalSeedAudit.cxx', 'FixedQopNuisancePrediction.cxx')
    (package/'CMakeLists.txt').write_text(cm)
    shutil.copyfile(SRC, package/SRC.name)
    for name, cmd in [('configure', ['cmake','-S',str(source),'-B',str(pre/'build'),'-DCalypso_DIR='+str(ROOT.parent/'calypso/run/cmake')]),
                      ('build', ['cmake','--build',str(pre/'build'),'-j','1'])]:
        with (pre/(name+'.log')).open('x') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        require(result.returncode == 0, name+' failed; preserve logs')
    binary = pre/'build/x86_64-el9-gcc13-opt/lib/libWB131Diagnostic.so'
    require(binary.is_file(), 'binary missing')
    write(pre/'receipt.json', {'binary':str(binary), 'binary_sha256':digest(binary), 'source_sha256':digest(SRC),
                               'new_reconstruction_calls':0, 'new_propagation_calls':0})


def freeze():
    require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='4station','branch')
    require(not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip(),'tracked dirty')
    hashes = {}
    # WB127 recovery_v5 already freezes runtime sources/binaries and identity payload.
    # Verify original frozen artifacts, not later amended parent narrative.
    parent_freeze = EXPORTS/'freeze.json'
    for path, expected in read(parent_freeze)['hashes'].items():
        p = Path(path) if Path(path).is_absolute() else ROOT/path
        require(digest(p)==expected, 'parent frozen identity '+str(p))
        hashes[str(p)] = expected
    receipt = read(OUT/'build_preflight/receipt.json')
    require(digest(receipt['binary'])==receipt['binary_sha256'] and digest(SRC)==receipt['source_sha256'], 'build changed')
    for path in SOURCES + [Path(receipt['binary']),OUT/'build_preflight/receipt.json',parent_freeze,
                           ROOT/'docs/wb129_sensor_surface_curve_result_manifest.json',ROOT/'docs/wb130_sensor_bounds_result_manifest.json',
                           FIELD_MAP, POOL_PAYLOAD]:
        hashes[str(path)] = digest(path)
    require(digest(POOL_PAYLOAD)==digest(PARENT/'identity_payload/tracker_alignment.pool.root'),'catalog-resolved pool payload')
    # Verify selected raw nominal responses against the published parent manifest.
    manifest = read(ROOT/'docs/wb129_sensor_surface_curve_result_manifest.json')['artifacts']
    (OUT/'events').mkdir(exist_ok=False)
    shutil.copyfile(PROTOCOL, OUT/'protocol.json')
    shutil.copyfile(WORKBOOK, OUT/'contract_workbook.md')
    hashes[str(OUT/'protocol.json')] = digest(OUT/'protocol.json')
    hashes[str(OUT/'contract_workbook.md')] = digest(OUT/'contract_workbook.md')
    rows = 0
    for index in read(PROTOCOL)['indices']:
        event = OUT/'events'/f'{index:02d}'
        event.mkdir()
        for name in ('fixture.json','export.json'):
            source = EXPORTS/'events'/event.name/name
            shutil.copyfile(source,event/name)
            hashes[str(source)] = digest(source)
            hashes[str(event/name)] = digest(event/name)
        nominal = NOMINAL/'events'/event.name/'curve_response.json'
        require(digest(nominal)==manifest[str(nominal.relative_to(ROOT))], 'parent nominal artifact changed')
        shutil.copyfile(nominal,event/'nominal.json')
        hashes[str(nominal)] = digest(nominal)
        hashes[str(event/'nominal.json')] = digest(event/'nominal.json')
        export = read(event/'export.json')
        fixture = read(event/'fixture.json')
        require(export['identity']['actual_event']==fixture['actual_event'] and
                export['identity']['actual_run']==fixture['actual_run'], 'event identity')
        source = Path(fixture['input_xaod'])
        stat = source.stat()
        require(stat.st_size==fixture['source_stat']['bytes'] and stat.st_mtime_ns==fixture['source_stat']['mtime_ns'], 'source stat changed')
        rows += len(export['rows'])
    require(rows==147, 'row population')
    write(OUT/'freeze.json', {'schema':'wb131_freeze_v1','hashes':hashes,
                            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                            'contract_workbook_sha256':digest(OUT/'contract_workbook.md'),
                            'max_new_propagation_calls':2646,'rows':rows,'held_out_access':False,'truth_access':False})


def verify():
    for path, expected in read(OUT/'freeze.json')['hashes'].items():
        require(digest(path)==expected, 'frozen input changed '+path)
    for index in read(PROTOCOL)['indices']:
        fixture = read(OUT/'events'/f'{index:02d}'/'fixture.json')
        stat = Path(fixture['input_xaod']).stat()
        require(stat.st_size==fixture['source_stat']['bytes'] and stat.st_mtime_ns==fixture['source_stat']['mtime_ns'], 'source changed')


def execute(event, phase, request):
    verify()
    work = event/phase
    work.mkdir(exist_ok=False)
    write(work/'request.json',request)
    binary = Path(read(OUT/'build_preflight/receipt.json')['binary'])
    platform = binary.parent.parent
    quote = lambda path: shlex.quote(str(path))
    command = '\n'.join(['set -eo pipefail','ulimit -c 0',
                          'source '+quote(ROOT/'scripts/setup_environment.sh')+' calypso',
                          'source '+quote(platform/'setup.sh'),
                          'export LD_LIBRARY_PATH='+quote(platform/'lib')+':${LD_LIBRARY_PATH:-}',
                          'python '+quote(ROOT/'scripts/wb131_athena.py')+' --work-dir '+quote(work)+
                          ' --sqlite '+quote(PARENT/'identity_payload/tracker_alignment.sqlite')])
    with (work/'command.sh').open('x') as stream:
        stream.write(command+'\n')
    with (work/'athena.log').open('x') as log:
        result = subprocess.run(['bash','-c',command],cwd=work,stdout=log,stderr=subprocess.STDOUT)
    write(work/'exit.json',{'exit_code':result.returncode})
    require(result.returncode==0,'execution failure '+phase)
    require((work/'response.json').is_file(),'response missing')
    return read(work/'response.json')


def run():
    verify()
    (OUT/'execution_lock').mkdir(exist_ok=False)
    protocol = read(PROTOCOL)
    events, calls, attempted = [], 0, 0
    for index in protocol['indices']:
        event = OUT/'events'/f'{index:02d}'
        export = read(event/'export.json')
        status = {'index':index, 'verdict':'UNKNOWN', 'rows':len(export['rows'])}
        print('WB131 event',index,'derivatives',flush=True)
        try:
            response = execute(event,'derivatives',derivative_request(export,protocol))
            calls += response['official_calls']
            fit = fit_event(export,response,read(event/'nominal.json'),protocol)
            write(event/'fit.json',fit)
            status.update(gate=fit['gate'], execution='PASS')
            if fit['gate']=='READY':
                print('WB131 event',index,'one replay',flush=True)
                replay = execute(event,'replay',{'schema':'wb131_request_v1','arms':[{'tag':'updated','seed':fit['updated_seed']}]})
                calls += replay['official_calls']
                metrics = replay_metrics(export,fit,replay,protocol)
                write(event/'metrics.json',metrics)
                status['verdict'] = metrics['verdict']
        except (ValueError, KeyError, OSError) as error:
            status['failure'] = str(error)
            status['execution'] = 'FAIL' if not (event/'derivatives/response.json').exists() else 'PASS_WITH_ANALYSIS_OR_REPLAY_FAILURE'
        # Count logged attempts even when no final response was produced.
        for log in event.glob('*/athena.log'):
            attempted += log.read_text(errors='replace').count('WB131_CALL_BEGIN id=')
        write(event/'status.json',status)
        events.append(status)
        print('WB131 event',index,status,flush=True)
    require(attempted<=protocol['max_new_propagation_calls'],'attempt budget exceeded')
    verify()
    verdicts = [e['verdict'] for e in events]
    science = ('PASS_CONDITIONAL_ONE_STEP' if all(v=='PASS_CONDITIONAL_ONE_STEP' for v in verdicts)
               else 'FAIL_ONE_STEP_HYPOTHESIS' if 'FAIL_ONE_STEP_HYPOTHESIS' in verdicts else 'UNKNOWN')
    write(OUT/'summary.json', {'schema':'wb131_summary_v1','events':events,'scientific_verdict':science,
                            'execution_contract':'PASS' if all(e.get('execution')=='PASS' for e in events) else 'FAIL_OR_UNKNOWN',
                            'completed_propagation_calls':calls,'attempted_propagation_calls':attempted,
                            'new_reconstruction_calls':0,'held_out_access':False,'truth_access':False})


def seal():
    verify()
    summary = read(OUT/'summary.json')
    independent = read(OUT/'independent_audit.json')
    require(independent['artifact_consistency']=='PASS','independent audit')
    artifacts = {str(p.relative_to(ROOT)):digest(p) for p in OUT.rglob('*') if p.is_file()}
    write(ROOT/'docs/wb131_fixed_qop_nuisance_result_manifest.json',
          {'schema':'wb131_result_manifest_v1','summary':summary,'independent_audit':independent,
           'sources':{str(p.relative_to(ROOT)):digest(p) for p in SOURCES},'artifacts':artifacts,
           'scientific_boundary':{'association':'INHERITED_CONDITIONAL','qop':'FIXED_PHYSICAL_SEED_NO_PRIOR',
                                  'covariance':'NOT_CALIBRATED','material':'NONE_CONDITIONAL_MODEL',
                                  'alignment':'NOT_ESTABLISHED','held_out':'NOT_ACCESSED','ML':'NOT_AUTHORIZED'}})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['build','freeze','verify','run','seal'])
    globals()[parser.parse_args().action]()
