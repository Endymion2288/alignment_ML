#!/usr/bin/env python3
"""Bounded read-only native CKF audit, exclusive outputs and immutable inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wb131_contract import digest, read, write, require, verify as verify_parent

PARENT=ROOT/'outputs/mc24_four_station_wb125_physical_seed_v1'
WB131=ROOT/'outputs/mc24_four_station_wb131_fixed_qop_nuisance_v1'
OUT=ROOT/'outputs/mc24_four_station_wb132_native_ckf_state_v1'
PROTOCOL=ROOT/'configs/research_review/wp132_native_ckf_state_contract.json'
WORKBOOK=ROOT/'workbook/2026-10-04_132_nativeCKF状态测量关联与strip预测审计.md'
SRC=ROOT/'research/wb132/NativeStateAudit.cxx'
SOURCES=[PROTOCOL,SRC,ROOT/'scripts/wb132_contract.py',ROOT/'scripts/wb132_athena.py',
         ROOT/'scripts/wb132_independent_audit.py',ROOT/'alignment/wb132_native_ckf_state.py',
         ROOT/'tests/test_wb132_native_ckf_state.py']
CKF=ROOT.parent/'calypso/Tracking/Acts/FaserActsKalmanFilter/src/CombinatorialKalmanFilterAlg.cxx'
PERSIST=Path('/cvmfs/atlas.cern.ch/repo/sw/software/24.0/Athena/24.0.41/InstallArea/x86_64-el9-gcc13-opt/src/Tracking/TrkEventCnv/TrkEventTPCnv/src/TrkParameters/TrackParametersCnv_p2.cxx')


def build():
    pre=OUT/'build_preflight';pre.mkdir(parents=True,exist_ok=False)
    source=pre/'source';pkg=source/'WB132Diagnostic';pkg.mkdir(parents=True)
    old=ROOT/'outputs/mc24_four_station_wb125_build_preflight_v2/source'
    (source/'CMakeLists.txt').write_text((old/'CMakeLists.txt').read_text().replace('WB125','WB132'))
    cm=(old/'WB125Diagnostic/CMakeLists.txt').read_text().replace('WB125Diagnostic','WB132Diagnostic')
    (pkg/'CMakeLists.txt').write_text(cm.replace('PersistedProvenance.cxx PhysicalSeedAudit.cxx','NativeStateAudit.cxx'))
    shutil.copyfile(SRC,pkg/SRC.name)
    for name,cmd in [('configure',['cmake','-S',str(source),'-B',str(pre/'build'),'-DCalypso_DIR='+str(ROOT.parent/'calypso/run/cmake')]),
                     ('build',['cmake','--build',str(pre/'build'),'-j','1'])]:
        with (pre/(name+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        require(r.returncode==0,name+' failure; preserve namespace')
    binary=pre/'build/x86_64-el9-gcc13-opt/lib/libWB132Diagnostic.so'
    write(pre/'receipt.json',{'binary':str(binary),'binary_sha256':digest(binary),'source_sha256':digest(SRC),
                             'new_reconstruction_calls':0,'new_propagation_calls':0})


def freeze():
    verify_parent()
    require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='4station','branch')
    require(not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip(),'tracked dirty')
    hashes=dict(read(WB131/'freeze.json')['hashes'])
    receipt=read(OUT/'build_preflight/receipt.json')
    require(digest(receipt['binary'])==receipt['binary_sha256'] and digest(SRC)==receipt['source_sha256'],'build identity')
    for path in SOURCES+[Path(receipt['binary']),OUT/'build_preflight/receipt.json',WB131/'freeze.json',
                         ROOT/'docs/wb131_fixed_qop_nuisance_result_manifest.json',CKF,PERSIST]:
        hashes[str(path)]=digest(path)
    old_hashes=read(PARENT/'freeze.json')['hashes']
    old_ckf=[h for path,h in old_hashes.items() if path.endswith('FaserActsKalmanFilter/src/CombinatorialKalmanFilterAlg.cxx')]
    write(OUT/'producer_evidence.json',{'current_ckf_source_sha256':digest(CKF),
            'wb125_frozen_ckf_source_sha256':old_ckf,'matches_wb125_freeze':old_ckf==[digest(CKF)],
            'historical_mc24_producer_execution_identity':'UNKNOWN_NOT_SAVED',
            'current_rules':{'Hole':'predicted','Outlier':'filtered','Measurement':'smoothed'},
            'parameter_conversion':'Acts sensor bound state -> Trk curvilinear; top-left covariance copied with qop unit scaling',
            'float_persistence_source_sha256':digest(PERSIST)})
    hashes[str(OUT/'producer_evidence.json')]=digest(OUT/'producer_evidence.json')
    shutil.copyfile(WORKBOOK,OUT/'contract_workbook.md');hashes[str(OUT/'contract_workbook.md')]=digest(OUT/'contract_workbook.md')
    (OUT/'events').mkdir(exist_ok=False)
    total=0
    for index in read(PROTOCOL)['indices']:
        event=OUT/'events'/f'{index:02d}';event.mkdir()
        for name in ('fixture.json','export.json'):
            source=WB131/'events'/event.name/name
            shutil.copyfile(source,event/name);hashes[str(event/name)]=digest(event/name)
        for source_name,target_name in [('provenance.json','parent_provenance.json'),('response.json','parent_response.json')]:
            source=PARENT/'events'/event.name/source_name
            require(digest(source)==hashes[str(source)],'parent artifact identity')
            shutil.copyfile(source,event/target_name);hashes[str(event/target_name)]=digest(event/target_name)
        total+=len(read(event/'export.json')['rows'])
    require(total==147,'row population')
    write(OUT/'freeze.json',{'schema':'wb132_freeze_v1','hashes':hashes,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                           'input_event_executions':6,'original_selected_rows':147,'expected_native_matches':111,
                           'new_reconstruction_calls':0,'new_propagation_calls':0,'algorithm_field_queries':0,
                           'held_out_access':False,'truth_access':False})


def verify():
    for path,expected in read(OUT/'freeze.json')['hashes'].items():require(digest(path)==expected,'frozen input '+path)
    for index in read(PROTOCOL)['indices']:
        fixture=read(OUT/'events'/f'{index:02d}'/'fixture.json');s=Path(fixture['input_xaod']).stat()
        require(s.st_size==fixture['source_stat']['bytes'] and s.st_mtime_ns==fixture['source_stat']['mtime_ns'],'input stat')


def execute(index):
    from alignment.wb132_native_ckf_state import audit_event
    event=OUT/'events'/f'{index:02d}'
    verify()
    binary=Path(read(OUT/'build_preflight/receipt.json')['binary']);platform=binary.parent.parent
    q=lambda p:shlex.quote(str(p))
    command='\n'.join(['set -eo pipefail','ulimit -c 0','source '+q(ROOT/'scripts/setup_environment.sh')+' calypso',
                       'source '+q(platform/'setup.sh'),'export LD_LIBRARY_PATH='+q(platform/'lib')+':${LD_LIBRARY_PATH:-}',
                       'python '+q(ROOT/'scripts/wb132_athena.py')+' --work-dir '+q(event)+' --sqlite '+q(PARENT/'identity_payload/tracker_alignment.sqlite')])
    with (event/'command.sh').open('x') as stream:stream.write(command+'\n')
    print('WB132 event',index,'read-only export',flush=True)
    with (event/'athena.log').open('x') as log:r=subprocess.run(['bash','-c',command],cwd=event,stdout=log,stderr=subprocess.STDOUT)
    write(event/'exit.json',{'exit_code':r.returncode})
    status={'index':index,'execution':'FAIL' if r.returncode else 'PASS','native_prediction_hypothesis':'UNKNOWN'}
    try:
        require(r.returncode==0,'Athena execution failure')
        result=audit_event(read(event/'export.json'),read(event/'native_response.json'),read(event/'parent_provenance.json'),
                           read(event/'parent_response.json'),read(PROTOCOL),read(OUT/'producer_evidence.json'))
        write(event/'audit.json',result)
        status.update(interface=result['interface'],native_prediction_hypothesis=result['native_prediction_hypothesis'],
                      source_type=result['source']['type_description'],matched_rows=result['matched_rows'],unmatched_rows=result['unmatched_rows'])
    except (ValueError,KeyError,OSError,IndexError) as error:status['failure']=str(error)
    write(event/'status.json',status);print('WB132 event',index,status,flush=True)
    return status


def run():
    verify();(OUT/'execution_lock').mkdir(exist_ok=False)
    protocol=read(PROTOCOL)
    with ThreadPoolExecutor(max_workers=protocol['local_concurrency']) as pool:events=list(pool.map(execute,protocol['indices']))
    verify()
    verdicts=[e['native_prediction_hypothesis'] for e in events]
    verdict=('PASS_GROSS_SCREEN_ONLY' if all(v=='PASS_GROSS_SCREEN_ONLY' for v in verdicts)
             else 'FAIL_GROSS_NATIVE_RESIDUAL' if 'FAIL_GROSS_NATIVE_RESIDUAL' in verdicts else 'UNKNOWN')
    write(OUT/'summary.json',{'schema':'wb132_summary_v1','events':events,'native_prediction_hypothesis':verdict,
                            'execution_contract':'PASS' if all(e['execution']=='PASS' for e in events) else 'FAIL_OR_UNKNOWN',
                            'interface':'PASS' if all(e.get('interface')=='PASS' for e in events) else 'FAIL_OR_UNKNOWN',
                            'input_event_executions':6,'new_reconstruction_calls':0,'new_propagation_calls':0,
                            'algorithm_field_queries':0,'held_out_access':False,'truth_access':False})


def seal():
    verify();independent=read(OUT/'independent_audit.json');require(independent['artifact_consistency']=='PASS','independent saved audit')
    write(ROOT/'docs/wb132_native_ckf_state_result_manifest.json',{'schema':'wb132_result_manifest_v1',
          'summary':read(OUT/'summary.json'),'independent_audit':independent,'producer_evidence':read(OUT/'producer_evidence.json'),
          'sources':{str(p.relative_to(ROOT)):digest(p) for p in SOURCES},
          'artifacts':{str(p.relative_to(ROOT)):digest(p) for p in OUT.rglob('*') if p.is_file()},
          'scientific_boundary':{'association':'INHERITED_CONDITIONAL_36_UNMATCHED_SELECTED_ROWS','covariance':'NOT_CALIBRATED',
                                'historical_fit_semantics':'UNKNOWN_NOT_PROVEN_BY_CURRENT_RULES','qop_prior':'NOT_ESTABLISHED',
                                'alignment':'NOT_ESTABLISHED','held_out':'NOT_ACCESSED','ML':'NOT_AUTHORIZED'}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','freeze','verify','run','seal']);globals()[p.parse_args().action]()
