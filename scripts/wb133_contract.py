#!/usr/bin/env python3
"""Exclusive single-pass accepted-anchor experiment and immutable receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wb131_contract import digest, read, write, require, PARENT, FIELD_MAP, POOL_PAYLOAD
from wb132_typed_recovery import verify_recovery as verify_parent
from alignment.wb133_accepted_anchor import source_request, audit_event
NATIVE=ROOT/'outputs/mc24_four_station_wb132_native_ckf_state_v1'
BASELINE=ROOT/'outputs/mc24_four_station_wb129_sensor_surface_curve_v2/recovery_v2'
OUT=ROOT/'outputs/mc24_four_station_wb133_accepted_anchor_v1'
PROTOCOL=ROOT/'configs/research_review/wp133_accepted_anchor_contract.json'
WORKBOOK=ROOT/'workbook/2026-10-04_133_接受测量起点四站共同轨迹名义预测对照.md'
SRC=ROOT/'research/wb133/AcceptedAnchorPrediction.cxx'
SOURCES=[PROTOCOL,SRC,ROOT/'alignment/wb133_accepted_anchor.py',ROOT/'scripts/wb133_athena.py',
         ROOT/'scripts/wb133_contract.py',ROOT/'scripts/wb133_independent_audit.py',
         ROOT/'tests/test_wb133_accepted_anchor.py',ROOT/'scripts/setup_environment.sh']


def build():
    pre=OUT/'build_preflight';pre.mkdir(parents=True,exist_ok=False)
    source=pre/'source';pkg=source/'WB133Diagnostic';pkg.mkdir(parents=True)
    old=ROOT/'outputs/mc24_four_station_wb125_build_preflight_v2/source'
    (source/'CMakeLists.txt').write_text((old/'CMakeLists.txt').read_text().replace('WB125','WB133'))
    cm=(old/'WB125Diagnostic/CMakeLists.txt').read_text().replace('WB125Diagnostic','WB133Diagnostic')
    (pkg/'CMakeLists.txt').write_text(cm.replace('PersistedProvenance.cxx PhysicalSeedAudit.cxx','AcceptedAnchorPrediction.cxx'))
    shutil.copyfile(SRC,pkg/SRC.name)
    for name,cmd in [('configure',['cmake','-S',str(source),'-B',str(pre/'build'),'-DCalypso_DIR='+str(ROOT.parent/'calypso/run/cmake')]),
                     ('build',['cmake','--build',str(pre/'build'),'-j','1'])]:
        with (pre/(name+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        require(r.returncode==0,name+' failed; preserve logs')
    binary=pre/'build/x86_64-el9-gcc13-opt/lib/libWB133Diagnostic.so'
    write(pre/'receipt.json',{'binary':str(binary),'binary_sha256':digest(binary),'source_sha256':digest(SRC),
                            'new_reconstruction_calls':0,'new_propagation_calls':0})


def freeze():
    verify_parent()
    require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='4station','branch')
    require(not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip(),'tracked dirty')
    hashes=dict(read(NATIVE/'freeze.json')['hashes'])
    for group in ('sources','artifacts'):
        for path,expected in read(ROOT/'docs/wb132_native_ckf_state_result_manifest.json')[group].items():
            require(digest(ROOT/path)==expected,'WB132 manifest identity '+path)
    receipt=read(OUT/'build_preflight/receipt.json')
    require(digest(receipt['binary'])==receipt['binary_sha256'] and digest(SRC)==receipt['source_sha256'],'build identity')
    parents=[ROOT/'docs/wb132_native_ckf_state_result_manifest.json',ROOT/'docs/wb129_sensor_surface_curve_result_manifest.json',
             ROOT/'docs/wb130_sensor_bounds_result_manifest.json',NATIVE/'freeze.json',NATIVE/'saved_typed_coverage_v1/freeze.json']
    for p in SOURCES+parents+[Path(receipt['binary']),OUT/'build_preflight/receipt.json',FIELD_MAP,POOL_PAYLOAD]:hashes[str(p)]=digest(p)
    shutil.copyfile(WORKBOOK,OUT/'contract_workbook.md');hashes[str(OUT/'contract_workbook.md')]=digest(OUT/'contract_workbook.md')
    require(digest(POOL_PAYLOAD)==digest(PARENT/'identity_payload/tracker_alignment.pool.root'),'pool identity')
    (OUT/'events').mkdir(exist_ok=False);total=0;requests=[]
    nominal_manifest=read(ROOT/'docs/wb129_sensor_surface_curve_result_manifest.json')['artifacts']
    native_manifest=read(ROOT/'docs/wb132_native_ckf_state_result_manifest.json')['artifacts']
    for index in read(PROTOCOL)['indices']:
        event=OUT/'events'/f'{index:02d}';event.mkdir()
        for name in ('fixture.json','export.json','native_response.json'):
            origin=NATIVE/'events'/event.name/name
            require(digest(origin)==native_manifest[str(origin.relative_to(ROOT))],'native evidence identity')
            shutil.copyfile(origin,event/name);hashes[str(origin)]=digest(origin);hashes[str(event/name)]=digest(event/name)
        origin=BASELINE/'events'/event.name/'curve_response.json'
        require(digest(origin)==nominal_manifest[str(origin.relative_to(ROOT))],'baseline identity')
        shutil.copyfile(origin,event/'baseline.json');hashes[str(origin)]=digest(origin);hashes[str(event/'baseline.json')]=digest(event/'baseline.json')
        request=source_request(read(event/'export.json'),read(event/'native_response.json'))
        write(event/'request.json',request);hashes[str(event/'request.json')]=digest(event/'request.json')
        total+=len(read(event/'export.json')['rows']);requests.append({'index':index,'source':request['source']})
    require(total==147,'population')
    write(OUT/'freeze.json',{'schema':'wb133_freeze_v1','hashes':hashes,'requests':requests,
          'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'rows':147,'max_new_propagation_calls':141,'max_input_event_executions':6,'held_out_access':False,'truth_access':False})
    verify()


def verify():
    for path,expected in read(OUT/'freeze.json')['hashes'].items():require(digest(path)==expected,'frozen identity '+path)
    for index in read(PROTOCOL)['indices']:
        f=read(OUT/'events'/f'{index:02d}'/'fixture.json');stat=Path(f['input_xaod']).stat()
        require(stat.st_size==f['source_stat']['bytes'] and stat.st_mtime_ns==f['source_stat']['mtime_ns'],'input stat')


def execute(index):
    verify();event=OUT/'events'/f'{index:02d}'
    binary=Path(read(OUT/'build_preflight/receipt.json')['binary']);platform=binary.parent.parent
    q=lambda p:shlex.quote(str(p))
    command='\n'.join(['set -eo pipefail','ulimit -c 0','source '+q(ROOT/'scripts/setup_environment.sh')+' calypso',
                       'source '+q(platform/'setup.sh'),'export LD_LIBRARY_PATH='+q(platform/'lib')+':${LD_LIBRARY_PATH:-}',
                       'python '+q(ROOT/'scripts/wb133_athena.py')+' --work-dir '+q(event)+' --sqlite '+q(PARENT/'identity_payload/tracker_alignment.sqlite')])
    with (event/'command.sh').open('x') as f:f.write(command+'\n')
    print('WB133 event',index,'prediction',flush=True)
    with (event/'athena.log').open('x') as log:result=subprocess.run(['bash','-c',command],cwd=event,stdout=log,stderr=subprocess.STDOUT)
    write(event/'exit.json',{'exit_code':result.returncode})
    status={'index':index,'execution':'PASS' if result.returncode==0 else 'FAIL','scientific_verdict':'UNKNOWN',
            'interface':'UNKNOWN','expected_propagation_calls':len(read(event/'export.json')['rows'])-1}
    if (event/'response.json').exists():status['completed_response_calls']=read(event/'response.json')['official_calls']
    try:
        require(result.returncode==0,'Athena execution failure')
        audit=audit_event(read(event/'export.json'),read(event/'native_response.json'),read(event/'request.json'),
                          read(event/'response.json'),read(event/'baseline.json'),read(PROTOCOL))
        write(event/'audit.json',audit)
        status.update(interface=audit['interface'],scientific_verdict=audit['scientific_verdict'],
                      valid_rows=audit['valid_rows'],original_rows=audit['original_rows'],complete_coverage=audit['complete_coverage'])
    except (ValueError,KeyError,OSError,IndexError) as error:status['failure']=str(error)
    log=(event/'athena.log').read_text(errors='replace')
    begins=[int(v) for v in re.findall(r'WB133_CALL_BEGIN id=(\d+)',log)]
    ends=[int(v) for v in re.findall(r'WB133_CALL_END id=(\d+)',log)]
    require(begins==list(range(1,len(begins)+1)) and ends==list(range(1,len(ends)+1)) and len(ends)<=len(begins)<=status['expected_propagation_calls'],'call log coverage')
    require('message limit (500) reached for WB133AcceptedAnchorPrediction' not in log,'call log saturation')
    status.update(attempted_calls=len(begins),completed_logged_calls=len(ends),call_accounting='PASS_LOG_LEDGER')
    if 'completed_response_calls' in status:require(status['completed_response_calls']==len(ends)==len(begins),'response/log call identity')
    write(event/'status.json',status);print(status,flush=True);return status


def run():
    verify();(OUT/'execution_lock').mkdir(exist_ok=False)
    protocol=read(PROTOCOL)
    with ThreadPoolExecutor(max_workers=protocol['local_concurrency']) as pool:events=list(pool.map(execute,protocol['indices']))
    verify();calls=sum(e['attempted_calls'] for e in events);require(calls<=protocol['max_new_propagation_calls'],'call budget')
    verdicts=[e['scientific_verdict'] for e in events]
    science=('PASS_GROSS_MODEL_ONLY' if all(v=='PASS_GROSS_MODEL_ONLY' for v in verdicts) else 'FAIL_GROSS_MODEL' if 'FAIL_GROSS_MODEL' in verdicts else 'UNKNOWN')
    write(OUT/'summary.json',{'schema':'wb133_summary_v1','events':events,'scientific_verdict':science,
          'execution_contract':'PASS' if all(e['execution']=='PASS' for e in events) else 'FAIL_OR_UNKNOWN',
          'interface':'PASS' if all(e['interface']=='PASS' for e in events) else 'FAIL_OR_UNKNOWN',
          'complete_coverage':all(e.get('complete_coverage',False) for e in events),
          'attempted_propagation_calls':calls,'completed_propagation_calls':sum(e['completed_logged_calls'] for e in events),
          'input_event_executions':len(events),'new_reconstruction_calls':0,'held_out_access':False,'truth_access':False})


def seal():
    verify();independent=read(OUT/'independent_audit.json');require(independent['artifact_consistency']=='PASS','independent artifact audit')
    write(ROOT/'docs/wb133_accepted_anchor_result_manifest.json',{'schema':'wb133_result_manifest_v1','summary':read(OUT/'summary.json'),
          'independent_audit':independent,'sources':{str(p.relative_to(ROOT)):digest(p) for p in SOURCES},
          'artifacts':{str(p.relative_to(ROOT)):digest(p) for p in OUT.rglob('*') if p.is_file()},
          'scientific_boundary':{'association':'EXACT_LINKS_AUDITED_PARTICLE_ASSOCIATION_CONDITIONAL',
          'source':'WHOLE_ACCEPTED_STATE_INTERVENTION_NOT_QOP_ONLY','qop_prior':'NOT_ESTABLISHED',
          'covariance':'NOT_CALIBRATED_NO_PRIOR','material':'NONE_CONDITIONAL_MODEL','alignment':'NOT_ESTABLISHED',
          'held_out':'NOT_ACCESSED','ML':'NOT_AUTHORIZED'}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','freeze','verify','run','seal']);globals()[p.parse_args().action]()
