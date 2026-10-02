#!/usr/bin/env python3
"""Saved-artifact WB102 aggregation recovery; never builds or propagates."""
import argparse, json, os, re, shlex, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT, read_public, write_new
from wb100_contract import digest
import wb102_contract as parent
import audit_wb102_results as audit_module

OUT = ROOT/'outputs/mc24_four_station_wb102_aggregation_recovery_v1'
PRIMARY = parent.OUT
SOURCES = [Path(__file__), ROOT/'scripts/run_wb102_recovery_condor.sh', ROOT/'tests/test_wb102_recovery.py']


def verify_primary():
    f = read_public(PRIMARY/'freeze.json')
    for path, h in f['hashes'].items():
        if digest(path) != h: raise ValueError('original freeze identity '+path)
    if read_public(PRIMARY/'worker_exit.json')['exit_code'] != 1:
        raise ValueError('original failure receipt')
    error = (PRIMARY/'condor.1195061.err').read_text()
    if "AttributeError: 'str' object has no attribute 'read_bytes'" not in error:
        raise ValueError('unexpected original failure')
    if any((PRIMARY/x).exists() for x in ('summary.json','trial_integrity_audit.json','scalar_summary_audit.json','result_integrity.json')):
        raise ValueError('original outputs changed')
    return f


def freeze(out):
    original = verify_primary()
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip() != '4station':
        raise ValueError('wrong branch')
    out.mkdir(exist_ok=False)
    write_new(out/'protocol.json', read_public(PRIMARY/'protocol.json'))
    paths = [p for p in PRIMARY.rglob('*') if p.is_file()] + SOURCES + [out/'protocol.json']
    hashes = dict(original['hashes'])
    hashes.update({str(p):digest(p) for p in paths})
    write_new(out/'freeze.json', {'schema':'wb102_saved_aggregation_recovery_v1','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'hashes':hashes,'original':str(PRIMARY),'original_execution':'FAIL','original_implementation_commit':original['commit'],
        'new_physical_calls':0,'new_reference_integrations':0,'threshold_change':False,'qualification':'NOT_EVALUATED'})


def verify(out):
    frozen = read_public(out/'freeze.json')
    for path,h in frozen['hashes'].items():
        if digest(path) != h: raise ValueError('recovery identity changed '+path)
    if read_public(out/'protocol.json') != read_public(PRIMARY/'protocol.json'):
        raise ValueError('scientific protocol changed')
    return frozen


def recovery_aggregate(source, destination):
    # Both frozen modules resolve a digest global. Only the path/streaming IO
    # adapter changes; the original analysis and decision functions are reused.
    parent.digest = digest
    original_write = parent.write_new
    def exclusive_output(path, value):
        path = Path(path)
        if path.parent != source or path.name not in ('summary.json','trial_integrity_audit.json','scalar_summary_audit.json'):
            raise ValueError('unexpected recovery write '+str(path))
        write_new(destination/path.name, value)
    parent.write_new = exclusive_output
    try:
        return parent.aggregate(source)
    finally:
        parent.write_new = original_write


def run(out):
    verify(out)
    write_new(out/'environment.json', {'host':os.uname().nodename,'python':sys.version,'condor_ad':os.environ.get('_CONDOR_JOB_AD'),
        'physical_calls':0,'read_only_parent':str(PRIMARY)})
    # The internal saved-trial audit imports digest locally. Frozen WB90 remains
    # unchanged; adapt that binding in this worker process only.
    import alignment.wb90_measurement_contract as io
    io.digest = digest
    recovery_aggregate(PRIMARY,out)
    write_new(out/'lambda_1_preflight.json', {'gate':'PASS','complete_saved_matrix_and_reference_audit':True,
                                               'new_physical_calls':0,'new_reference_integrations':0})
    verify(out)


def submit(out):
    verify(out)
    sub = out/'wb102_recovery.sub'
    with sub.open('x') as stream:
        stream.write('\n'.join(['universe = vanilla',f'executable = {ROOT}/scripts/run_wb102_recovery_condor.sh',f'arguments = {ROOT} {out}',
            f'output = {out}/condor.$(ClusterId).out',f'error = {out}/condor.$(ClusterId).err',f'log = {out}/condor.$(ClusterId).log',
            'request_cpus = 1','request_memory = 12000','request_disk = 1000000','+JobFlavour = "tomorrow"',
            'getenv = False','should_transfer_files = NO','on_exit_remove = True','periodic_remove = (NumJobStarts > 1)','queue 1','']))
    cmd = 'source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(sub))
    r = subprocess.run(['bash','-c',cmd],text=True,capture_output=True)
    write_new(out/'submission.json', {'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
    print(r.stdout); print(r.stderr,file=sys.stderr)
    if r.returncode: raise RuntimeError('recovery submission failed')


def history(out,cluster,expected):
    cmd='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null && condor_history '+str(cluster)+' -limit 1 -json'
    r=subprocess.run(['bash','-c',cmd],capture_output=True,text=True,check=True,timeout=55)
    ads=json.loads(r.stdout)
    if len(ads)!=1 or ads[0].get('JobStatus')!=4 or ads[0].get('ExitCode')!=expected or ads[0].get('NumJobStarts')!=1 or ads[0].get('ExitBySignal',False):
        raise ValueError('scheduler terminal '+str(cluster))
    return ads


def report(out):
    verify(out)
    original=history(out,1195061,1)
    cluster=re.search(r'cluster (\d+)',read_public(out/'submission.json')['stdout']).group(1)
    recovered=history(out,cluster,0)
    if read_public(out/'worker_exit.json')['exit_code']!=0: raise ValueError('recovery worker failed')
    write_new(out/'original_scheduler_history.json',original)
    write_new(out/'scheduler_history.json',recovered)
    write_new(out/'scheduler_terminal_audit.json',{'gate':'PASS','original_execution':'FAIL','recovery_execution':'PASS',
        'cluster':cluster,'original_cluster':1195061,'new_physical_calls':0,'original_wall_seconds':original[0]['RemoteWallClockTime'],
        'recovery_wall_seconds':recovered[0]['RemoteWallClockTime'],'worker':recovered[0].get('LastRemoteHost'),
        'RequestCpus':recovered[0].get('RequestCpus'),'RequestMemory':recovered[0].get('RequestMemory'),'ShouldTransferFiles':recovered[0].get('ShouldTransferFiles')})


def seal(out):
    frozen=verify(out);summary=read_public(out/'summary.json')
    for name in ('trial_integrity_audit','scalar_summary_audit','scheduler_terminal_audit'):
        if read_public(out/(name+'.json'))['gate']!='PASS': raise ValueError('incomplete recovery audit '+name)
    workbook=ROOT/'workbook/2026-10-02_102_四站五维Jacobian扰动尺度平台诊断.md'
    paths=[p for folder in (PRIMARY,out) for p in folder.rglob('*') if p.is_file()]+[workbook]
    value={'schema':'wb102_result_integrity_v1','original_execution':'FAIL','recovery_execution':'PASS',
        'implementation_commit':frozen['original_implementation_commit'],'recovery_commit':frozen['commit'],
        'execution_contract':summary['execution_contract'],'hypothesis':summary['hypothesis'],'new_physical_calls':0,
        'qualification':'NOT_EVALUATED','physics_screening':'NOT_EVALUATED','final_oracle':'NOT_EVALUATED',
        'held_out_access':False,'population':1,'production_backend_changed':False,
        'artifacts':{str(p.relative_to(ROOT)):digest(p) for p in paths},'workbook_sha256':digest(workbook)}
    write_new(out/'result_integrity.json',value)
    write_new(ROOT/'docs/wb102_jacobian_scale_result_manifest.json',value|{'result_integrity_sha256':digest(out/'result_integrity.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','verify','run','submit','report','seal'));p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();out=a.output_root.resolve()
    if out!=OUT: raise ValueError('exclusive recovery output')
    globals()[a.action](out)
