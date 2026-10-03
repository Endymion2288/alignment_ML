#!/usr/bin/env python3
"""Single seen-event passive ACTS bound-conversion attribution."""
import argparse,hashlib,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from wb100_contract import digest
from wb92_contract import ACTS,EXTERNAL,run as command_run
from wb107_sources import files,athena_source,OFFICIAL

BASE=ROOT/'outputs/mc24_four_station_wb106_surface_trace_v1'
OUT=ROOT/'outputs/mc24_four_station_wb107_bound_attribution_v1'
WORKBOOK=ROOT/'workbook/2026-10-03_107_四站ACTS终态绑定错误归因前瞻诊断.md'

def verify(out):
    frozen=read_public(out/'freeze.json')
    for path,h in frozen['hashes'].items():
        if digest(path)!=h:raise ValueError('frozen identity '+path)
    return frozen

def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='4station':raise ValueError('branch')
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip():raise ValueError('uncommitted tracked changes')
    parent=read_public(BASE/'freeze.json');hashes=parent['hashes'].copy()
    for path,h in hashes.items():
        if digest(path)!=h:raise ValueError('parent identity '+path)
    result=read_public(ROOT/'docs/wb106_surface_trace_result_manifest.json')
    for path,h in result['artifacts'].items():
        if digest(ROOT/path)!=h:raise ValueError('parent result '+path)
        hashes[str(ROOT/path)]=h
    if result['classification']!='UNKNOWN':raise ValueError('historical classification')
    paths=[ROOT/'docs/wb106_surface_trace_result_manifest.json',BASE/'result_integrity.json',
      ROOT/'scripts/wb107_contract.py',ROOT/'scripts/wb107_sources.py',ROOT/'scripts/audit_wb107_bound.py',
      ROOT/'scripts/run_wb107_condor.sh',ROOT/'research/wb107/BoundTrace.h',ROOT/'tests/test_wb107_bound.py',
      ROOT/'tests/wb107_stepper_compile.cxx',
      ROOT/'scripts/wb101_sources.py',ROOT/'scripts/wb106_contract.py',ROOT/'scripts/wb100_contract.py',
      ROOT/'scripts/wb92_contract.py',ROOT/'scripts/audit_wb106_trace.py',
      OFFICIAL/'FaserActsExtrapolationTool.h',OFFICIAL/'FaserActsExtrapolationTool.cxx']
    paths+=[ACTS/'include'/p for p in ('Acts/Propagator/Propagator.hpp','Acts/Propagator/Propagator.ipp',
      'Acts/Propagator/EigenStepper.hpp','Acts/Propagator/EigenStepper.ipp',
      'Acts/Definitions/Tolerance.hpp','Acts/Propagator/MaterialInteractor.hpp',
      'Acts/Propagator/StepperConcept.hpp','Acts/Utilities/TypeTraits.hpp',
      'Acts/Propagator/detail/ParameterTraits.hpp')]
    for path in paths:hashes[str(path)]=digest(path)
    f=read_public(BASE/'fixture.json')
    if (f['index'],f['ordinal'],f['actual_run'],f['actual_event'])!=(12,2268,100044,2268):raise ValueError('allowlist')
    # Validate all marker substitutions before creating an execution root.
    generated=files(); runner=athena_source()
    out.mkdir(exist_ok=False)
    shutil.copyfile(WORKBOOK,out/'contract_workbook.md');write_new(out/'fixture.json',f)
    for path in (out/'contract_workbook.md',out/'fixture.json'):hashes[str(path)]=digest(path)
    write_new(out/'generation_expectation.json',{'files':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in generated.items()},
      'athena_sha256':hashlib.sha256(runner.encode()).hexdigest()})
    hashes[str(out/'generation_expectation.json')]=digest(out/'generation_expectation.json')
    write_new(out/'freeze.json',{'schema':'wb107_bound_freeze_v1','hashes':hashes,'branch':'4station',
      'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'population':1,'max_official_calls':101,'max_diagnostic_calls':101,'held_out_access':False,'qualification':'NOT_EVALUATED'})

def build(out):
    source=out/'isolated_source';source.mkdir(exist_ok=False)
    generated=files();expected=read_public(out/'generation_expectation.json')
    for name,content in generated.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:stream.write(content)
        if digest(path)!=expected['files'][name]:raise ValueError('generated source differs')
    write_new(out/'generated_source_manifest.json',{name:digest(source/name) for name in generated})
    command_run(['cmake','-S',str(source),'-B',str(out/'isolated_build'),'-DCalypso_DIR='+str(EXTERNAL/'run/cmake')],out,out/'configure.log')
    command_run(['cmake','--build',str(out/'isolated_build'),'-j','1'],out,out/'build.log')
    binary=out/'isolated_build/x86_64-el9-gcc13-opt/lib/libWB107Diagnostic.so'
    write_new(out/'binary_manifest.json',{'binary':str(binary),'sha256':digest(binary)})
    return binary

def run(out):
    verify(out);(out/'execution_lock').mkdir(exist_ok=False)
    f=read_public(out/'fixture.json');raw=Path(f['input_xaod']);before=raw.stat()
    previous=read_public(BASE/'raw_identity.json')
    actual=digest(raw);after=raw.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('raw changed while hashing')
    if actual!=previous['sha256'] or {'bytes':after.st_size,'mtime_ns':after.st_mtime_ns}!=f['source_stat']:raise ValueError('raw identity changed')
    write_new(out/'raw_identity.json',{'path':str(raw),'sha256':actual,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'timing':'before build and event access'})
    binary=build(out)
    shutil.copytree(BASE/'identity_payload',out/'identity_payload')
    event=out/'event';event.mkdir(exist_ok=False)
    shutil.copyfile(out/'fixture.json',event/'fixture.json')
    with (event/'athena.py').open('x') as stream:stream.write(athena_source())
    if digest(event/'athena.py')!=read_public(out/'generation_expectation.json')['athena_sha256']:raise ValueError('runner source differs')
    platform=binary.parent.parent
    cmd='\n'.join(['export PYTHONPATH='+shlex.quote(str(ROOT))+':"${PYTHONPATH:-}"',
      'source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso',
      'source '+shlex.quote(str(platform/'setup.sh')),
      'export LD_LIBRARY_PATH='+shlex.quote(str(platform/'lib'))+':"$LD_LIBRARY_PATH"',
      'python '+shlex.quote(str(event/'athena.py'))+' --work-dir '+shlex.quote(str(event))+
      ' --sqlite '+shlex.quote(str(out/'identity_payload/tracker_alignment.sqlite'))])
    write_new(event/'command.json',{'script':cmd,'athena_sha256':digest(event/'athena.py')})
    with (event/'athena.log').open('x') as log:
        r=subprocess.run(['bash','-c',cmd],cwd=event,stdout=log,stderr=subprocess.STDOUT)
    write_new(out/'athena_exit.json',{'exit_code':r.returncode,'host':os.uname().nodename})
    verify(out)
    from audit_wb107_bound import audit
    write_new(out/'summary.json',audit(out))

def submit(out):
    verify(out)
    setup='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null'
    q=subprocess.check_output(['bash','-c',setup+' && condor_q -json'],text=True,timeout=55)
    # CERN condor_q emits an empty string for zero jobs; capture totals too.
    totals=subprocess.check_output(['bash','-c',setup+' && condor_q -totals'],text=True,timeout=55)
    slots=subprocess.check_output(['bash','-c',setup+' && condor_status -constraint \'State=="Unclaimed" && Activity=="Idle"\' -af Name'],text=True,timeout=55)
    queue=json.loads(q) if q.strip() else []
    write_new(out/'scheduler_preflight.json',{'queue':queue,'raw_queue_json':q,'queue_totals':totals,
      'idle_slots':len(slots.splitlines()),'schedd':'bigbird24'})
    if not slots.strip():raise RuntimeError('no idle slots; no submission')
    sub=out/'wb107.sub'
    with sub.open('x') as stream:
        stream.write('\n'.join(['universe = vanilla',f'executable = {ROOT}/scripts/run_wb107_condor.sh',
          f'arguments = {ROOT} {out}',f'output = {out}/condor.$(ClusterId).out',
          f'error = {out}/condor.$(ClusterId).err',f'log = {out}/condor.$(ClusterId).log',
          'request_cpus = 1','request_memory = 8000','request_disk = 8000000',
          '+JobFlavour = "workday"','getenv = False','should_transfer_files = NO',
          'on_exit_remove = True','periodic_remove = (NumJobStarts > 1)','queue 1','']))
    r=subprocess.run(['bash','-c',setup+' && condor_submit '+shlex.quote(str(sub))],capture_output=True,text=True,timeout=55)
    write_new(out/'submission.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'submit_sha256':digest(sub)})
    print(r.stdout,r.stderr)
    if r.returncode:raise RuntimeError('submission failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','verify','run','submit'));p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();out=a.output_root.resolve()
    if out!=OUT:p.error('exclusive WB107 root required')
    globals()[a.action](out)
