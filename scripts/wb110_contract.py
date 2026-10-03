#!/usr/bin/env python3
"""Single seen-event geometry topology audit, with no propagation."""
import argparse,hashlib,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from wb100_contract import digest
from wb92_contract import ACTS,EXTERNAL,run as command_run
from wb109_contract import OUT as BASE
from wb107_contract import verify
from wb110_sources import files,athena_source

OUT=ROOT/'outputs/mc24_four_station_wb110_boundary_topology_v1'
PREFLIGHT=ROOT/'outputs/mc24_four_station_wb110_tool_compile_preflight_v2'
WORKBOOK=ROOT/'workbook/2026-10-03_110_四站内部volume边界连接前瞻审计.md'

def probe():
    rows=[json.loads(x) for x in (BASE/'event/acts.json.calls.ndjson').read_text().splitlines()]
    checks=[r for r in rows if r.get('call_id')==124 and r['record']=='end_world_check']
    if len([r for r in checks if r['returned']])!=1 or not checks[-1]['returned']:raise ValueError('historical terminal abort')
    result=checks[-1].copy();result['last_nonnull_volume_name']=checks[-2]['current_volume_name']
    return result

def compile_check():
    PREFLIGHT.mkdir(exist_ok=False);generated=files()
    for name,content in generated.items():
        p=PREFLIGHT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as f:f.write(content)
    flags=BASE/'isolated_build/WB107Diagnostic/CMakeFiles/WB107Diagnostic.dir/flags.make'
    lines=flags.read_text().splitlines()
    parse=lambda key:shlex.split(next(x.split('=',1)[1] for x in lines if x.startswith(key+' =')))
    includes=[('-I'+str(PREFLIGHT/'WB110Diagnostic')) if x=='-I'+str(BASE/'isolated_source/WB107Diagnostic') else x for x in parse('CXX_INCLUDES')]
    defines=[x.replace('WB107Diagnostic','WB110Diagnostic') for x in parse('CXX_DEFINES')]
    cmd=['g++','-std=c++20','-fsyntax-only',*defines,*includes,str(PREFLIGHT/'WB110Diagnostic/BoundaryTopology.cxx')]
    with (PREFLIGHT/'compile.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    write_new(PREFLIGHT/'receipt.json',{'command':cmd,'returncode':r.returncode,'scope':'whole geometry-only algorithm; no link/event/propagation',
      'source_hashes':{k:digest(PREFLIGHT/k) for k in generated},'flags_sha256':digest(flags),
      'compiler':subprocess.check_output(['which','g++'],text=True).strip()})
    if r.returncode:raise RuntimeError('compile-only check failed')

def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='4station':raise ValueError('branch')
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip():raise ValueError('uncommitted tracked changes')
    hashes=verify(BASE)['hashes'].copy()
    result=read_public(ROOT/'docs/wb109_world_abort_result_manifest.json')
    if (result['execution_contract'],result['classification'],result['actual_outer_world_exit'])!=('PASS','END_OF_WORLD_BEFORE_TARGET','CONTRADICTED_FOR_THIS_CALL'):raise ValueError('historical status')
    for name,h in result['artifacts'].items():
        if digest(ROOT/name)!=h:raise ValueError('historical artifact '+name)
        hashes[str(ROOT/name)]=h
    generated=files();receipt=read_public(PREFLIGHT/'receipt.json')
    if receipt['returncode']!=0:raise ValueError('compile gate')
    for name,text in generated.items():
        if hashlib.sha256(text.encode()).hexdigest()!=receipt['source_hashes'][name]:raise ValueError('compile source differs')
    paths=[ROOT/'scripts/wb110_contract.py',ROOT/'scripts/wb110_sources.py',ROOT/'scripts/audit_wb110_topology.py',
      ROOT/'scripts/run_wb110_condor.sh',ROOT/'research/wb110/BoundaryTopology.cxx',ROOT/'tests/test_wb110_topology.py',
      ROOT/'docs/wb109_world_abort_result_manifest.json',BASE/'result_integrity.json',
      BASE/'isolated_build/WB107Diagnostic/CMakeFiles/WB107Diagnostic.dir/flags.make',Path(receipt['compiler'])]
    paths+=list(PREFLIGHT.rglob('*'))
    paths+=list((ROOT/'outputs/mc24_four_station_wb110_tool_compile_preflight_v1').rglob('*'))
    paths+=[ACTS/'include'/p for p in ('Acts/Geometry/TrackingVolume.hpp','Acts/Geometry/TrackingGeometry.hpp',
      'Acts/Geometry/BoundarySurfaceT.hpp','Acts/Geometry/Volume.hpp','Acts/Geometry/VolumeBounds.hpp','Acts/Surfaces/RectangleBounds.hpp')]
    paths+=[EXTERNAL/'Tracking/Acts/FaserActsGeometryInterfaces/FaserActsGeometryInterfaces/IFaserActsTrackingGeometryTool.h',
      EXTERNAL/'Tracking/Acts/FaserActsGeometry/FaserActsGeometry/FaserActsGeometryContext.h']
    for p in paths:
        if p.is_dir():continue
        hashes[str(p)]=digest(p)
    f=read_public(BASE/'fixture.json')
    if (f['index'],f['ordinal'],f['actual_run'],f['actual_event'])!=(12,2268,100044,2268):raise ValueError('allowlist')
    out.mkdir(exist_ok=False);shutil.copyfile(WORKBOOK,out/'contract_workbook.md')
    write_new(out/'fixture.json',f);write_new(out/'probe.json',probe())
    write_new(out/'generation_expectation.json',{'files':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in generated.items()},
      'athena_sha256':hashlib.sha256(athena_source().encode()).hexdigest()})
    for n in ('fixture.json','probe.json','contract_workbook.md','generation_expectation.json'):hashes[str(out/n)]=digest(out/n)
    write_new(out/'freeze.json',{'schema':'wb110_geometry_topology_freeze_v1','hashes':hashes,'branch':'4station',
      'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'population':1,'new_propagation_calls':0,'algorithm_field_queries':0,'held_out_access':False,'qualification':'NOT_EVALUATED'})

def build(out):
    source=out/'isolated_source';source.mkdir(exist_ok=False);generated=files()
    expected=read_public(out/'generation_expectation.json')
    for name,text in generated.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:f.write(text)
        if digest(path)!=expected['files'][name]:raise ValueError('generation identity')
    write_new(out/'generated_source_manifest.json',{k:digest(source/k) for k in generated})
    command_run(['cmake','-S',str(source),'-B',str(out/'isolated_build'),'-DCalypso_DIR='+str(EXTERNAL/'run/cmake')],out,out/'configure.log')
    command_run(['cmake','--build',str(out/'isolated_build'),'-j','1'],out,out/'build.log')
    binary=out/'isolated_build/x86_64-el9-gcc13-opt/lib/libWB110Diagnostic.so'
    write_new(out/'binary_manifest.json',{'binary':str(binary),'sha256':digest(binary)})
    return binary

def run(out):
    verify(out);(out/'execution_lock').mkdir(exist_ok=False)
    fixture=read_public(out/'fixture.json');raw=Path(fixture['input_xaod']);before=raw.stat()
    h=digest(raw);after=raw.stat();previous=read_public(BASE/'raw_identity.json')
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('raw changed while hashing')
    if h!=previous['sha256'] or fixture['source_stat']!={'bytes':after.st_size,'mtime_ns':after.st_mtime_ns}:raise ValueError('raw identity')
    write_new(out/'raw_identity.json',{'path':str(raw),'sha256':h,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'timing':'before build and single seen event context access'})
    binary=build(out);shutil.copytree(BASE/'identity_payload',out/'identity_payload')
    event=out/'event';event.mkdir(exist_ok=False)
    for n in ('fixture.json','probe.json'):shutil.copyfile(out/n,event/n)
    with (event/'athena.py').open('x') as f:f.write(athena_source())
    if digest(event/'athena.py')!=read_public(out/'generation_expectation.json')['athena_sha256']:raise ValueError('runner identity')
    platform=binary.parent.parent
    cmd='\n'.join(['export PYTHONPATH='+shlex.quote(str(ROOT))+':"${PYTHONPATH:-}"',
      'source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso',
      'source '+shlex.quote(str(platform/'setup.sh')),
      'export LD_LIBRARY_PATH='+shlex.quote(str(platform/'lib'))+':"$LD_LIBRARY_PATH"',
      'python '+shlex.quote(str(event/'athena.py'))+' --work-dir '+shlex.quote(str(event))+' --sqlite '+shlex.quote(str(out/'identity_payload/tracker_alignment.sqlite'))])
    write_new(event/'command.json',{'script':cmd,'athena_sha256':digest(event/'athena.py')})
    with (event/'athena.log').open('x') as log:r=subprocess.run(['bash','-c',cmd],cwd=event,stdout=log,stderr=subprocess.STDOUT)
    write_new(out/'athena_exit.json',{'exit_code':r.returncode,'host':os.uname().nodename})
    verify(out)
    from audit_wb110_topology import audit
    write_new(out/'summary.json',audit(out))

def submit(out):
    verify(out)
    setup='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null'
    q=subprocess.check_output(['bash','-c',setup+' && condor_q -json'],text=True,timeout=55)
    totals=subprocess.check_output(['bash','-c',setup+' && condor_q -totals'],text=True,timeout=55)
    slots=subprocess.check_output(['bash','-c',setup+' && condor_status -constraint \'State=="Unclaimed" && Activity=="Idle"\' -af Name'],text=True,timeout=55)
    write_new(out/'scheduler_preflight.json',{'queue':json.loads(q) if q.strip() else [],'raw_queue_json':q,'queue_totals':totals,
      'idle_slots':len(slots.splitlines()),'schedd':'bigbird24'})
    if not slots.strip():raise RuntimeError('no idle slots; no submission')
    sub=out/'wb110.sub'
    content=(BASE/'wb109.sub').read_text().replace(str(BASE),str(out)).replace('run_wb109_condor.sh','run_wb110_condor.sh')
    with sub.open('x') as f:f.write(content)
    r=subprocess.run(['bash','-c',setup+' && condor_submit '+shlex.quote(str(sub))],capture_output=True,text=True,timeout=55)
    write_new(out/'submission.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'submit_sha256':digest(sub)})
    print(r.stdout,r.stderr)
    if r.returncode:raise RuntimeError('submission failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('compile','freeze','verify','run','submit'));p.add_argument('--output-root',type=Path)
    a=p.parse_args()
    if a.action=='compile':
        if a.output_root:p.error('compile uses exclusive directory')
        compile_check()
    else:
        if a.output_root is None or a.output_root.resolve()!=OUT:p.error('exclusive WB110 output root required')
        if a.action=='verify':verify(OUT)
        else:globals()[a.action](OUT)
