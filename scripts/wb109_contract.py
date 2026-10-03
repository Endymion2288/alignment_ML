#!/usr/bin/env python3
"""Bounded world-aborter observation with frozen physical controls."""
import argparse,hashlib,json,shlex,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import wb107_contract as parent
from wb108_contract import OUT as BASE
from wb109_sources import files,athena_source
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from wb100_contract import digest
from wb92_contract import ACTS,EXTERNAL

OUT=ROOT/'outputs/mc24_four_station_wb109_world_abort_trace_v1'
PREFLIGHT=ROOT/'outputs/mc24_four_station_wb109_tool_compile_preflight_v1'
WORKBOOK=ROOT/'workbook/2026-10-03_109_四站ACTS世界边界提前终止前瞻诊断.md'

def compile_check():
    PREFLIGHT.mkdir(exist_ok=False)
    generated=files()
    for name,content in generated.items():
        p=PREFLIGHT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as stream:stream.write(content)
    flags=BASE/'isolated_build/WB107Diagnostic/CMakeFiles/WB107Diagnostic.dir/flags.make'
    lines=flags.read_text().splitlines()
    parse=lambda key:shlex.split(next(x.split('=',1)[1] for x in lines if x.startswith(key+' =')))
    includes=[('-I'+str(PREFLIGHT/'WB107Diagnostic')) if x=='-I'+str(BASE/'isolated_source/WB107Diagnostic') else x for x in parse('CXX_INCLUDES')]
    commands=[]
    for source in ('WB107ExtrapolationTool.cxx','CommonSeedAudit.cxx'):
        cmd=['g++','-std=c++20','-fsyntax-only',*parse('CXX_DEFINES'),*includes,str(PREFLIGHT/'WB107Diagnostic'/source)]
        with (PREFLIGHT/(source+'.compile.log')).open('x') as log:
            r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        commands.append({'command':cmd,'returncode':r.returncode})
        if r.returncode:break
    write_new(PREFLIGHT/'receipt.json',{'scope':'whole tool and algorithm translation units; no link/execution',
      'commands':commands,'returncode':commands[-1]['returncode'],'compiler':subprocess.check_output(['which','g++'],text=True).strip(),
      'source_hashes':{k:digest(PREFLIGHT/k) for k in generated},'flags_sha256':digest(flags)})
    if commands[-1]['returncode'] or len(commands)!=2:raise RuntimeError('compile-only preflight failed')

def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='4station':raise ValueError('branch')
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip():raise ValueError('uncommitted tracked change')
    hashes=parent.verify(BASE)['hashes'].copy()
    result=read_public(ROOT/'docs/wb108_bound_attribution_result_manifest.json')
    if (result['recovery_execution'],result['classification'],result['deeper_navigation_or_field_mechanism'])!=('PASS','FINAL_TARGET_BOUND_CONVERSION','UNKNOWN'):raise ValueError('historical result')
    for path,h in result['artifacts'].items():
        if digest(ROOT/path)!=h:raise ValueError('historical artifact '+path)
        hashes[str(ROOT/path)]=h
    generated=files();pre=read_public(PREFLIGHT/'receipt.json')
    if pre['returncode']!=0 or len(pre['commands'])!=2:raise ValueError('whole component syntax check')
    for name,content in generated.items():
        if hashlib.sha256(content.encode()).hexdigest()!=pre['source_hashes'][name]:raise ValueError('preflight source differs')
    paths=[ROOT/'scripts/wb109_contract.py',ROOT/'scripts/wb109_sources.py',ROOT/'scripts/audit_wb109_abort.py',
      ROOT/'scripts/run_wb109_condor.sh',ROOT/'research/wb109/AbortTrace.h',ROOT/'tests/test_wb109_abort.py',
      ROOT/'docs/wb108_bound_attribution_result_manifest.json',BASE/'result_integrity.json',
      BASE/'isolated_build/WB107Diagnostic/CMakeFiles/WB107Diagnostic.dir/flags.make',Path(pre['compiler'])]
    paths+=list(PREFLIGHT.rglob('*'))
    paths+=[ACTS/'include'/p for p in ('Acts/Propagator/StandardAborters.hpp','Acts/Propagator/AbortList.hpp',
      'Acts/Propagator/detail/abort_list_implementation.hpp','Acts/Propagator/Navigator.hpp','Acts/Propagator/Propagator.hpp',
      'Acts/Propagator/Propagator.ipp','Acts/Geometry/Volume.hpp','Acts/Geometry/VolumeBounds.hpp',
      'Acts/Geometry/CuboidVolumeBounds.hpp','Acts/Geometry/TrackingVolume.hpp','Acts/Geometry/BoundarySurfaceT.hpp')]
    paths+=[EXTERNAL/'Tracking/Acts/FaserActsGeometry/src'/p for p in ('FaserActsTrackingGeometrySvc.cxx','FaserActsLayerBuilder.cxx','CuboidVolumeBuilder.cxx')]
    for path in paths:
        if path.is_dir():continue
        hashes[str(path)]=digest(path)
    out.mkdir(exist_ok=False)
    shutil.copyfile(WORKBOOK,out/'contract_workbook.md');shutil.copyfile(BASE/'fixture.json',out/'fixture.json')
    write_new(out/'generation_expectation.json',{'files':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in generated.items()},
      'athena_sha256':hashlib.sha256(athena_source().encode()).hexdigest()})
    for n in ('contract_workbook.md','fixture.json','generation_expectation.json'):hashes[str(out/n)]=digest(out/n)
    write_new(out/'freeze.json',{'schema':'wb109_world_abort_freeze_v1','hashes':hashes,'branch':'4station',
      'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'population':1,'max_official_calls':101,'max_diagnostic_calls':101,'held_out_access':False,'qualification':'NOT_EVALUATED'})

def run(out):
    import audit_wb107_bound as old_audit
    from audit_wb109_abort import audit
    original_files,original_audit=parent.files,old_audit.audit
    try:
        parent.files=files;old_audit.audit=audit
        parent.run(out)
    finally:parent.files=original_files;old_audit.audit=original_audit
    write_new(out/'diagnostic_receipt.json',{'schema':'wb109_diagnostic_receipt_v1','summary_sha256':digest(out/'summary.json'),
      'base':'WB108','only_new_observation':'original EndOfWorldReached evaluation wrapper',
      'production_backend_changed':False,'held_out_access':False})

def submit(out):
    parent.verify(out)
    setup='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null'
    q=subprocess.check_output(['bash','-c',setup+' && condor_q -json'],text=True,timeout=55)
    totals=subprocess.check_output(['bash','-c',setup+' && condor_q -totals'],text=True,timeout=55)
    slots=subprocess.check_output(['bash','-c',setup+' && condor_status -constraint \'State=="Unclaimed" && Activity=="Idle"\' -af Name'],text=True,timeout=55)
    write_new(out/'scheduler_preflight.json',{'queue':json.loads(q) if q.strip() else [],'raw_queue_json':q,
      'queue_totals':totals,'idle_slots':len(slots.splitlines()),'schedd':'bigbird24'})
    if not slots.strip():raise RuntimeError('no idle slots; no submission')
    sub=out/'wb109.sub'
    content=(BASE/'wb108.sub').read_text().replace(str(BASE),str(out)).replace('run_wb108_condor.sh','run_wb109_condor.sh')
    with sub.open('x') as f:f.write(content)
    r=subprocess.run(['bash','-c',setup+' && condor_submit '+shlex.quote(str(sub))],capture_output=True,text=True,timeout=55)
    write_new(out/'submission.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'submit_sha256':digest(sub)})
    print(r.stdout,r.stderr)
    if r.returncode:raise RuntimeError('submission failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('compile','freeze','verify','run','submit'));p.add_argument('--output-root',type=Path)
    a=p.parse_args()
    if a.action=='compile':
        if a.output_root:p.error('compile uses exclusive preflight directory')
        compile_check()
    else:
        if a.output_root is None or a.output_root.resolve()!=OUT:p.error('exclusive WB109 output root required')
        if a.action=='verify':parent.verify(OUT)
        else:globals()[a.action](OUT)
