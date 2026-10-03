#!/usr/bin/env python3
"""Versioned build recovery; never edits frozen WB107 code or artifacts."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import wb107_contract as parent
from wb108_sources import files
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from wb100_contract import digest

OLD=parent.OUT
OUT=ROOT/'outputs/mc24_four_station_wb108_bound_build_recovery_v1'
WORKBOOK=ROOT/'workbook/2026-10-03_108_四站终态绑定诊断编译恢复前瞻合同.md'

def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='4station':raise ValueError('branch')
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip():raise ValueError('uncommitted tracked change')
    hashes=parent.verify(OLD)['hashes'].copy()
    result=read_public(ROOT/'docs/wb107_bound_attribution_result_manifest.json')
    if (result['execution_contract'],result['scientific_classification'],result['new_physical_calls'])!=('FAIL','UNKNOWN',0):raise ValueError('historical outcome')
    for path,h in result['artifacts'].items():
        if digest(ROOT/path)!=h:raise ValueError('historical artifact '+path)
        hashes[str(ROOT/path)]=h
    paths=[ROOT/'scripts/wb108_contract.py',ROOT/'scripts/wb108_sources.py',ROOT/'scripts/run_wb108_condor.sh',
      ROOT/'scripts/check_wb108_tool_compile.py',ROOT/'tests/test_wb108_recovery.py',
      ROOT/'docs/wb107_bound_attribution_result_manifest.json',OLD/'result_integrity.json',
      OLD/'isolated_build/WB107Diagnostic/CMakeFiles/WB107Diagnostic.dir/flags.make']
    generated=files()
    preflight=ROOT/'outputs/mc24_four_station_wb108_tool_compile_preflight_v1'
    receipt=read_public(preflight/'receipt.json')
    if receipt['returncode']!=0:raise ValueError('whole-tool compilation preflight failed')
    for name,content in generated.items():
        if hashlib.sha256(content.encode()).hexdigest()!=receipt['source_hashes'][name]:raise ValueError('preflight source differs '+name)
    paths+=[preflight/'receipt.json',preflight/'compile.log',Path(receipt['compiler'])]
    paths+=[preflight/name for name in generated]
    expectation=read_public(OLD/'generation_expectation.json')
    for name,content in generated.items():
        if name!='WB107Diagnostic/WB107ExtrapolationTool.h' and hashlib.sha256(content.encode()).hexdigest()!=expectation['files'][name]:
            raise ValueError('non-header scientific source changed '+name)
    for path in paths:hashes[str(path)]=digest(path)
    out.mkdir(exist_ok=False)
    shutil.copyfile(WORKBOOK,out/'contract_workbook.md')
    shutil.copyfile(OLD/'fixture.json',out/'fixture.json')
    write_new(out/'generation_expectation.json',{'files':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in generated.items()},
      'athena_sha256':expectation['athena_sha256']})
    for name in ('contract_workbook.md','fixture.json','generation_expectation.json'):hashes[str(out/name)]=digest(out/name)
    write_new(out/'freeze.json',{'schema':'wb108_build_recovery_freeze_v1','hashes':hashes,
      'branch':'4station','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'historical_execution':'FAIL','recovery_of':str(OLD),'population':1,'max_official_calls':101,
      'max_diagnostic_calls':101,'held_out_access':False,'qualification':'NOT_EVALUATED',
      'only_source_change':'two header interface identifiers restored to IFaserActsExtrapolationTool'})

def run(out):
    parent.files=files
    parent.run(out)
    write_new(out/'recovery_receipt.json',{'schema':'wb108_recovery_receipt_v1','original_execution':'FAIL',
      'recovery_summary_sha256':digest(out/'summary.json'),'scientific_contract':'unchanged WB107',
      'production_backend_changed':False,'held_out_access':False})

def submit(out):
    # Reuse frozen scheduler policy, while selecting the versioned worker.
    parent.verify(out)
    setup='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null'
    q=subprocess.check_output(['bash','-c',setup+' && condor_q -json'],text=True,timeout=55)
    totals=subprocess.check_output(['bash','-c',setup+' && condor_q -totals'],text=True,timeout=55)
    slots=subprocess.check_output(['bash','-c',setup+' && condor_status -constraint \'State=="Unclaimed" && Activity=="Idle"\' -af Name'],text=True,timeout=55)
    write_new(out/'scheduler_preflight.json',{'queue':json.loads(q) if q.strip() else [],'raw_queue_json':q,
      'queue_totals':totals,'idle_slots':len(slots.splitlines()),'schedd':'bigbird24'})
    if not slots.strip():raise RuntimeError('no idle slots; no submission')
    # Preserve exactly the previous submit policy with only worker/root paths changed.
    text=(OLD/'wb107.sub').read_text().replace(str(OLD),str(out)).replace('run_wb107_condor.sh','run_wb108_condor.sh')
    sub=out/'wb108.sub'
    with sub.open('x') as f:f.write(text)
    import shlex
    r=subprocess.run(['bash','-c',setup+' && condor_submit '+shlex.quote(str(sub))],capture_output=True,text=True,timeout=55)
    write_new(out/'submission.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'submit_sha256':digest(sub)})
    print(r.stdout,r.stderr)
    if r.returncode:raise RuntimeError('submission failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','verify','run','submit'));p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();out=a.output_root.resolve()
    if out!=OUT:p.error('exclusive WB108 output required')
    if a.action=='verify':parent.verify(out)
    else:globals()[a.action](out)
