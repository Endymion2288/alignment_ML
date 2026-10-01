#!/usr/bin/env python3
"""Repair two export-reader key names only. No physical calls or threshold changes."""
import argparse,os,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest
import wb99_contract as original
import wb97_contract as local

PARENT=ROOT/'outputs/mc24_four_station_wb99_direction_step_doubling_v1'
SOURCE=ROOT/'alignment/wb99_direction_step_doubling.py'
def corrected_source():
    text=SOURCE.read_text()
    for suffix in ('position_closure_mm','direction_closure'):
        old="row[name+'_%s']"%suffix
        new="row[('accepted' if name=='accepted_branch' else name)+'_%s']"%suffix
        if text.count(old)!=1:raise ValueError('exact recovery patch identity')
        text=text.replace(old,new)
    return text
def load_corrected():
    namespace={};exec(compile(corrected_source(),'WB99_key_recovery_only.py','exec'),namespace)
    return namespace['aggregate']
def freeze(out):
    prior=original.verify(PARENT)
    if read_public(PARENT/'worker_exit.json')['exit_code']!=1 or read_public(PARENT/'execution_error.json')['error']!="KeyError('accepted_branch_position_closure_mm')":raise ValueError('wrong recovery cause')
    out.mkdir(exist_ok=False);(out/'event').mkdir()
    for name in ('controls.json','protocol.json','fixture.json','inventory.json','binary_manifest.json','control_binary_manifest.json','generated_source_manifest.json','environment.json','controls'):
        (out/name).symlink_to(PARENT/name)
    for name in ('acts.json','acts.json.doubling.ndjson','fixture.json','athena.log','athena_manifest.json','command.json'):(out/'event'/name).symlink_to(PARENT/'event'/name)
    for name in ('isolated_source','identity_payload'):(out/name).symlink_to(PARENT/name)
    with (out/'corrected_aggregation.py').open('x') as f:f.write(corrected_source())
    paths=[Path(x) for x in prior['hashes']]+[SOURCE,Path(__file__),ROOT/'scripts/run_wb99_recovery_condor.sh',ROOT/'tests/test_wb99_aggregation_recovery.py',out/'corrected_aggregation.py',
      PARENT/'freeze.json',PARENT/'worker_exit.json',PARENT/'execution_error.json',PARENT/'event/acts.json',PARENT/'event/acts.json.doubling.ndjson',PARENT/'controls.json',PARENT/'event/athena.log',PARENT/'binary_manifest.json']
    write_new(out/'freeze.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
      'original_implementation_commit':prior['commit'],'hashes':{str(x):digest(x) for x in paths},'parent':str(PARENT),
      'only_change':'two closure-field reader key names; equations, units, controls, budgets and decisions unchanged',
      'physical_rerun':False,'qualification':'NOT_EVALUATED','held_out_access':False})
def run(out):
    original.verify(out);original.verify(PARENT)
    if (out/'corrected_aggregation.py').read_text()!=corrected_source():raise ValueError('patch changed')
    import numpy as np
    namespace={};exec(compile((out/'corrected_aggregation.py').read_text(),str(out/'corrected_aggregation.py'),'exec'),namespace)
    raw=read_public(out/'event/acts.json');prior=read_public(local.WB95/'summary.json');log=(out/'event/athena.log').read_text(errors='replace')
    import re
    maps=[Path(x.strip()) for x in re.findall(r'Initialized the field map from\s+([^\n]+)',log)]
    libs=[Path(x) for x in raw['loaded_libraries'] if any(y in x for y in ('WB99Diagnostic','FaserActs','ActsCore','MagField'))]
    bm=read_public(out/'binary_manifest.json')
    if not maps or Path(bm['binary']) not in libs or digest(Path(bm['binary']))!=bm['sha256'] or str(PARENT/'identity_payload/tracker_alignment.sqlite') not in log or 'Using FASER magnetic field service' not in log:raise ValueError('physical path/binary')
    for path in maps:
        if str(path) not in prior['field_map_hashes'] or digest(path)!=prior['field_map_hashes'][str(path)]:raise ValueError('field map changed')
    for path in libs:
        if str(path) in prior['loaded_scientific_libraries'] and digest(path)!=prior['loaded_scientific_libraries'][str(path)]:raise ValueError('official library changed')
    for name,h in read_public(out/'generated_source_manifest.json').items():
        if digest(out/'isolated_source'/name)!=h:raise ValueError('component source changed')
    nodes=np.fromfile(local.NODES,dtype='<i2').reshape(81,81,861,3)
    s=namespace['aggregate'](out,read_public(out/'inventory.json'),read_public(out/'protocol.json'),original.WB97,read_public(local.OLD),nodes)
    s.update(loaded_scientific_libraries={str(x):digest(x) for x in libs},field_map_hashes={str(x):digest(x) for x in maps},
      artifact_hashes={name:digest(out/name) for name in ('controls.json','event/acts.json','event/acts.json.doubling.ndjson','metrics.ndjson','freeze.json','event/athena.log')},
      original_execution={'cluster':1172309,'exit_code':1,'cause':read_public(PARENT/'execution_error.json')['error'],'physical_fidelity':'PASS','raw_coverage':'156/135355','rerun':False},
      overlap_errors=re.findall(r'Layers are overlapping at: ([^\n]+)',log))
    write_new(out/'summary.json',s);original.verify(out);original.verify(PARENT)
def submit(out):
    original.verify(out);sub=out/'wb99_recovery.sub'
    with sub.open('x') as f:f.write('\n'.join(['universe = vanilla',f'executable = {ROOT}/scripts/run_wb99_recovery_condor.sh',f'arguments = {ROOT} {out}',
      f'output = {out}/condor.$(ClusterId).out',f'error = {out}/condor.$(ClusterId).err',f'log = {out}/condor.$(ClusterId).log',
      'request_cpus = 1','request_memory = 8000','request_disk = 8000000','+JobFlavour = "workday"','getenv = False','should_transfer_files = NO',
      'on_exit_remove = True','periodic_remove = (NumJobStarts > 1)','queue 1','']))
    cmd='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(sub))
    r=subprocess.run(['bash','-c',cmd],capture_output=True,text=True);write_new(out/'submission.json',{'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});print(r.stdout)
    if r.returncode:raise RuntimeError(r.stderr)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','submit']);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();out=a.output_root.resolve()
    if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb99_aggregation_recovery_'):p.error('exclusive recovery output')
    {'freeze':freeze,'run':run,'submit':submit}[a.action](out)
