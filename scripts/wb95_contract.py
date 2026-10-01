#!/usr/bin/env python3
"""Exclusive WB95 same-node float/double diagnostic; preserves WB94 and backend."""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,digest,read_public,write_new,FORBIDDEN
from alignment.wb95_field_precision import analyze
import wb94_contract as previous
from wb92_contract import run as command_run

PROTOCOL=ROOT/'configs/research_review/wp95_field_precision_contract.json'
WB94=ROOT/'outputs/mc24_four_station_wb94_field_boundary_v1'
SOURCES=[PROTOCOL,ROOT/'scripts/wb95_contract.py',ROOT/'scripts/wb95_athena.py',ROOT/'scripts/run_wb95_condor.sh',
  ROOT/'research/wb95/AuditBody.inc',ROOT/'research/wb95/PrecisionControl.h',ROOT/'research/wb95/CompensatedRK4.h',ROOT/'research/wb94/ReferenceRK4.h',
  ROOT/'alignment/wb95_field_precision.py',ROOT/'alignment/wb94_field_boundary.py',ROOT/'alignment/wb93_transport_error.py',
  ROOT/'scripts/wb94_contract.py',ROOT/'scripts/wb93_contract.py',ROOT/'scripts/wb92_contract.py',
  ROOT/'scripts/setup_environment.sh',ROOT/'alignment/wb90_measurement_contract.py']

def verify(out):
    f=read_public(out/'freeze.json')
    for path,h in f['hashes'].items():
        if digest(Path(path))!=h:raise ValueError('frozen identity changed: '+path)
    if digest(out/'fixture.json')!=f['fixture_sha256'] or read_public(out/'protocol.json')!=read_public(PROTOCOL):raise ValueError('fixture/protocol changed')
    return f

def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='4station':raise ValueError('4station required')
    prior=previous.verify(WB94);s=read_public(WB94/'summary.json')
    if s['execution_contract']!='PASS' or s['mechanism_hypothesis']!='UNKNOWN' or s['WB92_gate']!='FAIL':raise ValueError('WB94 prior changed')
    if digest(WB94/'event/acts.json')!=s['acts_sha256']:raise ValueError('prior saved values changed')
    f=copy.deepcopy(read_public(WB94/'fixture.json'))
    if any(x in f['input_xaod'].lower() for x in FORBIDDEN) or (f['index'],f['ordinal'],f['actual_run'],f['actual_event'])!=(0,2270,100043,2270):raise ValueError('unauthorized pilot')
    f['wb95_protocol']=read_public(PROTOCOL)
    paths=SOURCES+[Path(x) for x in prior['hashes']]
    paths+=[WB94/x for x in ('fixture.json','freeze.json','summary.json','event/acts.json','event/athena.log','result_integrity.json',
       'generated_source_manifest.json','isolated_source/CMakeLists.txt','isolated_source/WB94Diagnostic/CMakeLists.txt')]
    paths+=[Path(x) for x in s['loaded_scientific_libraries']]
    paths.append(previous.EXTERNAL/'MagneticField/MagFieldElements/MagFieldElements/BFieldVector.h')
    out.mkdir(parents=True,exist_ok=False);write_new(out/'fixture.json',f);write_new(out/'protocol.json',read_public(PROTOCOL))
    write_new(out/'freeze.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'branch':'4station','hashes':{str(x):digest(x) for x in paths},'fixture_sha256':digest(out/'fixture.json'),
      'population':1,'qualification':'NOT_EVALUATED','held_out_access':False})

def generated_source():
    s=previous.generated_source();prefix=s[:s.index('  SG::ReadCondHandleKey<FaserFieldMapCondObj>')]
    prefix=prefix.replace('namespace WB94','namespace WB95').replace('FieldBoundaryAudit','FieldPrecisionAudit')
    return '#include "PrecisionControl.h"\n#include "CompensatedRK4.h"\n'+prefix+(ROOT/'research/wb95/AuditBody.inc').read_text()+'\n};\n}\nDECLARE_COMPONENT(WB95::FieldPrecisionAudit)\n'

def build(out):
    source=out/'isolated_source';source.mkdir(exist_ok=False);(source/'WB95Diagnostic').mkdir()
    top=(WB94/'isolated_source/CMakeLists.txt').read_text().replace('WB94','WB95')
    component=(WB94/'isolated_source/WB94Diagnostic/CMakeLists.txt').read_text().replace('WB94','WB95').replace('FieldBoundaryAudit','FieldPrecisionAudit')
    files={'CMakeLists.txt':top,'WB95Diagnostic/CMakeLists.txt':component,
      'WB95Diagnostic/FieldPrecisionAudit.cxx':generated_source(),
      'WB95Diagnostic/ReferenceRK4.h':(ROOT/'research/wb94/ReferenceRK4.h').read_text(),
      'WB95Diagnostic/PrecisionControl.h':(ROOT/'research/wb95/PrecisionControl.h').read_text(),
      'WB95Diagnostic/CompensatedRK4.h':(ROOT/'research/wb95/CompensatedRK4.h').read_text()}
    for name,content in files.items():
        with (source/name).open('x') as stream:stream.write(content)
    write_new(out/'generated_source_manifest.json',{name:digest(source/name) for name in files})
    command_run(['cmake','-S',str(source),'-B',str(out/'isolated_build'),'-DCalypso_DIR='+str(previous.EXTERNAL/'run/cmake')],out,out/'configure.log')
    command_run(['cmake','--build',str(out/'isolated_build'),'-j','1'],out,out/'build.log')
    binary=out/'isolated_build/x86_64-el9-gcc13-opt/lib/libWB95Diagnostic.so'
    write_new(out/'binary_manifest.json',{'binary':str(binary),'sha256':digest(binary),'generated_manifest_sha256':digest(out/'generated_source_manifest.json')})

def execute(out):
    verify(out);write_new(out/'environment.json',{'host':os.uname().nodename,'condor_ad':os.environ.get('_CONDOR_JOB_AD'),
      'environment':{k:os.environ.get(k) for k in ('CMAKE_PREFIX_PATH','LD_LIBRARY_PATH','AtlasVersion','AtlasProject','CMTCONFIG')}})
    build(out);shutil.copytree(WB94/'identity_payload',out/'identity_payload')
    work=out/'event';work.mkdir(exist_ok=False);shutil.copyfile(out/'fixture.json',work/'fixture.json')
    platform=out/'isolated_build/x86_64-el9-gcc13-opt'
    script='\n'.join(['source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso',
      'source '+shlex.quote(str(platform/'setup.sh')),'export LD_LIBRARY_PATH='+shlex.quote(str(platform/'lib'))+':"$LD_LIBRARY_PATH"',
      'python '+shlex.quote(str(ROOT/'scripts/wb95_athena.py'))+' --work-dir '+shlex.quote(str(work))+' --sqlite '+shlex.quote(str(out/'identity_payload/tracker_alignment.sqlite'))])
    write_new(work/'command.json',{'script':script});command_run(['bash','-c',script],work,work/'athena.log');aggregate(out)

def aggregate(out):
    verify(out);work=out/'event';binary=read_public(out/'binary_manifest.json');p=read_public(out/'protocol.json')
    if digest(Path(binary['binary']))!=binary['sha256'] or digest(out/'generated_source_manifest.json')!=binary['generated_manifest_sha256']:raise ValueError('binary/source identity changed')
    for name,h in read_public(out/'generated_source_manifest.json').items():
        if digest(out/'isolated_source'/name)!=h:raise ValueError('generated source changed')
    if digest(work/'fixture.json')!=digest(out/'fixture.json'):raise ValueError('worker fixture changed')
    r=read_public(work/'acts.json');nodepath=work/'acts.json.nodes_le_i16.bin'
    if Path(r['nodes_binary'])!=nodepath or digest(nodepath)!=p['expected_node_interleaved_le_i16_sha256'] or nodepath.stat().st_size!=6*p['expected_nodes']:
        raise ValueError('actual conditions nodes differ from frozen ROOT payload')
    nodes=np.fromfile(nodepath,dtype='<i2').reshape(-1,3)
    s=analyze(r,read_public(out/'fixture.json'),read_public(WB94/'event/acts.json'),p,nodes)
    log=(work/'athena.log').read_text(errors='replace');prior=read_public(WB94/'summary.json')
    if str(out/'identity_payload/tracker_alignment.sqlite') not in log or 'Using FASER magnetic field service' not in log:raise ValueError('wrong physical path')
    field_paths=[Path(x.strip()) for x in re.findall(r'Initialized the field map from\s+([^\n]+)',log)]
    libs=[Path(x) for x in r['loaded_libraries'] if any(y in x for y in ('WB95Diagnostic','FaserActs','ActsCore','MagField'))]
    if not field_paths or Path(binary['binary']) not in libs:raise ValueError('missing actual map/isolated binary')
    for path in libs:
        if str(path) in prior['loaded_scientific_libraries'] and digest(path)!=prior['loaded_scientific_libraries'][str(path)]:raise ValueError('runtime changed')
    for path in field_paths:
        if str(path) not in prior['field_map_hashes'] or digest(path)!=prior['field_map_hashes'][str(path)]:raise ValueError('field map changed')
    s.update({'acts_sha256':digest(work/'acts.json'),'log_sha256':digest(work/'athena.log'),'nodes_sha256':digest(nodepath),'binary':binary,
      'loaded_scientific_libraries':{str(x):digest(x) for x in libs},'field_map_hashes':{str(x):digest(x) for x in field_paths},
      'overlap_errors':re.findall(r'Layers are overlapping at: ([^\n]+)',log)})
    write_new(out/'summary.json',s)

def submit(out):
    verify(out);sub=out/'wb95.sub'
    with sub.open('x') as stream:stream.write('\n'.join(['universe = vanilla',f'executable = {ROOT}/scripts/run_wb95_condor.sh',
      f'arguments = {ROOT} {out}',f'output = {out}/condor.$(ClusterId).out',f'error = {out}/condor.$(ClusterId).err',f'log = {out}/condor.$(ClusterId).log',
      'request_cpus = 1','request_memory = 8000','request_disk = 8000000','requirements = (Arch == "X86_64")','+JobFlavour = "workday"','getenv = False','should_transfer_files = NO','queue 1','']))
    command='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(sub))
    r=subprocess.run(['bash','-c',command],capture_output=True,text=True)
    write_new(out/'submission.json',{'command':command,'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode})
    print(r.stdout);print(r.stderr,file=sys.stderr)
    if r.returncode:raise RuntimeError('submission failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','submit','aggregate']);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();out=a.output_root.resolve()
    if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb95_'):p.error('WB95 exclusive output required')
    try:{'freeze':freeze,'run':execute,'submit':submit,'aggregate':aggregate}[a.action](out)
    except Exception as e:
        if a.action=='run':write_new(out/'execution_error.json',{'error':repr(e),'qualification':'NOT_EVALUATED'})
        raise
