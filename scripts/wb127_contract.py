#!/usr/bin/env python3
import argparse, hashlib, json, os, shutil, subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/mc24_four_station_wb125_physical_seed_v1'
OUT=ROOT/'outputs/mc24_four_station_wb127_strip_measurement_v1'
PROTOCOL=ROOT/'configs/research_review/wp127_strip_measurement_contract.json'
WORKBOOK=ROOT/'workbook/2026-10-04_127_四站逐strip测量与直线压缩预测闭合.md'
SRC=ROOT/'research/wb127/StripMeasurementExport.cxx'
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')
def req(x,m):
 if not x:raise ValueError(m)
def build():
 pre=OUT/'build_preflight_v4';pre.mkdir(parents=True);src=pre/'source';(src/'WB127Diagnostic').mkdir(parents=True)
 (src/'WB127Diagnostic'/'StripMeasurementExport.cxx').write_text(SRC.read_text())
 top=(ROOT/'outputs/mc24_four_station_wb125_build_preflight_v2/source/CMakeLists.txt').read_text().replace('WB125','WB127')
 pkg=(ROOT/'outputs/mc24_four_station_wb125_build_preflight_v2/source/WB125Diagnostic/CMakeLists.txt').read_text().replace('WB125Diagnostic','WB127Diagnostic').replace('PersistedProvenance.cxx PhysicalSeedAudit.cxx','StripMeasurementExport.cxx').replace('GeneratorObjects FaserActsGeometryLib FaserActsGeometryInterfacesLib ActsCore','GeneratorObjects TrackerReadoutGeometry')
 (src/'CMakeLists.txt').write_text(top);(src/'WB127Diagnostic'/'CMakeLists.txt').write_text(pkg)
 def run(cmd,log):
  with log.open('w') as f:r=subprocess.run(cmd,cwd=pre,stdout=f,stderr=subprocess.STDOUT)
  req(r.returncode==0,'build '+str(log))
 run(['cmake','-S',str(src),'-B',str(pre/'build'),'-DCalypso_DIR='+str(ROOT.parent/'calypso/run/cmake')],pre/'configure.log')
 run(['cmake','--build',str(pre/'build'),'-j','1'],pre/'build.log')
 binary=pre/'build/x86_64-el9-gcc13-opt/lib/libWB127Diagnostic.so';req(binary.is_file(),'binary')
 write(pre/'receipt.json',{'binary':str(binary),'binary_sha256':digest(binary),'source_sha256':digest(SRC),'new_reconstruction_calls':0,'new_propagation_calls':0})
def freeze():
 req(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='4station','branch');req(not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip(),'tracked dirty')
 r=read(OUT/'build_preflight_v4/receipt.json');req(digest(r['binary'])==r['binary_sha256'] and digest(SRC)==r['source_sha256'],'build identity')
 OUT.mkdir(exist_ok=False);(OUT/'events').mkdir();shutil.copyfile(PROTOCOL,OUT/'protocol.json');(OUT/'contract_workbook.md').write_text(WORKBOOK.read_text());hashes={}
 parent=read(ROOT/'docs/wb125_physical_seed_result_manifest.json')
 for section in ('sources','artifacts'):
  for n,h in parent[section].items():req(digest(ROOT/n)==h,'parent identity');hashes[n]=h
 for p in [PROTOCOL,SRC,ROOT/'scripts/wb127_athena.py',ROOT/'scripts/wb127_contract.py',WORKBOOK,OUT/'protocol.json',OUT/'contract_workbook.md',Path(r['binary']),OUT/'build_preflight_v4/receipt.json']:hashes[str(p)]=digest(p)
 for idx in read(PROTOCOL)['indices']:
  e=PARENT/'events'/f'{idx:02d}';d=OUT/'events'/f'{idx:02d}';d.mkdir();shutil.copyfile(e/'fixture.json',d/'fixture.json');resp=read(e/'response.json');write(d/'p_seed.json',resp['seeds']['P']);hashes[str(d/'fixture.json')]=digest(d/'fixture.json');hashes[str(d/'p_seed.json')]=digest(d/'p_seed.json')
 write(OUT/'freeze.json',{'hashes':hashes,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'population':6,'new_reconstruction_calls':0,'new_propagation_calls':0,'held_out_access':False});verify();print('FREEZE_COMPLETE',len(hashes))
def verify():
 for p,h in read(OUT/'freeze.json')['hashes'].items():req(digest(p)==h,'frozen '+p)
def run():
 verify();(OUT/'execution_lock').mkdir(exist_ok=False);total=0
 for idx in read(PROTOCOL)['indices']:
  e=OUT/'events'/f'{idx:02d}';script='\n'.join(['source '+str(ROOT/'scripts/setup_environment.sh')+' calypso','source '+str(OUT/'build_preflight_v4/build/x86_64-el9-gcc13-opt/setup.sh'),'export LD_LIBRARY_PATH='+str(OUT/'build_preflight_v4/build/x86_64-el9-gcc13-opt/lib')+':$LD_LIBRARY_PATH','python '+str(ROOT/'scripts/wb127_athena.py')+' --work-dir '+str(e)+' --sqlite '+str(PARENT/'identity_payload/tracker_alignment.sqlite')]);(e/'command.sh').write_text(script);r=subprocess.run(['bash','-c',script],cwd=e,stdout=(e/'athena.log').open('w'),stderr=subprocess.STDOUT);write(e/'exit.json',{'exit_code':r.returncode});req(r.returncode==0,'Athena '+str(idx));total+=1;verify()
 write(OUT/'summary.json',{'schema':'wb127_strip_measurement_summary_v1','execution_contract':'PASS','population':6,'new_reconstruction_calls':total,'new_propagation_calls':0,'held_out_access':False,'measurement_export':'PASS','direct_curve_prediction':'NOT_EXECUTED','straight_compression':'NOT_EVALUATED','association':'INHERITED_CONDITIONAL','qualification':'NOT_EVALUATED','freeze_sha256':digest(OUT/'freeze.json')});print('RUN_COMPLETE',total)
def seal():
 verify();s=read(OUT/'summary.json');arts={str(p.relative_to(ROOT)):digest(p) for p in OUT.rglob('*') if p.is_file()};write(ROOT/'docs/wb127_strip_measurement_result_manifest.json',{'schema':'wb127_strip_measurement_result_manifest_v1','execution_contract':s['execution_contract'],'measurement_export':s['measurement_export'],'direct_curve_prediction':'NOT_EXECUTED','new_reconstruction_calls':s['new_reconstruction_calls'],'new_propagation_calls':0,'held_out_access':False,'sources':{str(p.relative_to(ROOT)):digest(p) for p in [SRC,ROOT/'scripts/wb127_athena.py',ROOT/'scripts/wb127_contract.py',PROTOCOL]},'artifacts':arts})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['build','freeze','run','verify','seal']);globals()[a.parse_args().action]()
