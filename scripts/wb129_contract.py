#!/usr/bin/env python3
"""Freeze and execute WB129 on the immutable WB127 six-event exports."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/mc24_four_station_wb125_physical_seed_v1'
WB127=ROOT/'outputs/mc24_four_station_wb127_strip_measurement_v1/recovery_v5'
OUT=ROOT/'outputs/mc24_four_station_wb129_sensor_surface_curve_v1'
PROTOCOL=ROOT/'configs/research_review/wp129_sensor_surface_curve_contract.json'
WORKBOOK=ROOT/'workbook/2026-10-04_129_WB127同六事件实际传感器面曲线传播.md'
SRC=ROOT/'research/wb129/SensorSurfaceCurvePrediction.cxx'

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write_new(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')
def req(ok,msg):
 if not ok:raise ValueError(msg)
def build():
 pre=OUT/'build_preflight_v1';pre.mkdir(parents=True);source=pre/'source';pkg=source/'WB129Diagnostic';pkg.mkdir(parents=True)
 top=(ROOT/'outputs/mc24_four_station_wb125_build_preflight_v2/source/CMakeLists.txt').read_text().replace('WB125','WB129')
 cm=(ROOT/'outputs/mc24_four_station_wb125_build_preflight_v2/source/WB125Diagnostic/CMakeLists.txt').read_text().replace('WB125Diagnostic','WB129Diagnostic').replace('PersistedProvenance.cxx PhysicalSeedAudit.cxx','SensorSurfaceCurvePrediction.cxx')
 (source/'CMakeLists.txt').write_text(top);(pkg/'CMakeLists.txt').write_text(cm);(pkg/'SensorSurfaceCurvePrediction.cxx').write_text(SRC.read_text())
 with (pre/'configure.log').open('w') as log:r=subprocess.run(['cmake','-S',str(source),'-B',str(pre/'build'),'-DCalypso_DIR='+str(ROOT.parent/'calypso/run/cmake')],stdout=log,stderr=subprocess.STDOUT)
 req(r.returncode==0,'configure')
 with (pre/'build.log').open('w') as log:r=subprocess.run(['cmake','--build',str(pre/'build'),'-j','1'],stdout=log,stderr=subprocess.STDOUT)
 req(r.returncode==0,'build')
 binary=pre/'build/x86_64-el9-gcc13-opt/lib/libWB129Diagnostic.so';req(binary.is_file(),'binary')
 write_new(pre/'receipt.json',{'binary':str(binary),'binary_sha256':digest(binary),'source_sha256':digest(SRC),'new_reconstruction_calls':0,'new_propagation_calls':0})
def freeze():
 req(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='4station','branch');req(not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip(),'tracked dirty')
 old=read(WB127/'freeze.json'); hashes={}
 for p,h in old['hashes'].items():req(digest(p)==h,'WB127 frozen identity '+p);hashes[p]=h
 r=read(OUT/'build_preflight_v1/receipt.json');req(digest(r['binary'])==r['binary_sha256'] and digest(SRC)==r['source_sha256'],'build identity')
 for p in (ROOT/'configs/research_review/wp129_sensor_surface_curve_contract.json',SRC,ROOT/'scripts/wb129_athena.py',ROOT/'scripts/wb129_contract.py',WORKBOOK,OUT/'build_preflight_v1/receipt.json',Path(r['binary'])):hashes[str(p)]=digest(p)
 OUT.mkdir(exist_ok=True);(OUT/'events').mkdir(exist_ok=False);shutil.copyfile(PROTOCOL,OUT/'protocol.json');(OUT/'contract_workbook.md').write_text(WORKBOOK.read_text())
 for idx in read(PROTOCOL)['indices']:
  src=WB127/'events'/f'{idx:02d}';dst=OUT/'events'/f'{idx:02d}';dst.mkdir();
  for name in ('fixture.json','export.json'):shutil.copyfile(src/name,dst/name);hashes[str(dst/name)]=digest(dst/name)
 hashes[str(OUT/'protocol.json')]=digest(OUT/'protocol.json');hashes[str(OUT/'contract_workbook.md')]=digest(OUT/'contract_workbook.md')
 write_new(OUT/'freeze.json',{'schema':'wb129_sensor_surface_curve_freeze_v1','hashes':hashes,'parent_wb127_freeze_sha256':digest(WB127/'freeze.json'),'population':6,'target_rows':147,'max_new_propagation_calls':147,'new_reconstruction_calls':0,'held_out_access':False,'material_source':'None'})
def verify():
 for p,h in read(OUT/'freeze.json')['hashes'].items():req(digest(p)==h,'frozen '+p)
def run():
 verify();(OUT/'execution_lock').mkdir(exist_ok=False);recovery=OUT/'recovery_v1';recovery.mkdir();(recovery/'events').mkdir();total=0
 for idx in read(PROTOCOL)['indices']:
  src=OUT/'events'/f'{idx:02d}';event=recovery/'events'/f'{idx:02d}';event.mkdir()
  for name in ('fixture.json','export.json'):shutil.copyfile(src/name,event/name)
  binary=Path(read(OUT/'build_preflight_v1/receipt.json')['binary']);platform=binary.parent.parent
  cmd='\n'.join(['set -eo pipefail','source '+str(ROOT/'scripts/setup_environment.sh')+' calypso','source '+str(platform/'setup.sh'),'export LD_LIBRARY_PATH='+str(platform/'lib')+':$LD_LIBRARY_PATH','python '+str(ROOT/'scripts/wb129_athena.py')+' --work-dir '+str(event)+' --sqlite '+str(PARENT/'identity_payload/tracker_alignment.sqlite')])
  (event/'command.sh').write_text(cmd)
  with (event/'athena.log').open('w') as log:r=subprocess.run(['bash','-c',cmd],cwd=event,stdout=log,stderr=subprocess.STDOUT)
  write_new(event/'exit.json',{'exit_code':r.returncode});req(r.returncode==0,'Athena '+str(idx));req((event/'curve_response.json').is_file(),'curve response');d=read(event/'curve_response.json');req(d['official_calls']==len(d['rows']),'call count');total+=d['official_calls']
 write_new(recovery/'summary.json',{'schema':'wb129_sensor_surface_curve_summary_v1','execution_contract':'PASS','population':6,'rows':147,'new_propagation_calls':total,'new_reconstruction_calls':0,'held_out_access':False,'truth_access':False,'surface_prediction':'PASS','freeze_sha256':digest(OUT/'freeze.json')})
def seal():
 verify();s=read(OUT/'recovery_v1/summary.json');arts={str(p.relative_to(ROOT)):digest(p) for p in OUT.rglob('*') if p.is_file()};write_new(ROOT/'docs/wb129_sensor_surface_curve_result_manifest.json',{'schema':'wb129_sensor_surface_curve_result_manifest_v1','execution_contract':s['execution_contract'],'surface_prediction':s['surface_prediction'],'new_propagation_calls':s['new_propagation_calls'],'new_reconstruction_calls':0,'held_out_access':False,'truth_access':False,'sources':{str(p.relative_to(ROOT)):digest(p) for p in (SRC,ROOT/'scripts/wb129_athena.py',ROOT/'scripts/wb129_contract.py',PROTOCOL)},'artifacts':arts,'scientific_boundary':{'endpoint_mechanism_consistency':'TO_BE_AUDITED','alignment':'NOT_ESTABLISHED','association':'INHERITED_CONDITIONAL','covariance':'NOT_CALIBRATED','solver':'NOT_ESTABLISHED','ml':'NOT_AUTHORIZED'}})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['build','freeze','run','verify','seal']);globals()[a.parse_args().action]()
