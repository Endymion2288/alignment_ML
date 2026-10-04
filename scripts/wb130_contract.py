#!/usr/bin/env python3
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/mc24_four_station_wb129_sensor_surface_curve_v2'
WB129=PARENT
OUT=ROOT/'outputs/mc24_four_station_wb130_sensor_bounds_v1'
PROTOCOL=ROOT/'configs/research_review/wp130_sensor_bounds_contract.json'
WORKBOOK=ROOT/'workbook/2026-10-04_130_WB129事件12与20传感器边界及测量帧审计.md'
SRC=ROOT/'research/wb130/SensorBoundsAudit.cxx'
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write_new(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')
def req(x,m):
 if not x:raise ValueError(m)
def build():
 pre=OUT/'build_preflight_v1';pre.mkdir(parents=True);source=pre/'source';pkg=source/'WB130Diagnostic';pkg.mkdir(parents=True)
 (source/'CMakeLists.txt').write_text((ROOT/'outputs/mc24_four_station_wb125_build_preflight_v2/source/CMakeLists.txt').read_text().replace('WB125','WB130'))
 (pkg/'CMakeLists.txt').write_text((ROOT/'outputs/mc24_four_station_wb125_build_preflight_v2/source/WB125Diagnostic/CMakeLists.txt').read_text().replace('WB125Diagnostic','WB130Diagnostic').replace('PersistedProvenance.cxx PhysicalSeedAudit.cxx','SensorBoundsAudit.cxx'))
 (pkg/'SensorBoundsAudit.cxx').write_text(SRC.read_text())
 with (pre/'configure.log').open('w') as log:r=subprocess.run(['cmake','-S',str(source),'-B',str(pre/'build'),'-DCalypso_DIR='+str(ROOT.parent/'calypso/run/cmake')],stdout=log,stderr=subprocess.STDOUT)
 req(r.returncode==0,'configure')
 with (pre/'build.log').open('w') as log:r=subprocess.run(['cmake','--build',str(pre/'build'),'-j','1'],stdout=log,stderr=subprocess.STDOUT)
 req(r.returncode==0,'build')
 binary=pre/'build/x86_64-el9-gcc13-opt/lib/libWB130Diagnostic.so';req(binary.is_file(),'binary');write_new(pre/'receipt.json',{'binary':str(binary),'binary_sha256':digest(binary),'source_sha256':digest(SRC),'new_propagation_calls':0,'new_reconstruction_calls':0})
def freeze():
 req(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='4station','branch');req(not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip(),'tracked dirty')
 old=read(WB129/'freeze.json');hashes={}
 for p,h in old['hashes'].items():
  if p.endswith('workbook/2026-10-04_129_WB127同六事件实际传感器面曲线传播.md'):continue
  req(digest(p)==h,'WB129 frozen '+p);hashes[p]=h
 r=read(OUT/'build_preflight_v1/receipt.json');req(digest(r['binary'])==r['binary_sha256'] and digest(SRC)==r['source_sha256'],'build identity')
 for p in (ROOT/'configs/research_review/wp130_sensor_bounds_contract.json',SRC,ROOT/'scripts/wb130_athena.py',ROOT/'scripts/wb130_contract.py',WORKBOOK,OUT/'build_preflight_v1/receipt.json',Path(r['binary'])):hashes[str(p)]=digest(p)
 OUT.mkdir(exist_ok=True);(OUT/'events').mkdir(exist_ok=False);shutil.copyfile(PROTOCOL,OUT/'protocol.json');(OUT/'contract_workbook.md').write_text(WORKBOOK.read_text())
 for idx in read(PROTOCOL)['indices']:
  src=WB129/'recovery_v2/events'/f'{idx:02d}';dst=OUT/'events'/f'{idx:02d}';dst.mkdir()
  for n in ('fixture.json','export.json','curve_response.json'):shutil.copyfile(src/n,dst/n);hashes[str(dst/n)]=digest(dst/n)
 hashes[str(OUT/'protocol.json')]=digest(OUT/'protocol.json');hashes[str(OUT/'contract_workbook.md')]=digest(OUT/'contract_workbook.md')
 write_new(OUT/'freeze.json',{'schema':'wb130_sensor_bounds_freeze_v1','hashes':hashes,'parent_wb129_freeze_sha256':digest(WB129/'freeze.json'),'parent_workbook_amendment':'WB129 result was appended after its preregistration freeze; all WB129 binary/export identities remain checked','population':2,'rows':51,'new_propagation_calls':0,'new_reconstruction_calls':0,'field_queries':0,'held_out_access':False})
def verify():
 for p,h in read(OUT/'freeze.json')['hashes'].items():req(digest(p)==h,'frozen '+p)
def run():
 verify();(OUT/'execution_lock').mkdir(exist_ok=False);rec=OUT/'recovery_v1';rec.mkdir();(rec/'events').mkdir()
 for idx in read(PROTOCOL)['indices']:
  src=OUT/'events'/f'{idx:02d}';ev=rec/'events'/f'{idx:02d}';ev.mkdir()
  for n in ('fixture.json','export.json','curve_response.json'):shutil.copyfile(src/n,ev/n)
  binary=Path(read(OUT/'build_preflight_v1/receipt.json')['binary']);platform=binary.parent.parent;cmd='\n'.join(['set -eo pipefail','source '+str(ROOT/'scripts/setup_environment.sh')+' calypso','source '+str(platform/'setup.sh'),'export LD_LIBRARY_PATH='+str(platform/'lib')+':$LD_LIBRARY_PATH','python '+str(ROOT/'scripts/wb130_athena.py')+' --work-dir '+str(ev)+' --sqlite '+str(PARENT/'identity_payload/tracker_alignment.sqlite')]);(ev/'command.sh').write_text(cmd)
  with (ev/'athena.log').open('w') as log:r=subprocess.run(['bash','-c',cmd],cwd=ev,stdout=log,stderr=subprocess.STDOUT)
  write_new(ev/'exit.json',{'exit_code':r.returncode});req(r.returncode==0,'Athena '+str(idx));req((ev/'bounds_audit.json').is_file(),'bounds output')
 write_new(rec/'summary.json',{'schema':'wb130_sensor_bounds_summary_v1','execution_contract':'PASS','population':2,'rows':51,'new_propagation_calls':0,'new_reconstruction_calls':0,'field_queries':0,'held_out_access':False})
def seal():
 verify();s=read(OUT/'recovery_v1/summary.json');arts={str(p.relative_to(ROOT)):digest(p) for p in OUT.rglob('*') if p.is_file()};write_new(ROOT/'docs/wb130_sensor_bounds_result_manifest.json',{'schema':'wb130_sensor_bounds_result_manifest_v1','execution_contract':s['execution_contract'],'new_propagation_calls':0,'new_reconstruction_calls':0,'field_queries':0,'held_out_access':False,'sources':{str(p.relative_to(ROOT)):digest(p) for p in (SRC,ROOT/'scripts/wb130_athena.py',ROOT/'scripts/wb130_contract.py',PROTOCOL)},'artifacts':arts,'scientific_boundary':{'alignment':'NOT_ESTABLISHED','association':'INHERITED_CONDITIONAL','curve_endpoint_active_surface':'TO_BE_AUDITED'}})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['build','freeze','run','verify','seal']);globals()[a.parse_args().action]()
