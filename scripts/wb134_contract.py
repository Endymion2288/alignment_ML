#!/usr/bin/env python3
"""Exclusive read-only material-input inventory; zero propagation budget."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import shlex,shutil,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wb131_contract import digest,read,write,require,PARENT
from wb133_contract import verify as verify_parent
from alignment.wb134_material_inventory import audit_event
PARENT_STAGE=ROOT/'outputs/mc24_four_station_wb133_accepted_anchor_v1'
OUT=ROOT/'outputs/mc24_four_station_wb134_material_inventory_v1'
PROTOCOL=ROOT/'configs/research_review/wp134_material_contract.json'
WORKBOOK=ROOT/'workbook/2026-10-05_134_四站材料输入与条件过程噪声合同审计.md'
SRC=ROOT/'research/wb134/MaterialInventory.cxx'
SOURCES=[PROTOCOL,SRC,ROOT/'alignment/wb134_material_inventory.py',ROOT/'scripts/wb134_contract.py',
 ROOT/'scripts/wb134_athena.py',ROOT/'scripts/wb134_independent_audit.py',ROOT/'tests/test_wb134_material_inventory.py',ROOT/'scripts/setup_environment.sh']
CALYPSO=ROOT.parent/'calypso'
EXTERNAL=Path('/cvmfs/atlas.cern.ch/repo/sw/software/24.0/AthenaExternals/24.0.41/InstallArea/x86_64-el9-gcc13-opt')
EVIDENCE=[CALYPSO/'Tracking/Acts/FaserActsGeometry'/p for p in ['python/ActsGeometryConfig.py','src/FaserActsTrackingGeometrySvc.cxx',
 'src/FaserActsTrackingGeometrySvc.h','src/FaserActsLayerBuilder.cxx','src/FaserActsDetectorElement.cxx','src/CuboidVolumeBuilder.cxx',
 'src/FaserActsExtrapolationTool.cxx']]
EVIDENCE += [EXTERNAL/'include/Acts'/p for p in ['Material/ProtoSurfaceMaterial.hpp','Material/MaterialSlab.hpp','Material/Interactions.hpp',
 'Propagator/MaterialInteractor.hpp','Propagator/detail/PointwiseMaterialInteraction.hpp','Geometry/TrackingVolume.hpp']]
EVIDENCE += [CALYPSO/'Tracking/Acts/FaserActsKalmanFilter/src/TrackFinderFunction.cxx',EXTERNAL/'lib/libActsCore.so']


def build():
 pre=OUT/'build_preflight_v2';pre.mkdir(parents=True,exist_ok=False);source=pre/'source';pkg=source/'WB134Diagnostic';pkg.mkdir(parents=True)
 old=ROOT/'outputs/mc24_four_station_wb125_build_preflight_v2/source'
 (source/'CMakeLists.txt').write_text((old/'CMakeLists.txt').read_text().replace('WB125','WB134'))
 cm=(old/'WB125Diagnostic/CMakeLists.txt').read_text().replace('WB125Diagnostic','WB134Diagnostic')
 (pkg/'CMakeLists.txt').write_text(cm.replace('PersistedProvenance.cxx PhysicalSeedAudit.cxx','MaterialInventory.cxx'))
 shutil.copyfile(SRC,pkg/SRC.name)
 for name,cmd in [('configure',['cmake','-S',str(source),'-B',str(pre/'build'),'-DCalypso_DIR='+str(CALYPSO/'run/cmake')]),('build',['cmake','--build',str(pre/'build'),'-j','1'])]:
  with (pre/(name+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  require(r.returncode==0,name+' failure; preserve logs')
 binary=pre/'build/x86_64-el9-gcc13-opt/lib/libWB134Diagnostic.so'
 write(pre/'receipt.json',{'binary':str(binary),'binary_sha256':digest(binary),'source_sha256':digest(SRC),'new_propagation_calls':0})


def freeze():
 verify_parent();require(not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip(),'tracked dirty')
 require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='4station','branch')
 manifest=read(ROOT/'docs/wb133_accepted_anchor_result_manifest.json')
 for group in ('sources','artifacts'):
  for name,h in manifest[group].items():require(digest(ROOT/name)==h,'parent manifest identity '+name)
 hashes=dict(read(PARENT_STAGE/'freeze.json')['hashes']);receipt=read(OUT/'build_preflight_v2/receipt.json')
 require(digest(receipt['binary'])==receipt['binary_sha256'] and digest(SRC)==receipt['source_sha256'],'build identity')
 for p in SOURCES+EVIDENCE+[ROOT/'docs/wb133_accepted_anchor_result_manifest.json',PARENT_STAGE/'freeze.json',Path(receipt['binary']),OUT/'build_preflight_v2/receipt.json']:hashes[str(p)]=digest(p)
 shutil.copyfile(WORKBOOK,OUT/'contract_workbook.md');hashes[str(OUT/'contract_workbook.md')]=digest(OUT/'contract_workbook.md')
 availability=[]
 for name in read(PROTOCOL)['map_candidate_paths']:
  p=(ROOT/name).resolve();entry={'path':str(p),'exists':p.exists(),'qualified':False}
  if p.is_file():entry.update(bytes=p.stat().st_size,sha256=digest(p));hashes[str(p)]=digest(p)
  availability.append(entry)
 write(OUT/'material_input_evidence.json',{'schema':'wb134_material_input_evidence_v1','candidate_paths':availability,
  'search_claim':'bounded listed paths only; not proof of global absence','configured_material_source':'None','map_loaded':False,
  'physical_budget_gate':'UNKNOWN_UNQUALIFIED_MAP_GEOMETRY_PATH','historical_mc24_material_identity':'UNKNOWN',
  'source_hashes':{str(p):digest(p) for p in EVIDENCE}})
 hashes[str(OUT/'material_input_evidence.json')]=digest(OUT/'material_input_evidence.json')
 (OUT/'events').mkdir(exist_ok=False);total=0
 for index in read(PROTOCOL)['indices']:
  event=OUT/'events'/f'{index:02d}';event.mkdir()
  for name in ('fixture.json','export.json','request.json','audit.json'):
   original=PARENT_STAGE/'events'/event.name/name;require(digest(original)==manifest['artifacts'][str(original.relative_to(ROOT))],'parent event identity')
   dest=event/('parent_audit.json' if name=='audit.json' else name);shutil.copyfile(original,dest);hashes[str(original)]=digest(original);hashes[str(dest)]=digest(dest)
  total+=len(read(event/'export.json')['rows'])
 require(total==147,'population')
 write(OUT/'freeze.json',{'schema':'wb134_freeze_v1','hashes':hashes,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  'original_rows':147,'max_input_event_executions':6,'max_new_propagation_calls':0,'held_out_access':False,'truth_access':False})
 verify()


def verify():
 for path,h in read(OUT/'freeze.json')['hashes'].items():require(digest(path)==h,'frozen identity '+path)
 for i in read(PROTOCOL)['indices']:
  f=read(OUT/'events'/f'{i:02d}'/'fixture.json');s=Path(f['input_xaod']).stat();require(s.st_size==f['source_stat']['bytes'] and s.st_mtime_ns==f['source_stat']['mtime_ns'],'input stat')


def execute(index):
 verify();event=OUT/'events'/f'{index:02d}';binary=Path(read(OUT/'build_preflight_v2/receipt.json')['binary']);platform=binary.parent.parent;q=lambda p:shlex.quote(str(p))
 command='\n'.join(['set -eo pipefail','ulimit -c 0','source '+q(ROOT/'scripts/setup_environment.sh')+' calypso','source '+q(platform/'setup.sh'),
  'export LD_LIBRARY_PATH='+q(platform/'lib')+':${LD_LIBRARY_PATH:-}','python '+q(ROOT/'scripts/wb134_athena.py')+' --work-dir '+q(event)+' --sqlite '+q(PARENT/'identity_payload/tracker_alignment.sqlite')])
 with (event/'command.sh').open('x') as f:f.write(command+'\n')
 print('WB134 event',index,'inventory',flush=True)
 with (event/'athena.log').open('x') as log:r=subprocess.run(['bash','-c',command],cwd=event,stdout=log,stderr=subprocess.STDOUT)
 write(event/'exit.json',{'exit_code':r.returncode});status={'index':index,'execution':'PASS' if r.returncode==0 else 'FAIL','interface':'UNKNOWN','physical_budget':'UNKNOWN'}
 try:
  require(r.returncode==0,'Athena execution failure')
  a=audit_event(read(event/'export.json'),read(event/'request.json'),read(event/'response.json'),read(PROTOCOL));write(event/'audit.json',a)
  status.update(interface=a['interface'],physical_budget=a['physical_budget'],targets=a['targets'],surface_counts=a['surface_counts'],volume_counts=a['volume_counts'])
 except (ValueError,KeyError,OSError,IndexError) as error:status['failure']=str(error)
 write(event/'status.json',status);print(status,flush=True);return status


def run():
 verify();(OUT/'execution_lock').mkdir(exist_ok=False)
 with ThreadPoolExecutor(max_workers=read(PROTOCOL)['local_concurrency']) as pool:events=list(pool.map(execute,read(PROTOCOL)['indices']))
 verify();write(OUT/'summary.json',{'schema':'wb134_summary_v1','events':events,'execution':'PASS' if all(e['execution']=='PASS' for e in events) else 'FAIL_OR_UNKNOWN',
  'interface':'PASS' if all(e['interface']=='PASS' for e in events) else 'UNKNOWN','physical_budget':'UNKNOWN_MATERIAL_INPUT_AND_PATH_NOT_QUALIFIED',
  'input_event_executions':6,'new_propagation_calls':0,'new_reconstruction_calls':0,'algorithm_field_queries':0,'held_out_access':False,'truth_access':False})


def seal():
 verify();independent=read(OUT/'independent_audit.json');require(independent['artifact_consistency']=='PASS','independent evidence')
 write(ROOT/'docs/wb134_material_result_manifest.json',{'schema':'wb134_result_manifest_v1','summary':read(OUT/'summary.json'),'independent_audit':independent,
  'material_input_evidence':read(OUT/'material_input_evidence.json'),'sources':{str(p.relative_to(ROOT)):digest(p) for p in SOURCES},
  'artifacts':{str(p.relative_to(ROOT)):digest(p) for p in OUT.rglob('*') if p.is_file()},
  'scientific_boundary':{'physical_material':'UNKNOWN_NO_QUALIFIED_MAP_PATH','process_noise':'NOT_ESTABLISHED_SYNTHETIC_CONTROLS_ONLY',
  'historical_mc24_material':'UNKNOWN','source_covariance':'NO_PRIOR','WB133':'FAIL_PRESERVED','alignment':'NOT_ESTABLISHED','held_out':'NOT_ACCESSED','ML':'NOT_AUTHORIZED'}})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['build','freeze','verify','run','seal']);globals()[p.parse_args().action]()
