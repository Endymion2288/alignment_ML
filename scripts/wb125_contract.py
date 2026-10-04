#!/usr/bin/env python3
"""Exclusive six-seen-event WB125 execution; production and old outputs immutable."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from wb92_contract import EXTERNAL,ACTS

PARENT=ROOT/'outputs/mc24_four_station_wb92_wb104_event_fixture_preflight_v2'
OUT=ROOT/'outputs/mc24_four_station_wb125_physical_seed_v1'
PREFLIGHT=ROOT/'outputs/mc24_four_station_wb125_build_preflight_v2'
PROTOCOL=ROOT/'configs/research_review/wp125_physical_seed_contract.json'
WORKBOOK=ROOT/'workbook/2026-10-04_125_四站物理共同轨迹种子六事件对照.md'
ATHENA=Path('/cvmfs/atlas.cern.ch/repo/sw/software/24.0/Athena/24.0.41/InstallArea/x86_64-el9-gcc13-opt')
SOURCES=[Path(__file__),ROOT/'scripts/wb125_athena.py',ROOT/'scripts/wb125_audit.py',
    ROOT/'scripts/wb125_test_env.py',
    ROOT/'scripts/audit_wb113_provenance.py',PROTOCOL,ROOT/'scripts/setup_environment.sh',
    ROOT/'research/wb125/PersistedProvenance.cxx',ROOT/'research/wb125/PhysicalSeedAudit.cxx',
    ROOT/'tests/test_wb125_physical_seed.py']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def generated_files():
    previous=ROOT/'outputs/mc24_four_station_wb110_boundary_topology_v1/isolated_source'
    top=(previous/'CMakeLists.txt').read_text().replace('WB110','WB125')
    package=(previous/'WB110Diagnostic/CMakeLists.txt').read_text()
    package=package[:package.index('atlas_add_component(')].replace('WB110','WB125')
    package+='atlas_add_component(WB125Diagnostic PersistedProvenance.cxx PhysicalSeedAudit.cxx LINK_LIBRARIES AthenaBaseComps StoreGateLib xAODFaserEventInfo xAODTruth TrackerIdentifier TrackerPrepRawData TrackerSimData TrackerRIO_OnTrack TrkTrack TrkParameters TrkSurfaces GeneratorObjects FaserActsGeometryLib FaserActsGeometryInterfacesLib ActsCore PRIVATE_LINK_LIBRARIES nlohmann_json::nlohmann_json)\n'
    return {'CMakeLists.txt':top,'WB125Diagnostic/CMakeLists.txt':package,
        **{'WB125Diagnostic/'+p.name:p.read_text() for p in SOURCES if p.suffix=='.cxx'}}

def command(args,cwd,log):
    with log.open('x') as f:r=subprocess.run(args,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('command exit '+str(r.returncode)+'; '+str(log))

def build_preflight():
    PREFLIGHT.mkdir(exist_ok=False);source=PREFLIGHT/'source';source.mkdir()
    expected={}
    for name,content in generated_files().items():
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as f:f.write(content)
        expected[name]=digest(p)
    command(['cmake','-S',str(source),'-B',str(PREFLIGHT/'build'),'-DCalypso_DIR='+str(EXTERNAL/'run/cmake')],PREFLIGHT,PREFLIGHT/'configure.log')
    command(['cmake','--build',str(PREFLIGHT/'build'),'-j','1'],PREFLIGHT,PREFLIGHT/'build.log')
    platform=PREFLIGHT/'build/x86_64-el9-gcc13-opt';binary=platform/'lib/libWB125Diagnostic.so'
    write_new(PREFLIGHT/'receipt.json',{'returncode':0,'generated':expected,'binary':str(binary),'binary_sha256':digest(binary),
        'platform':str(platform),'new_propagation_calls':0,'event_entries_decoded':0,'field_queries':0})

def require_branch():
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='4station':raise ValueError('branch')
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip():raise ValueError('uncommitted tracked sources')

def frozen_runtime():
    anchors={}
    for index in (1,4,8,16,20):
        d=read_public(PARENT/'events'/f'{index:02d}'/'validation.json')
        for section in ('field_map_hashes','loaded_scientific_libraries'):
            for p,h in d[section].items():
                if 'libWB92Diagnostic' in p:continue
                if p in anchors and anchors[p]!=h:raise ValueError('historical runtime inconsistency')
                anchors[p]=h
    for p,h in anchors.items():
        if digest(p)!=h:raise ValueError('historical default runtime changed: '+p)
    return anchors

def freeze():
    require_branch();protocol=read_public(PROTOCOL);receipt=read_public(PREFLIGHT/'receipt.json')
    if receipt['returncode']!=0:raise ValueError('build preflight')
    for name,content in generated_files().items():
        if hashlib.sha256(content.encode()).hexdigest()!=receipt['generated'][name]:raise ValueError('compiled source changed')
    if digest(receipt['binary'])!=receipt['binary_sha256']:raise ValueError('binary changed')
    for name in ('wb125_calypso_test_preflight_v2.json','wb125_ml_test_preflight_v2.json'):
        test=read_public(ROOT/'outputs'/name)
        if test['pytest_exit']!=0 or test['test_sha256']!=digest(ROOT/'tests/test_wb125_physical_seed.py') or test['audit_sha256']!=digest(ROOT/'scripts/wb125_audit.py'):raise ValueError('dual-environment test identity')
    old=read_public(PARENT/'freeze.json');hashes={}
    for p,h in old['hashes'].items():
        if digest(p)!=h:raise ValueError('historical source/input changed: '+p)
        hashes[p]=h
    hashes.update(frozen_runtime())
    paths=SOURCES+[PARENT/'freeze.json',PARENT/'wb104_manifest.json',PARENT/'identity_payload_manifest.json',PREFLIGHT/'receipt.json',Path(receipt['binary']),
        ROOT/'outputs/wb125_calypso_test_preflight_v2.json',ROOT/'outputs/wb125_ml_test_preflight_v2.json']
    for part in ('TrkParameters/TrkParameters/TrackParameters.h','TrkParametersBase/TrkParametersBase/CurvilinearParametersT.h'):
        paths.append(ATHENA/'src/Tracking/TrkEvent'/part)
    paths+=[ATHENA/'src/Tracking/TrkDetDescr/TrkSurfaces/TrkSurfaces/Surface.h',
        ATHENA/'src/Tracking/TrkEvent/TrkEventPrimitives/TrkEventPrimitives/SurfaceTypes.h',
        ATHENA/'src/Tracking/TrkEvent/TrkParametersBase/TrkParametersBase/CurvilinearParametersT.icc',
        ATHENA/'src/Tracking/TrkEvent/TrkParametersBase/TrkParametersBase/Charged.h',
        EXTERNAL/'Tracking/Acts/FaserActsKalmanFilter/src/CombinatorialKalmanFilterAlg.cxx',
        ACTS/'include/Acts/Definitions/Tolerance.hpp',ACTS/'include/Acts/Surfaces/PlaneSurface.hpp']
    for row in protocol['selected']:
        idx=row['index'];directory=PARENT/'events'/f'{idx:02d}'
        for p in directory.iterdir():
            if p.is_file() and p.suffix in ('.json','.log'):paths.append(p)
        f=read_public(directory/'fixture.json')
        if f['row']['role']!='development' or any(f[k]!=row[k] for k in ('input_xaod','ordinal','actual_run','actual_event','index')):raise ValueError('six-event identity')
        if hashlib.sha256(json.dumps(read_public(PARENT/'fixtures.json')[idx],sort_keys=True,separators=(',',':')).encode()).hexdigest()!=row['row_sha256']:raise ValueError('original row hash')
    OUT.mkdir(exist_ok=False);(OUT/'events').mkdir();(OUT/'raw_input_freeze').mkdir()
    shutil.copyfile(WORKBOOK,OUT/'contract_workbook.md');shutil.copyfile(PROTOCOL,OUT/'protocol.json')
    shutil.copytree(PARENT/'identity_payload',OUT/'identity_payload')
    for name,h in read_public(PARENT/'identity_payload_manifest.json')['hashes'].items():
        for p in (PARENT/'identity_payload'/name,OUT/'identity_payload'/name):
            if digest(p)!=h:raise ValueError('payload identity')
            hashes[str(p)]=h
    for p in paths:hashes[str(p)]=digest(p)
    for name in ('contract_workbook.md','protocol.json'):hashes[str(OUT/name)]=digest(OUT/name)
    write_new(OUT/'source_freeze.json',{'hashes':hashes,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'binary':receipt,'generated':receipt['generated'],'population':6,'new_propagation_calls':0,'held_out_access':False})
    for row in protocol['selected']:
        event=OUT/'events'/f'{row["index"]:02d}';event.mkdir()
        shutil.copyfile(PARENT/'events'/f'{row["index"]:02d}'/'fixture.json',event/'fixture.json')
        write_new(event/'control.json',{'track_keys':['CKFTrackCollection']})
    def raw_hash(row):
        idx=row['index'];f=read_public(OUT/'events'/f'{idx:02d}'/'fixture.json');p=Path(f['input_xaod']);before=p.stat()
        if f['source_stat']!={'bytes':before.st_size,'mtime_ns':before.st_mtime_ns}:raise ValueError('historical stat changed')
        h=digest(p);after=p.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('raw changed during hash')
        historical='UNKNOWN_NO_HISTORICAL_SHA256'
        if idx==12:
            previous=read_public(ROOT/'outputs/mc24_four_station_wb114_persisted_type_provenance_v1/raw_identity_before.json')
            if h!=previous['sha256']:raise ValueError('index12 historical raw SHA changed')
            historical='MATCH_HISTORICAL_SHA256'
        identity={'path':str(p),'sha256':h,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'historical_byte_identity':historical}
        write_new(OUT/'raw_input_freeze'/f'{idx:02d}.json',identity)
        print('RAW_SHA_FROZEN',idx,h,flush=True)
        return identity
    with ThreadPoolExecutor(max_workers=2) as pool:raws=list(pool.map(raw_hash,protocol['selected']))
    freeze_data=read_public(OUT/'source_freeze.json');freeze_data['raw_inputs']=raws
    for p in list((OUT/'raw_input_freeze').glob('*.json'))+list((OUT/'events').glob('*/*.json')):freeze_data['hashes'][str(p)]=digest(p)
    freeze_data['hashes'][str(OUT/'source_freeze.json')]=digest(OUT/'source_freeze.json')
    write_new(OUT/'freeze.json',freeze_data);verify()
    print('FREEZE_COMPLETE',len(freeze_data['hashes']),flush=True)

def verify():
    f=read_public(OUT/'freeze.json')
    for p,h in f['hashes'].items():
        if digest(p)!=h:raise ValueError('frozen identity changed: '+p)
    for raw in f['raw_inputs']:
        st=Path(raw['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(raw['bytes'],raw['mtime_ns']):raise ValueError('frozen raw stat')
    return f

def run():
    f=verify();(OUT/'execution_lock').mkdir(exist_ok=False)
    write_new(OUT/'execution_environment.json',{'host':os.uname().nodename,'local_bounded_execution':True,
        'new_reconstruction_calls':0,'max_propagation_calls':88,'held_out_access':False})
    for row in read_public(PROTOCOL)['selected']:
        verify();event=OUT/'events'/f'{row["index"]:02d}'
        script='\n'.join(['ulimit -c 0','export PYTHONDONTWRITEBYTECODE=1',
            'source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso',
            'source '+shlex.quote(f['binary']['platform']+'/setup.sh'),
            'export LD_LIBRARY_PATH='+shlex.quote(f['binary']['platform']+'/lib')+':"$LD_LIBRARY_PATH"',
            'python '+shlex.quote(str(ROOT/'scripts/wb125_athena.py'))+' --work-dir '+shlex.quote(str(event))+
            ' --sqlite '+shlex.quote(str(OUT/'identity_payload/tracker_alignment.sqlite'))])
        write_new(event/'command.json',{'script':script,'expected_index':row['index']})
        with (event/'athena.log').open('x') as log:r=subprocess.run(['bash','-c',script],cwd=event,stdout=log,stderr=subprocess.STDOUT)
        write_new(event/'exit.json',{'exit_code':r.returncode,'host':os.uname().nodename})
        print('EVENT_FINISHED',row['index'],'exit',r.returncode,flush=True)
    verify();audit()

def audit():
    from wb125_audit import audit_event
    freeze_data=verify();events=[];runtime=frozen_runtime();new_runtime={}
    for selected in read_public(PROTOCOL)['selected']:
        idx=selected['index'];event=OUT/'events'/f'{idx:02d}'
        try:
            if read_public(event/'exit.json')['exit_code']!=0:raise ValueError('Athena execution failed')
            response=read_public(event/'response.json');fixture=read_public(event/'fixture.json');provenance=read_public(event/'provenance.json')
            old=read_public(PARENT/'events'/f'{idx:02d}'/'acts.json') if idx!=12 else None
            summary=audit_event(fixture,provenance,response,old)
            text=(event/'athena.log').read_text(errors='replace')
            if 'Reading folder /Tracker/Align from sqlite' not in text or 'Using FASER magnetic field service' not in text:raise ValueError('actual field/sqlite path')
            begins=re.findall(r'WB125_CALL_BEGIN id=(\d+) label=([^ ]+) station=(\d+)',text)
            if [int(r[0]) for r in begins]!=list(range(1,response['propagation_calls']+1)):raise ValueError('actual call log/counter')
            for p,h in runtime.items():
                if '.so' in p and Path(p).resolve() not in {Path(q).resolve() for q in response['loaded_libraries']}:raise ValueError('actual default runtime not loaded: '+p)
            for p in response['loaded_libraries']:
                if any(n in p for n in ('WB125','TrkTrack','TrkParameters','TrkEventAthenaPool','TrackerEvent','GeneratorObjects','TrackerSimData','TrkSurfaces')):
                    new_runtime[p]=digest(p)
            maps=re.findall(r'Initialized the field map from\s+([^\n]+)',text)
            if not maps:raise ValueError('actual field map absent')
            for p in maps:
                p=p.strip().strip('"')
                if p not in runtime or digest(p)!=runtime[p]:raise ValueError('actual field map identity')
            old_text=(PARENT/'events'/f'{idx:02d}'/'athena.log').read_text(errors='replace')
            scale_pattern=r'initialized FaserFieldCacheCondObj and cache with scale factor ([^\s]+)'
            actual_scales=re.findall(scale_pattern,text);old_scales=re.findall(scale_pattern,old_text)
            if not actual_scales or actual_scales!=old_scales:raise ValueError('actual field scale differs from old default log')
            summary['actual_field_scales']=actual_scales
            summary['current_raw_identity']='FROZEN_SHA256_STAT_HEADER_MATCH'
            summary['historical_raw_identity']=read_public(OUT/'raw_input_freeze'/f'{idx:02d}.json')['historical_byte_identity']
        except Exception as e:
            summary={'index':idx,'integrity':'UNKNOWN','error':str(e),'hypothesis':'UNKNOWN','qualification':'NOT_EVALUATED'}
        write_new(event/'summary.json',summary);events.append(summary)
    total=sum(read_public(OUT/'events'/f'{r["index"]:02d}'/'response.json')['propagation_calls'] for r in events if (OUT/'events'/f'{r["index"]:02d}'/'response.json').is_file())
    if total>88:raise ValueError('total call cap')
    primary=next(r for r in events if r['index']==12)
    summary={'schema':'wb125_physical_seed_summary_v1','population':6,'events':events,'propagation_calls':total,
        'execution_contract':'PASS' if all(r['integrity']=='PASS' for r in events) else 'UNKNOWN',
        'hypothesis':primary['hypothesis'],'physical_fixture_supported':all(r.get('physical_fixture_supported',False) for r in events),
        'historical_raw_identity':'FIVE_UNKNOWN_ONE_HISTORICAL_SHA_MATCH','new_reconstruction_calls':0,'held_out_access':False,
        'qualification':'NOT_EVALUATED','physics_screening':'NOT_EVALUATED','final_oracle':'NOT_EVALUATED',
        'next_stage_executed':False,'freeze_sha256':digest(OUT/'freeze.json')}
    write_new(OUT/'loaded_edm_runtime_manifest.json',new_runtime);write_new(OUT/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='events'},indent=2),flush=True)

def seal():
    summary=read_public(OUT/'summary.json');verify();artifacts={}
    for p in sorted(OUT.rglob('*')):
        if p.is_file():artifacts[str(p.relative_to(ROOT))]=digest(p)
    for preflight in (ROOT/'outputs/mc24_four_station_wb125_build_preflight_v1',PREFLIGHT):
        for p in preflight.rglob('*'):
            if p.is_file() and (p.suffix in ('.cxx','.json','.log') or p.name=='CMakeLists.txt'):artifacts[str(p.relative_to(ROOT))]=digest(p)
    for p in (ROOT/'outputs').glob('wb125_*test_preflight_v*.json'):artifacts[str(p.relative_to(ROOT))]=digest(p)
    write_new(ROOT/'docs/wb125_physical_seed_result_manifest.json',{'schema':'wb125_physical_seed_result_manifest_v1',
        'execution_contract':summary['execution_contract'],'hypothesis':summary['hypothesis'],'physical_fixture_supported':summary['physical_fixture_supported'],
        'population':6,'propagation_calls':summary['propagation_calls'],'new_reconstruction_calls':0,'held_out_access':False,
        'qualification':'NOT_EVALUATED','physics_screening':'NOT_EVALUATED','final_oracle':'NOT_EVALUATED',
        'implementation_commit':read_public(OUT/'freeze.json')['git_commit'],'preregistration_commit':'8370121',
        'sources':{str(p.relative_to(ROOT)):digest(p) for p in SOURCES},'artifacts':artifacts})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build-preflight','freeze','verify','run','audit','seal']);args=parser.parse_args()
    if args.action=='build-preflight':build_preflight()
    else:globals()[args.action.replace('-','_')]()
