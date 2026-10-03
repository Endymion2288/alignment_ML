#!/usr/bin/env python3
"""Exclusive single-event persisted provenance read; no production mutation."""
import argparse, hashlib, json, os, shlex, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT, read_public, write_new
from wb100_contract import digest, verify
from wb92_contract import EXTERNAL, run as command_run
from audit_wb113_provenance import classify

BASE=ROOT/'outputs/mc24_four_station_wb111_navigation_reachability_v1'
PREVIOUS=ROOT/'outputs/mc24_four_station_wb112_shared_seed_audit_v1'
OUT=ROOT/'outputs/mc24_four_station_wb113_persisted_provenance_v1'
PREFLIGHT=ROOT/'outputs/mc24_four_station_wb113_compile_preflight_v3'
WORKBOOK=ROOT/'workbook/2026-10-03_113_四站已见事件持久化关联与生成条件来源审计.md'
ATHENA=Path('/cvmfs/atlas.cern.ch/repo/sw/software/24.0/Athena/24.0.41/InstallArea/x86_64-el9-gcc13-opt')

def files():
    old=ROOT/'outputs/mc24_four_station_wb110_boundary_topology_v1/isolated_source'
    cm=(old/'WB110Diagnostic/CMakeLists.txt').read_text().replace('WB110Diagnostic','WB113Diagnostic')
    cm=cm[:cm.index('atlas_add_component(')]+'''atlas_add_component(WB113Diagnostic PersistedProvenance.cxx LINK_LIBRARIES AthenaBaseComps StoreGateLib xAODFaserEventInfo xAODTruth TrackerIdentifier TrackerPrepRawData TrackerSimData TrackerRIO_OnTrack TrkTrack TrkParameters GeneratorObjects PRIVATE_LINK_LIBRARIES nlohmann_json::nlohmann_json)
'''
    return {'CMakeLists.txt':(old/'CMakeLists.txt').read_text().replace('WB110','WB113'),
        'WB113Diagnostic/CMakeLists.txt':cm,
        'WB113Diagnostic/PersistedProvenance.cxx':(ROOT/'research/wb113/PersistedProvenance.cxx').read_text()}

def compile_check():
    PREFLIGHT.mkdir(exist_ok=False)
    for name,content in files().items():
        p=PREFLIGHT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as f:f.write(content)
    prior=ROOT/'outputs/mc24_four_station_wb113_compile_preflight_v2'
    for name in ('CMakeLists.txt','WB113Diagnostic/CMakeLists.txt'):
        if digest(prior/name)!=digest(PREFLIGHT/name):raise ValueError('CMake dependencies changed; fresh configure required')
    flags=prior/'build/WB113Diagnostic/CMakeFiles/WB113Diagnostic.dir/flags.make'
    lines=flags.read_text().splitlines()
    parse=lambda k:shlex.split(next(x.split('=',1)[1] for x in lines if x.startswith(k+' =')))
    includes=parse('CXX_INCLUDES')
    cmd=['g++','-std=c++20','-fsyntax-only',*parse('CXX_DEFINES'),*includes,str(PREFLIGHT/'WB113Diagnostic/PersistedProvenance.cxx')]
    with (PREFLIGHT/'compile.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    write_new(PREFLIGHT/'receipt.json',{'returncode':r.returncode,'command':cmd,'flags_sha256':digest(flags),
        'source_hashes':{n:digest(PREFLIGHT/n) for n in files()},'scope':'syntax-only isolated algorithm; no event or ROOT access',
        'compiler':subprocess.check_output(['which','g++'],text=True).strip()})
    if r.returncode:raise RuntimeError('compile-only preflight failed; preserve directory')

def freeze():
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='4station':raise ValueError('branch')
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip():raise ValueError('tracked changes')
    hashes=verify(PREVIOUS)['hashes'].copy()
    previous=read_public(ROOT/'docs/wb112_shared_seed_result_manifest.json')
    for p,h in previous['artifacts'].items():
        if digest(ROOT/p)!=h:raise ValueError('WB112 result changed '+p)
        hashes[str(ROOT/p)]=h
    receipt=read_public(PREFLIGHT/'receipt.json')
    if receipt['returncode']!=0:raise ValueError('compile gate')
    for n,c in files().items():
        if hashlib.sha256(c.encode()).hexdigest()!=receipt['source_hashes'][n]:raise ValueError('preflight source changed')
    paths=[ROOT/'scripts/wb113_contract.py',ROOT/'scripts/wb113_athena.py',ROOT/'scripts/audit_wb113_provenance.py',
        ROOT/'scripts/run_wb113_condor.sh',ROOT/'research/wb113/PersistedProvenance.cxx',ROOT/'tests/test_wb113_provenance.py',
        ROOT/'docs/wb112_shared_seed_result_manifest.json',BASE/'raw_identity.json',PREVIOUS/'summary.json',Path(receipt['compiler']),
        ATHENA/'python/PyUtils/MetaReader.py',ATHENA/'src/Generators/GeneratorObjects/GeneratorObjects/HepMcParticleLink.h',
        ATHENA/'src/Event/xAOD/xAODTruthCnv/src/xAODTruthCnvAlg.cxx',ATHENA/'src/Event/xAOD/xAODTruth/xAODTruth/versions/TruthParticle_v1.h',
        EXTERNAL/'Tracking/Acts/FaserActsKalmanFilter/src/TrackTruthMatchingTool.cxx',
        EXTERNAL/'Tracker/TrackerRawEvent/TrackerSimData/TrackerSimData/TrackerSimData.h',
        EXTERNAL/'Tracker/TrackerDetDescr/TrackerIdentifier/TrackerIdentifier/FaserSCT_ID.h']
    paths+=list(PREFLIGHT.rglob('*'))
    paths+=list((ROOT/'outputs/mc24_four_station_wb113_compile_preflight_v1').rglob('*'))
    paths+=list((ROOT/'outputs/mc24_four_station_wb113_compile_preflight_v2').rglob('*'))
    paths += [EXTERNAL/'run/lib'/n for n in ('libTrackerSimData.so','libTrackerPrepRawData.so','libTrackerRIO_OnTrack.so',
        'libTrackerSimDataDict.so','libTrackerPrepRawDataDict.so')]
    paths += [ATHENA/'lib'/n for n in ('libxAODTruth.so','libGeneratorObjects.so','libTrkTrack.so','libxAODTruthDict.so')]
    for p in paths:
        if p.is_file():hashes[str(p)]=digest(p)
        elif not p.is_dir():raise FileNotFoundError(p)
    f=read_public(BASE/'fixture.json')
    if (f['index'],f['ordinal'],f['actual_run'],f['actual_event'])!=(12,2268,100044,2268) or f['row']['role']!='development':raise ValueError('allowlist')
    OUT.mkdir(exist_ok=False);shutil.copyfile(WORKBOOK,OUT/'contract_workbook.md');shutil.copyfile(BASE/'fixture.json',OUT/'fixture.json')
    write_new(OUT/'generation_expectation.json',{'files':{n:hashlib.sha256(c.encode()).hexdigest() for n,c in files().items()}})
    for n in ('contract_workbook.md','fixture.json','generation_expectation.json'):hashes[str(OUT/n)]=digest(OUT/n)
    write_new(OUT/'freeze.json',{'schema':'wb113_persisted_provenance_freeze_v1','hashes':hashes,'branch':'4station',
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'population':1,
        'root_access':'file metadata plus exactly ordinal2268','truth_access':'seen-event diagnostic only',
        'held_out_access':False,'new_reconstruction_calls':0,'new_propagation_calls':0})

def raw_identity():
    f=read_public(OUT/'fixture.json');p=Path(f['input_xaod']);before=p.stat();h=digest(p);after=p.stat();previous=read_public(BASE/'raw_identity.json')
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('raw changed while hashing')
    if h!=previous['sha256'] or f['source_stat']!={'bytes':after.st_size,'mtime_ns':after.st_mtime_ns}:raise ValueError('raw identity changed')
    return {'path':str(p),'sha256':h,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns}

def metadata(raw):
    import ROOT as R
    from PyUtils.MetaReader import read_metadata, convert_itemList
    # read_metadata full reads MetaData tree and POOL params, never CollectionTree entries.
    md=read_metadata([raw['path']],mode='full',promote=False,unique_tag_info_values=False)
    write_new(OUT/'source_metadata.json',md)
    r=R.TFile.Open(raw['path'],'READ')
    if not r or r.IsZombie():raise ValueError('ROOT file open failed')
    tree=r.Get('CollectionTree')
    if not tree:raise ValueError('CollectionTree absent')
    branches=[{'name':b.GetName(),'class':b.GetClassName(),'title':b.GetTitle()} for b in tree.GetListOfBranches()]
    write_new(OUT/'root_branch_metadata.json',{'source':raw,'tree':'CollectionTree','entries':int(tree.GetEntries()),
        'event_entries_decoded':0,'branches':branches,'root_uuid':r.GetUUID().AsString()});r.Close()
    items=convert_itemList(md[raw['path']],layout=None)
    if not isinstance(items,list):raise ValueError('actual event metadata items unavailable; no guessed track keys')
    for item in items:
        if not isinstance(item,(list,tuple)) or len(item)!=2:raise ValueError('event metadata item shape')
    return {'track_keys':[k for t,k in items if t=='TrackCollection'],'eventdata_items':items}

def run():
    verify(OUT);(OUT/'execution_lock').mkdir(exist_ok=False)
    raw=raw_identity();write_new(OUT/'raw_identity_before.json',raw)
    control=metadata(raw);write_new(OUT/'control.json',control)
    source=OUT/'isolated_source';source.mkdir(exist_ok=False)
    expected=read_public(OUT/'generation_expectation.json')['files']
    for n,c in files().items():
        p=source/n;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as f:f.write(c)
        if digest(p)!=expected[n]:raise ValueError('generated source differs')
    command_run(['cmake','-S',str(source),'-B',str(OUT/'isolated_build'),'-DCalypso_DIR='+str(EXTERNAL/'run/cmake')],OUT,OUT/'configure.log')
    command_run(['cmake','--build',str(OUT/'isolated_build'),'-j','1'],OUT,OUT/'build.log')
    platform=OUT/'isolated_build/x86_64-el9-gcc13-opt';binary=platform/'lib/libWB113Diagnostic.so'
    write_new(OUT/'binary_manifest.json',{'binary':str(binary),'sha256':digest(binary)})
    shutil.copytree(BASE/'identity_payload',OUT/'identity_payload');event=OUT/'event';event.mkdir(exist_ok=False)
    for n in ('fixture.json','control.json'):shutil.copyfile(OUT/n,event/n)
    cmd='\n'.join(['source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso',
        'source '+shlex.quote(str(platform/'setup.sh')),'export LD_LIBRARY_PATH='+shlex.quote(str(platform/'lib'))+':"$LD_LIBRARY_PATH"',
        'python '+shlex.quote(str(ROOT/'scripts/wb113_athena.py'))+' --work-dir '+shlex.quote(str(event))+' --sqlite '+shlex.quote(str(OUT/'identity_payload/tracker_alignment.sqlite'))])
    write_new(event/'command.json',{'script':cmd})
    with (event/'athena.log').open('x') as log:r=subprocess.run(['bash','-c',cmd],cwd=event,stdout=log,stderr=subprocess.STDOUT)
    write_new(OUT/'athena_exit.json',{'exit_code':r.returncode,'host':os.uname().nodename})
    if r.returncode:raise RuntimeError('Athena failed; no scientific verdict')
    write_new(OUT/'raw_identity_after.json',raw_identity());verify(OUT)
    d=read_public(event/'provenance.json');f=read_public(OUT/'fixture.json');s=classify(f,d)
    write_new(OUT/'summary.json',s)

def submit():
    verify(OUT)
    setup='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null'
    q=subprocess.check_output(['bash','-c',setup+' && condor_q -json'],text=True,timeout=55)
    totals=subprocess.check_output(['bash','-c',setup+' && condor_q -totals'],text=True,timeout=55)
    slots=subprocess.check_output(['bash','-c',setup+' && condor_status -constraint \'State=="Unclaimed" && Activity=="Idle"\' -af Name'],text=True,timeout=55)
    write_new(OUT/'scheduler_preflight.json',{'queue':json.loads(q) if q.strip() else [],'queue_totals':totals,
        'idle_slots':len(slots.splitlines()),'schedd':'bigbird24'})
    if not slots.strip():raise RuntimeError('no idle slots; no submission')
    text=(BASE/'wb111.sub').read_text().replace(str(BASE),str(OUT)).replace('run_wb111_condor.sh','run_wb113_condor.sh')
    with (OUT/'wb113.sub').open('x') as f:f.write(text)
    r=subprocess.run(['bash','-c',setup+' && condor_submit '+shlex.quote(str(OUT/'wb113.sub'))],capture_output=True,text=True,timeout=55)
    write_new(OUT/'submission.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'submit_sha256':digest(OUT/'wb113.sub')})
    print(r.stdout,r.stderr)
    if r.returncode:raise RuntimeError('submission failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('compile','freeze','verify','run','submit'));a=p.parse_args()
    if a.action=='compile':compile_check()
    elif a.action=='verify':verify(OUT)
    else:globals()[a.action]()
