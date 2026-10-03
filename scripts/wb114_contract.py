#!/usr/bin/env python3
"""Same frozen WB113 algorithm; repair only its file metadata entry contract."""
import argparse, hashlib, json, os, shlex, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from wb100_contract import digest,verify
from wb92_contract import EXTERNAL,run as command_run
from wb113_contract import files,ATHENA,PREFLIGHT
from wb114_metadata import control_from_metadata
from audit_wb113_provenance import classify

BASE=ROOT/'outputs/mc24_four_station_wb113_persisted_provenance_v1'
PHYSICAL_BASE=ROOT/'outputs/mc24_four_station_wb111_navigation_reachability_v1'
OUT=ROOT/'outputs/mc24_four_station_wb114_persisted_type_provenance_v1'
WORKBOOK=ROOT/'workbook/2026-10-03_114_四站持久化类型驱动的单事件来源读取前瞻修复.md'
CONVERTER=ATHENA/'src/Tracking/TrkEventCnv/TrkEventAthenaPool/src'

def freeze():
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='4station':raise ValueError('branch')
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip():raise ValueError('uncommitted tracked changes')
    hashes=verify(BASE)['hashes'].copy();previous=read_public(ROOT/'docs/wb113_persisted_provenance_result_manifest.json')
    if (previous['execution_contract'],previous['failure_type'],previous['event_entries_decoded'])!=('FAIL','INFRASTRUCTURE_METADATA_ITEMLIST_ABSENT',0):raise ValueError('historical failure changed')
    for p,h in previous['artifacts'].items():
        if digest(ROOT/p)!=h:raise ValueError('historical artifact '+p)
        hashes[str(ROOT/p)]=h
    expected=read_public(BASE/'generation_expectation.json');receipt=read_public(PREFLIGHT/'receipt.json')
    if receipt['returncode']!=0:raise ValueError('unchanged syntax preflight')
    for n,c in files().items():
        h=hashlib.sha256(c.encode()).hexdigest()
        if h!=expected['files'][n] or h!=receipt['source_hashes'][n]:raise ValueError('diagnostic algorithm changed')
    paths=[Path(__file__),ROOT/'scripts/wb114_metadata.py',ROOT/'scripts/run_wb114_condor.sh',ROOT/'tests/test_wb114_metadata.py',
        ROOT/'docs/wb113_persisted_provenance_result_manifest.json',CONVERTER/'TrackCollectionCnv.h',CONVERTER/'TrackCollectionCnv.cxx',
        EXTERNAL/'Tracker/TrackerEventCnv/TrackerEventAthenaPool/src/FaserSCT_ClusterContainerCnv.h',
        ATHENA/'src/Tracking/TrkEvent/TrkTrack/TrkTrack/TrackCollection.h']
    paths += [EXTERNAL/'Tracking/TrkEventCnv/TrkEventAthenaPool/src'/n for n in ('TrackCollectionCnv.h','TrackCollectionCnv.cxx')]
    paths += [EXTERNAL/'Tracking/TrkEventCnv/TrkEventCnvTools'/n for n in ('src/EventCnvSuperTool.cxx','TrkEventCnvTools/EventCnvSuperTool.h')]
    paths += [EXTERNAL/'Tracker/TrackerEventCnv/TrackerEventTPCnv'/n for n in (
        'src/TrackerRIO_OnTrack/FaserSCT_ClusterOnTrackCnv_p2.cxx','TrackerEventTPCnv/TrackerRIO_OnTrack/FaserSCT_ClusterOnTrackCnv_p2.h')]
    paths += [EXTERNAL/'Tracker/TrackerEventCnv/TrackerEventCnvTools/src/TrackerEventCnvTool.cxx']
    for prefix in ('run/lib','build/x86_64-el9-gcc13-opt/lib'):
        for n in ('libTrkEventAthenaPoolPoolCnv.so','libTrackerEventTPCnv.so','libTrkEventTopLevelCnv.so','libTrkEventCnvTools.so','libTrackerEventCnvTools.so'):
            library=EXTERNAL/prefix/n
            if library.is_file():paths.append(library)
    for p in paths:hashes[str(p)]=digest(p)
    f=read_public(BASE/'fixture.json')
    if (f['index'],f['ordinal'],f['actual_run'],f['actual_event'])!=(12,2268,100044,2268) or f['row']['role']!='development':raise ValueError('single event allowlist')
    # Real already-seen file metadata preflight, no ROOT or event-level decoding.
    control=control_from_metadata(read_public(BASE/'root_branch_metadata.json'),read_public(BASE/'root_branch_metadata.json'),read_public(BASE/'source_metadata.json'))
    OUT.mkdir(exist_ok=False)
    for source,name in ((WORKBOOK,'contract_workbook.md'),(BASE/'fixture.json','fixture.json'),(BASE/'generation_expectation.json','generation_expectation.json')):
        shutil.copyfile(source,OUT/name);hashes[str(OUT/name)]=digest(OUT/name)
    write_new(OUT/'saved_metadata_preflight.json',{'control':control,'root_access':False,'event_entries_decoded':0,
        'branch_metadata_sha256':digest(BASE/'root_branch_metadata.json'),'source_metadata_sha256':digest(BASE/'source_metadata.json')})
    hashes[str(OUT/'saved_metadata_preflight.json')]=digest(OUT/'saved_metadata_preflight.json')
    write_new(OUT/'freeze.json',{'schema':'wb114_persisted_type_freeze_v1','hashes':hashes,'branch':'4station',
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'population':1,
        'new_reconstruction_calls':0,'new_propagation_calls':0,'held_out_access':False,'diagnostic_algorithm_changed':False})

def raw_identity():
    f=read_public(OUT/'fixture.json');p=Path(f['input_xaod']);before=p.stat();h=digest(p);after=p.stat()
    previous=read_public(BASE/'raw_identity_before.json')
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('raw changed while hashing')
    if h!=previous['sha256'] or f['source_stat']!={'bytes':after.st_size,'mtime_ns':after.st_mtime_ns}:raise ValueError('raw identity changed')
    return {'path':str(p),'sha256':h,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns}

def metadata(raw):
    import ROOT as R
    from PyUtils.MetaReader import read_metadata,convert_itemList
    md=read_metadata([raw['path']],mode='full',promote=False,unique_tag_info_values=False)
    write_new(OUT/'source_metadata.json',md)
    if md!=read_public(BASE/'source_metadata.json'):raise ValueError('frozen source metadata changed')
    r=R.TFile.Open(raw['path'],'READ')
    if not r or r.IsZombie():raise ValueError('ROOT open failed')
    try:
        tree=r.Get('CollectionTree')
        if not tree:raise ValueError('CollectionTree absent')
        branches=[{'name':b.GetName(),'class':b.GetClassName(),'title':b.GetTitle()} for b in tree.GetListOfBranches()]
        bm={'source':raw,'tree':'CollectionTree','entries':int(tree.GetEntries()),'event_entries_decoded':0,
            'branches':branches,'root_uuid':r.GetUUID().AsString()}
    finally:r.Close()
    write_new(OUT/'root_branch_metadata.json',bm)
    return control_from_metadata(bm,read_public(BASE/'root_branch_metadata.json'),md,convert_itemList(md[raw['path']],layout=None))

def run():
    verify(OUT);(OUT/'execution_lock').mkdir(exist_ok=False)
    write_new(OUT/'raw_identity_before.json',raw_identity())
    write_new(OUT/'control.json',metadata(read_public(OUT/'raw_identity_before.json')))
    source=OUT/'isolated_source';source.mkdir(exist_ok=False);expected=read_public(OUT/'generation_expectation.json')['files']
    for n,c in files().items():
        p=source/n;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as f:f.write(c)
        if digest(p)!=expected[n]:raise ValueError('unchanged source identity')
    write_new(OUT/'generated_source_manifest.json',{n:digest(source/n) for n in files()})
    command_run(['cmake','-S',str(source),'-B',str(OUT/'isolated_build'),'-DCalypso_DIR='+str(EXTERNAL/'run/cmake')],OUT,OUT/'configure.log')
    command_run(['cmake','--build',str(OUT/'isolated_build'),'-j','1'],OUT,OUT/'build.log')
    platform=OUT/'isolated_build/x86_64-el9-gcc13-opt';binary=platform/'lib/libWB113Diagnostic.so'
    write_new(OUT/'binary_manifest.json',{'binary':str(binary),'sha256':digest(binary)})
    shutil.copytree(PHYSICAL_BASE/'identity_payload',OUT/'identity_payload');event=OUT/'event';event.mkdir(exist_ok=False)
    for n in ('fixture.json','control.json'):shutil.copyfile(OUT/n,event/n)
    cmd='\n'.join(['source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso',
        'source '+shlex.quote(str(platform/'setup.sh')),'export LD_LIBRARY_PATH='+shlex.quote(str(platform/'lib'))+':"$LD_LIBRARY_PATH"',
        'python '+shlex.quote(str(ROOT/'scripts/wb113_athena.py'))+' --work-dir '+shlex.quote(str(event))+' --sqlite '+shlex.quote(str(OUT/'identity_payload/tracker_alignment.sqlite'))])
    write_new(event/'command.json',{'script':cmd,'unchanged_athena_source':str(ROOT/'scripts/wb113_athena.py'),
        'athena_sha256':digest(ROOT/'scripts/wb113_athena.py')})
    with (event/'athena.log').open('x') as log:r=subprocess.run(['bash','-c',cmd],cwd=event,stdout=log,stderr=subprocess.STDOUT)
    write_new(OUT/'athena_exit.json',{'exit_code':r.returncode,'host':os.uname().nodename})
    # Always preserve post-Athena source identity; an Athena failure is not a scientific verdict.
    write_new(OUT/'raw_identity_after.json',raw_identity());verify(OUT)
    if r.returncode:raise RuntimeError('Athena failed; no scientific verdict')
    s=classify(read_public(OUT/'fixture.json'),read_public(event/'provenance.json'))
    s['audit_workbook']=114;s['diagnostic_algorithm_workbook']=113;s['metadata_control']=read_public(OUT/'control.json')
    write_new(OUT/'summary.json',s)

def submit():
    verify(OUT)
    setup='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null'
    q=subprocess.check_output(['bash','-c',setup+' && condor_q -json'],text=True,timeout=55)
    totals=subprocess.check_output(['bash','-c',setup+' && condor_q -totals'],text=True,timeout=55)
    slots=subprocess.check_output(['bash','-c',setup+' && condor_status -constraint \'State=="Unclaimed" && Activity=="Idle"\' -af Name'],text=True,timeout=55)
    write_new(OUT/'scheduler_preflight.json',{'queue':json.loads(q) if q.strip() else [],'queue_totals':totals,
        'idle_slots':len(slots.splitlines()),'schedd':'bigbird24.cern.ch'})
    if not slots.strip():raise RuntimeError('no idle slots; no submission')
    sub=(BASE/'wb113.sub').read_text().replace(str(BASE),str(OUT)).replace('run_wb113_condor.sh','run_wb114_condor.sh')
    with (OUT/'wb114.sub').open('x') as f:f.write(sub)
    r=subprocess.run(['bash','-c',setup+' && condor_submit '+shlex.quote(str(OUT/'wb114.sub'))],capture_output=True,text=True,timeout=55)
    write_new(OUT/'submission.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'submit_sha256':digest(OUT/'wb114.sub')})
    print(r.stdout,r.stderr)
    if r.returncode:raise RuntimeError('submission failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','verify','run','submit'));a=p.parse_args()
    if a.action=='verify':verify(OUT)
    else:globals()[a.action]()
