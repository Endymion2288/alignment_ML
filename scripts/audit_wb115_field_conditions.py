#!/usr/bin/env python3
"""Read-only comparison of two field folders in one frozen COOL snapshot."""
import argparse, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from wb100_contract import digest,verify

BASE=ROOT/'outputs/mc24_four_station_wb114_persisted_type_provenance_v1'
OUT=ROOT/'outputs/mc24_four_station_wb115_field_conditions_provenance_v1'
WORKBOOK=ROOT/'workbook/2026-10-03_115_四站磁场条件标签差异的只读载荷来源审计.md'
EXTERNAL=ROOT.parent/'calypso'
DB_LINK=Path('/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/sqlite200/ALLP200.db')
DB=DB_LINK.resolve()
MAP_DIR=Path('/cvmfs/faser.cern.ch/repo/sw/software/22.0/faser/offline/ReleaseData/v20/MagneticFieldMaps')
MAPS={n:MAP_DIR/n for n in ('FaserFieldTable_v2.root','FaserFieldTable_v100.root','FaserFieldTable_v101.root')}
TAGS=('OFLCOND-FASER-05','OFLCOND-FASER-06')
FOLDERS=('/GLOBAL/BField/Maps','/GLOBAL/BField/Scales')
MAX_RECORDS=256

def check(value,message):
    if not value:raise ValueError(message)

def validate_rows(data):
    check(data['status']=='READ','folder/tag unreadable')
    rows=data['records'];check(0<len(rows)<=MAX_RECORDS,'empty/excessive IOV rows')
    names={x['name'] for x in data['payload_specification']}
    check(len(names)==len(data['payload_specification']),'duplicate payload fields')
    keys=set();last={}
    for row in sorted(rows,key=lambda r:(r['channel'],r['since'],r['until'])):
        check(isinstance(row['since'],int) and isinstance(row['until'],int) and row['since']<row['until'],'invalid IOV')
        k=(row['channel'],row['since'],row['until']);check(k not in keys,'duplicate IOV');keys.add(k)
        check(row['since']>=last.get(row['channel'],-1),'overlapping same-channel IOV');last[row['channel']]=row['until']
        check(set(row['payload'])==names,'payload specification mismatch')
        for v in row['payload'].values():
            check(v is None or isinstance(v,(str,int,float,bool)),'unhandled payload type')
            if isinstance(v,float):check(math.isfinite(v),'nonfinite payload')
    return sorted(rows,key=lambda r:(r['channel'],r['since'],r['until']))

def compare(left,right):
    try:l=validate_rows(left);r=validate_rows(right)
    except (ValueError,KeyError,TypeError) as e:return {'classification':'UNKNOWN','reason':str(e)}
    checks={'records_equal':l==r,'specification_equal':left['payload_specification']==right['payload_specification'],
        'channel_names_equal':left['channel_names']==right['channel_names'],'folder_description_equal':left['description']==right['description']}
    equal=all(checks.values())
    return {'classification':('EQUIVALENT_FIELD_FOLDER_RECORDS_IN_FROZEN_SNAPSHOT' if equal else 'DIFFERENT_FIELD_FOLDER_RECORDS_IN_FROZEN_SNAPSHOT'),
        'checks':checks,'leaf_tags_equal':left['leaf_tag']==right['leaf_tag'],
        'left_record_count':len(l),'right_record_count':len(r),
        'note':'row/spec/name equality only; no historical generation-override or physical field validity claim'}

def freeze():
    check(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='4station','branch')
    check(not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip(),'uncommitted tracked changes')
    hashes=verify(BASE)['hashes'].copy();m=read_public(ROOT/'docs/wb114_persisted_type_provenance_result_manifest.json')
    check(m['execution_contract']=='PASS' and m['frozen_association_decision']=='UNKNOWN_OR_AMBIGUOUS','historical verdict')
    for n,h in m['artifacts'].items():check(digest(ROOT/n)==h,'historical artifact '+n);hashes[str(ROOT/n)]=h
    paths=[Path(__file__),ROOT/'tests/test_wb115_field_conditions.py',ROOT/'docs/wb114_persisted_type_provenance_result_manifest.json',
        DB,*MAPS.values(),MAP_DIR/'README',EXTERNAL/'run/XML/FaserAuthentication/dblookup.xml',
        EXTERNAL/'MagneticField/MagFieldServices/python/MagFieldServicesConfig.py',
        EXTERNAL/'MagneticField/MagFieldServices/src/FaserFieldMapCondAlg.cxx',
        EXTERNAL/'MagneticField/MagFieldServices/src/FaserFieldCacheCondAlg.cxx',
        EXTERNAL/'MagneticField/MagFieldServices/src/FaserFieldCacheCondAlg.h',
        ROOT/'outputs/mc24_four_station_wb109_world_abort_trace_v1/event/athena.log',
        ROOT/'outputs/mc24_four_station_wb93_transport_error_v1/postrun_map_metadata.json']
    for p in paths:hashes[str(p)]=digest(p)
    OUT.mkdir(exist_ok=False)
    with (OUT/'contract_workbook.md').open('x') as f:f.write(WORKBOOK.read_text())
    write_new(OUT/'control.json',{'database':str(DB),'database_link':str(DB_LINK),'link_target':os.readlink(DB_LINK),
        'bytes':DB.stat().st_size,'database_sha256':hashes[str(DB)],'instance':'OFLP200','readonly':True,
        'tags':TAGS,'folders':FOLDERS,'max_records_per_folder_tag':MAX_RECORDS,
        'maps':{n:{'path':str(p),'sha256':hashes[str(p)]} for n,p in MAPS.items()},
        'executor_anchor':{'tag':'OFLCOND-FASER-06','maps_leaf':'GLOBAL-BField-Maps-03','scales_leaf':'GLOBAL-BField-Scale-03',
            'map_uri':'file:MagneticFieldMaps/FaserFieldTable_v2.root','map_path':str(MAPS['FaserFieldTable_v2.root']),
            'map_sha256':'a799fb227b85fab98e8ab88c1911a4b7c6bc65844717db27fb98f4726bca8c51','scale':1.},
        'population':0,'event_root_access':False,'truth_access':False,'new_field_queries':0})
    for n in ('contract_workbook.md','control.json'):hashes[str(OUT/n)]=digest(OUT/n)
    write_new(OUT/'freeze.json',{'schema':'wb115_field_conditions_freeze_v1','hashes':hashes,'branch':'4station',
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'root_event_access':False,'held_out_access':False,'new_reconstruction_calls':0,'new_propagation_calls':0})

def read_folder(db,path,tag,cool):
    result={'folder':path,'global_tag':tag,'status':'UNKNOWN'}
    try:
        folder=db.getFolder(path);leaf=folder.resolveTag(tag);spec=folder.payloadSpecification()
        fields=[{'name':spec[i].name(),'storage_type':spec[i].storageType().name()} for i in range(spec.size())]
        channels=[int(x) for x in folder.listChannels()]
        result.update({'leaf_tag':str(leaf),'description':str(folder.description()),'versioning_mode':int(folder.versioningMode()),
            'payload_specification':fields,'channel_names':{str(c):str(folder.channelName(c)) for c in channels},
            'tag_lock_status':int(folder.tagLockStatus(leaf)),
            'browse_range':{'since':int(cool.ValidityKeyMin),'until':int(cool.ValidityKeyMax),'all_channels':True},'records':[]})
        cursor=folder.browseObjects(cool.ValidityKeyMin,cool.ValidityKeyMax,cool.ChannelSelection.all(),leaf)
        try:
            while cursor.goToNext():
                check(len(result['records'])<MAX_RECORDS,'record limit exceeded; no partial scientific comparison')
                obj=cursor.currentRef();payload=obj.payload();values={f['name']:payload[f['name']] for f in fields}
                json.dumps(values,allow_nan=False)
                result['records'].append({'channel':int(obj.channelId()),'since':int(obj.since()),'until':int(obj.until()),'payload':values})
        finally:cursor.close()
        result['status']='READ';validate_rows(result)
    except Exception as e:
        result['status']='UNKNOWN';result['error_type']=type(e).__name__;result['error']=str(e)
    return result

def summarize(records,control):
    check(len(records)==4 and {(r['folder'],r['global_tag']) for r in records}=={(f,t) for f in FOLDERS for t in TAGS},'two-by-two identity')
    by={(r['folder'],r['global_tag']):r for r in records};comparisons={f:compare(by[f,TAGS[0]],by[f,TAGS[1]]) for f in FOLDERS}
    classes=[v['classification'] for v in comparisons.values()]
    overall='UNKNOWN' if 'UNKNOWN' in classes else ('EQUIVALENT_FIELD_FOLDER_RECORDS_IN_FROZEN_SNAPSHOT' if all(v=='EQUIVALENT_FIELD_FOLDER_RECORDS_IN_FROZEN_SNAPSHOT' for v in classes) else 'DIFFERENT_FIELD_FOLDER_RECORDS_IN_FROZEN_SNAPSHOT')
    map_rows=[]
    for tag in TAGS:
        for row in by[FOLDERS[0],tag].get('records',[]):
            p=row['payload'];uri=p.get('MapFileName');known={('file:MagneticFieldMaps/'+n):v for n,v in control['maps'].items()}
            map_rows.append({'global_tag':tag,'channel':row['channel'],'since':row['since'],'until':row['until'],'field_type':p.get('FieldType'),
                'map_uri':uri,'known_frozen_map_file':known.get(uri),'content_identity':'KNOWN_FROZEN_FILE' if uri in known else 'UNKNOWN'})
    anchor=control['executor_anchor'];maps=by[FOLDERS[0],TAGS[1]];scales=by[FOLDERS[1],TAGS[1]]
    valid=maps['status']==scales['status']=='READ'
    anchor_checks={'maps_leaf_matches':maps.get('leaf_tag')==anchor['maps_leaf'],
        'scales_leaf_matches':scales.get('leaf_tag')==anchor['scales_leaf'],
        'globalmap_uri_present':any(r['payload'].get('FieldType')=='GlobalMap' and r['payload'].get('MapFileName')==anchor['map_uri'] for r in maps.get('records',[])),
        'channel1_scale1_present':any(r['channel']==1 and r['payload'].get('value')==anchor['scale'] for r in scales.get('records',[])),
        'observed_map_hash_matches':control['maps']['FaserFieldTable_v2.root']['sha256']==anchor['map_sha256']}
    return {'schema':'wb115_field_conditions_summary_v1','classification':overall,'folder_comparisons':comparisons,
        'map_file_identities':map_rows,'executor_anchor_checks':anchor_checks,
        'executor_anchor_status':'SUPPORTED_LIMITED_LEAF_MAP_SCALE_SNAPSHOT_CHECK' if valid and all(anchor_checks.values()) else 'UNKNOWN_OR_MISMATCH',
        'historical_generation_field_conditions':'UNKNOWN','historical_database_replica_identity':'UNKNOWN',
        'source_geometry_alignment_compatibility':'UNKNOWN','wb114_association_decision':'UNKNOWN_OR_AMBIGUOUS',
        'event_root_access':False,'truth_access':False,'held_out_access':False,'new_field_queries':0,
        'new_reconstruction_calls':0,'new_propagation_calls':0,'qualification':'NOT_EVALUATED','physics_screening':'NOT_EVALUATED','final_oracle':'NOT_EVALUATED'}

def run():
    start=time.monotonic();f=verify(OUT);(OUT/'execution_lock').mkdir(exist_ok=False);control=read_public(OUT/'control.json')
    from PyCool import cool
    conn='sqlite://;schema='+control['database']+';dbname='+control['instance']
    db=cool.DatabaseSvcFactory.databaseService().openDatabase(conn,True)
    try:
        write_new(OUT/'database_open_receipt.json',{'connection':conn,'requested_readonly':True,'database_id':str(db.databaseId()),'network_fallback':False})
        records=[]
        for path in FOLDERS:
            for tag in TAGS:
                r=read_folder(db,path,tag,cool);records.append(r)
                write_new(OUT/('folder_'+path.rsplit('/',1)[1]+'_'+tag+'.json'),r)
    finally:db.closeDatabase()
    write_new(OUT/'summary.json',summarize(records,control));verify(OUT)
    write_new(OUT/'execution_receipt.json',{'exit_code':0,'elapsed_seconds':time.monotonic()-start,'frozen_identities_verified':len(f['hashes']),
        'mode':'local_readonly_sqlite_conditions_only','folder_tag_reads_attempted':4,
        'folder_tag_reads_successful':sum(r['status']=='READ' for r in records),
        'execution_contract':'PASS' if all(r['status']=='READ' for r in records) else 'FAIL_INCOMPLETE_CONDITIONS_READ',
        'event_root_access':False,'truth_access':False,
        'database_before_after_sha256_identical':True,'new_field_queries':0,'new_reconstruction_calls':0,'new_propagation_calls':0})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','verify','run'));a=p.parse_args()
    if a.action=='verify':verify(OUT)
    else:globals()[a.action]()
