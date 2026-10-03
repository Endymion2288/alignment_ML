#!/usr/bin/env python3
"""Batch execution of unchanged WB115 conditions audit on an identical copy."""
import argparse,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from wb100_contract import digest,verify
from audit_wb115_field_conditions import OUT as LOCAL,FOLDERS,TAGS,MAX_RECORDS,read_folder,summarize,WORKBOOK

OUT=ROOT/'outputs/mc24_four_station_wb115_field_conditions_provenance_v2'

def freeze():
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='4station':raise ValueError('branch')
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip():raise ValueError('tracked changes')
    hashes=verify(LOCAL)['hashes'].copy();interrupted=read_public(LOCAL/'local_interruption_receipt.json')
    if interrupted['saved_folder_results']!=0 or (LOCAL/'summary.json').exists():raise ValueError('local attempt changed')
    paths=[Path(__file__),ROOT/'scripts/run_wb115_condor.sh',LOCAL/'freeze.json',LOCAL/'database_open_receipt.json',LOCAL/'local_interruption_receipt.json']
    for p in paths:hashes[str(p)]=digest(p)
    OUT.mkdir(exist_ok=False)
    shutil.copyfile(WORKBOOK,OUT/'contract_workbook.md');shutil.copyfile(LOCAL/'control.json',OUT/'control.json')
    for n in ('contract_workbook.md','control.json'):hashes[str(OUT/n)]=digest(OUT/n)
    write_new(OUT/'freeze.json',{'schema':'wb115_batch_field_conditions_freeze_v1','hashes':hashes,'branch':'4station',
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'scientific_rule_changed':False,
        'event_root_access':False,'held_out_access':False,'new_propagation_calls':0,'new_field_queries':0})

def run():
    started=time.monotonic();f=verify(OUT);(OUT/'execution_lock').mkdir(exist_ok=False);control=read_public(OUT/'control.json')
    # Private copy permits local filesystem access; pin bytes before any COOL read.
    source=Path(control['database']);private=OUT/'readonly_condition_snapshot.db'
    with source.open('rb') as src,private.open('xb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
    private.chmod(0o444)
    if digest(private)!=control['database_sha256'] or digest(source)!=control['database_sha256']:raise ValueError('snapshot content drift')
    write_new(OUT/'snapshot_copy_receipt.json',{'source':str(source),'private_copy':str(private),'sha256':control['database_sha256'],
        'bytes':private.stat().st_size,'private_mode':oct(private.stat().st_mode&0o777),'scientific_input_changed':False})
    from PyCool import cool
    conn='sqlite://;schema='+str(private)+';dbname='+control['instance']
    db=cool.DatabaseSvcFactory.databaseService().openDatabase(conn,True)
    try:
        write_new(OUT/'database_open_receipt.json',{'connection':conn,'requested_readonly':True,'database_id':str(db.databaseId()),
            'source_connection':read_public(LOCAL/'database_open_receipt.json')['connection'],'network_fallback':False})
        records=[]
        for path in FOLDERS:
            for tag in TAGS:
                row=read_folder(db,path,tag,cool);records.append(row)
                write_new(OUT/('folder_'+path.rsplit('/',1)[1]+'_'+tag+'.json'),row)
    finally:db.closeDatabase()
    if digest(private)!=control['database_sha256']:raise ValueError('private conditions copy mutated')
    write_new(OUT/'summary.json',summarize(records,control));verify(OUT)
    write_new(OUT/'execution_receipt.json',{'exit_code':0,'elapsed_seconds':time.monotonic()-started,'frozen_identities_verified':len(f['hashes']),
        'mode':'single_condor_readonly_identical_conditions_copy','folder_tag_reads_attempted':4,
        'folder_tag_reads_successful':sum(r['status']=='READ' for r in records),
        'execution_contract':'PASS' if all(r['status']=='READ' for r in records) else 'FAIL_INCOMPLETE_CONDITIONS_READ',
        'source_and_private_before_after_sha256_identical':True,'event_root_access':False,'new_field_queries':0,'new_propagation_calls':0})

def submit():
    verify(OUT)
    setup='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null'
    q=subprocess.check_output(['bash','-c',setup+' && condor_q -json'],text=True,timeout=55)
    totals=subprocess.check_output(['bash','-c',setup+' && condor_q -totals'],text=True,timeout=55)
    slots=subprocess.check_output(['bash','-c',setup+' && condor_status -constraint \'State=="Unclaimed" && Activity=="Idle"\' -af Name'],text=True,timeout=55)
    write_new(OUT/'scheduler_preflight.json',{'queue':json.loads(q) if q.strip() else [],'queue_totals':totals,'idle_slots':len(slots.splitlines()),'schedd':'bigbird24.cern.ch'})
    if not slots.strip():raise RuntimeError('no idle slots; no submission')
    prior=ROOT/'outputs/mc24_four_station_wb114_persisted_type_provenance_v1'
    sub=(prior/'wb114.sub').read_text().replace(str(prior),str(OUT)).replace('run_wb114_condor.sh','run_wb115_condor.sh')
    with (OUT/'wb115.sub').open('x') as f:f.write(sub)
    r=subprocess.run(['bash','-c',setup+' && condor_submit '+shlex.quote(str(OUT/'wb115.sub'))],capture_output=True,text=True,timeout=55)
    write_new(OUT/'submission.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'submit_sha256':digest(OUT/'wb115.sub')})
    print(r.stdout,r.stderr)
    if r.returncode:raise RuntimeError('submission failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','verify','run','submit'));a=p.parse_args()
    if a.action=='verify':verify(OUT)
    else:globals()[a.action]()
