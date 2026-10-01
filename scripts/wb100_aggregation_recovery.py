#!/usr/bin/env python3
"""Reuse immutable completed C++ artifacts; normalize NumPy JSON count scalars only."""
import argparse,os,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from wb100_contract import verify,digest,OLD,NODES
from alignment.wb100_cell_direction_envelope import aggregate
from audit_wb100_numerics import audit

PARENT=ROOT/'outputs/mc24_four_station_wb100_cell_direction_envelope_v1'
OUTPUT=ROOT/'outputs/mc24_four_station_wb100_aggregation_recovery_v1'
LINKS=('metrics.ndjson','controls.json','bounds.json','terminal.json','inventory.json','protocol.json',
       'diagnostic','binary_manifest.json','ControlReference.inc','reference_source_manifest.json','selftest.json')

def json_scalars(value):
    """Lossless schema conversion; no scientific value or decision changes."""
    import numpy as np
    if isinstance(value,dict):return {k:json_scalars(v) for k,v in value.items()}
    if isinstance(value,list):return [json_scalars(x) for x in value]
    if isinstance(value,np.generic):return value.item()
    return value

def freeze(out):
    if out!=OUTPUT:raise ValueError('new exclusive recovery output')
    prior=verify(PARENT)
    if read_public(PARENT/'worker_exit.json')['exit_code']==0:raise ValueError('no recovery of successful job')
    error=(PARENT/'condor.1173729.err').read_text()
    if 'Object of type int64 is not JSON serializable' not in error:raise ValueError('different failure; investigate first')
    terminal=read_public(PARENT/'terminal.json')
    if terminal['steps']!=135355 or len(terminal['traces'])!=156:raise ValueError('C++ not complete')
    out.mkdir(exist_ok=False)
    for name in LINKS:(out/name).symlink_to(PARENT/name)
    paths=[Path(x) for x in prior['hashes']]+[PARENT/x for x in LINKS]+[PARENT/'freeze.json',PARENT/'worker_exit.json',PARENT/'condor.1173729.err',PARENT/'condor.1173729.log',PARENT/'submission.json',
        Path(__file__).resolve(),ROOT/'scripts/run_wb100_recovery_condor.sh',ROOT/'tests/test_wb100_json_recovery.py',ROOT/'scripts/finalize_wb100_results.py']
    paths.append(PARENT/'numerical_audit.json') # Preserve even the failed partial JSON.
    write_new(out/'freeze.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),'branch':'4station',
        'hashes':{str(x):digest(x) for x in paths},'parent':str(PARENT),'parent_implementation_commit':prior['commit'],
        'repair':'lossless recursive np.generic.item() JSON normalization only','C++_rerun':False,'new_wrapper_queries':0,'held_out_access':False,'qualification':'NOT_EVALUATED'})

def run(out):
    verify(out)
    import numpy as np
    write_new(out/'environment.json',{'host':os.uname().nodename,'python':sys.version,'condor_ad':os.environ.get('_CONDOR_JOB_AD')})
    result=audit(out,read_public(OLD),np.fromfile(NODES,dtype='<i2').reshape(81,81,861,3),read_public(out/'protocol.json'))
    write_new(out/'numerical_audit.json',json_scalars(result))
    write_new(out/'summary.json',json_scalars(aggregate(out,read_public(out/'protocol.json'),read_public(out/'inventory.json'))))
    verify(out)

def submit(out):
    verify(out);sub=out/'wb100_recovery.sub'
    with sub.open('x') as f:f.write('\n'.join(['universe = vanilla',f'executable = {ROOT}/scripts/run_wb100_recovery_condor.sh',f'arguments = {ROOT} {out}',
        f'output = {out}/condor.$(ClusterId).out',f'error = {out}/condor.$(ClusterId).err',f'log = {out}/condor.$(ClusterId).log',
        'request_cpus = 1','request_memory = 8000','request_disk = 8000000','requirements = (Arch == "X86_64")','+JobFlavour = "workday"',
        'getenv = False','should_transfer_files = NO','on_exit_remove = True','periodic_remove = (NumJobStarts > 1)','queue 1','']))
    cmd='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(sub))
    r=subprocess.run(['bash','-c',cmd],text=True,capture_output=True)
    write_new(out/'submission.json',{'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});print(r.stdout);print(r.stderr,file=sys.stderr)
    if r.returncode:raise RuntimeError('submission failure')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','run','verify','submit'));parser.add_argument('--output-root',type=Path,default=OUTPUT)
    args=parser.parse_args();out=args.output_root.resolve()
    if out!=OUTPUT:raise ValueError('wrong recovery output')
    globals()[args.action](out)
