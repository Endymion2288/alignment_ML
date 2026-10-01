#!/usr/bin/env python3
"""Single saved-data diagnostic. No Athena entrypoint or new physical queries."""
import argparse, hashlib, json, os, shlex, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from alignment.wb100_cell_direction_envelope import aggregate
from wb92_contract import ACTS
from wb97_contract import EIGEN_INCLUDE,JSON_INCLUDE,OLD,NODES

PROTOCOL=ROOT/'configs/research_review/wp100_cell_direction_envelope_contract.json'
WB99=ROOT/'outputs/mc24_four_station_wb99_direction_step_doubling_v1'
RECOVERY=ROOT/'outputs/mc24_four_station_wb99_aggregation_recovery_v1'
WB97=ROOT/'outputs/mc24_four_station_wb97_rkn_local_defect_v1'
RAW=WB99/'event/acts.json.doubling.ndjson'
SOURCES=[PROTOCOL,ROOT/'research/wb100/Envelope.h',ROOT/'research/wb100/Diagnostic.cxx',ROOT/'research/wb99/Controls.cxx',
         ROOT/'research/wb99/DirectionDoubling.h',ROOT/'research/wb97/LocalDefect.cxx',ROOT/'research/wb95/CompensatedRK4.h',ROOT/'research/wb94/ReferenceRK4.h',
         ROOT/'alignment/wb100_cell_direction_envelope.py',ROOT/'scripts/wb100_contract.py',ROOT/'scripts/run_wb100_condor.sh',
         ROOT/'tests/test_wb100_cell_direction_envelope.py',ROOT/'scripts/setup_environment.sh']
SOURCES += [ROOT/'scripts/audit_wb100_numerics.py']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def reference_source():
    s=(ROOT/'research/wb99/Controls.cxx').read_text()
    start=s.index('using namespace WB99;');end=s.index('int main(')
    # Exactly the original Field/rk4/reference implementation, isolated namespace.
    return '#include <algorithm>\n#include <limits>\nnamespace ControlReference {\n'+s[start:end]+'\n}\n'

def build(out):
    write_new(out/'reference_source_manifest.json',{'source':str(ROOT/'research/wb99/Controls.cxx'),'sha256':digest(ROOT/'research/wb99/Controls.cxx')})
    header=out/'ControlReference.inc'
    with header.open('x') as f:f.write(reference_source())
    binary=out/'diagnostic'
    if binary.exists():raise FileExistsError(binary)
    cmd=['g++','-std=c++20','-O2','-DNDEBUG','-I'+str(ACTS/'include'),'-I'+str(EIGEN_INCLUDE),'-I'+str(JSON_INCLUDE),
         '-I'+str(ROOT/'research/wb95'),'-I'+str(ROOT/'research/wb94'),'-I'+str(out),str(ROOT/'research/wb100/Diagnostic.cxx'),
         '-L'+str(ACTS/'lib'),'-Wl,-rpath,'+str(ACTS/'lib'),'-lActsCore','-o',str(binary)]
    with (out/'build.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    write_new(out/'binary_manifest.json',{'command':cmd,'sha256':digest(binary),'reference_header_sha256':digest(header),
              'ldd':subprocess.check_output(['ldd',str(binary)],text=True)})
    return binary

def verify(out):
    f=read_public(out/'freeze.json')
    for path,h in f['hashes'].items():
        if digest(path)!=h:raise ValueError('frozen identity changed '+path)
    return f

def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'],text=True,cwd=ROOT).strip()!='4station':raise ValueError('wrong branch')
    p=read_public(PROTOCOL)
    for path,key in ((RAW,'source_wb99_raw_sha256'),(WB99/'controls.json','source_wb99_controls_sha256'),(WB97/'steps.ndjson','source_wb97_steps_sha256'),(NODES,'source_nodes_sha256')):
        if digest(path)!=p[key]:raise ValueError('source identity '+str(path))
    prior=read_public(ROOT/'docs/wb99_direction_step_doubling_result_manifest.json')
    for path,h in prior['artifacts'].items():
        if digest(ROOT/path)!=h:raise ValueError('WB99 final artifact '+path)
    if read_public(RECOVERY/'summary.json')['hypothesis']!='NOT_SUPPORTED':raise ValueError('WB99 status')
    priorfreeze=read_public(RECOVERY/'freeze.json')
    for path,h in priorfreeze['hashes'].items():
        if digest(path)!=h:raise ValueError('WB99 frozen dependency '+path)
    out.mkdir(exist_ok=False)
    write_new(out/'protocol.json',p);write_new(out/'inventory.json',read_public(WB97/'inventory.json'))
    paths=SOURCES+[RAW,WB99/'controls.json',WB97/'steps.ndjson',OLD,NODES,RECOVERY/'summary.json',ROOT/'docs/wb99_direction_step_doubling_result_manifest.json']
    paths+=[Path(x) for x in priorfreeze['hashes']]+[out/'protocol.json',out/'inventory.json']
    paths+=[Path(subprocess.check_output(['which','g++'],text=True).strip())]
    write_new(out/'freeze.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
              'branch':'4station','hashes':{str(x):digest(x) for x in paths},'held_out_access':False,'new_wrapper_queries':0,'qualification':'NOT_EVALUATED'})

def run(out,smoke=False):
    if smoke:out.mkdir(exist_ok=False)
    else:verify(out)
    write_new(out/'environment.json',{'host':os.uname().nodename,'python':sys.version,'compiler':subprocess.check_output(['g++','--version'],text=True),
              'condor_ad':os.environ.get('_CONDOR_JOB_AD'),'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH')})
    binary=build(out)
    with (out/'selftest.json').open('x') as f:subprocess.run([str(binary),'--self-test'],stdout=f,check=True)
    cmd=[str(binary),str(PROTOCOL),str(OLD),str(NODES),str(RAW),str(WB97/'steps.ndjson'),str(WB99/'controls.json'),str(out)]
    if smoke:cmd.append('smoke')
    with (out/'diagnostic.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not smoke:
        from audit_wb100_numerics import audit
        import numpy as np
        write_new(out/'numerical_audit.json',audit(out,read_public(OLD),np.fromfile(NODES,dtype='<i2').reshape(81,81,861,3),read_public(out/'protocol.json')))
        write_new(out/'summary.json',aggregate(out,read_public(out/'protocol.json'),read_public(out/'inventory.json')));verify(out)

def submit(out):
    verify(out);sub=out/'wb100.sub'
    with sub.open('x') as f:f.write('\n'.join(['universe = vanilla',f'executable = {ROOT}/scripts/run_wb100_condor.sh',f'arguments = {ROOT} {out}',
        f'output = {out}/condor.$(ClusterId).out',f'error = {out}/condor.$(ClusterId).err',f'log = {out}/condor.$(ClusterId).log',
        'request_cpus = 1','request_memory = 8000','request_disk = 8000000','requirements = (Arch == "X86_64")','+JobFlavour = "workday"',
        'getenv = False','should_transfer_files = NO','on_exit_remove = True','periodic_remove = (NumJobStarts > 1)','queue 1','']))
    cmd='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(sub))
    r=subprocess.run(['bash','-c',cmd],capture_output=True,text=True)
    write_new(out/'submission.json',{'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});print(r.stdout);print(r.stderr,file=sys.stderr)
    if r.returncode:raise RuntimeError('submission failed')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('smoke','freeze','run','verify','submit'));parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();out=args.output_root.resolve()
    if out.parent!=ROOT/'outputs' or not out.name.startswith('mc24_four_station_wb100_'):raise ValueError('exclusive WB100 output path')
    if args.action=='smoke':run(out,True)
    else:globals()[args.action](out)
