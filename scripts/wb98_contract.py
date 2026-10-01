#!/usr/bin/env python3
"""Exclusive saved-value transport job, no physical executor or conditions mutation."""
import argparse,os,subprocess,sys,shlex
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest
from alignment.wb98_defect_transport import aggregate
import wb97_contract as previous
from wb92_contract import ACTS
PROTOCOL=ROOT/'configs/research_review/wp98_defect_transport_contract.json'
WB97=ROOT/'outputs/mc24_four_station_wb97_rkn_local_defect_v1'
SOURCE=ROOT/'research/wb98/DefectTransport.cxx'
SOURCES=[PROTOCOL,SOURCE,ROOT/'scripts/wb98_contract.py',ROOT/'scripts/run_wb98_condor.sh',ROOT/'alignment/wb98_defect_transport.py',ROOT/'tests/test_wb98_defect_transport.py']

def verify(out):
    frozen=read_public(out/'freeze.json')
    for name,h in frozen['hashes'].items():
        if digest(Path(name))!=h:raise ValueError('frozen identity changed '+name)
    return frozen
def build(binary):
    binary.parent.mkdir(parents=True,exist_ok=True)
    if binary.exists():raise FileExistsError(binary)
    cmd=['g++','-std=c++20','-O2','-DNDEBUG','-I'+str(ACTS/'include'),'-I'+str(previous.EIGEN_INCLUDE),'-I'+str(previous.JSON_INCLUDE),
      '-I'+str(ROOT/'research/wb97'),'-I'+str(ROOT/'research/wb95'),'-I'+str(ROOT/'research/wb94'),str(SOURCE),'-L'+str(ACTS/'lib'),'-Wl,-rpath,'+str(ACTS/'lib'),'-lActsCore','-o',str(binary)]
    subprocess.run(cmd,check=True);return cmd
def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'],text=True,cwd=ROOT).strip()!='4station':raise ValueError('wrong branch')
    prior=previous.verify(WB97);summary=read_public(WB97/'summary.json');p=read_public(PROTOCOL)
    if summary['execution_contract']!='PASS' or summary['reference_contract']!='PASS' or summary['hypothesis']!='NOT_SUPPORTED':raise ValueError('WB97 prerequisite changed')
    for path,h in read_public(WB97/'result_completion_integrity.json')['artifacts'].items():
        if digest(ROOT/path)!=h:raise ValueError('WB97 artifact changed '+path)
    for file,key in ((previous.RAW,'source_acts_sha256'),(previous.NODES,'source_nodes_sha256'),(WB97/'steps.ndjson','source_wb97_steps_sha256')):
        if digest(file)!=p[key]:raise ValueError('pilot input changed')
    out.mkdir(exist_ok=False);write_new(out/'protocol.json',p);write_new(out/'inventory.json',read_public(WB97/'inventory.json'))
    paths=SOURCES+[Path(x) for x in prior['hashes']]+[WB97/x for x in ('freeze.json','summary.json','steps.ndjson','saved_step_source_audit.json','result_completion_integrity.json')]
    paths+=list((previous.EIGEN_INCLUDE/'unsupported/Eigen').rglob('*'))
    paths=[x for x in paths if x.is_file()]+[out/'inventory.json',out/'protocol.json']
    write_new(out/'freeze.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
        'hashes':{str(x):digest(x) for x in paths},'population':1,'held_out_access':False,'qualification':'NOT_EVALUATED'})
def run(out):
    verify(out);binary=out/'defect_transport';command=build(binary)
    write_new(out/'environment.json',{'host':os.uname().nodename,'python':sys.version,'compiler':subprocess.check_output(['g++','--version'],text=True),
        'condor_ad':os.environ.get('_CONDOR_JOB_AD'),'ld_library_path':os.environ.get('LD_LIBRARY_PATH')})
    write_new(out/'binary_manifest.json',{'command':command,'sha256':digest(binary),'ldd':subprocess.check_output(['ldd',str(binary)],text=True)})
    with (out/'analysis.log').open('x') as f:
        subprocess.run([str(binary),str(PROTOCOL),str(previous.RAW),str(previous.OLD),str(previous.NODES),str(out/'steps.ndjson'),str(out/'traces.json')],stdout=f,stderr=subprocess.STDOUT,check=True)
    verify(out);s=aggregate(out,read_public(out/'inventory.json'),read_public(out/'protocol.json'))
    s.update(steps_sha256=digest(out/'steps.ndjson'),traces_sha256=digest(out/'traces.json'),freeze_sha256=digest(out/'freeze.json'))
    write_new(out/'summary.json',s)
def submit(out):
    verify(out);sub=out/'wb98.sub'
    with sub.open('x') as f:f.write('\n'.join(['universe = vanilla',f'executable = {ROOT}/scripts/run_wb98_condor.sh',f'arguments = {ROOT} {out}',
      f'output = {out}/condor.$(ClusterId).out',f'error = {out}/condor.$(ClusterId).err',f'log = {out}/condor.$(ClusterId).log',
      'request_cpus = 1','request_memory = 8000','request_disk = 8000000','requirements = (Arch == "X86_64")','+JobFlavour = "workday"',
      'getenv = False','should_transfer_files = NO','on_exit_remove = True','periodic_remove = (NumJobStarts > 1)','queue 1','']))
    cmd='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(sub))
    r=subprocess.run(['bash','-c',cmd],capture_output=True,text=True);write_new(out/'submission.json',{'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
    print(r.stdout);print(r.stderr,file=sys.stderr)
    if r.returncode:raise RuntimeError('submission failed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','freeze','run','submit']);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();out=a.output_root.resolve()
    if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb98_'):p.error('WB98 exclusive output required')
    if a.action=='build':build(out/'defect_transport')
    else:{'freeze':freeze,'run':run,'submit':submit}[a.action](out)
