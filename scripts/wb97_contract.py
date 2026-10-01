#!/usr/bin/env python3
"""Frozen saved-value standalone job. Never imports or executes an Athena job."""
import argparse, json, os, shlex, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest
from alignment.wb97_rkn_local_defect import aggregate
from wb92_contract import ACTS

PROTOCOL=ROOT/'configs/research_review/wp97_rkn_local_defect_contract.json'
WB96=ROOT/'outputs/mc24_four_station_wb96_acts_tolerance_v2'
WB95=ROOT/'outputs/mc24_four_station_wb95_field_precision_v1'
RAW=WB96/'event/acts.json';OLD=WB95/'event/acts.json';NODES=WB95/'event/acts.json.nodes_le_i16.bin'
SOURCE=ROOT/'research/wb97/LocalDefect.cxx'
SOURCES=[PROTOCOL,SOURCE,ROOT/'alignment/wb97_rkn_local_defect.py',ROOT/'scripts/wb97_contract.py',
         ROOT/'scripts/run_wb97_condor.sh',ROOT/'tests/test_wb97_rkn_local_defect.py',ROOT/'scripts/setup_environment.sh',
         ROOT/'research/wb95/CompensatedRK4.h',ROOT/'research/wb94/ReferenceRK4.h',ROOT/'alignment/wb90_measurement_contract.py']
JSON_INCLUDE=Path('/cvmfs/sft.cern.ch/lcg/releases/LCG_104d_ATLAS_7/jsonmcpp/3.10.5/x86_64-el9-gcc13-opt/include')
EIGEN_INCLUDE=Path('/cvmfs/sft.cern.ch/lcg/releases/LCG_104d_ATLAS_7/eigen/3.4.0/x86_64-el9-gcc13-opt/include/eigen3')

def build(binary):
    binary.parent.mkdir(parents=True,exist_ok=True)
    if binary.exists():raise FileExistsError(binary)
    cmd=['g++','-std=c++20','-O2','-DNDEBUG','-I'+str(ACTS/'include'),'-I'+str(EIGEN_INCLUDE),'-I'+str(JSON_INCLUDE),
         '-I'+str(ROOT/'research/wb95'),'-I'+str(ROOT/'research/wb94'),str(SOURCE),'-L'+str(ACTS/'lib'),
         '-Wl,-rpath,'+str(ACTS/'lib'),'-lActsCore','-o',str(binary)]
    subprocess.run(cmd,check=True);return cmd

def verify(out):
    freeze=read_public(out/'freeze.json')
    for path,h in freeze['hashes'].items():
        if digest(Path(path))!=h:raise ValueError('frozen identity changed '+path)
    return freeze

def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'],text=True,cwd=ROOT).strip()!='4station':raise ValueError('wrong branch')
    p=read_public(PROTOCOL)
    if digest(RAW)!=p['source_acts_sha256'] or digest(NODES)!=p['source_nodes_sha256']:raise ValueError('raw/node identity changed')
    for prior in (WB96,WB95):
        for path,h in read_public(prior/'freeze.json')['hashes'].items():
            if digest(Path(path))!=h:raise ValueError('prior freeze changed '+path)
    r=read_public(RAW);traces=[]
    for setting in r['settings']:
        calls=[('entry',0,0,setting['entry_nominal'])]
        for target in setting['targets']:
            calls.extend(('direct',target['station'],i,target['samples'][i]) for i in p['trace_sample_indices'])
            calls.append(('fixed_start',target['station'],0,target['fixed_reference_start_nominal']))
        for path,station,sample,q in calls:
            if q['status']!='PASS' or len(q['accepted_trace'])!=q['accepted_steps']:raise ValueError('incomplete source trace')
            traces.append({'tolerance':setting['tolerance'],'cap_m':setting['cap_m'],'path':path,'station':station,'sample':sample,
                           'steps':q['accepted_steps'],'last_position_mm':q['accepted_trace'][-1]['position_mm']})
    if len(traces)!=p['expected_traces']:raise ValueError('trace inventory changed')
    out.mkdir(exist_ok=False)
    write_new(out/'inventory.json',{'traces':traces,'steps':sum(x['steps'] for x in traces)})
    write_new(out/'protocol.json',p)
    paths=SOURCES+[RAW,OLD,NODES,WB96/'summary.json',WB95/'summary.json',WB96/'freeze.json',WB95/'freeze.json',
                   WB96/'result_integrity.json',WB95/'result_integrity.json',ROOT/'research/wb92/CommonSeedAudit.cxx',ROOT/'research/wb96/ObservedPropagation.h',ROOT/'research/wb96/AuditBody.inc']
    paths+=list((ACTS/'include/Acts').rglob('*.hpp'))+list((ACTS/'include/Acts').rglob('*.ipp'))
    paths+=list((JSON_INCLUDE/'nlohmann').rglob('*.hpp'))+[x for x in (EIGEN_INCLUDE/'Eigen').rglob('*') if x.is_file()]
    paths+=[ACTS/'lib/libActsCore.so',Path(subprocess.check_output(['which','g++'],text=True).strip()),out/'inventory.json',out/'protocol.json']
    write_new(out/'freeze.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
                               'hashes':{str(x):digest(x) for x in paths},'population':1,'held_out_access':False,'qualification':'NOT_EVALUATED'})

def run(out):
    verify(out);write_new(out/'environment.json',{'host':os.uname().nodename,'python':sys.version,
       'compiler':subprocess.check_output(['g++','--version'],text=True),'ld_library_path':os.environ.get('LD_LIBRARY_PATH'),
       'condor_ad':os.environ.get('_CONDOR_JOB_AD')})
    binary=out/'local_defect';cmd=build(binary)
    write_new(out/'binary_manifest.json',{'command':cmd,'sha256':digest(binary),'ActsCore_sha256':digest(ACTS/'lib/libActsCore.so')})
    write_new(out/'linked_libraries.json',{'ldd':subprocess.check_output(['ldd',str(binary)],text=True)})
    with (out/'analysis.log').open('x') as log:
        subprocess.run([str(binary),str(PROTOCOL),str(RAW),str(OLD),str(NODES),str(out/'steps.ndjson')],stdout=log,stderr=subprocess.STDOUT,check=True)
    verify(out);summary=aggregate(out/'steps.ndjson',read_public(out/'inventory.json'),read_public(out/'protocol.json'))
    summary['steps_sha256']=digest(out/'steps.ndjson');summary['freeze_sha256']=digest(out/'freeze.json')
    write_new(out/'summary.json',summary)

def submit(out):
    verify(out);sub=out/'wb97.sub'
    with sub.open('x') as f:f.write('\n'.join(['universe = vanilla',f'executable = {ROOT}/scripts/run_wb97_condor.sh',f'arguments = {ROOT} {out}',
      f'output = {out}/condor.$(ClusterId).out',f'error = {out}/condor.$(ClusterId).err',f'log = {out}/condor.$(ClusterId).log',
      'request_cpus = 1','request_memory = 8000','request_disk = 8000000','requirements = (Arch == "X86_64")',
      '+JobFlavour = "workday"','getenv = False','should_transfer_files = NO','on_exit_remove = True','periodic_remove = (NumJobStarts > 1)','queue 1','']))
    command='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(sub))
    result=subprocess.run(['bash','-c',command],capture_output=True,text=True)
    write_new(out/'submission.json',{'command':command,'stdout':result.stdout,'stderr':result.stderr,'returncode':result.returncode})
    print(result.stdout);print(result.stderr,file=sys.stderr)
    if result.returncode:raise RuntimeError('submission failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','submit','build']);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();out=a.output_root.resolve()
    if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb97_'):p.error('exclusive WB97 output required')
    if a.action=='build':build(out/'local_defect')
    else:{'freeze':freeze,'run':run,'submit':submit}[a.action](out)
