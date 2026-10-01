#!/usr/bin/env python3
"""Exclusive single-pilot same-version ACTS controls; production sources immutable."""
from __future__ import annotations
import argparse,copy,os,re,shlex,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,digest,read_public,write_new,FORBIDDEN
from alignment.wb96_acts_tolerance import analyze
import wb95_contract as previous
from wb92_contract import ACTS,EXTERNAL,run as command_run

PROTOCOL=ROOT/'configs/research_review/wp96_acts_tolerance_contract.json'
WB95=ROOT/'outputs/mc24_four_station_wb95_field_precision_v1'
WB94=previous.WB94
SOURCES=[PROTOCOL,ROOT/'scripts/wb96_contract.py',ROOT/'scripts/wb96_athena.py',ROOT/'scripts/run_wb96_condor.sh',
  ROOT/'research/wb96/AuditBody.inc',ROOT/'research/wb96/ObservedPropagation.h',ROOT/'alignment/wb96_acts_tolerance.py',ROOT/'tests/test_wb96_acts_tolerance.py']
ACTS_HEADERS=['Propagator/Propagator.hpp','Propagator/Propagator.ipp','Propagator/EigenStepper.hpp','Propagator/EigenStepper.ipp',
  'Propagator/Navigator.hpp','Propagator/ConstrainedStep.hpp','Propagator/MaterialInteractor.hpp','Propagator/StandardAborters.hpp',
  'Propagator/VoidNavigator.hpp','Propagator/DefaultExtension.hpp','Propagator/detail/GenericDefaultExtension.hpp',
  'Propagator/ActionList.hpp','Propagator/AbortList.hpp','Definitions/Tolerance.hpp','MagneticField/ConstantBField.hpp',
  'MagneticField/MagneticFieldProvider.hpp']

def verify(out):
    freeze=read_public(out/'freeze.json')
    for path,h in freeze['hashes'].items():
        if digest(Path(path))!=h:raise ValueError('frozen identity changed: '+path)
    if digest(out/'fixture.json')!=freeze['fixture_sha256'] or read_public(out/'protocol.json')!=read_public(PROTOCOL):raise ValueError('fixture/protocol changed')
    return freeze

def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='4station':raise ValueError('4station required')
    prior=previous.verify(WB95);s=read_public(WB95/'summary.json')
    if s['execution_contract']!='PASS' or s['mechanism_hypothesis']!='SUPPORTED_BUT_LIMITED':raise ValueError('WB95 prerequisite changed')
    if digest(WB95/'event/acts.json')!=s['acts_sha256']:raise ValueError('reference identity changed')
    f=copy.deepcopy(read_public(WB95/'fixture.json'))
    if any(x in f['input_xaod'].lower() for x in FORBIDDEN) or (f['index'],f['ordinal'],f['actual_run'],f['actual_event'])!=(0,2270,100043,2270):raise ValueError('unauthorized pilot')
    raw=read_public(WB95/'event/acts.json');inner=next(m for m in raw['modes'] if m['name']=='mesh_z_double')['ladders'][-1]['samples'][0]['interior']
    f['wb96_fixed_start']={'seed':[x[0] if isinstance(x,list) else x for x in inner['h']]+[inner['q_over_p_per_MeV']],
      'z_mm':inner['z_mm'],'time_Acts':inner['time_Acts']};f['wb96_protocol']=read_public(PROTOCOL)
    f['wb96_saved_baseline']=[]
    for cap in read_public(WB94/'event/acts.json')['caps']:
        targets=[]
        for target in cap['targets']:
            rows=[target['nominal']['direct']]+[next(a for a in d['samples'] if a['multiplier']==1)[name]['direct']
              for d in target['directions'] for name in ('central_plus','central_minus','full','half')]
            targets.append(rows)
        f['wb96_saved_baseline'].append({'entry':cap['targets'][0]['nominal']['entry'],'targets':targets})
    paths=SOURCES+[Path(x) for x in prior['hashes']]+[ACTS/'include/Acts'/x for x in ACTS_HEADERS]
    paths+=[WB95/x for x in ('freeze.json','fixture.json','protocol.json','summary.json','event/acts.json','event/athena.log','event/acts.json.nodes_le_i16.bin',
      'result_integrity.json','isolated_source/CMakeLists.txt','isolated_source/WB95Diagnostic/CMakeLists.txt')]
    paths+=[ROOT/'docs/wb95_field_precision_result_manifest.json']+[Path(x) for x in s['loaded_scientific_libraries']]
    out.mkdir(parents=True,exist_ok=False);write_new(out/'fixture.json',f);write_new(out/'protocol.json',read_public(PROTOCOL))
    write_new(out/'freeze.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'branch':'4station','hashes':{str(x):digest(x) for x in paths},'fixture_sha256':digest(out/'fixture.json'),
      'population':1,'held_out_access':False,'qualification':'NOT_EVALUATED'})

def generated_source():
    original=previous.generated_source();prefix=original[:original.index('  SG::ReadCondHandleKey<FaserFieldMapCondObj>')]
    prefix=prefix.replace('namespace WB95','namespace WB96').replace('FieldPrecisionAudit','ToleranceNavigationAudit')
    return '#include "ObservedPropagation.h"\n'+prefix+(ROOT/'research/wb96/AuditBody.inc').read_text()+'\n};\n}\nDECLARE_COMPONENT(WB96::ToleranceNavigationAudit)\n'

def build(out):
    source=out/'isolated_source';source.mkdir(exist_ok=False);(source/'WB96Diagnostic').mkdir()
    top=(WB95/'isolated_source/CMakeLists.txt').read_text().replace('WB95','WB96')
    component=(WB95/'isolated_source/WB95Diagnostic/CMakeLists.txt').read_text().replace('WB95','WB96').replace('FieldPrecisionAudit','ToleranceNavigationAudit')
    files={'CMakeLists.txt':top,'WB96Diagnostic/CMakeLists.txt':component,'WB96Diagnostic/ToleranceNavigationAudit.cxx':generated_source(),
      'WB96Diagnostic/ObservedPropagation.h':(ROOT/'research/wb96/ObservedPropagation.h').read_text(),
      'WB96Diagnostic/ReferenceRK4.h':(ROOT/'research/wb94/ReferenceRK4.h').read_text(),
      'WB96Diagnostic/PrecisionControl.h':(ROOT/'research/wb95/PrecisionControl.h').read_text(),
      'WB96Diagnostic/CompensatedRK4.h':(ROOT/'research/wb95/CompensatedRK4.h').read_text()}
    for name,content in files.items():
        with (source/name).open('x') as stream:stream.write(content)
    write_new(out/'generated_source_manifest.json',{name:digest(source/name) for name in files})
    command_run(['cmake','-S',str(source),'-B',str(out/'isolated_build'),'-DCalypso_DIR='+str(EXTERNAL/'run/cmake')],out,out/'configure.log')
    command_run(['cmake','--build',str(out/'isolated_build'),'-j','1'],out,out/'build.log')
    binary=out/'isolated_build/x86_64-el9-gcc13-opt/lib/libWB96Diagnostic.so'
    write_new(out/'binary_manifest.json',{'binary':str(binary),'sha256':digest(binary),'generated_manifest_sha256':digest(out/'generated_source_manifest.json')})

def execute(out):
    verify(out);write_new(out/'environment.json',{'host':os.uname().nodename,'condor_ad':os.environ.get('_CONDOR_JOB_AD'),
      'environment':{k:os.environ.get(k) for k in ('CMAKE_PREFIX_PATH','LD_LIBRARY_PATH','AtlasVersion','AtlasProject','CMTCONFIG')}})
    build(out);shutil.copytree(WB95/'identity_payload',out/'identity_payload')
    work=out/'event';work.mkdir(exist_ok=False);shutil.copyfile(out/'fixture.json',work/'fixture.json')
    platform=out/'isolated_build/x86_64-el9-gcc13-opt'
    script='\n'.join(['source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso',
      'source '+shlex.quote(str(platform/'setup.sh')),'export LD_LIBRARY_PATH='+shlex.quote(str(platform/'lib'))+':"$LD_LIBRARY_PATH"',
      'python '+shlex.quote(str(ROOT/'scripts/wb96_athena.py'))+' --work-dir '+shlex.quote(str(work))+' --sqlite '+shlex.quote(str(out/'identity_payload/tracker_alignment.sqlite'))])
    write_new(work/'command.json',{'script':script});command_run(['bash','-c',script],work,work/'athena.log');aggregate(out)

def aggregate(out):
    verify(out);work=out/'event';binary=read_public(out/'binary_manifest.json')
    if digest(Path(binary['binary']))!=binary['sha256'] or digest(out/'generated_source_manifest.json')!=binary['generated_manifest_sha256']:raise ValueError('binary/source changed')
    for name,h in read_public(out/'generated_source_manifest.json').items():
        if digest(out/'isolated_source'/name)!=h:raise ValueError('generated source changed')
    if digest(work/'fixture.json')!=digest(out/'fixture.json'):raise ValueError('worker fixture changed')
    r=read_public(work/'acts.json');s=analyze(r,read_public(out/'fixture.json'),read_public(WB94/'event/acts.json'),
      read_public(WB95/'event/acts.json'),read_public(WB95/'summary.json'),read_public(out/'protocol.json'))
    log=(work/'athena.log').read_text(errors='replace');prior=read_public(WB95/'summary.json')
    if str(out/'identity_payload/tracker_alignment.sqlite') not in log or 'Using FASER magnetic field service' not in log:raise ValueError('wrong runtime conditions path')
    maps=[Path(x.strip()) for x in re.findall(r'Initialized the field map from\s+([^\n]+)',log)]
    libs=[Path(x) for x in r['loaded_libraries'] if any(y in x for y in ('WB96Diagnostic','FaserActs','ActsCore','MagField'))]
    if not maps or Path(binary['binary']) not in libs:raise ValueError('missing actual map/diagnostic library')
    for path in libs:
        if str(path) in prior['loaded_scientific_libraries'] and digest(path)!=prior['loaded_scientific_libraries'][str(path)]:raise ValueError('runtime changed')
    for path in maps:
        if str(path) not in prior['field_map_hashes'] or digest(path)!=prior['field_map_hashes'][str(path)]:raise ValueError('field map changed')
    s.update({'acts_sha256':digest(work/'acts.json'),'log_sha256':digest(work/'athena.log'),'binary':binary,
      'loaded_scientific_libraries':{str(x):digest(x) for x in libs},'field_map_hashes':{str(x):digest(x) for x in maps},
      'overlap_errors':re.findall(r'Layers are overlapping at: ([^\n]+)',log)})
    write_new(out/'summary.json',s)

def submit(out):
    verify(out);sub=out/'wb96.sub'
    with sub.open('x') as stream:stream.write('\n'.join(['universe = vanilla',f'executable = {ROOT}/scripts/run_wb96_condor.sh',
      f'arguments = {ROOT} {out}',f'output = {out}/condor.$(ClusterId).out',f'error = {out}/condor.$(ClusterId).err',f'log = {out}/condor.$(ClusterId).log',
      'request_cpus = 1','request_memory = 8000','request_disk = 8000000','requirements = (Arch == "X86_64")','+JobFlavour = "workday"',
      'getenv = False','should_transfer_files = NO','on_exit_remove = True','periodic_remove = (NumJobStarts > 1)','queue 1','']))
    command='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(sub))
    r=subprocess.run(['bash','-c',command],capture_output=True,text=True)
    write_new(out/'submission.json',{'command':command,'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode})
    print(r.stdout);print(r.stderr,file=sys.stderr)
    if r.returncode:raise RuntimeError('submission failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','submit','aggregate']);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();out=a.output_root.resolve()
    if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb96_'):p.error('WB96 exclusive output required')
    try:{'freeze':freeze,'run':execute,'submit':submit,'aggregate':aggregate}[a.action](out)
    except Exception as e:
        if a.action=='run':write_new(out/'execution_error.json',{'error':repr(e),'qualification':'NOT_EVALUATED'})
        raise
