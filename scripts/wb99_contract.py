#!/usr/bin/env python3
"""Exclusive seen-pilot component and estimator controls; production immutable."""
import argparse,copy,os,shlex,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest,FORBIDDEN
from alignment.wb99_direction_step_doubling import aggregate
import wb96_contract as physical
import wb97_contract as local
import wb98_contract as transport
from wb92_contract import ACTS,EXTERNAL,run as command_run

PROTOCOL=ROOT/'configs/research_review/wp99_direction_step_doubling_contract.json'
WB96=local.WB96;WB95=local.WB95;WB97=transport.WB97
WB98=ROOT/'outputs/mc24_four_station_wb98_defect_transport_v1'
SOURCES=[PROTOCOL,ROOT/'research/wb99/DirectionDoubling.h',ROOT/'research/wb99/Controls.cxx',ROOT/'research/wb99/ExportBody.inc',
  ROOT/'alignment/wb99_direction_step_doubling.py',ROOT/'tests/test_wb99_direction_step_doubling.py',
  ROOT/'scripts/wb99_contract.py',ROOT/'scripts/wb99_athena.py',ROOT/'scripts/run_wb99_condor.sh',ROOT/'scripts/finalize_wb99_results.py',ROOT/'scripts/wb96_athena.py']
def verify(out):
    f=read_public(out/'freeze.json')
    for path,h in f['hashes'].items():
        if digest(Path(path))!=h:raise ValueError('frozen identity changed '+path)
    return f
def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'],text=True,cwd=ROOT).strip()!='4station':raise ValueError('4station required')
    prior=transport.verify(WB98);pp=physical.verify(WB96);p=read_public(PROTOCOL)
    if read_public(WB98/'summary.json')['hypothesis']!='SUPPORTED_BUT_LIMITED':raise ValueError('WB98 prerequisite')
    for path,h in read_public(WB98/'result_integrity.json')['artifacts'].items():
        if digest(ROOT/path)!=h:raise ValueError('WB98 artifact changed '+path)
    for path,key in ((local.RAW,'source_acts_sha256'),(local.NODES,'source_nodes_sha256'),(WB97/'steps.ndjson','source_wb97_steps_sha256')):
        if digest(path)!=p[key]:raise ValueError('pilot source changed')
    f=copy.deepcopy(read_public(WB96/'fixture.json'))
    if (f['ordinal'],f['actual_run'],f['actual_event'])!=(2270,100043,2270) or any(x in f['input_xaod'].lower() for x in FORBIDDEN):raise ValueError('pilot allowlist')
    f['wb99_protocol']=p;f['wb99_saved_raw']=str(local.RAW)
    out.mkdir(exist_ok=False);write_new(out/'protocol.json',p);write_new(out/'fixture.json',f);write_new(out/'inventory.json',read_public(WB97/'inventory.json'))
    paths=SOURCES+[Path(x) for x in prior['hashes']]+[Path(x) for x in pp['hashes']]
    paths+=[WB98/x for x in ('summary.json','freeze.json','result_integrity.json')]+[WB97/'steps.ndjson',WB96/'fixture.json',
      WB96/'isolated_source/CMakeLists.txt',WB96/'isolated_source/WB96Diagnostic/CMakeLists.txt',ROOT/'scripts/wb96_contract.py',
      ROOT/'scripts/wb95_contract.py',ROOT/'scripts/wb94_contract.py',ROOT/'scripts/wb93_contract.py',ROOT/'scripts/wb92_contract.py']
    paths+=[out/x for x in ('protocol.json','fixture.json','inventory.json')]
    write_new(out/'freeze.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
      'hashes':{str(x):digest(x) for x in paths},'branch':'4station','population':1,'held_out_access':False,'qualification':'NOT_EVALUATED'})
def generated_source():
    s=physical.generated_source().replace('namespace WB96 {','namespace WB99Component {').replace('WB96::ToleranceNavigationAudit','WB99Component::DirectionDoublingAudit').replace('ToleranceNavigationAudit','DirectionDoublingAudit')
    marker='    std::ifstream maps("/proc/self/maps");'
    if s.count(marker)!=1:raise ValueError('source injection identity')
    s=s.replace(marker,'    doubling(out,g,mctx,mh->fieldMap());\n'+marker)
    marker='  SG::ReadCondHandleKey<FaserFieldMapCondObj>'
    s=s.replace(marker,(ROOT/'research/wb99/ExportBody.inc').read_text()+'\n'+marker,1)
    return '#include "DirectionDoubling.h"\n#include <cstring>\n#include <type_traits>\n'+s
def build_controls(binary):
    binary.parent.mkdir(exist_ok=True,parents=True)
    if binary.exists():raise FileExistsError(binary)
    cmd=['g++','-std=c++20','-O2','-DNDEBUG','-I'+str(ACTS/'include'),'-I'+str(local.EIGEN_INCLUDE),'-I'+str(local.JSON_INCLUDE),
      str(ROOT/'research/wb99/Controls.cxx'),'-L'+str(ACTS/'lib'),'-Wl,-rpath,'+str(ACTS/'lib'),'-lActsCore','-o',str(binary)]
    subprocess.run(cmd,check=True);return cmd
def build_physical(out):
    source=out/'isolated_source';source.mkdir(exist_ok=False);(source/'WB99Diagnostic').mkdir()
    top=(WB96/'isolated_source/CMakeLists.txt').read_text().replace('WB96','WB99')
    cmake=(WB96/'isolated_source/WB96Diagnostic/CMakeLists.txt').read_text().replace('WB96','WB99').replace('ToleranceNavigationAudit','DirectionDoublingAudit')
    files={'CMakeLists.txt':top,'WB99Diagnostic/CMakeLists.txt':cmake,'WB99Diagnostic/DirectionDoublingAudit.cxx':generated_source(),
      'WB99Diagnostic/DirectionDoubling.h':(ROOT/'research/wb99/DirectionDoubling.h').read_text()}
    for path in ('research/wb96/ObservedPropagation.h','research/wb95/PrecisionControl.h','research/wb95/CompensatedRK4.h','research/wb94/ReferenceRK4.h'):
        files['WB99Diagnostic/'+Path(path).name]=(ROOT/path).read_text()
    for name,content in files.items():
        with (source/name).open('x') as f:f.write(content)
    write_new(out/'generated_source_manifest.json',{name:digest(source/name) for name in files})
    command_run(['cmake','-S',str(source),'-B',str(out/'isolated_build'),'-DCalypso_DIR='+str(EXTERNAL/'run/cmake')],out,out/'configure.log')
    command_run(['cmake','--build',str(out/'isolated_build'),'-j','1'],out,out/'build.log')
    binary=out/'isolated_build/x86_64-el9-gcc13-opt/lib/libWB99Diagnostic.so'
    write_new(out/'binary_manifest.json',{'binary':str(binary),'sha256':digest(binary),'generated_manifest_sha256':digest(out/'generated_source_manifest.json')})
def execute(out):
    verify(out);write_new(out/'environment.json',{'host':os.uname().nodename,'python':sys.version,'condor_ad':os.environ.get('_CONDOR_JOB_AD'),
      'environment':{k:os.environ.get(k) for k in ('CMAKE_PREFIX_PATH','LD_LIBRARY_PATH','AtlasVersion','AtlasProject','CMTCONFIG')}})
    control=out/'controls';cmd=build_controls(control);write_new(out/'control_binary_manifest.json',{'command':cmd,'sha256':digest(control),
      'ldd':subprocess.check_output(['ldd',str(control)],text=True)})
    command_run([str(control),str(PROTOCOL),str(out/'controls.json')],out,out/'controls.log')
    # Scientific control failure is recorded, not an infrastructure failure or a skip of actual data.
    build_physical(out);shutil.copytree(WB95/'identity_payload',out/'identity_payload')
    work=out/'event';work.mkdir(exist_ok=False);shutil.copyfile(out/'fixture.json',work/'fixture.json');platform=out/'isolated_build/x86_64-el9-gcc13-opt'
    script='\n'.join(['source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso','source '+shlex.quote(str(platform/'setup.sh')),
      'export LD_LIBRARY_PATH='+shlex.quote(str(platform/'lib'))+':"$LD_LIBRARY_PATH"',
      'python '+shlex.quote(str(ROOT/'scripts/wb99_athena.py'))+' --work-dir '+shlex.quote(str(work))+' --sqlite '+shlex.quote(str(out/'identity_payload/tracker_alignment.sqlite'))])
    write_new(work/'command.json',{'script':script});command_run(['bash','-c',script],work,work/'athena.log')
    aggregate_results(out)
def aggregate_results(out):
    import numpy as np
    verify(out)
    if digest(out/'event/fixture.json')!=digest(out/'fixture.json'):raise ValueError('worker fixture')
    bm=read_public(out/'binary_manifest.json')
    if digest(Path(bm['binary']))!=bm['sha256']:raise ValueError('binary changed')
    for name,h in read_public(out/'generated_source_manifest.json').items():
        if digest(out/'isolated_source'/name)!=h:raise ValueError('generated source changed')
    raw=read_public(out/'event/acts.json');prior=read_public(WB95/'summary.json');log=(out/'event/athena.log').read_text(errors='replace')
    import re
    maps=[Path(x.strip()) for x in re.findall(r'Initialized the field map from\s+([^\n]+)',log)]
    libs=[Path(x) for x in raw['loaded_libraries'] if any(y in x for y in ('WB99Diagnostic','FaserActs','ActsCore','MagField'))]
    if not maps or Path(bm['binary']) not in libs or str(out/'identity_payload/tracker_alignment.sqlite') not in log or 'Using FASER magnetic field service' not in log:raise ValueError('wrong physical path')
    for path in maps:
        if str(path) not in prior['field_map_hashes'] or digest(path)!=prior['field_map_hashes'][str(path)]:raise ValueError('field changed')
    for path in libs:
        if str(path) in prior['loaded_scientific_libraries'] and digest(path)!=prior['loaded_scientific_libraries'][str(path)]:raise ValueError('official library changed')
    old=read_public(local.OLD);nodes=np.fromfile(local.NODES,dtype='<i2').reshape(81,81,861,3)
    s=aggregate(out,read_public(out/'inventory.json'),read_public(out/'protocol.json'),WB97,old,nodes)
    s.update(loaded_scientific_libraries={str(x):digest(x) for x in libs},field_map_hashes={str(x):digest(x) for x in maps},
      artifact_hashes={name:digest(out/name) for name in ('controls.json','event/acts.json','event/acts.json.doubling.ndjson','metrics.ndjson','freeze.json','event/athena.log')},
      overlap_errors=re.findall(r'Layers are overlapping at: ([^\n]+)',log))
    write_new(out/'summary.json',s);verify(out)
def submit(out):
    verify(out);sub=out/'wb99.sub'
    with sub.open('x') as f:f.write('\n'.join(['universe = vanilla',f'executable = {ROOT}/scripts/run_wb99_condor.sh',f'arguments = {ROOT} {out}',
      f'output = {out}/condor.$(ClusterId).out',f'error = {out}/condor.$(ClusterId).err',f'log = {out}/condor.$(ClusterId).log',
      'request_cpus = 1','request_memory = 8000','request_disk = 8000000','requirements = (Arch == "X86_64")','+JobFlavour = "workday"',
      'getenv = False','should_transfer_files = NO','on_exit_remove = True','periodic_remove = (NumJobStarts > 1)','queue 1','']))
    cmd='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(sub))
    r=subprocess.run(['bash','-c',cmd],capture_output=True,text=True);write_new(out/'submission.json',{'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
    print(r.stdout);print(r.stderr,file=sys.stderr)
    if r.returncode:raise RuntimeError('submission failed')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','submit','build-controls']);parser.add_argument('--output-root',type=Path,required=True)
    a=parser.parse_args();out=a.output_root.resolve()
    if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb99_'):parser.error('exclusive WB99 output')
    if a.action=='build-controls':build_controls(out/'controls')
    else:
        try:{'freeze':freeze,'run':execute,'submit':submit}[a.action](out)
        except Exception as e:
            if a.action=='run':write_new(out/'execution_error.json',{'error':repr(e),'qualification':'NOT_EVALUATED'})
            raise
