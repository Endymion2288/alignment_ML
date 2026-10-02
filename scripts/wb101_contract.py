#!/usr/bin/env python3
"""Frozen actual-path experiment; no production source/build mutation."""
import argparse,copy,json,os,re,shlex,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,FORBIDDEN
from alignment.wb101_direction_acceptance import analyze
from wb100_contract import digest
from wb92_contract import ACTS,EXTERNAL,run as command_run
from wb97_contract import EIGEN_INCLUDE,JSON_INCLUDE
from wb101_sources import files,step_source,WB96
WB95=ROOT/'outputs/mc24_four_station_wb95_field_precision_v1'
WB100=ROOT/'outputs/mc24_four_station_wb100_aggregation_recovery_v1'
PROTOCOL=ROOT/'configs/research_review/wp101_direction_acceptance_contract.json'
SOURCES=[PROTOCOL,ROOT/'scripts/wb101_contract.py',ROOT/'scripts/wb101_sources.py',ROOT/'scripts/wb101_athena.py',ROOT/'scripts/run_wb101_condor.sh',
         ROOT/'scripts/finalize_wb101_results.py',ROOT/'scripts/audit_wb101_trials.py',ROOT/'alignment/wb101_direction_acceptance.py',ROOT/'tests/test_wb101_direction_acceptance.py']
SOURCES += list((ROOT/'research/wb101').glob('*'))
SOURCES += [ROOT/'scripts/setup_environment.sh']

def verify(out):
    freeze=read_public(out/'freeze.json')
    for path,h in freeze['hashes'].items():
        if digest(path)!=h:raise ValueError('frozen identity changed '+path)
    return freeze

def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='4station':raise ValueError('wrong branch')
    hashes={}
    for name,prior in (('wb100_cell_direction_envelope',WB100),('wb96_acts_tolerance',WB96)):
        index=read_public(ROOT/f'docs/{name}_result_manifest.json')
        for path,h in index['artifacts'].items():
            if digest(ROOT/path)!=h:raise ValueError('prior artifact changed '+path)
        for path,h in read_public(prior/'freeze.json')['hashes'].items():
            if digest(path)!=h:raise ValueError('prior dependency '+path)
            hashes[path]=h
    if read_public(WB100/'summary.json')['hypothesis']!='SUPPORTED_BUT_LIMITED':raise ValueError('WB100 prerequisite')
    f=copy.deepcopy(read_public(WB96/'fixture.json'));p=read_public(PROTOCOL)
    if (f['ordinal'],f['actual_run'],f['actual_event'])!=(2270,100043,2270) or any(x in f['input_xaod'].lower() for x in FORBIDDEN):raise ValueError('pilot allowlist')
    baseline=read_public(WB96/'event/acts.json')
    guards=[]
    for cap in p['max_step_sizes_m']:
        setting=next(x for x in baseline['settings'] if x['cap_m']==cap and x['tolerance']==p['step_tolerance'])
        rows=[setting['entry_nominal']]
        for target in setting['targets']:rows.extend(target['samples']);rows.append(target['fixed_reference_start_nominal'])
        guards.append([{k:r[k] for k in ('accepted_steps','rejected_trials','field_counts','sensitive_sequence','options')} for r in rows])
    f.update(wb101_protocol=p,wb101_field_source=str(WB95/'event/acts.json'),wb101_bounds_source=str(WB100/'bounds.json'),wb101_nodes_source=str(WB95/'event/acts.json.nodes_le_i16.bin'),wb101_baseline_guards=guards)
    out.mkdir(exist_ok=False);write_new(out/'protocol.json',p);write_new(out/'fixture.json',f)
    paths=SOURCES+[WB95/'summary.json',WB95/'event/acts.json',WB96/'event/acts.json',WB100/'summary.json',WB100/'bounds.json',WB95/'event/acts.json.nodes_le_i16.bin',
        ROOT/'docs/wb100_cell_direction_envelope_result_manifest.json',ROOT/'docs/wb96_acts_tolerance_result_manifest.json',out/'fixture.json',out/'protocol.json']
    paths+=list((WB95/'identity_payload').glob('*'))+[WB96/'isolated_source/WB96Diagnostic/ToleranceNavigationAudit.cxx',
         WB96/'isolated_source/CMakeLists.txt',WB96/'isolated_source/WB96Diagnostic/CMakeLists.txt',ROOT/'research/wb100/Envelope.h',ROOT/'research/wb99/DirectionDoubling.h']
    if any(not x.is_file() for x in paths):raise ValueError('missing frozen source')
    hashes.update({str(x):digest(x) for x in paths})
    write_new(out/'freeze.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),'branch':'4station','hashes':hashes,
              'population':1,'held_out_access':False,'qualification':'NOT_EVALUATED','production_backend_change':False})

def build_controls(out):
    controlDir=out/'control_build';controlDir.mkdir(exist_ok=False)
    sourcefiles=files()
    for name in ('Acceptance.h','Envelope.h','DirectionDoubling.h','DirectionStep.inc'):
        with (controlDir/name).open('x') as f:f.write(sourcefiles['WB101Diagnostic/'+name])
    binary=controlDir/'controls';cmd=['g++','-std=c++20','-O2','-DNDEBUG','-I'+str(ACTS/'include'),'-I'+str(EIGEN_INCLUDE),'-I'+str(JSON_INCLUDE),'-I'+str(controlDir),
      str(ROOT/'research/wb101/Controls.cxx'),'-L'+str(ACTS/'lib'),'-Wl,-rpath,'+str(ACTS/'lib'),'-lActsCore','-o',str(binary)]
    command_run(cmd,ROOT,out/'controls_build.log');command_run([str(binary)],ROOT,out/'controls.json')
    if read_public(out/'controls.json')['gate']!='PASS':raise ValueError('ACTS math controls')
    write_new(out/'controls_manifest.json',{'binary_sha256':digest(binary),'command':cmd,'sources':{x.name:digest(x) for x in controlDir.iterdir() if x.is_file()},
        'ldd':subprocess.check_output(['ldd',str(binary)],text=True)})

def build(out):
    source=out/'isolated_source';source.mkdir(exist_ok=False);(source/'WB101Diagnostic').mkdir()
    for name,content in files().items():
        with (source/name).open('x') as f:f.write(content)
    write_new(out/'generated_source_manifest.json',{name:digest(source/name) for name in files()})
    command_run(['cmake','-S',str(source),'-B',str(out/'isolated_build'),'-DCalypso_DIR='+str(EXTERNAL/'run/cmake')],out,out/'configure.log')
    command_run(['cmake','--build',str(out/'isolated_build'),'-j','1'],out,out/'build.log')
    binary=out/'isolated_build/x86_64-el9-gcc13-opt/lib/libWB101Diagnostic.so'
    write_new(out/'binary_manifest.json',{'binary':str(binary),'sha256':digest(binary),'generated_manifest_sha256':digest(out/'generated_source_manifest.json')})

def run(out):
    verify(out);write_new(out/'environment.json',{'host':os.uname().nodename,'python':sys.version,'condor_ad':os.environ.get('_CONDOR_JOB_AD'),
      'compiler':subprocess.check_output(['g++','--version'],text=True),'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH')})
    build_controls(out);build(out);shutil.copytree(WB95/'identity_payload',out/'identity_payload')
    work=out/'event';work.mkdir(exist_ok=False);shutil.copyfile(out/'fixture.json',work/'fixture.json')
    platform=out/'isolated_build/x86_64-el9-gcc13-opt'
    script='\n'.join(['source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso','source '+shlex.quote(str(platform/'setup.sh')),
        'export LD_LIBRARY_PATH='+shlex.quote(str(platform/'lib'))+':"$LD_LIBRARY_PATH"',
        'python '+shlex.quote(str(ROOT/'scripts/wb101_athena.py'))+' --work-dir '+shlex.quote(str(work))+' --sqlite '+shlex.quote(str(out/'identity_payload/tracker_alignment.sqlite'))])
    write_new(work/'command.json',{'script':script});command_run(['bash','-c',script],work,work/'athena.log');aggregate(out)

def aggregate(out):
    verify(out);bm=read_public(out/'binary_manifest.json')
    if digest(bm['binary'])!=bm['sha256']:raise ValueError('binary changed')
    for name,h in read_public(out/'generated_source_manifest.json').items():
        if digest(out/'isolated_source'/name)!=h:raise ValueError('generated source changed')
    if digest(out/'event/fixture.json')!=digest(out/'fixture.json'):raise ValueError('worker fixture')
    raw=read_public(out/'event/acts.json');log=(out/'event/athena.log').read_text(errors='replace');prior=read_public(WB95/'summary.json')
    if str(out/'identity_payload/tracker_alignment.sqlite') not in log or 'Using FASER magnetic field service' not in log:raise ValueError('physical conditions path')
    maps=[Path(x.strip()) for x in re.findall(r'Initialized the field map from\s+([^\n]+)',log)]
    libs=[Path(x) for x in raw['loaded_libraries'] if any(y in x for y in ('WB101Diagnostic','FaserActs','ActsCore','MagField'))]
    if not maps or Path(bm['binary']) not in libs:raise ValueError('physical libraries missing')
    for x in maps:
        if str(x) not in prior['field_map_hashes'] or digest(x)!=prior['field_map_hashes'][str(x)]:raise ValueError('field changed')
    for x in libs:
        if str(x) in prior['loaded_scientific_libraries'] and digest(x)!=prior['loaded_scientific_libraries'][str(x)]:raise ValueError('official library changed')
    summary=analyze(raw,read_public(out/'fixture.json'),read_public(WB96/'event/acts.json'),read_public(WB95/'event/acts.json'),prior,read_public(PROTOCOL))
    from audit_wb101_trials import audit
    write_new(out/'trial_integrity_audit.json',audit(out,raw,read_public(WB96/'event/acts.json'),read_public(PROTOCOL)))
    summary.update(loaded_scientific_libraries={str(x):digest(x) for x in libs},field_map_hashes={str(x):digest(x) for x in maps},
      overlap_errors=re.findall(r'Layers are overlapping at: ([^\n]+)',log))
    write_new(out/'summary.json',summary);verify(out)

def submit(out):
    verify(out);sub=out/'wb101.sub'
    with sub.open('x') as f:f.write('\n'.join(['universe = vanilla',f'executable = {ROOT}/scripts/run_wb101_condor.sh',f'arguments = {ROOT} {out}',
      f'output = {out}/condor.$(ClusterId).out',f'error = {out}/condor.$(ClusterId).err',f'log = {out}/condor.$(ClusterId).log',
      'request_cpus = 1','request_memory = 8000','request_disk = 10000000','requirements = (Arch == "X86_64")','+JobFlavour = "workday"',
      'getenv = False','should_transfer_files = NO','on_exit_remove = True','periodic_remove = (NumJobStarts > 1)','queue 1','']))
    cmd='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(sub))
    r=subprocess.run(['bash','-c',cmd],text=True,capture_output=True);write_new(out/'submission.json',{'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});print(r.stdout);print(r.stderr,file=sys.stderr)
    if r.returncode:raise RuntimeError('submission failed')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','run','submit','aggregate','verify','build_smoke'));parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();out=args.output_root.resolve()
    if out.parent!=ROOT/'outputs' or not out.name.startswith('mc24_four_station_wb101_'):raise ValueError('exclusive WB101 output')
    if args.action=='build_smoke':out.mkdir(exist_ok=False);build_controls(out);build(out)
    else:globals()[args.action](out)
