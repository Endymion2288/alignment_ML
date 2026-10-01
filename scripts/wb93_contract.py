#!/usr/bin/env python3
"""Exclusive single-pilot WB93 build/run; immutable WB92 inputs and code."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,digest,read_public,write_new,FORBIDDEN
from alignment.wb93_transport_error import analytic_control,analyze
from wb92_contract import verify as verify_wb92,EXTERNAL,ACTS,EXTERNAL_SOURCES,HEADERS,run as command_run

PROTOCOL=ROOT/'configs/research_review/wp93_transport_error_budget.json'
WB92=ROOT/'outputs/mc24_four_station_wb92_common_seed_acts_v3'
SOURCES=[PROTOCOL,ROOT/'scripts/wb93_contract.py',ROOT/'scripts/wb93_athena.py',ROOT/'scripts/run_wb93_condor.sh',
    ROOT/'research/wb93/AuditBody.inc',ROOT/'research/wb92/CommonSeedAudit.cxx',ROOT/'alignment/wb93_transport_error.py',
    ROOT/'scripts/wb92_contract.py',ROOT/'scripts/setup_environment.sh',ROOT/'alignment/wb90_measurement_contract.py']

def verify(out):
    freeze=read_public(out/'freeze.json')
    for path,h in freeze['hashes'].items():
        if digest(Path(path))!=h:raise ValueError('frozen identity changed: '+path)
    if digest(out/'fixture.json')!=freeze['fixture_sha256']:raise ValueError('fixture changed')
    if read_public(out/'protocol.json')!=read_public(PROTOCOL):raise ValueError('protocol changed')
    return freeze

def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='4station':raise ValueError('4station required')
    verify_wb92(WB92)
    f=read_public(WB92/'events/00/fixture.json');prior=read_public(WB92/'events/00/validation.json')
    if prior['gate']!='FAIL' or digest(WB92/'events/00/acts.json')!=prior['acts_sha256']:raise ValueError('immutable pilot mismatch')
    if any(x in f['input_xaod'].lower() for x in FORBIDDEN):raise ValueError('forbidden source')
    if f['index']!=0 or f['ordinal']!=2270 or f['actual_run']!=100043 or f['actual_event']!=2270:raise ValueError('unexpected pilot')
    f=copy.deepcopy(f);f['wb93_protocol']=read_public(PROTOCOL);f['wb92_targets']=read_public(WB92/'events/00/acts.json')['targets']
    paths=SOURCES+[EXTERNAL/p for p in EXTERNAL_SOURCES]+[ACTS/p for p in HEADERS]
    paths+=[ACTS/'include/Acts/Propagator/Propagator.hpp',ACTS/'include/Acts/Propagator/detail/SteppingLogger.hpp',
            EXTERNAL/'MagneticField/MagFieldElements/MagFieldElements/FaserFieldCache.h']
    paths+=[WB92/'events/00'/p for p in ('fixture.json','validation.json','acts.json','athena.log')]
    paths+=[WB92/'freeze.json',WB92/'identity_payload_manifest.json']
    payload_manifest=read_public(WB92/'identity_payload_manifest.json')
    for path,h in payload_manifest['hashes'].items():
        source=WB92/'identity_payload'/path
        if digest(source)!=h:raise ValueError('payload changed')
        paths.append(source)
    for path,h in {**prior['loaded_scientific_libraries'],**prior['field_map_hashes']}.items():
        if digest(Path(path))!=h:raise ValueError('physical runtime changed')
        paths.append(Path(path))
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/'fixture.json',f);write_new(out/'protocol.json',read_public(PROTOCOL))
    write_new(out/'freeze.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'branch':'4station','hashes':{str(p):digest(p) for p in paths},'fixture_sha256':digest(out/'fixture.json'),
        'population':1,'qualification':'NOT_EVALUATED','held_out_access':False})
    control=analytic_control(f,read_public(PROTOCOL));write_new(out/'analytic_control.json',control)
    if control['gate']!='PASS':raise ValueError('analytic control failed')

def once(text,old,new):
    if text.count(old)!=1:raise ValueError('nonunique scientific fragment '+old[:60])
    return text.replace(old,new,1)

def generated_source():
    original=(ROOT/'research/wb92/CommonSeedAudit.cxx').read_text()
    prefix=original[:original.index('  void audit() {')]
    prefix=prefix.replace('namespace WB92','namespace WB93').replace('CommonSeedAudit','TransportErrorAudit')
    prefix=once(prefix,'ATH_CHECK(m_tool.retrieve());','ATH_CHECK(m_tools.retrieve());')
    prefix=once(prefix,'  ToolHandle<IFaserActsExtrapolationTool> m_tool{this,"ExtrapolationTool",""};',
        '  ToolHandleArray<IFaserActsExtrapolationTool> m_tools{this,"ExtrapolationTools",{}};\n'
        '  const IFaserActsExtrapolationTool* m_active=nullptr;')
    prefix=once(prefix,'m_tool->propagate(ctx,start,*target,','m_active->propagate(ctx,start,*target,')
    return prefix+(ROOT/'research/wb93/AuditBody.inc').read_text()+'\n};\n}\nDECLARE_COMPONENT(WB93::TransportErrorAudit)\n'

def build(out):
    source=out/'isolated_source';source.mkdir(exist_ok=False);(source/'WB93Diagnostic').mkdir()
    relocation='''
set(bad_external "${Calypso_INSTALL_DIR}/../../../../AthenaExternals/${AthenaExternals_VERSION}/InstallArea/${AthenaExternals_PLATFORM}")
foreach(name IN LISTS Calypso_TARGET_NAMES)
  if(TARGET Calypso::${name})
    foreach(property INTERFACE_INCLUDE_DIRECTORIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES INTERFACE_LINK_LIBRARIES)
      get_target_property(value Calypso::${name} ${property})
      if(value)
        string(REPLACE "${bad_external}" "${AthenaExternals_INSTALL_DIR}" fixed "${value}")
        set_target_properties(Calypso::${name} PROPERTIES ${property} "${fixed}")
      endif()
    endforeach()
  endif()
endforeach()
'''
    files={'CMakeLists.txt':'cmake_minimum_required(VERSION 3.11)\nproject(WB93 VERSION 1.0.0 LANGUAGES C CXX)\nfind_package(Calypso REQUIRED)\natlas_project(USE Calypso ${Calypso_VERSION})\n',
        'WB93Diagnostic/CMakeLists.txt':'atlas_subdir(WB93Diagnostic)\n'+relocation+
        'find_package(nlohmann_json REQUIRED)\natlas_add_component(WB93Diagnostic TransportErrorAudit.cxx LINK_LIBRARIES AthenaBaseComps StoreGateLib xAODFaserEventInfo TrackerIdentifier FaserActsGeometryLib FaserActsGeometryInterfacesLib ActsCore PRIVATE_LINK_LIBRARIES nlohmann_json::nlohmann_json)\n',
        'WB93Diagnostic/TransportErrorAudit.cxx':generated_source()}
    for name,content in files.items():
        with (source/name).open('x') as stream:stream.write(content)
    write_new(out/'generated_source_manifest.json',{name:digest(source/name) for name in files})
    command_run(['cmake','-S',str(source),'-B',str(out/'isolated_build'),'-DCalypso_DIR='+str(EXTERNAL/'run/cmake')],out,out/'configure.log')
    command_run(['cmake','--build',str(out/'isolated_build'),'-j','1'],out,out/'build.log')
    binary=out/'isolated_build/x86_64-el9-gcc13-opt/lib/libWB93Diagnostic.so'
    write_new(out/'binary_manifest.json',{'binary':str(binary),'sha256':digest(binary),
        'generated_manifest_sha256':digest(out/'generated_source_manifest.json')})

def execute(out):
    verify(out)
    if read_public(out/'analytic_control.json')['gate']!='PASS':raise ValueError('analytic control failed')
    write_new(out/'environment.json',{'host':os.uname().nodename,'condor_ad':os.environ.get('_CONDOR_JOB_AD'),
        'environment':{k:os.environ.get(k) for k in ('CMAKE_PREFIX_PATH','LD_LIBRARY_PATH','AtlasVersion','AtlasProject','CMTCONFIG')}})
    build(out)
    shutil.copytree(WB92/'identity_payload',out/'identity_payload')
    work=out/'event';work.mkdir(exist_ok=False);shutil.copyfile(out/'fixture.json',work/'fixture.json')
    platform_dir=out/'isolated_build/x86_64-el9-gcc13-opt'
    script='\n'.join(['source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso',
        'source '+shlex.quote(str(platform_dir/'setup.sh')),
        'export LD_LIBRARY_PATH='+shlex.quote(str(platform_dir/'lib'))+':"$LD_LIBRARY_PATH"',
        'python '+shlex.quote(str(ROOT/'scripts/wb93_athena.py'))+' --work-dir '+shlex.quote(str(work))+
        ' --sqlite '+shlex.quote(str(out/'identity_payload/tracker_alignment.sqlite'))])
    write_new(work/'command.json',{'script':script});command_run(['bash','-c',script],work,work/'athena.log')
    aggregate(out)

def aggregate(out):
    verify(out);work=out/'event';f=read_public(out/'fixture.json');p=read_public(out/'protocol.json')
    binary=read_public(out/'binary_manifest.json')
    if digest(Path(binary['binary']))!=binary['sha256']:raise ValueError('binary changed')
    for name,h in read_public(out/'generated_source_manifest.json').items():
        if digest(out/'isolated_source'/name)!=h:raise ValueError('generated source changed')
    if digest(work/'fixture.json')!=digest(out/'fixture.json'):raise ValueError('worker fixture differs')
    r=read_public(work/'acts.json');summary=analyze(r,f,read_public(WB92/'events/00/acts.json'),p)
    log=(work/'athena.log').read_text(errors='replace')
    if str(out/'identity_payload/tracker_alignment.sqlite') not in log or 'Using FASER magnetic field service' not in log:raise ValueError('wrong official physical path')
    field_paths=[Path(x.strip()) for x in re.findall(r'Initialized the field map from\s+([^\n]+)',log)]
    if not field_paths:raise ValueError('missing conditions map')
    libraries=[Path(x) for x in r['loaded_libraries'] if any(word in x for word in ('WB93Diagnostic','FaserActs','ActsCore','MagField'))]
    if Path(binary['binary']) not in libraries:raise ValueError('isolated binary not loaded')
    prior=read_public(WB92/'events/00/validation.json')
    for path in libraries:
        if str(path) in prior['loaded_scientific_libraries'] and digest(path)!=prior['loaded_scientific_libraries'][str(path)]:raise ValueError('physical binary changed')
    for path in field_paths:
        if str(path) not in prior['field_map_hashes'] or digest(path)!=prior['field_map_hashes'][str(path)]:raise ValueError('field map changed')
    summary.update({'acts_sha256':digest(work/'acts.json'),'log_sha256':digest(work/'athena.log'),
        'binary':binary,'field_map_hashes':{str(x):digest(x) for x in field_paths},
        'loaded_scientific_libraries':{str(x):digest(x) for x in libraries},
        'overlap_errors':re.findall(r'Layers are overlapping at: ([^\n]+)',log),
        'analytic_control':read_public(out/'analytic_control.json')})
    write_new(out/'summary.json',summary)

def submit(out):
    verify(out)
    worker=ROOT/'scripts/run_wb93_condor.sh';sub=out/'wb93.sub'
    with sub.open('x') as stream:stream.write('\n'.join(['universe = vanilla',f'executable = {worker}',
        f'arguments = {ROOT} {out}',f'output = {out}/condor.$(ClusterId).out',f'error = {out}/condor.$(ClusterId).err',
        f'log = {out}/condor.$(ClusterId).log','request_cpus = 1','request_memory = 8000','request_disk = 8000000',
        'requirements = (Arch == "X86_64")','+JobFlavour = "workday"','getenv = False','should_transfer_files = NO','queue 1','']))
    command='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(sub))
    r=subprocess.run(['bash','-c',command],capture_output=True,text=True)
    write_new(out/'submission.json',{'command':command,'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode})
    print(r.stdout);print(r.stderr,file=sys.stderr)
    if r.returncode:raise RuntimeError('submit failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','submit','aggregate']);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();out=a.output_root.resolve()
    if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb93_'):p.error('WB93 exclusive output required')
    try:
        {'freeze':freeze,'run':execute,'submit':submit,'aggregate':aggregate}[a.action](out)
    except Exception as e:
        if a.action=='run':write_new(out/'execution_error.json',{'error':repr(e),'qualification':'NOT_EVALUATED'})
        raise
