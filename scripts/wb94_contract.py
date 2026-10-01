#!/usr/bin/env python3
"""Exclusive WB94 one-event domain/transport study; no external mutation."""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,digest,read_public,write_new,FORBIDDEN
from alignment.wb94_field_boundary import analyze
from wb93_contract import verify as verify_wb93,once
from wb92_contract import EXTERNAL,ACTS,run as command_run

PROTOCOL=ROOT/'configs/research_review/wp94_field_boundary_contract.json'
WB93=ROOT/'outputs/mc24_four_station_wb93_transport_error_v1'
SOURCES=[PROTOCOL,ROOT/'scripts/wb94_contract.py',ROOT/'scripts/wb94_athena.py',ROOT/'scripts/run_wb94_condor.sh',
    ROOT/'research/wb94/AuditBody.inc',ROOT/'research/wb94/ReferenceRK4.h',ROOT/'research/wb92/CommonSeedAudit.cxx',
    ROOT/'alignment/wb94_field_boundary.py',ROOT/'alignment/wb93_transport_error.py',ROOT/'scripts/wb93_contract.py',
    ROOT/'scripts/wb92_contract.py',ROOT/'scripts/setup_environment.sh',ROOT/'alignment/wb90_measurement_contract.py']
EXTRA=[EXTERNAL/'MagneticField'/x for x in (
    'MagFieldConditions/MagFieldConditions/FaserFieldMapCondObj.h','MagFieldConditions/src/FaserFieldMapCondObj.cxx',
    'MagFieldConditions/MagFieldConditions/FaserFieldCacheCondObj.h','MagFieldConditions/src/FaserFieldCacheCondObj.cxx',
    'MagFieldElements/MagFieldElements/FaserFieldMap.h','MagFieldElements/src/FaserFieldMap.cxx',
    'MagFieldElements/MagFieldElements/BFieldZone.h','MagFieldElements/MagFieldElements/BFieldMesh.h',
    'MagFieldElements/MagFieldElements/BFieldCache.h','MagFieldServices/src/FaserFieldCacheCondAlg.cxx',
    'MagFieldServices/src/FaserFieldCacheCondAlg.h','MagFieldServices/src/FaserFieldMapCondAlg.h')]
EXTRA+=[ACTS/'include/Acts/Propagator/detail/GenericDefaultExtension.hpp',ACTS/'include/Acts/Propagator/EigenStepper.ipp',
        ACTS/'include/Acts/Propagator/DefaultExtension.hpp',ACTS/'include/Acts/EventData/ParticleHypothesis.hpp']

def verify(out):
    f=read_public(out/'freeze.json')
    for path,h in f['hashes'].items():
        if digest(Path(path))!=h:raise ValueError('frozen identity changed: '+path)
    if digest(out/'fixture.json')!=f['fixture_sha256'] or read_public(out/'protocol.json')!=read_public(PROTOCOL):
        raise ValueError('fixture/protocol changed')
    return f

def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='4station':raise ValueError('4station required')
    previous=verify_wb93(WB93);prior=read_public(WB93/'summary.json')
    if prior['execution_contract']!='PASS' or prior['WB92_gate']!='FAIL':raise ValueError('prior state mismatch')
    if digest(WB93/'event/acts.json')!=prior['acts_sha256']:raise ValueError('WB93 result changed')
    f=copy.deepcopy(read_public(WB93/'fixture.json'))
    if any(x in f['input_xaod'].lower() for x in FORBIDDEN) or (f['index'],f['ordinal'],f['actual_run'],f['actual_event'])!=(0,2270,100043,2270):
        raise ValueError('unexpected/forbidden pilot')
    f['wb94_protocol']=read_public(PROTOCOL)
    paths=SOURCES+EXTRA+[Path(x) for x in previous['hashes']]
    paths+=[WB93/x for x in ('freeze.json','summary.json','event/acts.json','event/athena.log','fixture.json','postrun_map_metadata.json','result_integrity.json')]
    out.mkdir(parents=True,exist_ok=False);write_new(out/'fixture.json',f);write_new(out/'protocol.json',read_public(PROTOCOL))
    write_new(out/'freeze.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'branch':'4station','hashes':{str(p):digest(p) for p in paths},'fixture_sha256':digest(out/'fixture.json'),
        'population':1,'qualification':'NOT_EVALUATED','held_out_access':False})

def generated_source():
    text=(ROOT/'research/wb92/CommonSeedAudit.cxx').read_text();prefix=text[:text.index('  void audit() {')]
    prefix=prefix.replace('namespace WB92','namespace WB94').replace('CommonSeedAudit','FieldBoundaryAudit')
    prefix=once(prefix,'ATH_CHECK(m_tool.retrieve());','ATH_CHECK(m_tools.retrieve());ATH_CHECK(m_mapKey.initialize());ATH_CHECK(m_cacheKey.initialize());')
    prefix=once(prefix,'  ToolHandle<IFaserActsExtrapolationTool> m_tool{this,"ExtrapolationTool",""};',
        '  ToolHandleArray<IFaserActsExtrapolationTool> m_tools{this,"ExtrapolationTools",{}};\n  const IFaserActsExtrapolationTool* m_active=nullptr;')
    prefix=once(prefix,'m_tool->propagate(ctx,start,*target,','m_active->propagate(ctx,start,*target,')
    includes='#include "StoreGate/ReadCondHandle.h"\n#include "MagFieldConditions/FaserFieldMapCondObj.h"\n#include "MagFieldConditions/FaserFieldCacheCondObj.h"\n#include "ReferenceRK4.h"\n'
    return includes+prefix+(ROOT/'research/wb94/AuditBody.inc').read_text()+'\n};\n}\nDECLARE_COMPONENT(WB94::FieldBoundaryAudit)\n'

def build(out):
    source=out/'isolated_source';source.mkdir(exist_ok=False);(source/'WB94Diagnostic').mkdir()
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
    files={'CMakeLists.txt':'cmake_minimum_required(VERSION 3.11)\nproject(WB94 VERSION 1.0.0 LANGUAGES C CXX)\nfind_package(Calypso REQUIRED)\natlas_project(USE Calypso ${Calypso_VERSION})\n',
        'WB94Diagnostic/CMakeLists.txt':'atlas_subdir(WB94Diagnostic)\n'+relocation+'find_package(nlohmann_json REQUIRED)\natlas_add_component(WB94Diagnostic FieldBoundaryAudit.cxx LINK_LIBRARIES AthenaBaseComps StoreGateLib xAODFaserEventInfo TrackerIdentifier FaserActsGeometryLib FaserActsGeometryInterfacesLib ActsCore MagFieldConditions MagFieldElements PRIVATE_LINK_LIBRARIES nlohmann_json::nlohmann_json)\n',
        'WB94Diagnostic/FieldBoundaryAudit.cxx':generated_source(),
        'WB94Diagnostic/ReferenceRK4.h':(ROOT/'research/wb94/ReferenceRK4.h').read_text()}
    for name,content in files.items():
        with (source/name).open('x') as stream:stream.write(content)
    write_new(out/'generated_source_manifest.json',{name:digest(source/name) for name in files})
    command_run(['cmake','-S',str(source),'-B',str(out/'isolated_build'),'-DCalypso_DIR='+str(EXTERNAL/'run/cmake')],out,out/'configure.log')
    command_run(['cmake','--build',str(out/'isolated_build'),'-j','1'],out,out/'build.log')
    binary=out/'isolated_build/x86_64-el9-gcc13-opt/lib/libWB94Diagnostic.so'
    write_new(out/'binary_manifest.json',{'binary':str(binary),'sha256':digest(binary),'generated_manifest_sha256':digest(out/'generated_source_manifest.json')})

def execute(out):
    verify(out);write_new(out/'environment.json',{'host':os.uname().nodename,'condor_ad':os.environ.get('_CONDOR_JOB_AD'),
      'environment':{k:os.environ.get(k) for k in ('CMAKE_PREFIX_PATH','LD_LIBRARY_PATH','AtlasVersion','AtlasProject','CMTCONFIG')}})
    build(out);shutil.copytree(WB93/'identity_payload',out/'identity_payload')
    work=out/'event';work.mkdir(exist_ok=False);shutil.copyfile(out/'fixture.json',work/'fixture.json')
    platform_dir=out/'isolated_build/x86_64-el9-gcc13-opt'
    script='\n'.join(['source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso',
        'source '+shlex.quote(str(platform_dir/'setup.sh')),
        'export LD_LIBRARY_PATH='+shlex.quote(str(platform_dir/'lib'))+':"$LD_LIBRARY_PATH"',
        'python '+shlex.quote(str(ROOT/'scripts/wb94_athena.py'))+' --work-dir '+shlex.quote(str(work))+' --sqlite '+shlex.quote(str(out/'identity_payload/tracker_alignment.sqlite'))])
    write_new(work/'command.json',{'script':script});command_run(['bash','-c',script],work,work/'athena.log');aggregate(out)

def aggregate(out):
    verify(out);work=out/'event';binary=read_public(out/'binary_manifest.json')
    if digest(Path(binary['binary']))!=binary['sha256'] or digest(out/'generated_source_manifest.json')!=binary['generated_manifest_sha256']:
        raise ValueError('binary/source manifest changed')
    for name,h in read_public(out/'generated_source_manifest.json').items():
        if digest(out/'isolated_source'/name)!=h:raise ValueError('generated source changed')
    if digest(work/'fixture.json')!=digest(out/'fixture.json'):raise ValueError('worker fixture changed')
    r=read_public(work/'acts.json');s=analyze(r,read_public(out/'fixture.json'),read_public(WB93/'event/acts.json'),read_public(out/'protocol.json'))
    log=(work/'athena.log').read_text(errors='replace');prior=read_public(WB93/'summary.json')
    if str(out/'identity_payload/tracker_alignment.sqlite') not in log or 'Using FASER magnetic field service' not in log:raise ValueError('wrong physical path')
    field_paths=[Path(x.strip()) for x in re.findall(r'Initialized the field map from\s+([^\n]+)',log)]
    libraries=[Path(x) for x in r['loaded_libraries'] if any(y in x for y in ('WB94Diagnostic','FaserActs','ActsCore','MagField'))]
    if not field_paths or Path(binary['binary']) not in libraries:raise ValueError('missing map/isolated binary')
    for path in libraries:
        if str(path) in prior['loaded_scientific_libraries'] and digest(path)!=prior['loaded_scientific_libraries'][str(path)]:raise ValueError('runtime library changed')
    for path in field_paths:
        if str(path) not in prior['field_map_hashes'] or digest(path)!=prior['field_map_hashes'][str(path)]:raise ValueError('field map changed')
    s.update({'acts_sha256':digest(work/'acts.json'),'log_sha256':digest(work/'athena.log'),'binary':binary,
        'loaded_scientific_libraries':{str(x):digest(x) for x in libraries},'field_map_hashes':{str(x):digest(x) for x in field_paths},
        'overlap_errors':re.findall(r'Layers are overlapping at: ([^\n]+)',log)})
    write_new(out/'summary.json',s)

def submit(out):
    verify(out);sub=out/'wb94.sub'
    with sub.open('x') as stream:stream.write('\n'.join(['universe = vanilla',f'executable = {ROOT}/scripts/run_wb94_condor.sh',
        f'arguments = {ROOT} {out}',f'output = {out}/condor.$(ClusterId).out',f'error = {out}/condor.$(ClusterId).err',f'log = {out}/condor.$(ClusterId).log',
        'request_cpus = 1','request_memory = 8000','request_disk = 8000000','requirements = (Arch == "X86_64")','+JobFlavour = "workday"','getenv = False','should_transfer_files = NO','queue 1','']))
    command='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(sub))
    r=subprocess.run(['bash','-c',command],capture_output=True,text=True)
    write_new(out/'submission.json',{'command':command,'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode})
    print(r.stdout);print(r.stderr,file=sys.stderr)
    if r.returncode:raise RuntimeError('submission failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','submit','aggregate']);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();out=a.output_root.resolve()
    if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb94_'):p.error('WB94 exclusive output required')
    try:{'freeze':freeze,'run':execute,'submit':submit,'aggregate':aggregate}[a.action](out)
    except Exception as e:
        if a.action=='run':write_new(out/'execution_error.json',{'error':repr(e),'qualification':'NOT_EVALUATED'})
        raise
