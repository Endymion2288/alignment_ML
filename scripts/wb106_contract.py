#!/usr/bin/env python3
"""One-event instrumented WB92 replay, without changing the official tool."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment.wb90_measurement_contract import read_public, write_new
from wb100_contract import digest
from wb101_sources import replace_one
from wb92_contract import run as command_run, EXTERNAL, ACTS

PARENT = ROOT/'outputs/mc24_four_station_wb92_common_seed_acts_v3'
EVENT = ROOT/'outputs/mc24_four_station_wb92_wb104_event_fixture_preflight_v2/events/12'
OUT = ROOT/'outputs/mc24_four_station_wb106_surface_trace_v1'
WORKBOOK = ROOT/'workbook/2026-10-03_106_四站SurfaceError导航机制前瞻合同.md'


def generated_source():
    s = (PARENT/'isolated_source/WB92Diagnostic/CommonSeedAudit.cxx').read_text()
    s = s.replace('namespace WB92 {', 'namespace WB106 {').replace('WB92::CommonSeedAudit', 'WB106::CommonSeedAudit')
    s = replace_one(s, '  Json m_fixture;', '''  Json m_fixture;
  mutable std::ofstream m_trace;
  mutable size_t m_call=0;
  int m_station=-1,m_axis=-1;
  std::string m_label="nominal";
  void record(const Json& row) const {
    m_trace<<row.dump()<<std::endl;
    if(!m_trace)throw std::runtime_error("WB106 trace write failed");
  }''')
    s = replace_one(s, '    return StatusCode::SUCCESS;\n  }\n  StatusCode execute()', '''    const std::string path=m_output.value()+".calls.ndjson";
    const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if(fd<0)return StatusCode::FAILURE;
    ::close(fd);m_trace.open(path);
    return StatusCode::SUCCESS;
  }
  StatusCode execute()''')
    s = replace_one(s, '    try {audit();} catch(const std::exception& e) {ATH_MSG_ERROR("WB92 fail-closed: "<<e.what());return StatusCode::FAILURE;}', '''    try {audit();record(Json{{"record","terminal"},{"status","SUCCESS"},{"calls",m_call}});}
    catch(const std::exception& e) {
      record(Json{{"record","terminal"},{"status","FAIL_CLOSED"},{"error",e.what()},{"calls",m_call}});
      ATH_MSG_ERROR("WB92 fail-closed: "<<e.what());return StatusCode::FAILURE;
    }''')
    s = replace_one(s, '    const auto start=bound(seed,z,g);', '''    const size_t cid=m_call++;
    record(Json{{"record","input"},{"call_id",cid},{"station",m_station},{"axis",m_axis},{"label",m_label},
      {"seed",encode(seed)},{"seed_z_mm",z},{"frame",encode(frame.matrix())}});
    const auto start=bound(seed,z,g);''')
    s = replace_one(s, '    if((frame.matrix()-seedFrame.matrix()).cwiseAbs().maxCoeff()<1e-12)return seed.head<4>();', '''    if((frame.matrix()-seedFrame.matrix()).cwiseAbs().maxCoeff()<1e-12) {
      record(Json{{"record","boundary"},{"call_id",cid},{"h",encode(seed.head<4>())}});
      return seed.head<4>();
    }''')
    s = replace_one(s, '    auto result=m_tool->propagate(ctx,start,*target,distance>=0?Acts::Direction::Forward:Acts::Direction::Backward);', '''    record(Json{{"record","before_official"},{"call_id",cid},{"start_position",encode(start.position(g))},
      {"start_direction",encode(start.direction())},{"distance",distance},{"forward",distance>=0},
      {"target_geometry_id",target->geometryId().value()},{"target_frame",encode(target->transform(g).matrix())}});
    auto result=m_tool->propagate(ctx,start,*target,distance>=0?Acts::Direction::Forward:Acts::Direction::Backward);
    record(Json{{"record","after_official"},{"call_id",cid},{"has_value",result.has_value()}});''')
    s = replace_one(s, '    if(!out.allFinite())throw std::runtime_error("nonfinite prediction");', '''    if(!out.allFinite())throw std::runtime_error("nonfinite prediction");
    record(Json{{"record","output"},{"call_id",cid},{"local",encode(local)},{"direction",encode(direction)},
      {"h",encode(out)},{"global_position",encode(result->position(g))}});''')
    s = replace_one(s, '    const auto* geometry=m_tool->trackingGeometryTool();', '''    Json libs=Json::array();std::ifstream loaded("/proc/self/maps");std::string mapline;std::set<std::string> unique;
    while(std::getline(loaded,mapline)){auto slash=mapline.find('/');
      if(slash!=std::string::npos && mapline.find(".so",slash)!=std::string::npos)unique.insert(mapline.substr(slash));}
    for(const auto& path:unique)libs.push_back(path);
    record(Json{{"record","event"},{"actual_run",header->runNumber()},{"actual_event",header->eventNumber()},
      {"input_xaod",m_fixture.at("input_xaod")},{"ordinal",m_fixture.at("ordinal")},{"loaded_libraries",libs}});
    const auto* geometry=m_tool->trackingGeometryTool();''')
    s = replace_one(s, '      const int station=ref.at("station");', '      const int station=ref.at("station");m_station=station;m_axis=-1;m_label="nominal";')
    s = replace_one(s, '      auto h=[&](const V5& x,const Acts::Transform3& t){return prediction(x,z,t,ctx,g);};', '''      record(Json{{"record","sensors"},{"station",station},{"sensors",sensors},{"frame",encode(frame.matrix())}});
      auto h=[&](const V5& x,const Acts::Transform3& t,const std::string& label="nominal"){
        m_label=label;return prediction(x,z,t,ctx,g);};''')
    s = replace_one(s, '{"repeat_h",encode(h(seed,frame))}', '{"repeat_h",encode(h(seed,frame,"repeat"))}')
    s = replace_one(s, '          V5 delta=V5::Zero();delta[k]=xs[k];', '          m_axis=k;V5 delta=V5::Zero();delta[k]=xs[k];')
    s = replace_one(s, '          const V4 plus=h(seed+delta,frame),minus=h(seed-delta,frame);', '          const V4 plus=h(seed+delta,frame,"xi_plus"),minus=h(seed-delta,frame,"xi_minus");')
    s = replace_one(s, '          const V4 hp=h(seed+0.5*delta,frame),hm=h(seed-0.5*delta,frame);', '          const V4 hp=h(seed+0.5*delta,frame,"xi_half_plus"),hm=h(seed-0.5*delta,frame,"xi_half_minus");')
    s = replace_one(s, '        V5 xd=xs*0.25;', '        m_axis=-1;V5 xd=xs*0.25;')
    s = replace_one(s, '{"full",encode(h(seed+xd,frame))},{"half",encode(h(seed+0.5*xd,frame))}', '{"full",encode(h(seed+xd,frame,"xi_mixed_full"))},{"half",encode(h(seed+0.5*xd,frame,"xi_mixed_half"))}')
    s = replace_one(s, '            V6 delta=V6::Zero();delta[k]=gs[k];', '            m_axis=k;V6 delta=V6::Zero();delta[k]=gs[k];')
    s = replace_one(s, '            const V4 plus=h(seed,exp(delta)*frame),minus=h(seed,exp(-delta)*frame);', '            const V4 plus=h(seed,exp(delta)*frame,"theta_plus"),minus=h(seed,exp(-delta)*frame,"theta_minus");')
    s = replace_one(s, '            const V4 hp=h(seed,exp(0.5*delta)*frame),hm=h(seed,exp(-0.5*delta)*frame);', '            const V4 hp=h(seed,exp(0.5*delta)*frame,"theta_half_plus"),hm=h(seed,exp(-0.5*delta)*frame,"theta_half_minus");')
    s = replace_one(s, '          V6 gd=gs*0.25;', '          m_axis=-1;V6 gd=gs*0.25;')
    return replace_one(s, '{"full",encode(h(seed,exp(gd)*frame))},{"half",encode(h(seed,exp(0.5*gd)*frame))}', '{"full",encode(h(seed,exp(gd)*frame,"theta_mixed_full"))},{"half",encode(h(seed,exp(0.5*gd)*frame,"theta_mixed_half"))}')


def verify(out):
    f = read_public(out/'freeze.json')
    for path, h in f['hashes'].items():
        if digest(path) != h:
            raise ValueError('frozen identity changed '+path)
    return f


def freeze(out):
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip() != '4station':
        raise ValueError('branch')
    if subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip():
        raise ValueError('uncommitted tracked changes')
    f = read_public(EVENT/'fixture.json')
    if (f['index'],f['ordinal'],f['actual_run'],f['actual_event']) != (12,2268,100044,2268):
        raise ValueError('allowlist identity')
    parent = read_public(PARENT/'freeze.json')
    hashes = parent['hashes'].copy()
    for path, h in hashes.items():
        if digest(path) != h: raise ValueError('parent dependency '+path)
    paths = [EVENT/'fixture.json',EVENT/'athena.log',
             ROOT/'scripts/wb106_contract.py',ROOT/'scripts/run_wb106_condor.sh',
             ROOT/'scripts/audit_wb106_trace.py',ROOT/'tests/test_wb106_trace.py',
             ROOT/'scripts/setup_environment.sh', ROOT/'scripts/wb92_athena.py',
             ACTS/'include/Acts/Surfaces/SurfaceError.hpp']
    paths += list((PARENT/'isolated_source').rglob('*'))
    paths += list((EVENT.parent.parent/'identity_payload').glob('*'))
    paths += [PARENT/'binary_manifest.json',PARENT/'freeze.json',PARENT/'fixtures.json']
    catalog=EVENT.parent.parent/'identity_payload/PoolFileCatalog.xml'
    for node in ET.parse(catalog).findall('.//pfn'):
        path=Path(node.attrib['name'])
        if not path.is_relative_to(ROOT/'outputs/mc24_four_station_wb91_covariance_repair_v6'):
            raise ValueError('unexpected historical POOL PFN')
        paths.append(path)
    for path in paths:
        if path.is_file(): hashes[str(path)] = digest(path)
    # Capture the actually loaded official libraries in the failed run's log.
    for path, h in read_public(EVENT.parent.parent/'events/08/validation.json')['loaded_scientific_libraries'].items():
        if digest(path) != h: raise ValueError('official library changed')
        hashes[path] = h
    for path,h in read_public(EVENT.parent.parent/'events/08/validation.json')['field_map_hashes'].items():
        if digest(path)!=h: raise ValueError('official field map changed')
        hashes[path]=h
    out.mkdir(exist_ok=False)
    shutil.copyfile(WORKBOOK,out/'contract_workbook.md')
    hashes[str(out/'contract_workbook.md')]=digest(out/'contract_workbook.md')
    write_new(out/'fixture.json',f)
    hashes[str(out/'fixture.json')] = digest(out/'fixture.json')
    write_new(out/'freeze.json',{'schema':'wb106_trace_freeze_v1','hashes':hashes,
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'branch':'4station','population':1,'held_out_access':False,'qualification':'NOT_EVALUATED'})


def run(out):
    verify(out)
    (out/'execution_lock').mkdir(exist_ok=False)
    f = read_public(out/'fixture.json')
    raw = Path(f['input_xaod']); before=raw.stat()
    if f['source_stat'] != {'bytes':before.st_size,'mtime_ns':before.st_mtime_ns}:
        raise ValueError('raw stat differs from historical fixture')
    h=digest(raw); after=raw.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('raw file changed while hashing')
    write_new(out/'raw_identity.json',{'path':str(raw),'sha256':h,'bytes':after.st_size,
              'mtime_ns':after.st_mtime_ns,'timing':'before build and event access; not a historical WB104 hash'})
    source=out/'isolated_source'; source.mkdir(exist_ok=False)
    files={'CMakeLists.txt':(PARENT/'isolated_source/CMakeLists.txt').read_text().replace('WB92','WB106'),
      'WB106Diagnostic/CMakeLists.txt':(PARENT/'isolated_source/WB92Diagnostic/CMakeLists.txt').read_text().replace('WB92','WB106'),
      'WB106Diagnostic/CommonSeedAudit.cxx':generated_source()}
    for name,content in files.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream: stream.write(content)
    write_new(out/'generated_source_manifest.json',{name:digest(source/name) for name in files})
    command_run(['cmake','-S',str(source),'-B',str(out/'isolated_build'),'-DCalypso_DIR='+str(EXTERNAL/'run/cmake')],out,out/'configure.log')
    command_run(['cmake','--build',str(out/'isolated_build'),'-j','1'],out,out/'build.log')
    binary=out/'isolated_build/x86_64-el9-gcc13-opt/lib/libWB106Diagnostic.so'
    write_new(out/'binary_manifest.json',{'binary':str(binary),'sha256':digest(binary)})
    shutil.copytree(EVENT.parent.parent/'identity_payload',out/'identity_payload')
    event=out/'event';event.mkdir(exist_ok=False)
    shutil.copyfile(out/'fixture.json',event/'fixture.json')
    athena=(ROOT/'scripts/wb92_athena.py').read_text().replace('CompFactory.WB92.CommonSeedAudit','CompFactory.WB106.CommonSeedAudit')
    with (event/'athena.py').open('x') as stream: stream.write(athena)
    platform=binary.parent.parent
    cmd='\n'.join(['source '+shlex.quote(str(ROOT/'scripts/setup_environment.sh'))+' calypso',
       'source '+shlex.quote(str(platform/'setup.sh')),
       'export LD_LIBRARY_PATH='+shlex.quote(str(platform/'lib'))+':"$LD_LIBRARY_PATH"',
       'python '+shlex.quote(str(event/'athena.py'))+' --work-dir '+shlex.quote(str(event))+
       ' --sqlite '+shlex.quote(str(out/'identity_payload/tracker_alignment.sqlite'))])
    # The copied Athena runner retains the original __file__ parent assumption.
    cmd='export PYTHONPATH='+shlex.quote(str(ROOT))+':"${PYTHONPATH:-}"\n'+cmd
    write_new(event/'command.json',{'script':cmd,'athena_sha256':digest(event/'athena.py')})
    with (event/'athena.log').open('x') as log:
        r=subprocess.run(['bash','-c',cmd],cwd=event,stdout=log,stderr=subprocess.STDOUT)
    write_new(out/'athena_exit.json',{'exit_code':r.returncode,'host':os.uname().nodename})
    verify(out)
    from audit_wb106_trace import audit
    result=audit(out)
    write_new(out/'summary.json',result)
    if result['integrity_gate'] != 'PASS': raise RuntimeError('diagnostic integrity did not pass')


def submit(out):
    verify(out)
    setup='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out'
    q=subprocess.run(['bash','-c',setup+' && condor_q -json'],capture_output=True,text=True,check=True,timeout=55)
    slots=subprocess.run(['bash','-c',setup+' && condor_status -constraint \'State=="Unclaimed" && Activity=="Idle"\' -af Name'],
                         capture_output=True,text=True,check=True,timeout=55)
    write_new(out/'scheduler_preflight.json',{'queue':json.loads(q.stdout),'idle_slots':len(slots.stdout.splitlines()),
              'schedd':'bigbird24','command':setup})
    sub=out/'wb106.sub'
    with sub.open('x') as stream:
        stream.write('\n'.join(['universe = vanilla',f'executable = {ROOT}/scripts/run_wb106_condor.sh',
           f'arguments = {ROOT} {out}',f'output = {out}/condor.$(ClusterId).out',
           f'error = {out}/condor.$(ClusterId).err',f'log = {out}/condor.$(ClusterId).log',
           'request_cpus = 1','request_memory = 8000','request_disk = 8000000',
           '+JobFlavour = "workday"','getenv = False','should_transfer_files = NO',
           'on_exit_remove = True','periodic_remove = (NumJobStarts > 1)','queue 1','']))
    r=subprocess.run(['bash','-c',setup+' && condor_submit '+shlex.quote(str(sub))],capture_output=True,text=True,timeout=55)
    write_new(out/'submission.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'submit_sha256':digest(sub)})
    print(r.stdout,r.stderr)
    if r.returncode: raise RuntimeError('submission failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','verify','run','submit'));p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();out=a.output_root.resolve()
    if out!=OUT: p.error('exclusive WB106 output required')
    globals()[a.action](out)
