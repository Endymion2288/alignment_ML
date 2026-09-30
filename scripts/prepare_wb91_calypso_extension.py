#!/usr/bin/env python3
"""Generate an isolated renamed diagnostic fitter from frozen original sources."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT as PROJECT,write_new,digest,read_public
from run_wb91_covariance_contract import verify,OUTPUT

def replace_once(text,old,new):
    if text.count(old)!=1: raise ValueError(f'expected unique source fragment: {old[:80]}')
    return text.replace(old,new,1)

def prepare(out):
    verify(out)
    if read_public(out/'kernel_summary.json')['gate']!='PASS': raise ValueError('kernel prerequisite failed')
    source=out/'isolated_source'
    if source.exists(): raise FileExistsError(source)
    origin=PROJECT.parent/'calypso/Tracker/TrackerRecAlgs/TrackerSegmentFit'
    header=(origin/'src/SegmentFitAlg.h').read_text().replace('SegmentFitAlg','WB91SegmentFitAlg').replace('FASERSEGMENTFIT_SEGMENTFITALG_H','WB91SEGMENTFITALG_H')
    header=replace_once(header,'#include <map>','#include <map>\n#include <fstream>')
    header=replace_once(header,'    static constexpr size_t compatibilityMaxSize',
        '    Gaudi::Property<bool> m_repair{this,"RepairCovariance",false};\n'
        '    Gaudi::Property<std::string> m_auditPath{this,"AuditPath",""};\n'
        '    mutable std::ofstream m_audit;\n\n    static constexpr size_t compatibilityMaxSize')
    code=(origin/'src/SegmentFitAlg.cxx').read_text().replace('SegmentFitAlg','WB91SegmentFitAlg')
    code=replace_once(code,'#include <list>','#include <list>\n#include "Audit.h"\n#include "GaudiKernel/ThreadLocalContext.h"\n#include "xAODEventInfo/EventInfo.h"\n#include <fcntl.h>\n#include <unistd.h>\n#include <sstream>')
    code=replace_once(code,'  // Define the edge region',
        '  const int auditFd=::open(m_auditPath.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);\n'
        '  if(auditFd<0) {ATH_MSG_ERROR("WB91 exclusive audit creation failed");return StatusCode::FAILURE;}\n'
        '  ::close(auditFd);m_audit.open(m_auditPath.value(),std::ios::app);\n'
        '  if(!m_audit) return StatusCode::FAILURE;\n\n  // Define the edge region')
    code=replace_once(code,'    std::unique_ptr<FaserSCT_ClusterOnTrack> rot = nullptr;',
        '    if(m_repair) {\n'
        '      Trk::CurvilinearParameters nominal(pos,phi,theta,qoverp);\n'
        '      *covPar5=WB91::nativeCovariance(fitResult,fitCovariance,zFit-fitInfo::zCenter,nominal);\n'
        '    }\n\n    std::unique_ptr<FaserSCT_ClusterOnTrack> rot = nullptr;')
    code=replace_once(code,'    // Create and store track',
        '    WB91::M4 normal;const auto& v=theFit->sums;\n'
        '    normal << v[1],v[2],v[4],v[5],v[2],v[3],v[5],v[6],\n'
        '              v[4],v[5],v[7],v[8],v[5],v[6],v[8],v[9];\n'
        '    std::ostringstream ids;ids << "[";bool first=true;\n'
        '    for(int ci:theFit->candidates) {if(!first)ids << ",";first=false;\n'
        '      ids << theFit->clusters[ci]->cluster.identify().get_compact();}\n'
        '    ids << "]";\n'
        '    const auto& eid=Gaudi::Hive::currentContext().eventID();\n'
        '    const xAOD::EventInfo* inputHeader=nullptr;ATH_CHECK(evtStore()->retrieve(inputHeader,"EventInfo"));\n'
        '    if(inputHeader->runNumber()!=eid.run_number() || inputHeader->eventNumber()!=eid.event_number())\n'
        '      {ATH_MSG_ERROR("WB91 input header/context mismatch");return StatusCode::FAILURE;}\n'
        '    const int station=m_idHelper->station(theFit->clusters[theFit->candidates.front()]->cluster.detectorElement()->identify());\n'
        '    int stateIndex=0;\n'
        '    for(const auto* state:*s) WB91::dumpState(m_audit,eid.run_number(),eid.event_number(),station,ids.str(),stateIndex++,\n'
        '       theFit->fitParams,theFit->fitCovariance,normal,fitInfo::zCenter,*state->trackParameters());\n'
        '    m_audit.flush();\n\n    // Create and store track')
    cmake=(origin/'CMakeLists.txt').read_text().replace('TrackerSegmentFit','WB91SegmentFit')
    cmake=cmake[:cmake.index('atlas_install_python_modules')]
    cmake=cmake.replace('TrackerEventTPCnv )','TrackerEventTPCnv xAODEventInfo )')
    generated={'CMakeLists.txt':
        'cmake_minimum_required(VERSION 3.11)\nproject(WB91 VERSION 1.0.0 LANGUAGES C CXX)\n'
        'find_package(Calypso REQUIRED)\natlas_project(USE Calypso ${Calypso_VERSION})\n',
        'WB91SegmentFit/CMakeLists.txt':cmake,
        'WB91SegmentFit/src/WB91SegmentFitAlg.h':header,
        'WB91SegmentFit/src/WB91SegmentFitAlg.cxx':code,
        'WB91SegmentFit/src/components/entries.cxx':'#include "../WB91SegmentFitAlg.h"\nDECLARE_COMPONENT(Tracker::WB91SegmentFitAlg)\n',
        'WB91SegmentFit/src/CovarianceContract.h':(PROJECT/'research/wb91/CovarianceContract.h').read_text(),
        'WB91SegmentFit/src/Audit.h':(PROJECT/'research/wb91/Audit.h').read_text()}
    for name,text in generated.items():
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as stream:stream.write(text)
    write_new(out/'extension_manifest.json',{'original_fitter_sha256':digest(origin/'src/SegmentFitAlg.cxx'),
        'generated_source_hashes':{name:digest(source/name) for name in generated},
        'generator_sha256':digest(Path(__file__)),
        'production_source_modified':False,'production_build_modified':False})
    print(source)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,default=OUTPUT)
    args=parser.parse_args();out=args.output_root.resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('mc24_four_station_wb91_'):parser.error('WB91 output required')
    prepare(out)
