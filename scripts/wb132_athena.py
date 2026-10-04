#!/usr/bin/env python3
"""Read six frozen entries; instantiate no reconstruction/extrapolation tool."""
import argparse
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wb132_contract import OUT, PARENT, PROTOCOL, read, write, digest, verify

p=argparse.ArgumentParser();p.add_argument('--work-dir',required=True,type=Path);p.add_argument('--sqlite',required=True,type=Path)
a=p.parse_args();work=a.work_dir.resolve();sqlite=a.sqlite.resolve();verify();fixture=read(work/'fixture.json')
if work.parent!=OUT/'events' or work.name!=f"{fixture['index']:02d}" or fixture['index'] not in read(PROTOCOL)['indices']:raise ValueError('allowlist')
if sqlite!=PARENT/'identity_payload/tracker_alignment.sqlite':raise ValueError('payload')
from AthenaCommon.Configurable import Configurable
from CalypsoConfiguration.AllConfigFlags import initConfigFlags
from CalypsoConfiguration.MainServicesConfig import MainServicesCfg
from AthenaPoolCnvSvc.PoolReadConfig import PoolReadCfg
from AthenaConfiguration.ComponentFactory import CompFactory
from FaserGeoModel.FaserGeoModelConfig import FaserGeometryCfg
Configurable.configurableRun3Behavior=True
flags=initConfigFlags();flags.Input.Files=[fixture['input_xaod']];flags.Input.isMC=True;flags.Input.ProjectName='data21'
flags.IOVDb.DatabaseInstance='OFLP200';flags.GeoModel.FaserVersion='FASERNU-04';flags.IOVDb.GlobalTag='OFLCOND-FASER-06'
flags.IOVDb.SqliteInput=str(sqlite);flags.IOVDb.SqliteFolders=('/Tracker/Align',)
flags.Exec.SkipEvents=fixture['ordinal'];flags.Exec.MaxEvents=1;flags.Common.isOnline=False;flags.Beam.NumberOfCollisions=0.
for detector in ('FaserSCT','Dipole','Veto','Trigger','Preshower','VetoNu','Ecal','Emulsion'):setattr(flags.Detector,'Geometry'+detector,True)
flags.TrackingGeometry.MaterialSource='None';flags.Concurrency.NumThreads=0;flags.Concurrency.NumConcurrentEvents=1;flags.lock()
acc=MainServicesCfg(flags);acc.merge(PoolReadCfg(flags));acc.merge(FaserGeometryCfg(flags))
acc.getService('PoolSvc').ReadCatalog+=['xmlcatalog_file:'+str(sqlite.with_name('PoolFileCatalog.xml'))]
acc.addEventAlgo(CompFactory.WB132.NativeStateAudit('WB132NativeStateAudit',FixturePath=str(work/'fixture.json'),
                  ExportPath=str(work/'export.json'),ParentProvenancePath=str(work/'parent_provenance.json'),OutputPath=str(work/'native_response.json')))
write(work/'athena_manifest.json',{'identity':{k:fixture[k] for k in ('input_xaod','ordinal','actual_run','actual_event')},
       'geometry':'FASERNU-04','global_tag':'OFLCOND-FASER-06','sqlite_sha256':digest(sqlite),
       'catalog_sha256':digest(sqlite.with_name('PoolFileCatalog.xml')),'new_reconstruction_calls':0,'new_propagation_calls':0,
       'algorithm_field_queries':0,'truth_access':False,'held_out_access':False,
       'field_tool_instantiated':False,'extrapolation_tool_instantiated':False})
sc=acc.run(maxEvents=1);sys.exit(0 if sc.isSuccess() else 1)
