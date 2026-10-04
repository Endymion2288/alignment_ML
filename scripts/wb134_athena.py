#!/usr/bin/env python3
"""Read-only runtime geometry inventory; no extrapolation tool."""
import argparse, sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import read_public, write_new, digest
from wb134_contract import OUT, PROTOCOL, PARENT, verify
p=argparse.ArgumentParser();p.add_argument('--work-dir',required=True,type=Path);p.add_argument('--sqlite',required=True,type=Path)
a=p.parse_args();work=a.work_dir.resolve();event=work;sqlite=a.sqlite.resolve()
verify()
fixture=read_public(event/'fixture.json');protocol=read_public(PROTOCOL);export=read_public(event/'export.json')
if fixture['index'] not in protocol['indices'] or event.parent!=OUT/'events' or event.name!=f"{fixture['index']:02d}":raise ValueError('allowlist')
if sqlite!=PARENT/'identity_payload/tracker_alignment.sqlite':raise ValueError('payload')
if read_public(event/'request.json')['identity']!=export['identity']:raise ValueError('request identity')
from AthenaCommon.Configurable import Configurable
from CalypsoConfiguration.AllConfigFlags import initConfigFlags
from CalypsoConfiguration.MainServicesConfig import MainServicesCfg
from AthenaPoolCnvSvc.PoolReadConfig import PoolReadCfg
from AthenaConfiguration.ComponentFactory import CompFactory
from FaserGeoModel.FaserGeoModelConfig import FaserGeometryCfg
from FaserActsGeometry.ActsGeometryConfig import ActsTrackingGeometryToolCfg
from MagFieldServices.MagFieldServicesConfig import MagneticFieldSvcCfg
Configurable.configurableRun3Behavior=True
flags=initConfigFlags();flags.Input.Files=[fixture['input_xaod']];flags.Input.isMC=True;flags.Input.ProjectName='data21'
flags.IOVDb.DatabaseInstance='OFLP200';flags.GeoModel.FaserVersion='FASERNU-04';flags.IOVDb.GlobalTag='OFLCOND-FASER-06';flags.IOVDb.SqliteInput=str(sqlite);flags.IOVDb.SqliteFolders=('/Tracker/Align',)
flags.Exec.SkipEvents=fixture['ordinal'];flags.Exec.MaxEvents=1;flags.Common.isOnline=False;flags.Beam.NumberOfCollisions=0.
for detector in ('FaserSCT','Dipole','Veto','Trigger','Preshower','VetoNu','Ecal','Emulsion'):setattr(flags.Detector,'Geometry'+detector,True)
flags.TrackingGeometry.MaterialSource='None';flags.Concurrency.NumThreads=0;flags.Concurrency.NumConcurrentEvents=1;flags.lock()
acc=MainServicesCfg(flags);acc.merge(PoolReadCfg(flags));acc.merge(FaserGeometryCfg(flags));acc.getService('PoolSvc').ReadCatalog += ['xmlcatalog_file:'+str(sqlite.with_name('PoolFileCatalog.xml'))]
acc.merge(MagneticFieldSvcCfg(flags));gacc,geometry=ActsTrackingGeometryToolCfg(flags);acc.merge(gacc)
acc.addEventAlgo(CompFactory.WB134.MaterialInventory('WB134MaterialInventory',TrackingGeometryTool=geometry,FixturePath=str(event/'fixture.json'),ExportPath=str(event/'export.json'),RequestPath=str(event/'request.json'),OutputPath=str(work/'response.json')))
write_new(work/'athena_manifest.json',{'identity':{k:fixture[k] for k in ('input_xaod','ordinal','actual_run','actual_event')},'geometry':'FASERNU-04','global_tag':'OFLCOND-FASER-06','field_mode':'FASER','material_source':'None','sqlite_sha256':digest(sqlite),'catalog_sha256':digest(sqlite.with_name('PoolFileCatalog.xml')),'new_reconstruction_calls':0,'new_propagation_calls':0,'algorithm_field_queries':0,'held_out_access':False,'truth_access':False})
sc=acc.run(maxEvents=1);sys.exit(0 if sc.isSuccess() else 1)
