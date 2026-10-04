#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import read_public, write_new, digest
from wb127_contract import OUT, PROTOCOL
p=argparse.ArgumentParser();p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--sqlite',type=Path,required=True);a=p.parse_args()
work=a.work_dir.resolve(); sqlite=a.sqlite.resolve(); f=read_public(work/'fixture.json'); selected_indices=read_public(PROTOCOL)['indices']
if not ((work==OUT/'events'/f'{f["index"]:02d}') or (work.parent.parent.parent==OUT and work.parent.parent.name.startswith('recovery_'))) or f['index'] not in selected_indices:raise ValueError('allowlist')
if sqlite not in (OUT/'identity_payload/tracker_alignment.sqlite', OUT.parent/'mc24_four_station_wb125_physical_seed_v1/identity_payload/tracker_alignment.sqlite'):raise ValueError('payload')
from AthenaCommon.Configurable import Configurable
from CalypsoConfiguration.AllConfigFlags import initConfigFlags
from CalypsoConfiguration.MainServicesConfig import MainServicesCfg
from AthenaPoolCnvSvc.PoolReadConfig import PoolReadCfg
from AthenaConfiguration.ComponentFactory import CompFactory
from FaserGeoModel.FaserGeoModelConfig import FaserGeometryCfg
Configurable.configurableRun3Behavior=True
flags=initConfigFlags();flags.Input.Files=[f['input_xaod']];flags.Input.isMC=True;flags.Input.ProjectName='data21';flags.IOVDb.DatabaseInstance='OFLP200';flags.GeoModel.FaserVersion='FASERNU-04';flags.IOVDb.GlobalTag='OFLCOND-FASER-06';flags.IOVDb.SqliteInput=str(sqlite);flags.IOVDb.SqliteFolders=('/Tracker/Align',);flags.Exec.SkipEvents=f['ordinal'];flags.Exec.MaxEvents=1;flags.Common.isOnline=False;flags.Beam.NumberOfCollisions=0.
for d in ('FaserSCT','Dipole','Veto','Trigger','Preshower','VetoNu','Ecal','Emulsion'):setattr(flags.Detector,'Geometry'+d,True)
flags.TrackingGeometry.MaterialSource='None';flags.Concurrency.NumThreads=0;flags.Concurrency.NumConcurrentEvents=1;flags.lock()
acc=MainServicesCfg(flags);acc.merge(PoolReadCfg(flags));acc.merge(FaserGeometryCfg(flags));acc.getService('PoolSvc').ReadCatalog += ['xmlcatalog_file:'+str(sqlite.with_name('PoolFileCatalog.xml'))]
acc.addEventAlgo(CompFactory.WB127.StripMeasurementExport('WB127StripMeasurementExport',FixturePath=str(work/'fixture.json'),SeedPath=str(work/'p_seed.json'),OutputPath=str(work/'export.json')))
write_new(work/'athena_manifest.json',{'identity':{k:f[k] for k in ('input_xaod','ordinal','actual_run','actual_event')},'geometry':'FASERNU-04','global_tag':'OFLCOND-FASER-06','sqlite_sha256':digest(sqlite),'catalog_sha256':digest(sqlite.with_name('PoolFileCatalog.xml')),'new_reconstruction_calls':1,'new_propagation_calls':0,'held_out_access':False})
sc=acc.run(maxEvents=1);sys.exit(0 if sc.isSuccess() else 1)
