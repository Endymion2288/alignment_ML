#!/usr/bin/env python3
"""Read only persisted EDM at one frozen seen entry; no reconstruction tools."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import read_public, write_new
from wb100_contract import digest

p=argparse.ArgumentParser();p.add_argument('--work-dir',required=True,type=Path);p.add_argument('--sqlite',required=True,type=Path)
a=p.parse_args();work=a.work_dir.resolve();sqlite=a.sqlite.resolve();f=read_public(work/'fixture.json')
if (f['index'],f['ordinal'],f['actual_run'],f['actual_event'])!=(12,2268,100044,2268):raise ValueError('single seen entry only')
from AthenaCommon.Configurable import Configurable
from CalypsoConfiguration.AllConfigFlags import initConfigFlags
from CalypsoConfiguration.MainServicesConfig import MainServicesCfg
from AthenaPoolCnvSvc.PoolReadConfig import PoolReadCfg
from AthenaConfiguration.ComponentFactory import CompFactory
from FaserSCT_GeoModel.FaserSCT_GeoModelConfig import FaserSCT_GeometryCfg
Configurable.configurableRun3Behavior=True
flags=initConfigFlags();flags.Input.Files=[f['input_xaod']];flags.Input.isMC=True;flags.Input.ProjectName='data21'
flags.IOVDb.DatabaseInstance='OFLP200';flags.GeoModel.FaserVersion='FASERNU-04';flags.IOVDb.GlobalTag='OFLCOND-FASER-06'
flags.IOVDb.SqliteInput=str(sqlite);flags.IOVDb.SqliteFolders=('/Tracker/Align',)
flags.Exec.SkipEvents=f['ordinal'];flags.Exec.MaxEvents=1
flags.Common.isOnline=False;flags.Beam.NumberOfCollisions=0.;flags.Detector.GeometryFaserSCT=True
flags.Concurrency.NumThreads=0;flags.Concurrency.NumConcurrentEvents=1;flags.lock()
acc=MainServicesCfg(flags);acc.merge(PoolReadCfg(flags));acc.merge(FaserSCT_GeometryCfg(flags))
acc.getService('PoolSvc').ReadCatalog+=['xmlcatalog_file:'+str(sqlite.with_name('PoolFileCatalog.xml'))]
acc.addEventAlgo(CompFactory.WB113.PersistedProvenance('WB113PersistedProvenance',
    FixturePath=str(work/'fixture.json'),ControlPath=str(work/'control.json'),AuditPath=str(work/'provenance.json')))
write_new(work/'athena_manifest.json',{'input_xaod':f['input_xaod'],'ordinal':f['ordinal'],'max_events':1,
    'geometry':'FASERNU-04','global_tag':'OFLCOND-FASER-06','sqlite_sha256':digest(sqlite),
    'pool_catalog_sha256':digest(sqlite.with_name('PoolFileCatalog.xml')),
    'rerefit':False,'acts':False,'field_queries':False,'seen_development_truth_diagnostic':True})
sc=acc.run(maxEvents=1);sys.exit(0 if sc.isSuccess() else 1)
