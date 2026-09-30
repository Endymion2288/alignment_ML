#!/usr/bin/env python3
"""One exact authorized development entry; paired isolated fitters, no ACTS."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import read_public,write_new,digest
from run_wb91_covariance_contract import verify

parser=argparse.ArgumentParser()
parser.add_argument('--output-root',required=True,type=Path)
parser.add_argument('--index',required=True,type=int)
parser.add_argument('--work-dir',required=True,type=Path)
parser.add_argument('--sqlite',required=True,type=Path)
args=parser.parse_args();out=args.output_root.resolve();verify(out)
row=read_public(out/'selection.json')['events'][args.index]
if row['role']!='development':raise ValueError('only development permitted')
work=args.work_dir.resolve()
if not work.is_relative_to(out) or not work.is_dir():raise ValueError('new WB91 workdir required')
sqlite=args.sqlite.resolve()
if not sqlite.is_relative_to(out) or not sqlite.is_file():raise ValueError('new WB91 sqlite required')
catalog=sqlite.with_name('PoolFileCatalog.xml')
if not catalog.is_file():raise ValueError('missing payload pool catalog')
from AthenaCommon.Configurable import Configurable
from CalypsoConfiguration.AllConfigFlags import initConfigFlags
from CalypsoConfiguration.MainServicesConfig import MainServicesCfg
from AthenaPoolCnvSvc.PoolReadConfig import PoolReadCfg
from AthenaConfiguration.ComponentFactory import CompFactory
from FaserSCT_GeoModel.FaserSCT_GeoModelConfig import FaserSCT_GeometryCfg
Configurable.configurableRun3Behavior=True
flags=initConfigFlags();flags.Input.Files=[row['input_xaod']]
flags.Input.isMC=True;flags.Input.ProjectName='data21'
flags.IOVDb.DatabaseInstance='OFLP200';flags.GeoModel.FaserVersion='FASERNU-04'
flags.IOVDb.GlobalTag='OFLCOND-FASER-06'
flags.IOVDb.SqliteInput=str(sqlite);flags.IOVDb.SqliteFolders=('/Tracker/Align',)
flags.Exec.SkipEvents=int(row['xaod_entry_index']);flags.Exec.MaxEvents=1
flags.Common.isOnline=False;flags.Beam.NumberOfCollisions=0.
flags.Detector.GeometryFaserSCT=True
flags.Concurrency.NumThreads=0;flags.Concurrency.NumConcurrentEvents=1
flags.lock()
acc=MainServicesCfg(flags);acc.merge(PoolReadCfg(flags));acc.merge(FaserSCT_GeometryCfg(flags))
acc.getService('PoolSvc').ReadCatalog += ['xmlcatalog_file:'+str(catalog)]
for mode in ('legacy','repair'):
    acc.addEventAlgo(CompFactory.Tracker.WB91SegmentFitAlg('WB91_'+mode,
        ClustersName='SCT_ClusterContainer',OutputCollection='WB91_'+mode+'_tracks',
        RepairCovariance=(mode=='repair'),AuditPath=str(work/(mode+'.jsonl'))))
write_new(work/'input_manifest.json',{'row':row,'sqlite':str(sqlite),'sqlite_sha256':digest(sqlite),
    'pool_catalog_sha256':digest(catalog),'geometry':'FASERNU-04','global_tag':'OFLCOND-FASER-06',
    'acts':False,'ghostbusters':False,'truth_association':False,
    'source_stat':{'bytes':Path(row['input_xaod']).stat().st_size}})
sc=acc.run(maxEvents=1)
sys.exit(0 if sc.isSuccess() else 1)
