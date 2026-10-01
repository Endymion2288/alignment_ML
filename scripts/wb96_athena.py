#!/usr/bin/env python3
"""One seen event; actual unchanged FASER geometry, conditions and official controls."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import read_public,write_new,digest
p=argparse.ArgumentParser();p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--sqlite',type=Path,required=True)
a=p.parse_args();work=a.work_dir.resolve();sqlite=a.sqlite.resolve();f=read_public(work/'fixture.json')
from AthenaCommon.Configurable import Configurable
from CalypsoConfiguration.AllConfigFlags import initConfigFlags
from CalypsoConfiguration.MainServicesConfig import MainServicesCfg
from AthenaPoolCnvSvc.PoolReadConfig import PoolReadCfg
from AthenaConfiguration.ComponentFactory import CompFactory
from FaserGeoModel.FaserGeoModelConfig import FaserGeometryCfg
from FaserActsGeometry.ActsGeometryConfig import ActsTrackingGeometryToolCfg
from MagFieldServices.MagFieldServicesConfig import MagneticFieldSvcCfg
Configurable.configurableRun3Behavior=True
flags=initConfigFlags();flags.Input.Files=[f['input_xaod']];flags.Input.isMC=True;flags.Input.ProjectName='data21'
flags.IOVDb.DatabaseInstance='OFLP200';flags.GeoModel.FaserVersion='FASERNU-04';flags.IOVDb.GlobalTag='OFLCOND-FASER-06'
flags.IOVDb.SqliteInput=str(sqlite);flags.IOVDb.SqliteFolders=('/Tracker/Align',)
flags.Exec.SkipEvents=f['ordinal'];flags.Exec.MaxEvents=1;flags.Common.isOnline=False;flags.Beam.NumberOfCollisions=0.
for detector in ('FaserSCT','Dipole','Veto','Trigger','Preshower','VetoNu','Ecal','Emulsion'):setattr(flags.Detector,'Geometry'+detector,True)
flags.TrackingGeometry.MaterialSource='None';flags.Concurrency.NumThreads=0;flags.Concurrency.NumConcurrentEvents=1;flags.lock()
acc=MainServicesCfg(flags);acc.merge(PoolReadCfg(flags));acc.merge(FaserGeometryCfg(flags))
acc.getService('PoolSvc').ReadCatalog+=['xmlcatalog_file:'+str(sqlite.with_name('PoolFileCatalog.xml'))]
acc.merge(MagneticFieldSvcCfg(flags));gacc,geometry=ActsTrackingGeometryToolCfg(flags);acc.merge(gacc)
tools=[]
for i,cap in enumerate(f['wb96_protocol']['max_step_sizes_m']):
    tool=CompFactory.FaserActsExtrapolationTool('WB96OfficialControl'+str(i),TrackingGeometryTool=geometry,FieldMode='FASER',
      MaxSteps=f['wb96_protocol']['max_steps'],MaxStepSize=cap,PtLoopers=f['wb96_protocol']['pt_loopers_MeV'],
      InteractionMultiScatering=False,InteractionEloss=False,InteractionRecord=False)
    acc.addPublicTool(tool);tools.append(tool)
acc.addEventAlgo(CompFactory.WB96.ToleranceNavigationAudit('WB96ToleranceNavigationAudit',ExtrapolationTools=tools,
  MapKey='fieldMapCondObj',CacheKey='fieldCondObj',FixturePath=str(work/'fixture.json'),AuditPath=str(work/'acts.json')))
write_new(work/'athena_manifest.json',{'source':f['input_xaod'],'ordinal':f['ordinal'],'fixture_sha256':digest(work/'fixture.json'),
  'sqlite':str(sqlite),'sqlite_sha256':digest(sqlite),'field_mode':'FASER','caps_m':f['wb96_protocol']['max_step_sizes_m'],
  'tolerances':f['wb96_protocol']['step_tolerances'],'rerefit':False,'truth':False,'covariance':False,'material_interactions':False,
  'engine':'isolated_same_version_EigenStepper_Navigator_actual_FASER_wrapper','production_backend_changed':False})
sc=acc.run(maxEvents=1);sys.exit(0 if sc.isSuccess() else 1)
