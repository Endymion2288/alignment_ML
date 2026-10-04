#!/usr/bin/env python3
"""One frozen seen ordinal: persisted reader then the bounded official seed intervention."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from wb125_contract import digest,OUT,PROTOCOL

p=argparse.ArgumentParser();p.add_argument('--work-dir',required=True,type=Path);p.add_argument('--sqlite',required=True,type=Path)
a=p.parse_args();work=a.work_dir.resolve();sqlite=a.sqlite.resolve();f=read_public(work/'fixture.json')
selected=next((r for r in read_public(PROTOCOL)['selected'] if r['index']==f['index']),None)
if selected is None or work!=OUT/'events'/f'{f["index"]:02d}' or any(f[k]!=selected[k] for k in ('input_xaod','ordinal','actual_run','actual_event')):raise ValueError('six seen entry allowlist')
if sqlite!=OUT/'identity_payload/tracker_alignment.sqlite':raise ValueError('frozen identity payload')

import ROOT as R
from PyUtils.MetaReader import read_metadata
md=read_metadata([f['input_xaod']],mode='full',promote=False,unique_tag_info_values=False)
write_new(work/'source_metadata.json',md)
rootfile=R.TFile.Open(f['input_xaod'],'READ')
if not rootfile or rootfile.IsZombie():raise ValueError('source open')
tree=rootfile.Get('CollectionTree')
if not tree:raise ValueError('CollectionTree absent')
branches=[{'name':b.GetName(),'class':b.GetClassName(),'title':b.GetTitle()} for b in tree.GetListOfBranches()]
matching=[b for b in branches if b['name']=='CKFTrackCollection']
if matching and (len(matching)!=1 or matching[0]['class']!='Trk::TrackCollection_tlp6'):raise ValueError('primary persistent type')
write_new(work/'root_metadata.json',{'branches':branches,'entries':int(tree.GetEntries()),'uuid':rootfile.GetUUID().AsString(),'event_entries_decoded':0})
rootfile.Close()

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
tool=CompFactory.FaserActsExtrapolationTool('WB125OfficialExtrapolation',TrackingGeometryTool=geometry,
    FieldMode='FASER',MaxSteps=10000,MaxStepSize=10.,InteractionMultiScatering=False,InteractionEloss=False,InteractionRecord=False)
acc.addPublicTool(tool)
acc.addEventAlgo(CompFactory.WB125.PersistedProvenance('WB125PersistedProvenance',FixturePath=str(work/'fixture.json'),
    ControlPath=str(work/'control.json'),AuditPath=str(work/'provenance.json')))
acc.addEventAlgo(CompFactory.WB125.PhysicalSeedAudit('WB125PhysicalSeedAudit',ExtrapolationTool=tool,
    FixturePath=str(work/'fixture.json'),ProvenancePath=str(work/'provenance.json'),AuditPath=str(work/'response.json')))
write_new(work/'athena_manifest.json',{'identity':{k:f[k] for k in ('input_xaod','ordinal','actual_run','actual_event')},
    'max_events':1,'geometry':'FASERNU-04','global_tag':'OFLCOND-FASER-06','field_mode':'FASER',
    'sqlite_sha256':digest(sqlite),'catalog_sha256':digest(sqlite.with_name('PoolFileCatalog.xml')),
    'primary_collection':'CKFTrackCollection','truth':'only_original_allowlisted_seen_RDO_links_and_related_particle_metadata',
    'material':False,'covariance_transport':False,'rerefit':False,'held_out_access':False})
sc=acc.run(maxEvents=1);sys.exit(0 if sc.isSuccess() else 1)
