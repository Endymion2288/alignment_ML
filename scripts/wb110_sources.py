"""Geometry-only component and runner using frozen official conditions."""
from alignment.wb90_measurement_contract import ROOT
from wb106_contract import PARENT
from wb101_sources import replace_one

def files():
    cm=(PARENT/'isolated_source/WB92Diagnostic/CMakeLists.txt').read_text().replace('WB92','WB110')
    cm=replace_one(cm,'CommonSeedAudit.cxx','BoundaryTopology.cxx')
    return {'CMakeLists.txt':(PARENT/'isolated_source/CMakeLists.txt').read_text().replace('WB92','WB110'),
      'WB110Diagnostic/CMakeLists.txt':cm,
      'WB110Diagnostic/BoundaryTopology.cxx':(ROOT/'research/wb110/BoundaryTopology.cxx').read_text()}

def athena_source():
    source=(ROOT/'scripts/wb92_athena.py').read_text()
    start=source.index('tool = CompFactory.FaserActsExtrapolationTool(')
    end=source.index("write_new(work/'athena_manifest.json'",start)
    source=source[:start]+'''acc.addEventAlgo(CompFactory.WB110.BoundaryTopology('WB110BoundaryTopology', TrackingGeometryTool=geometry,
    FixturePath=str(work/'fixture.json'), ProbePath=str(work/'probe.json'), AuditPath=str(work/'topology.json')))
'''+source[end:]
    source=replace_one(source,'acc.merge(MagneticFieldSvcCfg(flags))\n','')
    source=replace_one(source,"'field_mode': 'FASER'","'field_mode': 'NOT_QUERIED_GEOMETRY_ONLY', 'propagation': False")
    return source
