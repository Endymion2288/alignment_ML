"""New isolated same-field component; the production tool is unchanged."""
from alignment.wb90_measurement_contract import ROOT
from wb110_sources import files as previous_files
from wb101_sources import replace_one

def files():
    previous=previous_files()
    return {'CMakeLists.txt':previous['CMakeLists.txt'].replace('WB110','WB111'),
      'WB111Diagnostic/CMakeLists.txt':previous['WB110Diagnostic/CMakeLists.txt'].replace('WB110','WB111').replace('BoundaryTopology.cxx','NavigationReachability.cxx').replace('ActsCore PRIVATE','ActsCore MagFieldConditions MagFieldElements PRIVATE'),
      'WB111Diagnostic/NavigationReachability.cxx':(ROOT/'research/wb111/NavigationReachability.cxx').read_text()}

def athena_source():
    source=(ROOT/'scripts/wb92_athena.py').read_text()
    start=source.index('acc.addEventAlgo(CompFactory.WB92.CommonSeedAudit(')
    end=source.index("write_new(work/'athena_manifest.json'",start)
    source=source[:start]+'''acc.addEventAlgo(CompFactory.WB111.NavigationReachability('WB111NavigationReachability', ExtrapolationTool=tool,
    FixturePath=str(work/'fixture.json'), ControlPath=str(work/'control.json'), AuditPath=str(work/'reachability.json')))
'''+source[end:]
    return replace_one(source,"'field_mode': 'FASER'","'field_mode': 'FASER', 'diagnostic_navigator': 'Acts::VoidNavigator', 'max_official_calls': 3, 'max_diagnostic_calls': 4")
