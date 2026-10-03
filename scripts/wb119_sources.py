"""Isolated algorithm using unchanged production tool and original conditions runner."""
from alignment.wb90_measurement_contract import ROOT
from wb111_sources import files as previous_files
from wb101_sources import replace_one


def files():
    previous = previous_files()
    return {'CMakeLists.txt': previous['CMakeLists.txt'].replace('WB111', 'WB119'),
            'WB119Diagnostic/CMakeLists.txt': previous['WB111Diagnostic/CMakeLists.txt'].replace('WB111', 'WB119').replace('NavigationReachability.cxx', 'BoundedResponse.cxx'),
            'WB119Diagnostic/BoundedResponse.cxx': (ROOT/'research/wb119/BoundedResponse.cxx').read_text()}


def athena_source():
    source = (ROOT/'scripts/wb92_athena.py').read_text()
    source = replace_one(source, "fixture = read_public(work/'fixture.json')", "fixture = read_public(work/'fixture.json')\nif (fixture['index'],fixture['ordinal'],fixture['actual_run'],fixture['actual_event'])!=(12,2268,100044,2268):raise ValueError('single seen allowlist')")
    start = source.index('acc.addEventAlgo(CompFactory.WB92.CommonSeedAudit(')
    end = source.index("write_new(work/'athena_manifest.json'", start)
    source = source[:start]+"""acc.addEventAlgo(CompFactory.WB119.BoundedResponse('WB119BoundedResponse', ExtrapolationTool=tool,
    FixturePath=str(work/'fixture.json'), ControlPath=str(work/'control.json'), AuditPath=str(work/'response.json')))
"""+source[end:]
    return replace_one(source, "'field_mode': 'FASER'", "'field_mode': 'FASER', 'max_official_calls': 36, 'production_tool_unchanged': True")
