"""Independent interface invariants and exact recovery source-delta checks."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wb108_sources import files
from wb107_sources import athena_source,OFFICIAL
from wb107_contract import OUT as OLD

def test_official_interface_include_and_base_are_preserved():
    s=files()['WB107Diagnostic/WB107ExtrapolationTool.h']
    assert '#include "FaserActsGeometryInterfaces/IFaserActsExtrapolationTool.h"' in s
    assert 'extends<AthAlgTool, IFaserActsExtrapolationTool>' in s
    assert 'IWB107ExtrapolationTool' not in s

def test_only_two_interface_occurrences_changed_from_frozen_generated_files():
    expected=json.loads((OLD/'generation_expectation.json').read_text())
    for name,text in files().items():
        historical=(OLD/'isolated_source'/name).read_text()
        assert hashlib.sha256(historical.encode()).hexdigest()==expected['files'][name]
        if name.endswith('WB107ExtrapolationTool.h'):
            assert historical.count('IWB107ExtrapolationTool')==2
            assert text==historical.replace('IWB107ExtrapolationTool','IFaserActsExtrapolationTool')
        else:assert text==historical
    assert hashlib.sha256(athena_source().encode()).hexdigest()==expected['athena_sha256']

def test_recovered_header_restores_official_header_without_touching_interface():
    s=files()['WB107Diagnostic/WB107ExtrapolationTool.h']
    # Class rename is reversed as tokens, not inside interface identifiers.
    import re
    s=re.sub(r'\bWB107ExtrapolationTool\b','FaserActsExtrapolationTool',s)
    s=s.replace('WB107_EXTRAPOLATION_TOOL_H','FASERACTSGEOMETRY_ACTSEXTRAPOLATIONTOOL_H')
    s=s.replace('WB107ExtrapolationDetail','ActsExtrapolationDetail')
    assert s==(OFFICIAL/'FaserActsExtrapolationTool.h').read_text()
