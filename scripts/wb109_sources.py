"""WB108 sources with a passive wrapper around the existing world aborter."""
from wb108_sources import files as previous_files
from wb107_sources import athena_source
from wb101_sources import replace_one
from alignment.wb90_measurement_contract import ROOT

def files():
    generated=previous_files()
    name='WB107Diagnostic/WB107ExtrapolationTool.h'
    generated[name]=replace_one(generated[name],'#include "Acts/Utilities/Logger.hpp"',
      '#include "Acts/Utilities/Logger.hpp"\n#include "AbortTrace.h"')
    generated[name]=replace_one(generated[name],'using EndOfWorld = Acts::EndOfWorldReached;',
      'using EndOfWorld = WB109Trace::EndOfWorld;')
    generated['WB107Diagnostic/AbortTrace.h']=(ROOT/'research/wb109/AbortTrace.h').read_text()
    return generated
