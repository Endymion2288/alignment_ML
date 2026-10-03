"""Build-only recovery: retain the official interface in the isolated tool."""
import wb107_sources

def files():
    generated=wb107_sources.files()
    name='WB107Diagnostic/WB107ExtrapolationTool.h'
    source=generated[name]
    if source.count('IWB107ExtrapolationTool')!=2:
        raise ValueError('historical interface rename count')
    generated[name]=source.replace('IWB107ExtrapolationTool','IFaserActsExtrapolationTool')
    return generated
