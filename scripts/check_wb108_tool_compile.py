#!/usr/bin/env python3
"""Compile the whole isolated tool without executing Athena or opening ROOT."""
import argparse,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wb108_sources import files
from wb107_contract import OUT as OLD
from alignment.wb90_measurement_contract import ROOT,write_new
from wb100_contract import digest

def check(out):
    expected=ROOT/'outputs/mc24_four_station_wb108_tool_compile_preflight_v1'
    if out.resolve()!=expected:raise ValueError('exclusive compile-only directory')
    out.mkdir(exist_ok=False)
    generated=files()
    for name,content in generated.items():
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:f.write(content)
    flags=OLD/'isolated_build/WB107Diagnostic/CMakeFiles/WB107Diagnostic.dir/flags.make'
    lines=flags.read_text().splitlines()
    includes=shlex.split(next(x.split('=',1)[1] for x in lines if x.startswith('CXX_INCLUDES =')))
    defines=shlex.split(next(x.split('=',1)[1] for x in lines if x.startswith('CXX_DEFINES =')))
    includes=['-I'+str(out/'WB107Diagnostic') if x=='-I'+str(OLD/'isolated_source/WB107Diagnostic') else x for x in includes]
    # Match the component's include/define environment; syntax-only produces no object/library.
    cmd=['g++','-std=c++20','-fsyntax-only',*defines,*includes,str(out/'WB107Diagnostic/WB107ExtrapolationTool.cxx')]
    with (out/'compile.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    write_new(out/'receipt.json',{'command':cmd,'returncode':r.returncode,'scope':'whole Athena tool translation unit; no link or execution',
       'compiler':subprocess.check_output(['which','g++'],text=True).strip(),'source_hashes':{k:digest(out/k) for k in generated},'historical_flags_sha256':digest(flags)})
    if r.returncode:raise RuntimeError('whole tool syntax check failed: '+str(out/'compile.log'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True)
    check(p.parse_args().output_root)
