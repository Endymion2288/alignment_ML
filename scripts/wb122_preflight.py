"""Whole two-TU syntax checks and actual-header read-only cache self-test."""
import argparse, json, shlex, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,write_new
from wb111_contract import OUT as HISTORICAL
from wb92_contract import EXTERNAL
from wb100_contract import digest
from wb122_sources import files,step_source_and_proof

def run(out):
    out.mkdir(exist_ok=False);generated=files()
    for name,content in generated.items():
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
    old=json.loads((ROOT/'outputs/mc24_four_station_wb122_compile_preflight_v1/receipt.json').read_text())
    commands=[]; codes=[]
    for prior,tu in zip(old['commands'],('BoundedResponse.cxx','WB122ExtrapolationTool.cxx')):
        cmd=[s.replace('mc24_four_station_wb122_compile_preflight_v1',out.name) for s in prior]
        with (out/(tu+'.compile.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        commands.append(cmd);codes.append(r.returncode);print(tu,r.returncode,flush=True)
    source=(ROOT/'research/wb122/CacheSelftest.cxx').read_text();(out/'WB122Diagnostic/CacheSelftest.cxx').write_text(source)
    cmd=[s for s in commands[0][:-1] if s!='-fsyntax-only']+[str(out/'WB122Diagnostic/CacheSelftest.cxx'),'-o',str(out/'cache_selftest')]
    link=shlex.split((HISTORICAL/'isolated_build/WB111Diagnostic/CMakeFiles/WB111Diagnostic.dir/link.txt').read_text())
    cmd += ['-Wl,--as-needed']+[x for x in link if x.startswith('/') and '.so' in x]
    with (out/'cache_selftest.compile.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    commands.append(cmd);codes.append(r.returncode)
    if r.returncode==0:
        with (out/'cache_selftest.json').open('x') as log:r=subprocess.run([str(out/'cache_selftest')],stdout=log,stderr=subprocess.STDOUT)
        codes.append(r.returncode)
    write_new(out/'step_source_restoration_proof.json',step_source_and_proof()[1])
    write_new(out/'receipt.json',{'commands':commands,'returncodes':codes,'source_hashes':{k:digest(out/k) for k in generated},'cache_source_sha256':digest(out/'WB122Diagnostic/CacheSelftest.cxx'),'compiler':subprocess.check_output(['which','g++'],text=True).strip(),'scope':'two whole TUs syntax-only and actual-header synthetic cache read; no event/fieldmap'})
    if codes!=[0,0,0,0]:raise ValueError('compile/cache preflight failure')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');run(ROOT/'outputs'/p.parse_args().out)
