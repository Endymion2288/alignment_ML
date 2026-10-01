#!/usr/bin/env python3
"""Reporting-only immutable result index; no execution or gate changes."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,digest,read_public,write_new
p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();out=a.output_root.resolve()
if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb93_'):p.error('WB93 output required')
s=read_public(out/'summary.json');attempts=[]
for directory in sorted((ROOT/'outputs').glob('mc24_four_station_wb93_transport_error_v*')):
    files=[f for f in directory.iterdir() if f.is_file() and f.suffix in ('.json','.log','.sub','.out','.err')]
    if (directory/'event').is_dir():files.extend(f for f in (directory/'event').iterdir() if f.is_file() and f.suffix in ('.json','.log'))
    if (directory/'figures').is_dir():files.extend(f for f in (directory/'figures').iterdir() if f.is_file())
    attempts.append({'path':str(directory.relative_to(ROOT)),'hashes':{str(f.relative_to(ROOT)):digest(f) for f in files}})
manifest={'output_root':str(out.relative_to(ROOT)),'execution_contract':s['execution_contract'],
    'mechanism_hypothesis':s['mechanism_hypothesis'],'attempts':attempts,'reporting_script_sha256':digest(Path(__file__)),
    'population':1,'qualification':'NOT_EVALUATED','WB92_gate':'FAIL','held_out_access':False}
write_new(out/'result_integrity.json',manifest)
write_new(ROOT/'docs/wb93_transport_error_result_manifest.json',{**manifest,'result_integrity_sha256':digest(out/'result_integrity.json')})
print({k:manifest[k] for k in ('output_root','execution_contract','mechanism_hypothesis')})
