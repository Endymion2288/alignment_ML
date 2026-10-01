#!/usr/bin/env python3
"""Reporting-only exclusive result index, including raw loaded node payload."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest
p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();out=a.output_root.resolve()
if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb95_'):p.error('WB95 output required')
s=read_public(out/'summary.json');scheduler=read_public(out/'scheduler_terminal_audit.json');attempts=[]
for directory in sorted((ROOT/'outputs').glob('mc24_four_station_wb95_field_precision_v*')):
    files=[f for f in directory.iterdir() if f.is_file() and f.suffix in ('.json','.log','.sub','.out','.err')]
    for child in ('event','figures'):
        if (directory/child).is_dir():files.extend(f for f in (directory/child).iterdir() if f.is_file())
    attempts.append({'path':str(directory.relative_to(ROOT)),'hashes':{str(f.relative_to(ROOT)):digest(f) for f in files}})
manifest={'output_root':str(out.relative_to(ROOT)),'execution_contract':s['execution_contract'],'mechanism_hypothesis':s['mechanism_hypothesis'],
  'scheduler_terminal_state':scheduler['scheduler_terminal_state'],'scheduler_exit_code':scheduler['scheduler_exit_code'],
  'reference_by_mode':{k:v['reference']['gate'] for k,v in s['modes'].items()},'attempts':attempts,
  'nodes_sha256':s['nodes_sha256'],'reporting_script_sha256':digest(Path(__file__)),
  'population':1,'qualification':'NOT_EVALUATED','WB92_gate':'FAIL','WB94_mechanism':'UNKNOWN','held_out_access':False}
write_new(out/'result_integrity.json',manifest)
write_new(ROOT/'docs/wb95_field_precision_result_manifest.json',{**manifest,'result_integrity_sha256':digest(out/'result_integrity.json')})
print({k:manifest[k] for k in ('output_root','execution_contract','mechanism_hypothesis','reference_by_mode')})
