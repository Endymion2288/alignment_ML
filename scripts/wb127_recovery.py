#!/usr/bin/env python3
"""Bounded WB127 runner recovery; preserves the first runner failure receipts."""
import json, subprocess
from pathlib import Path
from wb127_contract import OUT, PARENT, PROTOCOL, digest, read, write, req, verify
ROOT=Path(__file__).resolve().parents[1]
BIN=OUT/'build_preflight_v4/build/x86_64-el9-gcc13-opt'
def main():
 recovery=OUT/'recovery_v4';recovery.mkdir(exist_ok=False);(recovery/'events').mkdir()
 old=read(OUT/'freeze.json');hashes={}
 for p,h in old['hashes'].items():
  if p in (str(ROOT/'scripts/wb127_athena.py'),str(ROOT/'scripts/wb127_contract.py')):continue
  req(digest(p)==h,'parent frozen identity '+p);hashes[p]=h
 for p in (ROOT/'scripts/wb127_athena.py',ROOT/'scripts/wb127_contract.py',Path(__file__)):
  hashes[str(p)]=digest(p)
 write(recovery/'freeze.json',{'hashes':hashes,'parent_freeze_sha256':digest(OUT/'freeze.json'),'interface_recovery':'allowlist wrapper indices and preserve old failed receipts','new_propagation_calls':0,'new_reconstruction_calls':0})
 recovery_verify(recovery)
 total=0
 for idx in read(PROTOCOL)['indices']:
  source=OUT/'events'/f'{idx:02d}';e=recovery/'events'/f'{idx:02d}';e.mkdir()
  for name in ('fixture.json','p_seed.json'): (e/name).write_bytes((source/name).read_bytes())
  script='\n'.join(['source '+str(ROOT/'scripts/setup_environment.sh')+' calypso','source '+str(BIN/'setup.sh'),'export LD_LIBRARY_PATH='+str(BIN/'lib')+':$LD_LIBRARY_PATH','python '+str(ROOT/'scripts/wb127_athena.py')+' --work-dir '+str(e)+' --sqlite '+str(PARENT/'identity_payload/tracker_alignment.sqlite')])
  (e/'command.sh').write_text(script)
  with (e/'recovery.log').open('w') as log:r=subprocess.run(['bash','-c',script],cwd=e,stdout=log,stderr=subprocess.STDOUT)
  write(e/'recovery_exit.json',{'exit_code':r.returncode});req(r.returncode==0,'recovery Athena '+str(idx));total+=1
  req((e/'export.json').is_file(),'missing export');recovery_verify(recovery)
 write(recovery/'summary.json',{'schema':'wb127_recovery_summary_v1','execution_contract':'PASS','population':6,'new_reconstruction_calls':total,'new_propagation_calls':0,'first_runner_failure':'PRESERVED','held_out_access':False,'freeze_sha256':digest(OUT/'freeze.json')})
 print('RECOVERY_COMPLETE',total)
def recovery_verify(recovery):
 for p,h in read(recovery/'freeze.json')['hashes'].items():req(digest(p)==h,'recovery frozen identity '+p)
if __name__=='__main__':main()
