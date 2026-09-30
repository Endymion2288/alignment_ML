#!/usr/bin/env python3
"""Submit one immutable WB91 build/pilot/conditional-development worker."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT as PROJECT,write_new,read_public,digest
from run_wb91_covariance_contract import verify,OUTPUT

p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,default=OUTPUT)
p.add_argument('--reuse-build-root',type=Path)
args=p.parse_args();out=args.output_root.resolve();verify(out)
if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('mc24_four_station_wb91_'):p.error('WB91 output required')
if read_public(out/'kernel_summary.json')['gate']!='PASS':raise ValueError('kernel prerequisite failed')
if (out/'wb91.sub').exists():raise FileExistsError('no duplicate submission')
worker=PROJECT/'scripts/run_wb91_condor.sh'
reuse=' '+str(args.reuse_build_root.resolve()) if args.reuse_build_root else ''
text='\n'.join(['universe = vanilla',f'executable = {worker}',f'arguments = {PROJECT} {out}{reuse}',
    f'output = {out}/condor.$(ClusterId).out',f'error = {out}/condor.$(ClusterId).err',f'log = {out}/condor.$(ClusterId).log',
    'request_cpus = 1','request_memory = 8000','request_disk = 8000000','requirements = (Arch == "X86_64")',
    '+JobFlavour = "workday"','getenv = False','should_transfer_files = NO','queue 1',''])
with (out/'wb91.sub').open('x') as stream:stream.write(text)
cmd='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && condor_submit '+shlex.quote(str(out/'wb91.sub'))
r=subprocess.run(['bash','-c',cmd],text=True,capture_output=True)
write_new(out/'submission.json',{'command':cmd,'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode,
    'submit_sha256':digest(out/'wb91.sub'),'worker_sha256':digest(worker)})
print(r.stdout);print(r.stderr,file=sys.stderr);sys.exit(r.returncode)
