#!/usr/bin/env python3
"""Preserve frozen reporter; bound the two job-specific history queries to one ad."""
import argparse,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from wb100_contract import verify,digest

OUT=ROOT/'outputs/mc24_four_station_wb100_aggregation_recovery_v1'
SOURCE=ROOT/'scripts/finalize_wb100_results.py'
def main(action):
    freeze=verify(OUT)
    if digest(SOURCE)!=freeze['hashes'][str(SOURCE)]:raise ValueError('frozen reporter identity')
    before=SOURCE.read_text();needle="+' -json'"
    if before.count(needle)!=2:raise ValueError('exact two history commands required')
    after=before.replace(needle,"+' -limit 1 -json'")
    generated=OUT/'bounded_reporting.py'
    if action=='report':
        with generated.open('x') as f:f.write(after)
        write_new(OUT/'bounded_reporting_manifest.json',{'original':str(SOURCE),'original_sha256':digest(SOURCE),
           'generated_sha256':digest(generated),'adapter':str(Path(__file__).resolve()),'adapter_sha256':digest(Path(__file__).resolve()),
           'replacement_count':2,'replacement':'job-specific condor_history -json becomes -limit 1 -json; no scientific changes'})
    elif generated.read_text()!=after:raise ValueError('generated reporter changed')
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'scripts')+':'+str(ROOT)+':'+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,str(generated),action,'--output-root',str(OUT)],env=env,check=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('report','seal'));main(parser.parse_args().action)
