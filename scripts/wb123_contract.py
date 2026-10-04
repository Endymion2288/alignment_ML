#!/usr/bin/env python3
"""Freeze/recover/rebuild WB123 using existing JSON and actual target-plane API only."""
import argparse,difflib,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from wb100_contract import digest,verify
from wb92_contract import ACTS
from wb123_recovery import generated_source,source_proof,load_reader,target_request,recover
from wb123_plane import PREFLIGHT,evaluate
ORIGINAL=ROOT/'outputs/mc24_four_station_wb122_complete_rk_cache_trace_v1'
OUT=ROOT/'outputs/mc24_four_station_wb123_saved_complete_trace_recovery_v1'
WORKBOOK=ROOT/'workbook/2026-10-04_123_四站保存步进数据的终止与完整覆盖接口语义恢复.md'
MANIFEST=ROOT/'docs/wb122_complete_rk_cache_result_manifest.json'


def require(ok,message):
    if not ok:raise ValueError(message)


def prior_hashes():
    previous=read_public(MANIFEST)
    require(previous['execution_contract']==previous['integrity']=='PASS' and previous['hypothesis']=='UNKNOWN_FIDELITY_OR_COVERAGE','immutable WB122 state')
    hashes=verify(ORIGINAL)['hashes'].copy()
    for name,h in previous['artifacts'].items():
        require(digest(ROOT/name)==h,'WB122 sealed artifact '+name);hashes[str(ROOT/name)]=h
    hashes[str(MANIFEST)]=digest(MANIFEST)
    return hashes


def freeze():
    require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='4station','branch')
    require(not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip(),'implementation must be committed')
    hashes=prior_hashes();source=generated_source();proof=source_proof()
    for env in ('calypso','ml'):
        p=ROOT/('outputs/wb123_'+env+'_test_preflight_v1.json');test=read_public(p)
        require(test['pytest_exit']==0 and test['imports'] is True and test['adapter_sha256']==digest(ROOT/'scripts/wb123_recovery.py') and test['generated_source_sha256']==proof['generated_sha256'] and test['test_sha256']==digest(ROOT/'tests/test_wb123_recovery.py'),'exact tested source '+env)
        hashes[str(p)]=digest(p)
    compile_receipt=read_public(PREFLIGHT/'compile_receipt.json');binary=read_public(PREFLIGHT/'binary_manifest.json')
    require(compile_receipt['returncode']==0 and compile_receipt['source_sha256']==digest(ROOT/'research/wb123/PlaneContract.cxx'),'actual plane source compile')
    require(digest(Path(binary['binary']))==binary['sha256'] and digest(Path(binary['libActsCore']))==binary['libActsCore_sha256'],'plane binary/library identity')
    require(read_public(PREFLIGHT/'synthetic_probe_receipt.json')['status']=='PASS','actual API synthetic probe')
    paths=[ROOT/'scripts'/n for n in ('wb123_recovery.py','wb123_plane.py','wb123_contract.py','wb123_test_env.py','audit_wb122_response.py','setup_environment.sh')]
    paths += [ROOT/'tests/test_wb123_recovery.py',ROOT/'tests/test_wb122_response.py',ROOT/'research/wb123/PlaneContract.cxx',Path(compile_receipt['compiler']),Path(binary['libActsCore'])]
    paths += [p for p in PREFLIGHT.rglob('*') if p.is_file()]
    paths += [ACTS/'include/Acts'/n for n in ('Definitions/Tolerance.hpp','Definitions/TrackParametrization.hpp','Definitions/Units.hpp','Surfaces/PlaneSurface.hpp','Propagator/Propagator.hpp','Propagator/Propagator.ipp','Propagator/StandardAborters.hpp','Propagator/detail/SteppingLogger.hpp','Utilities/Intersection.hpp')]
    for p in paths:hashes[str(p)]=digest(p)
    OUT.mkdir(exist_ok=False);shutil.copyfile(WORKBOOK,OUT/'contract_workbook.md')
    with (OUT/'generated_reader.py').open('x') as f:f.write(source)
    with (OUT/'terminal_semantics_only.diff').open('x') as f:f.write(''.join(difflib.unified_diff((ROOT/'scripts/audit_wb122_response.py').read_text().splitlines(True),source.splitlines(True),fromfile='frozen_wb122',tofile='wb123_stage_predicate')))
    write_new(OUT/'source_restoration_proof.json',proof)
    # A request uses saved actual target transforms/free states only; the 16GB event file is not read.
    runtime=read_public(ORIGINAL/'event/response.json')
    write_new(OUT/'target_requests.json',{'rows':[target_request(row) for row in runtime['rows'] if row['input']['arm']=='observer_enabled']})
    for n in ('contract_workbook.md','generated_reader.py','terminal_semantics_only.diff','source_restoration_proof.json','target_requests.json'):hashes[str(OUT/n)]=digest(OUT/n)
    write_new(OUT/'freeze.json',{'schema':'wb123_saved_semantics_recovery_freeze_v1','hashes':hashes,'branch':'4station','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'original_wb122_hypothesis':'UNKNOWN_FIDELITY_OR_COVERAGE','new_propagation_calls':0,'new_event_access':False,'data':'already seen WB122 saved streams; no independent qualification'})


def rebuild():
    verify(OUT)
    return recover(ORIGINAL,(OUT/'generated_reader.py').read_text(),read_public(OUT/'target_plane_results.json'))


def run():
    frozen=verify(OUT);(OUT/'execution_lock').mkdir(exist_ok=False)
    write_new(OUT/'environment.json',{'python':sys.version,'executable':sys.executable,'scope':'saved JSON and geometric API only; no ROOT EDM/Athena/event/field/propagation'})
    evaluate(OUT/'target_requests.json',OUT/'target_plane_results.json')
    write_new(OUT/'target_evaluation_receipt.json',{'helper_sha256':digest(PREFLIGHT/'plane_contract'),'request_sha256':digest(OUT/'target_requests.json'),'result_sha256':digest(OUT/'target_plane_results.json'),'geometry_api_calls':18,'new_propagation_calls':0,'event_access':False,'field_access':False})
    summary=rebuild();write_new(OUT/'summary.json',summary)
    original=load_reader((ROOT/'scripts/audit_wb122_response.py').read_text()).audit(ORIGINAL)
    require(original==read_public(ORIGINAL/'summary.json'),'original summary exact unchanged')
    verify(OUT);prior_hashes()
    write_new(OUT/'recovery_receipt.json',{'exit_code':0,'recovery_execution':summary['recovery_execution'],'hypothesis':summary['hypothesis'],'identities_verified':len(frozen['hashes']),'original_wb122_summary_exact':True,'new_physical_calls':0,'new_event_root_access':False,'causal_attribution':'UNVERIFIED','physical_derivative_accuracy':'UNVERIFIED','qualification':'NOT_EVALUATED','physics_screening':'NOT_EVALUATED','final_oracle':'NOT_EVALUATED'})
    print(summary['recovery_execution'],summary['hypothesis'],'profiles',len(summary['profiles']),'unknowns',summary['unknowns'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','verify','run','rebuild'));a=p.parse_args()
    if a.action=='verify':verify(OUT)
    elif a.action=='rebuild':print(json.dumps(rebuild(),sort_keys=True,allow_nan=False))
    else:globals()[a.action]()
