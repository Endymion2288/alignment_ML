"""Isolated actual ACTS plane API on saved vectors; never event/field/propagation."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,write_new,read_public
from wb92_contract import ACTS
from wb97_contract import EIGEN_INCLUDE,JSON_INCLUDE
from wb100_contract import digest
PREFLIGHT=ROOT/'outputs/mc24_four_station_wb123_plane_preflight_v1'
SOURCE=ROOT/'research/wb123/PlaneContract.cxx'

def probe_requests():
    import math
    frame=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]];rows=[]
    for z in (0.,-2e-4,-1e-4,-5e-5,5e-5,1e-4,2e-4):
        for cosine in (1.,.1):
            rows.append({'case_id':len(rows),'frame':frame,'position_mm':[[0.],[0.],[z]],'direction':[[math.sqrt(1-cosine*cosine)],[0.],[cosine]],'navigation_direction':1})
    return {'rows':rows}

def build():
    PREFLIGHT.mkdir(exist_ok=False);(PREFLIGHT/'PlaneContract.cxx').write_text(SOURCE.read_text())
    binary=PREFLIGHT/'plane_contract';cmd=['g++','-std=c++20','-O2','-I'+str(ACTS/'include'),'-I'+str(EIGEN_INCLUDE),'-I'+str(JSON_INCLUDE),str(PREFLIGHT/'PlaneContract.cxx'),'-L'+str(ACTS/'lib'),'-Wl,-rpath,'+str(ACTS/'lib'),'-lActsCore','-o',str(binary)]
    with (PREFLIGHT/'compile.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    write_new(PREFLIGHT/'compile_receipt.json',{'command':cmd,'returncode':r.returncode,'source_sha256':digest(SOURCE),'compiler':subprocess.check_output(['which','g++'],text=True).strip(),'scope':'standalone geometric API; no Stepper/Propagator/ROOT/field'})
    if r.returncode:raise ValueError('plane helper compile failure')
    write_new(PREFLIGHT/'binary_manifest.json',{'binary':str(binary),'sha256':digest(binary),'libActsCore':str(ACTS/'lib/libActsCore.so'),'libActsCore_sha256':digest(ACTS/'lib/libActsCore.so'),'ldd':subprocess.check_output(['ldd',str(binary)],text=True)})
    request=PREFLIGHT/'synthetic_request.json';result=PREFLIGHT/'synthetic_result.json';write_new(request,probe_requests())
    subprocess.run([str(binary),str(request),str(result)],check=True)
    data=read_public(result)
    assert len(data['rows'])==14 and data['propagation_calls']==0
    for row in data['rows']:
        z=row['input']['position_mm'][2][0];assert row['surface_tolerance_mm']==1e-4
        if z==0.:assert row['on_surface'] is True
        if abs(z)==2e-4:assert row['on_surface'] is False
    write_new(PREFLIGHT/'synthetic_probe_receipt.json',{'status':'PASS','actual_api_cases':14,'boundary_and_inclined_cases_reported_without_posthoc_threshold':True,'binary_sha256':digest(binary),'event_access':False,'propagation_calls':0})
    print([(r['input']['position_mm'][2][0],r['input']['direction'][2][0],r['on_surface']) for r in data['rows']])

def evaluate(request,result):
    binary=PREFLIGHT/'plane_contract';receipt=read_public(PREFLIGHT/'binary_manifest.json')
    if digest(binary)!=receipt['sha256']:raise ValueError('helper binary mutation')
    subprocess.run([str(binary),str(request),str(result)],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','evaluate'));p.add_argument('paths',nargs='*',type=Path);a=p.parse_args()
    if a.action=='build':build()
    else:
        if len(a.paths)!=2:raise ValueError('request/result paths')
        evaluate(*a.paths)
