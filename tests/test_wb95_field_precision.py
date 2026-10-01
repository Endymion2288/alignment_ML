import copy
import json
from pathlib import Path
import numpy as np
import pytest
from alignment.wb95_field_precision import trilinear,effects,mechanism_decision,field_contract
P=json.loads((Path(__file__).resolve().parents[1]/'configs/research_review/wp95_field_precision_contract.json').read_text())

def test_trilinear_rejects_units_coordinates_and_nonfinite():
    nodes=np.array([[((c>>2)&1)+2*((c>>1)&1)+3*(c&1),4*((c>>2)&1)-((c>>1)&1)+(c&1),7] for c in range(8)])
    assert np.allclose(trilinear(nodes,[.2,.3,.4],2),[4,1.8,14])
    assert not np.allclose(trilinear(nodes,[.4,.3,.2],2),[4,1.8,14])
    assert not np.allclose(trilinear(nodes,[.2,.3,.4],2000),[4,1.8,14])
    with pytest.raises(ValueError):trilinear(nodes,[-1,.3,.4],2)
    with pytest.raises(ValueError):trilinear(nodes,[float('nan'),.3,.4],2)

def test_effects_uses_coordinate_sum_and_detects_precision_loss():
    ladders=[]
    for step in P['mesh_rk4_dz_mm']:
        rows=[{'targets':[{'h':[0.,0,0,0]} for _ in range(3)]} for _ in range(25)]
        for k in range(6):
            for j,amplitude in enumerate((.5,-.5,.25,.125)):
                # Mixed direct central effect deliberately differs; main effect must use coordinates.
                value=amplitude*(1 if k<5 else 99)
                rows[1+4*k+j]['targets']=[{'h':[value,0,0,0]} for _ in range(3)]
        ladders.append({'dz_mm':step,'samples':rows})
    e=effects(ladders,P);assert e['all_pass'] and len(e['checks'])==18
    mixed=next(x for x in e['taylor'] if x['direction']=='alternating_mixed')
    assert mixed['effect'][0]==.25 and mixed['directional_vs_coordinate_effect']>1
    ladders[-1]['samples'][1]['targets'][0]['h'][0]+=.001
    assert not effects(ladders,P)['all_pass']

def test_mechanism_is_unknown_without_reference_and_not_support_when_controls_agree():
    d={'reference':{'gate':'NUMERICAL_REFERENCE_SUPPORTED','max_fine_difference':1e-10},'effects':{'all_pass':True}}
    f={'reference':{'gate':'UNKNOWN','max_fine_difference':1e-8}}
    r={'mesh_z_double':d,'mesh_z_float':f};field={'max_float_vs_double_T':1e-7}
    assert mechanism_decision(r,field,P)=='SUPPORTED_BUT_LIMITED'
    bad=copy.deepcopy(r);bad['mesh_z_double']['effects']['all_pass']=False
    assert mechanism_decision(bad,field,P)=='UNKNOWN'
    bad=copy.deepcopy(r);bad['mesh_z_float']['reference']['gate']='NUMERICAL_REFERENCE_SUPPORTED'
    assert mechanism_decision(bad,field,P)=='NOT_SUPPORTED'

def test_field_contract_rejects_different_nodes_and_zero_outside():
    p={**P,'expected_mesh_dimensions':[2,2,2],'expected_nodes':8}
    mesh=[list(x) for x in zip(p['expected_zone']['min_mm'],p['expected_zone']['max_mm'])]
    nodes=np.array([[c,c+1,c+2] for c in range(8)],dtype=np.int16)
    def probe(pos):
        inside=all(a<=x<=b for a,x,b in zip(p['expected_zone']['min_mm'],pos,p['expected_zone']['max_mm']))
        if not inside:return {'position_mm':pos,'zone_id':-1,'float_T':[1e-5]*3,'double_T':[1e-5]*3}
        fractions=[(x-a)/(b-a) for a,x,b in zip(p['expected_zone']['min_mm'],pos,p['expected_zone']['max_mm'])]
        b=trilinear(nodes,fractions,1e-4).tolist()
        return {'position_mm':pos,'zone_id':1,'float_T':b,'double_T':b,'cell':[0,0,0],'fractions':fractions,
                'nodes':nodes.tolist(),'independent_double_T':b}
    c={'map_key':p['map_key'],'cache_key':p['cache_key'],'wrapper_cache_identity':True,'map_IOV':'actual','cache_IOV':'actual',
       'min_mm':p['expected_zone']['min_mm'],'max_mm':p['expected_zone']['max_mm'],'zone_id':1,'scale':1.,'bscale_kT':1e-7}
    r={'conditions':c,'mesh_mm':mesh,'node_count':8,'probes':[probe([47,74,0]) for _ in range(19)],
       'domain_controls':[probe(pos) for pos in ([47,74,-1860],[47,74,0],[-1761,47,74],[201,74,0],[47,201,0])]}
    assert field_contract(r,p,nodes)['gate']=='PASS'
    bad=copy.deepcopy(r);bad['probes'][0]['nodes'][0][0]+=1
    with pytest.raises(ValueError,match='different field nodes'):field_contract(bad,p,nodes)
    bad=copy.deepcopy(r);bad['domain_controls'][0]['double_T']=[0,0,0]
    with pytest.raises(ValueError,match='outside fallback'):field_contract(bad,p,nodes)
