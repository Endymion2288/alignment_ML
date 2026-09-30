"""MC event headers repeat across ROOT sources; never silently deduplicate them."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

PROJECT=Path(__file__).resolve().parents[1]

def load_aggregator():
    sys.path.insert(0,str(PROJECT/'scripts'))
    spec=importlib.util.spec_from_file_location('wb91_aggregate_test',PROJECT/'scripts/aggregate_wb91_results.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def fixture(tmp_path,monkeypatch,same_source=False):
    module=load_aggregator();monkeypatch.setattr(module,'verify',lambda _:None)
    out=tmp_path/'output';pilot=tmp_path/'pilot';out.mkdir()
    selections=[{'input_xaod':'source-a.root','xaod_entry_index':5,'source_id':'a'},
                {'input_xaod':'source-a.root' if same_source else 'source-b.root','xaod_entry_index':5,'source_id':'b'}]
    for i,work in enumerate((pilot/'physical/00',out/'physical/01')):
        work.mkdir(parents=True)
        for mode in ('legacy','repair'):(work/f'{mode}.jsonl').write_text('{}\n')
        result={k:0. for k in ('paired_error','state_error','normal_inverse_error','fd_error','jacobian_roundtrip_error','covariance_roundtrip_error','legacy_covariance_error')}
        result['min_native_correlation_eigenvalue']=1.
        summary={'gate':'PASS','input_event_header_verified':True,'actual_event_header':[1,5],
                 'results':[result],'n_states':1,**{f'{mode}_sha256':module.digest(work/f'{mode}.jsonl') for mode in ('legacy','repair')}}
        module.write_new(work/'validation.json',summary)
        module.write_new(work/'input_manifest.json',{'row':selections[i]})
    module.write_new(out/'array_inputs.json',{'pilot_root':str(pilot),'pilot_validation_sha256':module.digest(pilot/'physical/00/validation.json'),'indices':[1]})
    module.write_new(out/'selection.json',{'events':selections})
    return module,out

def test_equal_headers_from_distinct_sources_are_recorded_not_dropped(tmp_path,monkeypatch):
    module,out=fixture(tmp_path,monkeypatch)
    module.aggregate(out)
    result=json.loads((out/'physical_summary.json').read_text())
    assert result['n_events']==2
    assert result['unique_run_event_pairs']==1
    assert len(result['cross_source_header_collisions'])==1
    assert result['gate']=='INCOMPLETE'  # Never qualify a partial population.

def test_same_source_duplicate_is_rejected(tmp_path,monkeypatch):
    module,out=fixture(tmp_path,monkeypatch,same_source=True)
    with pytest.raises(ValueError,match='duplicate event within same input source'):
        module.aggregate(out)
    assert not (out/'physical_summary.json').exists()
