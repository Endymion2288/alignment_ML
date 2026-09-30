"""Scientific control tests for dimensional normalization and source guards."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_covariance_comparison_detects_slope_error_despite_mm_scale():
    module=load('run_wb91_covariance_contract')
    c=np.diag([1.,4.,1.e-6,4.e-6]);bad=c.copy();bad[2,2]*=2.
    assert module.covariance_error(bad,c)>.1
    scale=np.diag([100.,100.,.001,.001])
    assert module.covariance_error(scale@bad@scale,scale@c@scale)==pytest.approx(module.covariance_error(bad,c))

def test_generator_refuses_ambiguous_or_missing_patch():
    module=load('prepare_wb91_calypso_extension')
    for source in ('absent','xx'):
        with pytest.raises(ValueError):module.replace_once(source,'x','y')
    assert module.replace_once('ax','x','y')=='ay'

def test_controls_preserve_full_rank_and_cross_covariances():
    import json
    config=json.loads((ROOT/'configs/research_review/wp91_covariance_repair.json').read_text())
    l=np.array(config['covariance_cholesky']);c=l@l.T
    assert np.linalg.eigvalsh(c).min()>0
    assert c[0,2]!=0 and c[1,3]!=0
    assert len(config['analytic_slopes'])==7
    assert config['analytic_slopes'][-1]==[.3,.4]
