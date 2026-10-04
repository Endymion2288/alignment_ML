"""Dual environment synthetic recovery validation; no real saved trajectory read."""
import argparse,hashlib,os,sys
from pathlib import Path
sys.path[:]=[p for p in sys.path if 'MadGraph' not in p and 'madgraph' not in p]
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'scripts'),str(root)]
from alignment.wb90_measurement_contract import write_new
from wb100_contract import digest
import wb123_recovery
import pytest
p=argparse.ArgumentParser();p.add_argument('result',type=Path);a=p.parse_args()
code=pytest.main(['-q',str(root/'tests/test_wb122_response.py'),str(root/'tests/test_wb123_recovery.py')])
write_new(a.result,{'python':sys.version,'version_info':list(sys.version_info),'pytest_exit':code,'imports':True,
 'adapter_sha256':digest(root/'scripts/wb123_recovery.py'),'generated_source_sha256':hashlib.sha256(wb123_recovery.generated_source().encode()).hexdigest(),
 'test_sha256':digest(root/'tests/test_wb123_recovery.py'),'scope':'synthetic source/target/order tests; old WB122 and new WB123; no real event/saved trajectory read'})
sys.exit(code)
