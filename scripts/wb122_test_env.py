"""Run exact WB122 tests/imports in either supplied environment, no events."""
import argparse,json,sys,os,hashlib
from pathlib import Path
# Calypso's MadGraph py.py shadows pytest's compatibility package.
sys.path[:]=[p for p in sys.path if 'MadGraph' not in p and 'madgraph' not in p]
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'));sys.path.insert(0,str(root))
from alignment.wb90_measurement_contract import write_new
from wb100_contract import digest
import audit_wb122_response,wb122_sources,wb122_contract
import pytest
p=argparse.ArgumentParser();p.add_argument('out');args=p.parse_args()
generated=wb122_sources.files();compile(wb122_sources.athena_source(),'generated_wb122_athena.py','exec')
code=pytest.main(['-q',str(root/'tests/test_wb122_response.py')])
write_new(Path(args.out),{'python':sys.version,'python_version_info':list(sys.version_info),'pytest_exit':code,'imports':True,'analyzer_sha256':digest(root/'scripts/audit_wb122_response.py'),'source_generator_sha256':digest(root/'scripts/wb122_sources.py'),'generated_source_hashes':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in generated.items()},'generated_athena_compile':True,'scope':'synthetic/import/source; no event calls'})
sys.exit(code)
