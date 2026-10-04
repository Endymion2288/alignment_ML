"""Synthetic WB125 tests in a pinned Python environment; no event decoding."""
import argparse
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path[:]=[p for p in sys.path if 'madgraph' not in p.lower()]
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'scripts'),str(root)]
import pytest
from alignment.wb90_measurement_contract import write_new
from wb125_contract import digest
parser=argparse.ArgumentParser();parser.add_argument('result',type=Path);args=parser.parse_args()
code=pytest.main(['-q','-p','no:cacheprovider',str(root/'tests/test_wb125_physical_seed.py')])
write_new(args.result,{'python':sys.version,'pytest_exit':code,'scope':'synthetic controls only; no event/field/propagation access',
    'test_sha256':digest(root/'tests/test_wb125_physical_seed.py'),'audit_sha256':digest(root/'scripts/wb125_audit.py')})
sys.exit(code)
