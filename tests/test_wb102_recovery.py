import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wb102_recovery import recovery_aggregate, digest, parent


def test_digest_path_string_and_streaming(tmp_path):
    file=tmp_path/'binary';file.write_bytes(b'payload')
    assert digest(file)==digest(str(file))


def test_saved_aggregation_redirect_and_fail_closed(tmp_path,monkeypatch):
    source=tmp_path/'source';destination=tmp_path/'recovery'
    source.mkdir();destination.mkdir()
    def aggregate(out):
        parent.write_new(out/'summary.json',{'hypothesis':'UNKNOWN'})
        return {'physical_calls':0}
    monkeypatch.setattr(parent,'aggregate',aggregate)
    assert recovery_aggregate(source,destination)=={'physical_calls':0}
    assert not (source/'summary.json').exists() and (destination/'summary.json').exists()
    with pytest.raises(FileExistsError): recovery_aggregate(source,destination)
    def forbidden(out): parent.write_new(out/'fixture.json',{})
    monkeypatch.setattr(parent,'aggregate',forbidden)
    with pytest.raises(ValueError,match='unexpected recovery write'): recovery_aggregate(source,destination)
