#!/usr/bin/env python3
"""Saved-only correction for saturated Athena INFO call logs; no propagation.

Never overwrites the frozen controller or its original summary. A complete
response with contiguous call IDs provides exact counts; a missing response
leaves counts UNKNOWN instead of interpreting truncated logs as exact.
"""
import argparse
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from wb131_contract import ROOT, OUT, SOURCES, read, write, digest, verify, require

RECOVERY=OUT/'saved_call_accounting_v1'


def count_response(request, response, rows):
    require(len(response['arms'])==len(request['arms']), 'arm coverage')
    identifiers=[]
    for arm, expected in zip(response['arms'],request['arms']):
        require(arm['tag']==expected['tag'] and arm['seed']==expected['seed'], 'arm identity')
        require(len(arm['rows'])==rows, 'row coverage')
        identifiers.extend(row['call_id'] for row in arm['rows'])
    expected=len(request['arms'])*rows
    require(identifiers==list(range(1,expected+1)), 'complete contiguous call IDs')
    require(response['official_calls']==response['new_propagation_calls']==expected, 'call counter')
    return expected


def freeze():
    verify()
    RECOVERY.mkdir(exist_ok=False)
    paths=[Path(__file__).resolve(),OUT/'freeze.json',OUT/'summary.json',OUT/'independent_audit.json']
    paths+=list((OUT/'events').glob('*/status.json'))
    paths+=list((OUT/'events').glob('*/*/response.json'))
    paths+=list((OUT/'events').glob('*/*/request.json'))
    paths+=list((OUT/'events').glob('*/*/athena.log'))
    write(RECOVERY/'freeze.json',{'schema':'wb131_saved_call_accounting_freeze_v1',
                                'hashes':{str(p):digest(p) for p in paths},
                                'scientific_contract_changes':False,'new_propagation_calls':0})


def run():
    verify()
    for path, expected in read(RECOVERY/'freeze.json')['hashes'].items():
        require(digest(path)==expected,'saved accounting identity '+path)
    # Negative control: a suppressed/missing call ID must not pass as complete.
    request={'arms':[{'tag':'synthetic','seed':[0,0,0,0,1]}]}
    good={'arms':[{'tag':'synthetic','seed':[0,0,0,0,1],'rows':[{'call_id':1},{'call_id':2}]}],
          'official_calls':2,'new_propagation_calls':2}
    require(count_response(request,good,2)==2,'counter positive control')
    bad=read_control_copy(good)
    bad['arms'][0]['rows'][1]['call_id']=1
    try:
        count_response(request,bad,2)
    except ValueError:
        pass
    else:
        raise ValueError('counter negative control failed')
    records=[]
    completed=0
    original=read(OUT/'summary.json')
    for event_status in original['events']:
        event=OUT/'events'/f"{event_status['index']:02d}"
        rows=len(read(event/'export.json')['rows'])
        for phase in ('derivatives','replay'):
            work=event/phase
            if not work.exists():
                continue
            require((work/'response.json').exists(),'UNKNOWN partial execution: exact count unavailable')
            response=read(work/'response.json')
            count=count_response(read(work/'request.json'),response,rows)
            completed+=count
            log=(work/'athena.log').read_text(errors='replace')
            visible=log.count('WB131_CALL_BEGIN id=')
            require(visible<=count,'extra logged calls')
            if visible<count:
                require('message limit (500) reached for WB131FixedQopNuisancePrediction' in log,'unexplained missing log calls')
            records.append({'index':event_status['index'],'phase':phase,'exact_completed_calls':count,
                            'visible_begin_lines':visible,'log_saturated':visible<count})
    require(completed==original['completed_propagation_calls'] and completed<=2646,'exact budget')
    require(sum(r['visible_begin_lines'] for r in records)==original['attempted_propagation_calls'],'original log counter')
    # All requests completed exactly once and all call IDs are present: attempts
    # equal completed calls. Partial/crashed executions deliberately fail above.
    result={'schema':'wb131_saved_call_accounting_v1','call_accounting':'PASS',
            'original_log_begin_count':original['attempted_propagation_calls'],
            'actual_attempted_propagation_calls':completed,'actual_completed_propagation_calls':completed,
            'controls':'PASS','records':records,'new_propagation_calls':0,
            'scientific_verdict_changed':False,'scientific_contract_changes':False}
    write(RECOVERY/'audit.json',result)
    corrected=dict(original)
    corrected['attempted_propagation_calls']=completed
    corrected['call_accounting']='CORRECTED_FROM_COMPLETE_RESPONSE_IDS'
    write(RECOVERY/'summary.json',corrected)
    independent=read(OUT/'independent_audit.json')
    require(independent['artifact_consistency']=='PASS','independent QR saved audit')
    artifacts={str(p.relative_to(ROOT)):digest(p) for p in OUT.rglob('*') if p.is_file()}
    write(ROOT/'docs/wb131_fixed_qop_nuisance_result_manifest.json',
          {'schema':'wb131_result_manifest_v1','summary':corrected,'original_summary':original,
           'call_accounting_audit':result,'independent_audit':independent,
           'independent_audit_boundary':'its attempted_propagation_calls field counts visible log lines; corrected exact counts above',
           'sources':{str(p.relative_to(ROOT)):digest(p) for p in SOURCES+[Path(__file__).resolve()]},
           'artifacts':artifacts,
           'scientific_boundary':{'association':'INHERITED_CONDITIONAL','qop':'FIXED_PHYSICAL_SEED_NO_PRIOR',
                                  'covariance':'NOT_CALIBRATED','material':'NONE_CONDITIONAL_MODEL',
                                  'alignment':'NOT_ESTABLISHED','held_out':'NOT_ACCESSED','ML':'NOT_AUTHORIZED'}})


def read_control_copy(value):
    import copy
    return copy.deepcopy(value)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','run'])
    globals()[parser.parse_args().action]()
