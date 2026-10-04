#!/usr/bin/env python3
"""Saved-only typed interface recovery; never reads event data or reruns Athena."""
import argparse
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from wb132_contract import ROOT, OUT, PROTOCOL, SOURCES, read, write, digest, require, verify
sys.path.insert(0,str(ROOT))
from alignment.wb132_typed_coverage import audit_typed_event

RECOVERY=OUT/'saved_typed_coverage_v1'
RECOVERY_PROTOCOL=ROOT/'configs/research_review/wp132_typed_coverage_recovery.json'
RECOVERY_SOURCES=[RECOVERY_PROTOCOL,Path(__file__).resolve(),ROOT/'alignment/wb132_typed_coverage.py',
                  ROOT/'scripts/wb132_typed_independent_audit.py',ROOT/'tests/test_wb132_typed_coverage.py']


def freeze():
    verify();RECOVERY.mkdir(exist_ok=False);(RECOVERY/'events').mkdir()
    paths=RECOVERY_SOURCES+[OUT/'freeze.json',OUT/'summary.json',OUT/'producer_evidence.json']
    for index in read(PROTOCOL)['indices']:
        event=OUT/'events'/f'{index:02d}'
        paths += [event/'native_response.json',event/'exit.json',event/'status.json',event/'athena.log']
        require(read(event/'exit.json')['exit_code']==0,'raw execution failure')
        require(read(event/'status.json').get('failure')=='old selected membership changed','recovery scope: different original failure')
    write(RECOVERY/'freeze.json',{'schema':'wb132_typed_recovery_freeze_v1','hashes':{str(p):digest(p) for p in paths},
          'scientific_threshold_changes':False,'new_input_event_executions':0,'new_propagation_calls':0,
          'new_reconstruction_calls':0,'raw_population':147,'primary_accepted_population':111})


def verify_recovery():
    verify()
    for path,expected in read(RECOVERY/'freeze.json')['hashes'].items():require(digest(path)==expected,'saved typed recovery identity '+path)


def run():
    verify_recovery();events=[]
    for index in read(PROTOCOL)['indices']:
        raw=OUT/'events'/f'{index:02d}';dest=RECOVERY/'events'/raw.name;dest.mkdir(exist_ok=False)
        result=audit_typed_event(read(raw/'export.json'),read(raw/'native_response.json'),read(raw/'parent_provenance.json'),
             read(raw/'parent_response.json'),read(PROTOCOL),read(OUT/'producer_evidence.json'))
        write(dest/'audit.json',result)
        status={'index':index,'interface':result['interface'],'native_prediction_hypothesis':result['native_prediction_hypothesis'],
                'accepted_matches':result['matched_rows'],'all_exact_links':result['all_exact_linked_rows'],
                'excluded_linked_rows':result['excluded_linked_rows'],'truly_unlinked_rows':result['truly_unlinked_rows'],
                'source_type':result['source']['type_description'],'source_measurement_anchor':result['source']['measurement_anchor']}
        write(dest/'status.json',status);events.append(status);print(status,flush=True)
    verdicts=[e['native_prediction_hypothesis'] for e in events]
    verdict=('PASS_GROSS_SCREEN_ONLY' if all(v=='PASS_GROSS_SCREEN_ONLY' for v in verdicts)
             else 'FAIL_GROSS_NATIVE_RESIDUAL' if 'FAIL_GROSS_NATIVE_RESIDUAL' in verdicts else 'UNKNOWN')
    verify_recovery()
    write(RECOVERY/'summary.json',{'schema':'wb132_typed_recovery_summary_v1','events':events,'interface':'PASS',
          'native_prediction_hypothesis':verdict,'original_execution_contract':read(OUT/'summary.json')['execution_contract'],
          'original_analysis_contract':'UNKNOWN_PRESERVED','all_exact_linked_rows':sum(e['all_exact_links'] for e in events),
          'accepted_measurement_rows':sum(e['accepted_matches'] for e in events),
          'excluded_linked_rows':sum(e['excluded_linked_rows'] for e in events),
          'truly_unlinked_rows':sum(e['truly_unlinked_rows'] for e in events),
          'new_input_event_executions':0,'new_propagation_calls':0,'new_reconstruction_calls':0,
          'scientific_threshold_changes':False,'held_out_access':False,'truth_access':False})


def seal():
    verify_recovery();independent=read(RECOVERY/'independent_audit.json');require(independent['artifact_consistency']=='PASS','independent evidence')
    write(ROOT/'docs/wb132_native_ckf_state_result_manifest.json',{'schema':'wb132_result_manifest_v1',
          'original_summary':read(OUT/'summary.json'),'typed_recovery_summary':read(RECOVERY/'summary.json'),
          'independent_audit':independent,'producer_evidence':read(OUT/'producer_evidence.json'),
          'sources':{str(p.relative_to(ROOT)):digest(p) for p in SOURCES+RECOVERY_SOURCES},
          'artifacts':{str(p.relative_to(ROOT)):digest(p) for p in OUT.rglob('*') if p.is_file()},
          'scientific_boundary':{'association':'EXACT_TSOS_PRD_LINKS_AUDITED_PARTICLE_ASSOCIATION_STILL_CONDITIONAL',
              'covariance':'NOT_CALIBRATED','historical_fit_semantics':'CURRENT_RULE_CONSISTENT_HISTORICAL_UNKNOWN',
              'qop_prior':'NOT_ESTABLISHED','alignment':'NOT_ESTABLISHED','held_out':'NOT_ACCESSED','ML':'NOT_AUTHORIZED'}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','verify_recovery','run','seal']);globals()[p.parse_args().action]()
