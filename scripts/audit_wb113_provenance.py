#!/usr/bin/env python3
"""Fail-closed descriptive classification of persisted, allowlisted SDO links."""
import math
from collections import Counter

def check(value, message):
    if not value:raise ValueError(message)

def classify(fixture, data):
    identity={k:fixture[k] for k in ('input_xaod','ordinal','actual_run','actual_event')}
    check(data['identity']==identity,'event/source/ordinal identity')
    allowed={int(c):r['station'] for r in fixture['references'] for c in r['clusters']}
    check(len(allowed)==sum(len(r['clusters']) for r in fixture['references']),'duplicate allowlist')
    check(len(data['clusters'])==len(allowed),'cluster coverage')
    seen=set();cluster_rows=[];unique=[];all_complete=True;resolved_by_link={}
    for c in data['clusters']:
        cid=c['cluster_id'];check(cid in allowed and cid not in seen,'cluster identity/duplicate');seen.add(cid)
        check(c['station']==allowed[cid],'station identity')
        links=set();barcodes=set();reasons=[];rdos=set()
        if not c['rdos']:reasons.append('NO_RDO')
        for r in c['rdos']:
            check(r['rdo_id'] not in rdos,'duplicate RDO');rdos.add(r['rdo_id']);local=set()
            if not r['sdo_present'] or not r['deposits']:reasons.append('MISSING_SDO_OR_DEPOSITS')
            for d in r['deposits']:
                check(math.isfinite(d['weight_native']),'nonfinite deposit weight')
                key=(d['event_collection'],d['event_index'],d['event_position'],d['barcode'])
                if key in local:reasons.append('DUPLICATE_DEPOSIT_LINK')
                local.add(key);barcodes.add(d['barcode']);links.add(key)
                if not d['valid'] or d['event_position'] is None or d['event_position']==4294967295 or d['event_index']==4294967295 or d['barcode']==0:
                    reasons.append('INVALID_OR_UNRESOLVED_LINK')
                if d['weight_native']<=0:reasons.append('NONPOSITIVE_DEPOSIT_WEIGHT')
                resolved=d['resolved_particle']
                if resolved is None:reasons.append('MISSING_RESOLVED_PARTICLE')
                else:
                    check(all(math.isfinite(x) for x in resolved['momentum_native']),'nonfinite particle momentum')
                    if key in resolved_by_link:check(resolved_by_link[key]==resolved,'inconsistent resolved particle for same persisted identity')
                    resolved_by_link[key]=resolved
                    if resolved['parent_event_number'] is None:reasons.append('UNKNOWN_PARENT_EVENT')
        complete=not reasons and len(links)==1
        if len(links)>1:reasons.append('MIXED_PARTICLE_IDENTITIES')
        all_complete=all_complete and complete
        if complete:unique.append(next(iter(links)))
        cluster_rows.append({'cluster_id':cid,'station':c['station'],'rdo_count':len(rdos),
            'particle_links':[list(k) for k in sorted(links,key=repr)],'barcodes':sorted(barcodes),
            'complete_unique_link':complete,'limitations':sorted(set(reasons))})
    check(seen==set(allowed),'cluster set')
    association='UNKNOWN_OR_AMBIGUOUS'
    if all_complete:
        association='SUPPORTED_PERSISTED_COMMON_PARTICLE' if len(set(unique))==1 else 'CONTRADICTED_COMMON_PARTICLE'
    by_station={}
    for s in range(4):
        counts=Counter(b for c in cluster_rows if c['station']==s for b in c['barcodes'])
        by_station[str(s)]={'cluster_count':sum(c['station']==s for c in cluster_rows),'barcode_cluster_counts':dict(sorted(counts.items()))}
    # Exact link agreement must not turn an ambiguous xAOD join into a charge oracle.
    relevant={b for c in cluster_rows for b in c['barcodes']};particles=data['related_truth_particles'];truth_counts=Counter(p['barcode'] for p in particles)
    check(all(p['barcode'] in relevant for p in particles),'unrelated truth export')
    for p in particles:check(math.isfinite(p['charge_e']) and all(math.isfinite(x) for x in p['momentum_native']),'nonfinite xAOD truth')
    unique_truth={b:truth_counts[b]==1 for b in relevant}
    return {'schema':'wb113_provenance_summary_v1','integrity_gate':'PASS','identity':identity,'association':association,
        'cluster_links':cluster_rows,'by_station':by_station,'xaod_barcode_join_unique':unique_truth,
        'related_truth_particles':particles,'persisted_tracks':data['persisted_tracks'],
        'generation_conditions_compatibility':'UNKNOWN_PENDING_METADATA_EVIDENCE',
        'new_reconstruction_calls':0,'new_propagation_calls':0,'held_out_access':False,
        'qualification':'NOT_EVALUATED','physics_screening':'NOT_EVALUATED','final_oracle':'NOT_EVALUATED'}
