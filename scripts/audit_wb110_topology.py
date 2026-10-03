"""Runtime geometry census audit without track propagation."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from audit_wb106_trace import check,digest
from audit_wb109_abort import frame,vector

def metadata(node):return {k:v for k,v in node.items() if k!='boundaries'}

def validate(result,probe,fixture):
    check(result['probe_used']==probe,'probe identity')
    check(result['identity']=={k:fixture[k] for k in ('actual_run','actual_event','input_xaod','ordinal')},'event/source identity')
    check(result['new_propagation_calls']==0 and result['algorithm_field_queries']==0,'geometry-only execution')
    check(result['world']==probe['world'],'world metadata differs from WB109')
    nodes=result['volumes'];check(bool(nodes),'empty volume tree')
    ids=[n['geometry_id'] for n in nodes];check(len(ids)==len(set(ids)) and all(isinstance(i,int) and i>0 for i in ids),'duplicate/invalid volume ID')
    lookup={n['geometry_id']:n for n in nodes};world_id=result['world']['geometry_id']
    check(world_id in lookup and metadata(lookup[world_id])==result['world'],'world tree identity')
    target=probe['target_volume'];check(target['geometry_id'] in lookup and metadata(lookup[target['geometry_id']])==target,'target volume identity')
    boundary_refs=[]
    for node in nodes:
        frame(node['transform']);bounds=np.asarray(node['bounds_values'])
        check(np.isfinite(bounds).all(),'nonfinite volume bounds')
        if node['bounds_type']==1:check(bounds.shape==(3,) and (bounds>0).all(),'cuboid bounds')
        for b in node['boundaries']:
            frame(b['frame']);check(np.isfinite(np.asarray(b['bounds_values'])).all(),'nonfinite boundary bounds')
            if b['geometry_id']==probe['current_surface']['geometry_id']:
                boundary_refs.append((node['geometry_id'],b))
    edges=result['edges'];check(len({(e['parent'],e['child'],e['kind']) for e in edges})==len(edges),'duplicate edge')
    check(all(e['parent'] in lookup and e['child'] in lookup and e['parent']!=e['child'] and e['kind'] in ('confined','dense') for e in edges),'invalid volume edge')
    adjacency={i:[] for i in ids}
    for e in edges:adjacency[e['parent']].append(e['child'])
    done=set();active=set()
    def visit(i):
        check(i not in active,'volume tree cycle')
        if i in done:return
        active.add(i)
        for j in adjacency[i]:visit(j)
        active.remove(i);done.add(i)
    visit(world_id);check(done==set(ids),'disconnected volume tree')
    matches=result['matches']
    check([(m['owner_geometry_id'],m['surface']) for m in matches]==boundary_refs,'matches detached from volume census')
    check(result['attachment_queries']==2*len(matches),'attachment query count')
    for m in matches:
        check(m['surface']==probe['current_surface']==probe['navigation_boundary_surface'],'terminating surface metadata')
        for key in ('forward_attachment','reverse_attachment'):
            a=m[key]
            if a is not None:check(a['geometry_id'] in lookup and metadata(lookup[a['geometry_id']])==a,'attachment identity absent/different')
    check(isinstance(result['world_inside_probe'],bool),'world containment flag')
    world=result['world'];check(world['bounds_type']==1,'world cuboid type')
    pos=vector(probe['position']);direction=vector(probe['direction'])
    world_local=(np.linalg.inv(frame(world['transform']))@np.r_[pos,1.])[:3]
    margin=np.asarray(world['bounds_values'])-np.abs(world_local)
    inside=bool(np.all(margin>=0));check(result['world_inside_probe']==inside,'world containment disagreement')
    summary={'classification':'OTHER_OR_UNRESOLVED_TOPOLOGY','matched_owners':len(matches),
      'world_inside_probe':inside,'world_interior_margins_mm':margin.tolist(),
      'volume_count':len(nodes),'edge_count':len(edges),'attachment_queries':result['attachment_queries'],
      'deeper_acceptance_or_seed_mechanism':'UNKNOWN','new_propagation_calls':0,'algorithm_field_queries':0}
    if len(matches)!=1:return summary
    match=matches[0];owner=lookup[match['owner_geometry_id']];summary.update(owner=metadata(owner),
      forward_attachment=match['forward_attachment'],reverse_attachment=match['reverse_attachment'])
    if owner['bounds_type']!=1:return summary
    transform=frame(owner['transform']);local=(np.linalg.inv(transform)@np.r_[pos,1.])[:3]
    half=np.asarray(owner['bounds_values']);tol=probe['options']['surfaceTolerance_mm']
    candidates=np.flatnonzero(np.abs(np.abs(local)-half)<=tol)
    axis=int(candidates[0]) if len(candidates)==1 else None
    outward=axis is not None and (transform[:3,:3].T@direction)[axis]*np.sign(local[axis])>0
    side=axis in (0,1) and np.all(np.abs(local)<=half+tol)
    summary.update(owner_local_position_mm=local.tolist(),face_axis=axis,face_sign=int(np.sign(local[axis])) if axis is not None else None,
      direction_outward=bool(outward),owner_parents=[e['parent'] for e in edges if e['child']==owner['geometry_id']])
    if (owner['name']==probe['last_nonnull_volume_name'] and owner['geometry_id']!=world_id and
        match['forward_attachment'] is None and match['reverse_attachment']==metadata(owner) and
        inside and bool(np.all(margin>tol)) and side and outward):
        summary['classification']='INTERNAL_CHILD_BOUNDARY_NULL_ATTACHMENT'
    return summary

def audit(out):
    from wb110_contract import BASE,verify
    frozen=verify(out);event=out/'event'
    fixture=json.loads((out/'fixture.json').read_text());probe=json.loads((out/'probe.json').read_text())
    result=json.loads((event/'topology.json').read_text())
    summary=validate(result,probe,fixture)
    for name in ('fixture.json','probe.json'):check(digest(event/name)==digest(out/name),'event input copy '+name)
    raw=json.loads((out/'raw_identity.json').read_text())
    previous_raw=json.loads((BASE/'raw_identity.json').read_text())
    check(raw['path']==fixture['input_xaod'] and raw['sha256']==previous_raw['sha256'] and
      {'bytes':raw['bytes'],'mtime_ns':raw['mtime_ns']}==fixture['source_stat'],'raw identity receipt')
    expected=json.loads((out/'generation_expectation.json').read_text())
    for name,h in expected['files'].items():check(digest(out/'isolated_source'/name)==h,'generated source')
    check(digest(event/'athena.py')==expected['athena_sha256'],'runner source')
    binary=json.loads((out/'binary_manifest.json').read_text())
    check(binary['binary'] in result['loaded_libraries'] and digest(binary['binary'])==binary['sha256'],'diagnostic binary')
    libraries=[p for p in result['loaded_libraries'] if any(k in p for k in ('FaserActs','ActsCore'))]
    check(bool(libraries),'official geometry libraries absent')
    for p in libraries:check(p in frozen['hashes'] and digest(p)==frozen['hashes'][p],'official library')
    for p in (out/'identity_payload').iterdir():
        if p.is_file():check(digest(p)==digest(BASE/'identity_payload'/p.name),'copied payload')
    for pfn in ET.parse(out/'identity_payload/PoolFileCatalog.xml').findall('.//pfn'):
        p=pfn.attrib['name'];check(digest(p)==frozen['hashes'][p],'actual POOL PFN')
    manifest=json.loads((event/'athena_manifest.json').read_text())
    check(manifest['sqlite_sha256']==digest(out/'identity_payload/tracker_alignment.sqlite'),'actual sqlite')
    check(manifest['pool_catalog_sha256']==digest(out/'identity_payload/PoolFileCatalog.xml'),'actual pool catalog')
    check((manifest['geometry'],manifest['global_tag'],manifest['field_mode'],manifest['propagation'])==
      ('FASERNU-04','OFLCOND-FASER-06','NOT_QUERIED_GEOMETRY_ONLY',False),'geometry-only manifest')
    log=(event/'athena.log').read_text()
    for required in ('start processing event #2268, run #100044',str(out/'identity_payload/tracker_alignment.sqlite'),'Acts TrackingGeometry construction completed'):
        check(required in log,'conditions/header receipt')
    check(json.loads((out/'athena_exit.json').read_text())['exit_code']==0,'Athena execution')
    return {'schema':'wb110_topology_summary_v1','integrity_gate':'PASS','identity':result['identity'],
      'held_out_access':False,'qualification':'NOT_EVALUATED',**summary}
