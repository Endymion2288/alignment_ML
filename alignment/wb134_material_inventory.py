"""Material input gates and synthetic noise controls; never fit detector noise."""
import math
import numpy as np


def require(ok, message):
    if not ok:raise ValueError(message)


def thin_theta(thickness_mm, x0_mm, momentum_MeV, mass_MeV):
    require(all(math.isfinite(v) for v in (thickness_mm,x0_mm,momentum_MeV,mass_MeV)) and
            thickness_mm>=0 and x0_mm>0 and momentum_MeV>0 and mass_MeV>=0, 'noise units/inputs')
    if thickness_mm==0:return 0.
    t=thickness_mm/x0_mm;beta=momentum_MeV/math.hypot(momentum_MeV,mass_MeV)
    return 13.6/(beta*momentum_MeV)*math.sqrt(t)*(1+0.038*math.log(t/(beta*beta)))


def toy_process_q(theta, lever_mm, direction):
    require(theta>=0 and math.isfinite(theta) and math.isfinite(lever_mm),'toy noise input')
    d=np.asarray(direction,float);require(d.shape==(3,) and np.all(np.isfinite(d)) and abs(np.linalg.norm(d)-1)<1e-12,'unit direction')
    transverse=np.eye(3)-np.outer(d,d)
    a=np.vstack((lever_mm*np.eye(3),np.eye(3)))
    return theta*theta*(a@transverse@a.T)


def classify_inventory(response):
    surfaces=response['surfaces'];volumes=response['volumes']
    require([s['index'] for s in surfaces]==list(range(len(surfaces))),'surface indices')
    require([v['index'] for v in volumes]==list(range(len(volumes))),'volume indices')
    require(bool(volumes) and bool(surfaces),'empty inventory')
    reached=set();active=set()
    def visit(index):
        require(0<=index<len(volumes),'child index')
        require(index not in active,'volume cycle')
        if index in reached:return
        reached.add(index);active.add(index)
        for child in volumes[index]['children']:visit(child)
        active.remove(index)
    visit(response['root_volume_index']);require(len(reached)==len(volumes),'disconnected volume')
    surface_counts={};volume_counts={};sensitive=set()
    for v in volumes:
        kind=v['material_kind'];require(kind in ('NONE','PROTO','NONPROTO_UNKNOWN'),'volume type')
        volume_counts[kind]=volume_counts.get(kind,0)+1
        require(len(set(v['surface_indices']))==len(v['surface_indices']),'duplicate volume surface')
        for index in v['surface_indices']:require(0<=index<len(surfaces),'surface link')
    for s in surfaces:
        kind=s['material_kind'];require(kind in ('NONE','PROTO_VACUUM','HOMOGENEOUS','BINNED','OTHER_NONPROTO_UNKNOWN'),'surface type')
        surface_counts[kind]=surface_counts.get(kind,0)+1
        require(bool(s['memberships']),'surface membership absent')
        for m in s['memberships']:
            require(m['role'] in ('BOUNDARY','LAYER_REPRESENTATION','SENSITIVE_ARRAY','APPROACH'),'role')
            require(s['index'] in volumes[m['volume_index']]['surface_indices'],'membership link')
            if m['role']=='SENSITIVE_ARRAY':sensitive.add(s['index'])
        sample=s['center_sample']
        if kind=='NONE':require(sample is None,'null material sample')
        if kind=='PROTO_VACUUM':require(not sample['valid'] and sample['thickness_mm']==sample['thickness_in_X0']==sample['thickness_in_L0']==0,'Proto not vacuum')
    require(response['sensitive_visit_count']==len(sensitive),'sensitive recursion coverage')
    return {'surface_counts':surface_counts,'volume_counts':volume_counts,'surfaces':len(surfaces),'volumes':len(volumes),'sensitive_surfaces':len(sensitive)}


def audit_event(export, request, response, protocol):
    require(response['identity']==export['identity']==request['identity'] and response['source']==request['source'],'event/source identity')
    require(response['material_source']=='None' and not response['map_loaded'],'material setting changed')
    for key in ('new_propagation_calls','new_reconstruction_calls','algorithm_field_queries'):require(response[key]==0,'unexpected work')
    require(response['path_integral'] is None and response['process_noise_covariance'] is None,'unqualified physical budget')
    inventory=classify_inventory(response);require(len(response['targets'])==len(export['rows']),'target coverage')
    for before,after in zip(export['rows'],response['targets']):
        require(all(before[k]==after[k] for k in ('cluster_id','wafer_id','station')),'target identity')
        s=response['surfaces'][after['surface_index']]
        require(after['geometry_id']==s['geometry_id'],'target geometry id')
        frame=np.asarray(s['transform']);require(np.max(np.abs(frame-before['sensor_transform']))<=protocol['frame_tolerance'] and after['frame_max_error']<=protocol['frame_tolerance'],'frame changed')
    c=response['controls'];require(c['synthetic_only'] and not c['vacuum']['valid'] and c['vacuum']['thickness_in_X0']==0,'synthetic/vacuum controls')
    expected=thin_theta(c['thin_slab']['thickness_mm'],93.7,c['momentum_GeV']*1000,c['mass_MeV'])
    require(abs(c['thin_slab']['thickness_in_X0']-.3/93.7)<1e-9,'slab units')
    require(abs(c['theta0_rad']-expected)<=max(1e-10,abs(expected)*1e-4),'ACTS/Highland control')
    q=toy_process_q(c['theta0_rad'],c['lever_mm'],[0,0,1]);require(np.linalg.eigvalsh(q).min()>=-1e-18,'noise PSD')
    return {'schema':'wb134_material_audit_v1','identity':export['identity'],'interface':'PASS',**inventory,
            'targets':len(response['targets']),'toy_controls':'PASS','toy_theta0_rad':c['theta0_rad'],'toy_q':q.tolist(),
            'physical_budget':'UNKNOWN_MATERIAL_INPUT_AND_PATH_NOT_QUALIFIED','source_uncertainty':'NOT_CALIBRATED_NO_PRIOR',
            'new_propagation_calls':0,'new_reconstruction_calls':0,'algorithm_field_queries':0,'held_out_access':False,'truth_access':False}
