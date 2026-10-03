"""Exact persistent-type inventory, justified by the pinned official converter."""
TRACK_CLASS='Trk::TrackCollection_tlp6'
TRACK_KEYS=('CKFTrackCollection','CKFTrackCollectionBackward',
    'CKFTrackCollectionBackwardWithoutIFT','CKFTrackCollectionWithoutIFT','SegmentFit','Segments')

def check(value,message):
    if not value:raise ValueError(message)

def control_from_metadata(current,expected,source_metadata,item_list=None):
    # Inventory is already seen file-level evidence, never selected by event outcomes.
    check(current['source']==expected['source'],'raw source identity')
    for k in ('tree','entries','root_uuid'):
        check(current[k]==expected[k],'ROOT identity '+k)
    check(current['tree']=='CollectionTree' and current['event_entries_decoded']==0,'metadata-only access')
    branches=current['branches'];seen=set();tracks=[]
    for b in branches:
        check(b['name'] not in seen,'duplicate branch');seen.add(b['name'])
        if b['class']==TRACK_CLASS:
            check(b['title']==b['name'],'track branch title/name')
            tracks.append(b['name'])
        if b['name'] in TRACK_KEYS:check(b['class']==TRACK_CLASS,'track persistent type changed')
    check(set(tracks)==set(TRACK_KEYS) and len(tracks)==len(TRACK_KEYS),'track inventory changed')
    # Cross-check the entire frozen branch name/type/title list without treating unknown classes as tracks.
    def rows(bs):return sorted((b['name'],b['class'],b['title']) for b in bs)
    check(rows(branches)==rows(expected['branches']),'frozen branch inventory changed')
    check(set(source_metadata)=={current['source']['path']},'metadata source identity')
    metadata=source_metadata[current['source']['path']]
    check(metadata.get('/TagInfo')=={'AtlasRelease':'Athena-24.0.41','GeoFaser':'FASERNU-04','IOVDbGlobalTag':'OFLCOND-FASER-05'},'source TagInfo changed')
    if item_list is not None:
        check(isinstance(item_list,list),'EventStreamInfo item list shape')
        check(all(isinstance(r,(list,tuple)) and len(r)==2 for r in item_list),'EventStreamInfo item shape')
        keys=[k for t,k in item_list if t=='TrackCollection']
        check(len(keys)==len(tracks) and set(keys)==set(tracks),'EventStreamInfo / ROOT track disagreement')
    return {'schema':'wb114_persistent_type_control_v1','metadata_origin':'ROOT_BRANCH_PERSISTENT_TYPE',
        'track_keys':sorted(tracks),'track_persistent_class':TRACK_CLASS,
        'track_transient_type':'TrackCollection','converter_p6_guid':'3228B252-2C5D-11E8-B170-0800271C02BC',
        'event_stream_itemlist_present':item_list is not None,
        'source_tag_info':metadata['/TagInfo'],
        'executor_tag_info':{'GeoFaser':'FASERNU-04','IOVDbGlobalTag':'OFLCOND-FASER-06'},
        'tag_string_comparison':'GEOMETRY_TAG_EQUAL_GLOBAL_TAG_DIFFERS',
        'physical_conditions_compatibility':'UNKNOWN','event_entries_decoded':0}
