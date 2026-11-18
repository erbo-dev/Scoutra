import importlib.util, itertools, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scoutra_lab.common import verify_repo

def no_link(tracks):
    return [{**{k:r[k] for k in ('clip_id','scene_id','frame_index','tracklet_id')},'player_id':r['tracklet_id'],'confidence':None,'state':'assigned','evidence':'local tracklet only; no physical identity claim'} for r in tracks]

def agentcoach(tracks,features,spec):
    verify_repo(spec['upstream'])
    file=Path(spec['upstream']['path'])/'goalinsight/track_consolidation/_consolidator.py'
    loader=importlib.util.spec_from_file_location('audited_agentcoach_consolidator',file)
    module=importlib.util.module_from_spec(loader); sys.modules[loader.name]=module; loader.loader.exec_module(module)
    grouped=defaultdict(list)
    for r in tracks:grouped[(r['clip_id'],r['tracklet_id'])].append(r)
    mapping={}; evidence=[]
    # No cross-highlight linking: independent clips unless match/clock alignment is established.
    for clip in sorted({k[0] for k in grouped}):
        ids=sorted(k[1] for k in grouped if k[0]==clip); index={tid:i for i,tid in enumerate(ids)}
        metas=[]; simultaneous=defaultdict(set)
        for tid in ids:
            rows=grouped[clip,tid]
            for r in rows:simultaneous[(r['scene_id'],r['frame_index'])].add(index[tid])
            vectors=[features[r['detection_id']] for r in rows if r['detection_id'] in features]
            if not vectors:raise ValueError(f'Missing embeddings for {tid}')
            centroid=module.compute_centroid(vectors)
            # Optional OCR/team evidence must be predicted, not supplied GT; initial condition is ReID-only.
            metas.append(module.TrackMeta(track_id=index[tid],team='unknown',role='player',frame_count=len(rows),jersey_number=None,jersey_confidence=0.,reid_centroid=centroid))
        cooccur=set(pair for tids in simultaneous.values() for pair in itertools.combinations(sorted(tids),2))
        out,clusters=module.consolidate(metas,cooccur_pairs=cooccur,**spec.get('parameters',{}))
        for tid,i in index.items():mapping[clip,tid]=f'{clip}:{out[i]}'
        evidence.extend({'clip_id':clip,**c.to_dict()} for c in clusters)
    result=[{**{k:r[k] for k in ('clip_id','scene_id','frame_index','tracklet_id')},'player_id':mapping[r['clip_id'],r['tracklet_id']],'confidence':None,'state':'assigned','evidence':'AgentCoach upstream ReID-only consolidation; threshold is not a calibrated probability'} for r in tracks]
    return result,evidence
