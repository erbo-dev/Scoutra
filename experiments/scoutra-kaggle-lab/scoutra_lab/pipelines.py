import json
from collections import defaultdict
from pathlib import Path
from .common import Run, gpu_guard, seed_all, sha256, dump, read_table, write_table
from .dataset import load_frames
from evaluation.metrics import load_gt,detection_metrics,identity_metrics,retrieval_metrics,retrieval_operating_point,restrict_gt

def blocked(config,kind,reason):
    run=Run(config,kind);run.metrics['reason']=reason;run.errors.append({'type':'BLOCKER','message':reason});run.finish();return run.path

def provenance(run,config,inputs):
    for name,path in inputs.items():
        run.provenance[name]={'path':str(path),'sha256':sha256(path)}
    run.provenance['measurement_caveat']='Single run is exploratory. Repeat isolated single-GPU runs, include cold and warm scopes separately; no speed claims from static preparation.'
    dump(run.path/'metadata.json',run.provenance)

def detect(config):
    if not config.get('execute'):return blocked(config,'detections','NOT EXECUTED: enable execute only inside confirmed Kaggle after installing the selected isolated profile and supplying reviewed assets')
    run=Run(config,'detections')
    try:
        torch,index=gpu_guard(config);seed_all(config['seed'])
        from adapters.detection import Detector
        from PIL import Image
        frames=load_frames(config['frames']);provenance(run,config,{'frame_cache':config['frames']})
        detector=Detector(config['model'],f'cuda:{index}');rows=[]
        with run.measured(torch,index):
            for f in frames:
                for n,d in enumerate(detector.predict(Image.open(f['path']).convert('RGB'))):
                    rows.append({**{k:f[k] for k in ('clip_id','scene_id','frame_index','timestamp_s')},**d,'detection_id':f'{f["clip_id"]}:{f["frame_index"]}:{n}'})
            write_table(run.path/'predictions.parquet',rows,'detections')
        run.metrics=detection_metrics(rows,restrict_gt(load_gt(config.get('ground_truth')),frames),score_threshold=config.get('evaluation_score_threshold',.25));run.finish()
    except Exception as exc:
        run.errors.append({'type':type(exc).__name__,'message':str(exc)});run.metrics['reason']=str(exc);run.finish();raise
    return run.path

def track(config):
    if not config.get('execute'):return blocked(config,'tracks','NOT EXECUTED: requires one frozen detector cache, dense frames, reviewed scene cuts, and Kaggle runtime')
    run=Run(config,'tracks')
    try:
        torch,index=gpu_guard(config);seed_all(config['seed'])
        import cv2
        from adapters.tracking import Tracker
        frames=load_frames(config['frames']);dets=read_table(config['detections']);by=defaultdict(list)
        for d in dets:by[d['clip_id'],d['frame_index']].append(d)
        valid={(f['clip_id'],f['frame_index']) for f in frames}
        if any(k not in valid for k in by):raise ValueError('Detections refer to frames outside cache')
        provenance(run,config,{'frame_cache':config['frames'],'detections':config['detections']})
        rows=[];tracker=None;previous=None;last_frame=None;epoch=-1
        with run.measured(torch,index):
            for f in frames:
                scene=(f['clip_id'],f['scene_id'])
                if previous!=scene or (last_frame is not None and f['frame_index']!=last_frame+1):
                    tracker=Tracker(config['tracker'],f['fps'],f'cuda:{index}');previous=scene;epoch+=1
                image=cv2.imread(f['path'])
                if image is None:raise ValueError('Unreadable shared cached frame')
                chosen=[d for d in by[f['clip_id'],f['frame_index']] if d['class_id'] in config.get('tracked_classes',[1])]
                for r in tracker.update(chosen,image):
                    rows.append({**{k:f[k] for k in ('clip_id','scene_id','frame_index','timestamp_s')},**{k:v for k,v in r.items() if k!='local_id'},'tracklet_id':f'{f["clip_id"]}:{f["scene_id"]}:epoch{epoch}:t{r["local_id"]}'})
                last_frame=f['frame_index']
            write_table(run.path/'predictions.parquet',rows,'tracks')
        anchors=json.loads(Path(config['anchors']).read_text()).get('anchors',[]) if config.get('anchors') and Path(config['anchors']).is_file() else []
        run.runtime['timing_scope']='dense frame read, tracker initialization/reset including ReID loading per scene, feature extraction, association and serialization; no warmup exclusion'
        run.metrics=identity_metrics(rows,restrict_gt(load_gt(config.get('ground_truth')),frames),anchors);run.finish()
    except Exception as exc:
        run.errors.append({'type':type(exc).__name__,'message':str(exc)});run.metrics['reason']=str(exc);run.finish();raise
    return run.path

def reid(config):
    if not config.get('execute'):return blocked(config,'embeddings','NOT EXECUTED: requires immutable crop manifest and reviewed checkpoint; full PRTReID/BPBReID need isolated upstream extraction and visibility-aware distances')
    run=Run(config,'embeddings')
    try:
        torch,index=gpu_guard(config);seed_all(config['seed'])
        import numpy as np
        cropfile=Path(config['crops']);crops=json.loads(cropfile.read_text());provenance(run,config,{'crop_manifest':cropfile})
        if not crops:raise ValueError('Empty crop manifest')
        for c in crops:
            if sha256(c['path'])!=c['sha256']:raise ValueError('Crop changed since labeling')
        if config['model']['method']!='osnet':raise RuntimeError('Full-part upstream bridge required; do not substitute global embeddings')
        from adapters.reid import OSNet
        extractor=OSNet(config['model'],f'cuda:{index}')
        with run.measured(torch,index):
            features=extractor.extract([c['path'] for c in crops]);np.save(run.path/'features.npy',features,allow_pickle=False)
            rows=[{'crop_id':c['crop_id'],'head':'global','part_index':0,'visibility':None,'feature_path':'features.npy','array_index':i} for i,c in enumerate(crops)]
            write_table(run.path/'predictions.parquet',rows,'embeddings')
        labeled=all(c.get('reviewed') is True and c.get('player_id') and c.get('source_match') and c.get('tracklet_id') for c in crops)
        if labeled:
            q=[i for i,c in enumerate(crops) if c.get('retrieval_role')=='query'];g=[i for i,c in enumerate(crops) if c.get('retrieval_role')=='gallery']
            if not q or not g:run.metrics={'status':'NOT EVALUATED','metrics':None,'reason':'Assign disjoint query/gallery crop roles before evaluation'}
            else:
                norms=np.linalg.norm(features,axis=1,keepdims=True)
                if (norms<1e-12).any():raise ValueError('Zero-norm embeddings')
                f=features/norms;distance=1-f[q]@f[g].T
                np.save(run.path/'distances.npy',distance,allow_pickle=False)
                queries=[crops[i] for i in q];gallery=[crops[i] for i in g]
                run.metrics=retrieval_metrics(distance,queries,gallery)
                if config.get('validation_distance_threshold') is not None:
                    run.metrics['metrics']['fixed_operating_point']=retrieval_operating_point(distance,queries,gallery,config['validation_distance_threshold'])
        else:run.metrics={'status':'NOT EVALUATED','metrics':None,'reason':'Unlabeled extraction only; physical IDs/source matches/review missing'}
        dump(run.path/'crop_manifest.json',crops);run.finish()
    except Exception as exc:
        run.errors.append({'type':type(exc).__name__,'message':str(exc)});run.metrics['reason']=str(exc);run.finish();raise
    return run.path

def recover(config):
    if not config.get('execute'):return blocked(config,'identities','NOT EXECUTED: requires frozen local tracks and compatible appearance features; external GTA/McByte exports remain separate conditions')
    # This offline step is cheap but still intentionally restricted to the configured Kaggle workflow.
    run=Run(config,'identities')
    try:
        torch,index=gpu_guard(config);seed_all(config['seed'])
        frames=load_frames(config['frames']);tracks=read_table(config['tracks']);provenance(run,config,{'tracks':config['tracks'],'frame_cache':config['frames']})
        from adapters.identity import no_link,agentcoach
        with run.measured():
            if config['method']=='no_link':rows=no_link(tracks)
            elif config['method']=='agentcoach_reid_only':
                import numpy as np
                features=np.load(config['features'],allow_pickle=False);crops=json.loads(Path(config['crop_manifest']).read_text())
                if features.ndim!=2 or len(features)!=len(crops):raise ValueError('Expected global NxD features aligned exactly with crop manifest')
                provenance(run,config,{'features':config['features'],'crop_manifest':config['crop_manifest']})
                rows,clusters=agentcoach(tracks,{c['detection_id']:features[i] for i,c in enumerate(crops)},config['agentcoach'])
                dump(run.path/'clusters.json',clusters)
            else:raise RuntimeError('Method not implemented in core runner; use validated external bridge, not a surrogate')
            write_table(run.path/'predictions.parquet',rows,'identities')
        run.runtime['timing_scope']='offline linking including source import and cached feature loading, constraints and serialization; CPU wall time'
        lookup={(r['clip_id'],r['frame_index'],r['tracklet_id']):r for r in rows}
        merged=[{**t,**lookup[t['clip_id'],t['frame_index'],t['tracklet_id']]} for t in tracks]
        anchors=json.loads(Path(config['anchors']).read_text()).get('anchors',[]) if config.get('anchors') and Path(config['anchors']).is_file() else []
        run.metrics=identity_metrics(merged,restrict_gt(load_gt(config.get('ground_truth')),frames),anchors);run.finish()
    except Exception as exc:
        run.errors.append({'type':type(exc).__name__,'message':str(exc)});run.metrics['reason']=str(exc);run.finish();raise
    return run.path
