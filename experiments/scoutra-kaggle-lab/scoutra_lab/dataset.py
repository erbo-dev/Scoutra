import csv, json
from pathlib import Path
from .common import dump, sha256

FIELDS=['clip_id','relative_path','source_match','duration_s','fps','width','height','camera_type','challenge_category','selected_player_id','detections_available','identities_available','events_available','split','sha256','reviewed_scene_boundaries']

def discover(root,output):
    import cv2
    root=Path(root)
    if (Path(output)/'manifest.csv').exists():raise FileExistsError('Audit manifest already exists; use a new audit output folder to preserve manual review')
    if not root.is_dir(): raise FileNotFoundError(f'Dataset root does not exist: {root}; inspect /kaggle/input and set DATASET_ROOT')
    files=sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.mp4','.mov','.mkv','.avi'))
    if not files: raise FileNotFoundError('No video files found beneath configured root')
    rows=[]; errors=[]; seen=set()
    for p in files:
        clip=p.stem
        if clip in seen: raise ValueError('Duplicate filename stems; assign unique clip IDs before proceeding')
        seen.add(clip); cap=cv2.VideoCapture(str(p))
        if not cap.isOpened(): errors.append({'file':str(p),'error':'Cannot open video'}); cap.release(); continue
        fps=float(cap.get(cv2.CAP_PROP_FPS)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rows.append(dict(zip(FIELDS,[clip,str(p.relative_to(root)),'',count/fps if fps>0 else '',fps,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),'unknown','unreviewed','','unknown','unknown','unknown','unassigned',sha256(p),'false'])))
        cap.release()
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    with open(output/'manifest.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    dump(output/'audit_errors.json',errors)
    dump(output/'audit_status.json',{'status':'EXECUTED','labels':'UNVERIFIED','metadata':'OpenCV container estimates; validate VFR and scene cuts manually','videos':len(rows)})
    return rows

def read_manifest(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))

def validate_split(rows):
    groups={}; errors=[]
    for r in rows:
        if not r['source_match'] or r['split']=='unassigned': errors.append(f"{r['clip_id']}: unresolved source match/split")
        if r['source_match']: groups.setdefault(r['source_match'],set()).add(r['split'])
    errors += [f'Match leakage: {m}: {sorted(s)}' for m,s in groups.items() if len(s)>1]
    return errors

def cache_segments(config):
    import cv2
    root=Path(config['dataset_root']); out=Path(config['cache_root'])
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use a new immutable cache path; do not overwrite a shared cache')
    rows={r['clip_id']:r for r in read_manifest(config['manifest'])}
    selected=config.get('segments',[])
    if not selected: raise ValueError('Select explicit 20–60 second segments after audit; no automatic whole-dataset processing')
    result=[]; frame_limit=config.get('max_total_frames',3000)
    for segment in selected:
        row=rows[segment['clip_id']]
        if not segment.get('scene_boundaries_reviewed'):raise ValueError('Review scene cuts and set scene_boundaries_reviewed=true')
        if not 0 < segment['duration_s'] <= 60: raise ValueError('Stage 1 requires positive segments <=60 seconds')
        boundaries=sorted(set(segment.get('scene_cut_frames',[])))
        video=(root/row['relative_path']).resolve()
        if not video.is_relative_to(root.resolve()):raise ValueError('Video path escapes dataset root')
        if sha256(video)!=row['sha256']:raise ValueError('Video changed since manifest audit')
        cap=cv2.VideoCapture(str(video)); fps=float(cap.get(cv2.CAP_PROP_FPS))
        if fps<=0:raise ValueError('Unknown FPS; provide a validated timestamp decode workflow')
        start=round(segment['start_s']*fps); stop=round((segment['start_s']+segment['duration_s'])*fps)
        if stop>int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):raise ValueError('Segment extends beyond reported video duration')
        # Sequential decode rather than inaccurate random seeking into inter-frame codecs.
        for index in range(stop):
            ok,frame=cap.read()
            if not ok: raise RuntimeError(f'Decode stopped at {video}:{index}')
            if index<start:continue
            if len(result)>=frame_limit:raise ValueError('Frame budget exceeded; reduce explicit selections')
            scene=sum(index>=cut for cut in boundaries)
            path=out/segment['clip_id']/f'{index:08d}.png'; path.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(path),frame):raise RuntimeError('Frame write failed')
            result.append({'clip_id':segment['clip_id'],'scene_id':f'{segment["clip_id"]}:scene{scene:03d}','frame_index':index,'timestamp_s':index/fps,'timestamp_source':'frame_index/fps_ESTIMATE','path':str(path),'sha256':sha256(path),'fps':fps,'width':frame.shape[1],'height':frame.shape[0]})
        cap.release()
    dump(out/'frames.json',result); dump(out/'cache_config.json',config)
    return result

def load_frames(path):
    frames=json.loads(Path(path).read_text())
    keys=[(r['clip_id'],r['frame_index']) for r in frames]
    if len(keys)!=len(set(keys)):raise ValueError('Overlapping/duplicate cached frames')
    for r in frames:
        if sha256(r['path'])!=r['sha256']:raise ValueError('Cached frame integrity failed')
    return sorted(frames,key=lambda r:(r['clip_id'],r['frame_index']))
