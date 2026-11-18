"""Export only dense complete reviewed sequences for official TrackEval (no proxy HOTA)."""
import csv,json
from pathlib import Path
from scoutra_lab.common import read_table,dump
from scoutra_lab.dataset import load_frames

def export(frames,pred,gt,output,tracker='method'):
    root=Path(output);sequences={};pred_by={}
    for f in frames:sequences.setdefault((f['clip_id'],f['scene_id']),[]).append(f)
    gt_by={(f['clip_id'],f['frame_index']):f for f in gt['frames']}
    names=[]
    for (clip,scene),seq in sequences.items():
        seq=sorted(seq,key=lambda f:f['frame_index']);indices=[f['frame_index'] for f in seq]
        if indices!=list(range(indices[0],indices[-1]+1)):raise ValueError('Official MOT export requires dense frames; split gaps into distinct sequences')
        for f in seq:
            ann=gt_by.get((clip,f['frame_index']))
            if not ann or not all(ann.get(k) is True for k in ('reviewed','boxes_exhaustive','identities_exhaustive')):raise ValueError('Official metrics blocked: incomplete dense annotation')
            if any(o.get('ignore') for o in ann['objects']):raise ValueError('Ignore handling requires an explicit official dataset policy')
            if any(o['class_id']==1 and not o.get('player_id') for o in ann['objects']):raise ValueError('Unknown physical ID in dense GT')
        name=f'{clip}__{scene.split(":")[-1]}';names.append(name);frame_map={f['frame_index']:i+1 for i,f in enumerate(seq)}
        gt_ids=sorted({str(o['player_id']) for f in seq for o in gt_by[clip,f['frame_index']]['objects'] if o['class_id']==1});gm={v:i+1 for i,v in enumerate(gt_ids)}
        selected=[p for p in pred if p['clip_id']==clip and p['scene_id']==scene and p['class_id']==1]
        predicted_ids=sorted({str(p.get('player_id',p.get('tracklet_id'))) for p in selected if p.get('player_id',p.get('tracklet_id')) is not None});pm={v:i+1 for i,v in enumerate(predicted_ids)}
        gt_dir=root/'gt'/name/'gt';gt_dir.mkdir(parents=True,exist_ok=True);tr_dir=root/'trackers'/tracker/'data';tr_dir.mkdir(parents=True,exist_ok=True)
        with open(gt_dir/'gt.txt','w',newline='') as stream:
            w=csv.writer(stream)
            for f in seq:
                for o in gt_by[clip,f['frame_index']]['objects']:
                    if o['class_id']==1:w.writerow([frame_map[f['frame_index']],gm[str(o['player_id'])],o['x1'],o['y1'],o['x2']-o['x1'],o['y2']-o['y1'],1,1,1])
        with open(tr_dir/(name+'.txt'),'w',newline='') as stream:
            w=csv.writer(stream)
            for p in selected:
                identity=p.get('player_id',p.get('tracklet_id'))
                if identity is not None:w.writerow([frame_map[p['frame_index']],pm[str(identity)],p['x1'],p['y1'],p['x2']-p['x1'],p['y2']-p['y1'],p.get('confidence') if p.get('confidence') is not None else 1,-1,-1,-1])
        (gt_dir.parent/'seqinfo.ini').write_text(f'[Sequence]\nname={name}\nimDir=img1\nframeRate={seq[0]["fps"]}\nseqLength={len(seq)}\nimWidth={seq[0]["width"]}\nimHeight={seq[0]["height"]}\nimExt=.png\n')
    (root/'seqmap.txt').write_text('name\n'+'\n'.join(names)+'\n')
    dump(root/'export_metadata.json',{'status':'EXPORTED_NOT_EVALUATED','sequences':names,'DO_PREPROC':False,'warning':'Confidence=1 in MOT file is a format-required acceptance field when confidence absent, not calibrated identity confidence; canonical person includes officials. Match scene-reset scope across all methods.'})
    return root
