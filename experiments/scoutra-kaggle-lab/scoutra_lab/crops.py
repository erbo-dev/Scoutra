"""Shared, lossless native-resolution crops; no model-specific preprocessing here."""
from pathlib import Path
from .common import dump,sha256

def make_crops(frames,boxes,output,source_matches=None):
    from PIL import Image
    import math
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    images={(f['clip_id'],f['frame_index']):f for f in frames};rows=[]
    for i,b in enumerate(boxes):
        if b['class_id']!=1:continue
        f=images[b['clip_id'],b['frame_index']];image=Image.open(f['path']).convert('RGB')
        box=(max(0,math.floor(b['x1'])),max(0,math.floor(b['y1'])),min(image.width,math.ceil(b['x2'])),min(image.height,math.ceil(b['y2'])))
        if box[2]<=box[0] or box[3]<=box[1]:continue
        path=out/f'crop_{i:08d}.png';image.crop(box).save(path)
        rows.append({'crop_id':f'crop_{i:08d}','detection_id':b.get('detection_id'),'clip_id':b['clip_id'],'frame_index':b['frame_index'],'timestamp_s':f['timestamp_s'],'path':str(path),'sha256':sha256(path),'source_match':(source_matches or {}).get(b['clip_id'],''),'tracklet_id':b.get('tracklet_id',''),'player_id':b.get('player_id'),'team_id':b.get('team_id'),'viewpoint':b.get('viewpoint','unknown'),'occlusion':b.get('occlusion','unknown'),'reviewed':b.get('reviewed',False),'retrieval_role':'unassigned','crop_width':box[2]-box[0],'crop_height':box[3]-box[1]})
    dump(out/'crops.json',rows);return rows
