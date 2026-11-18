from pathlib import Path
import hashlib
from collections import defaultdict

def render(frames,rows,output,every=30,limit=100,ground_truth=None):
    from PIL import Image,ImageDraw
    out=Path(output);out.mkdir(parents=True,exist_ok=True);by=defaultdict(list)
    for r in rows:by[r['clip_id'],r['frame_index']].append(r)
    truth={(f['clip_id'],f['frame_index']):f for f in (ground_truth or {}).get('frames',[])}
    for f in frames[::every][:limit]:
        image=Image.open(f['path']).convert('RGB');draw=ImageDraw.Draw(image)
        for r in by[f['clip_id'],f['frame_index']]:
            label=str(r.get('player_id',r.get('tracklet_id',r.get('detection_id','?'))));h=hashlib.sha256(label.encode()).digest();color=tuple(80+x%175 for x in h[:3])
            draw.rectangle([r[k] for k in ('x1','y1','x2','y2')],outline=color,width=3);draw.text((r['x1'],r['y1']),label,fill=color)
        for g in truth.get((f['clip_id'],f['frame_index']),{}).get('objects',[]):
            draw.rectangle([g[k] for k in ('x1','y1','x2','y2')],outline='white',width=1);draw.text((g['x1'],max(0,g['y1']-12)),'GT:'+str(g.get('player_id','?')),fill='white')
        draw.text((10,10),f"{f['clip_id']} frame={f['frame_index']} time~{f['timestamp_s']:.3f}s",fill='yellow')
        image.save(out/f"{f['clip_id']}_{f['frame_index']:08d}.jpg")
