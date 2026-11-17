import json
from pathlib import Path
from .common import dump

def merge(paths,output):
    frames=[];seen=set()
    for path in paths:
        data=json.loads(Path(path).read_text())
        if data.get('schema_version')!='scoutra-ground-truth/1.0':raise ValueError('Unknown annotation schema')
        for f in data['frames']:
            key=(f['clip_id'],f['frame_index'])
            if key in seen:raise ValueError(f'Duplicate frame annotations: {key}; reconcile review revisions manually')
            seen.add(key);frames.append(f)
    result={'schema_version':'scoutra-ground-truth/1.0','coordinate_system':'native_image_xyxy_pixels','frames':sorted(frames,key=lambda f:(f['clip_id'],f['frame_index']))}
    dump(output,result);return result
