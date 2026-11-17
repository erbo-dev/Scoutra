from pathlib import Path
from scoutra_lab.common import verify_asset

class Tracker:
    def __init__(self,spec,fps,device):
        self.spec=spec; self.method=spec['method']
        if self.method=='bytetrack':
            import supervision as sv
            self.model=sv.ByteTrack(frame_rate=round(fps),**spec.get('parameters',{}))
        elif self.method in ('botsort_boxmot','strongsort_boxmot'):
            from boxmot import BotSort,StrongSort
            import torch
            verify_asset(spec['reid_checkpoint'])
            kwargs=dict(reid_weights=Path(spec['reid_checkpoint']['path']),device=torch.device(device),half=spec.get('half',False),**spec.get('parameters',{}))
            if self.method=='botsort_boxmot':kwargs['frame_rate']=round(fps)
            self.model=(BotSort if self.method=='botsort_boxmot' else StrongSort)(**kwargs)
        else:raise RuntimeError('Method requires an audited external bridge; McByte++ is not a ByteTrack alias')
    def update(self,detections,image):
        import numpy as np
        if self.method=='bytetrack':
            import supervision as sv
            det=sv.Detections(xyxy=np.asarray([[d[k] for k in ('x1','y1','x2','y2')] for d in detections],dtype=np.float32).reshape(-1,4),confidence=np.asarray([d['confidence'] for d in detections],dtype=np.float32),class_id=np.asarray([d['class_id'] for d in detections],dtype=int),data={'detection_id':np.asarray([d['detection_id'] for d in detections],dtype=str)})
            out=self.model.update_with_detections(det)
            return [{'local_id':int(tid),'detection_id':str(out.data['detection_id'][i]),'class_id':int(out.class_id[i]),'confidence':float(out.confidence[i]),**dict(zip(('x1','y1','x2','y2'),map(float,out.xyxy[i])))} for i,tid in enumerate(out.tracker_id)]
        x=np.asarray([[d[k] for k in ('x1','y1','x2','y2','confidence','class_id')] for d in detections],dtype=np.float32).reshape(-1,6)
        result=self.model.update(x,image)
        rows=[]
        for r in result:
            if len(r)!=8:raise RuntimeError('Unexpected BoxMOT output contract; expected Nx8')
            i=int(r[7])
            if not 0<=i<len(detections):raise RuntimeError('Invalid detection provenance from tracker')
            rows.append({'local_id':int(r[4]),'confidence':float(r[5]),'class_id':int(r[6]),'detection_id':detections[i]['detection_id'],**dict(zip(('x1','y1','x2','y2'),map(float,r[:4])))})
        return rows
