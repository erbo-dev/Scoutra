from pathlib import Path
from scoutra_lab.common import verify_asset

class OSNet:
    def __init__(self,spec,device):
        verify_asset(spec['checkpoint'])
        if 'osnet' not in Path(spec['checkpoint']['path']).name.lower():raise ValueError('BoxMOT infers architecture from filename; retain OSNet architecture filename')
        from boxmot.reid.core.reid_handler import ReID
        import torch
        self.model=ReID(weights=Path(spec['checkpoint']['path']),device=torch.device(device),half=False).model
    def extract(self,paths):
        import cv2,numpy as np
        features=[]
        for path in paths:
            image=cv2.imread(str(path))
            if image is None:raise ValueError(f'Crop unreadable: {path}')
            f=self.model.get_features(np.asarray([[0,0,image.shape[1],image.shape[0]]],dtype=np.float32),image)
            if hasattr(f,'detach'):f=f.detach().cpu().numpy()
            features.append(np.asarray(f)[0])
        return np.asarray(features,dtype=np.float32)

# Part-aware methods intentionally preserve upstream feature and visibility contracts.
# See reports/MODEL_INVENTORY.md and external_bridge.py; no global-vector substitute.
