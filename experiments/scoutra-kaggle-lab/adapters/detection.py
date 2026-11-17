from pathlib import Path
from scoutra_lab.common import verify_asset

class Detector:
    def __init__(self,spec,device):
        self.spec=spec; verify_asset(spec['checkpoint']); name=spec['method']
        if not spec.get('class_map'):raise ValueError('Explicit raw-model-class -> canonical class_map is required')
        self.class_map={int(k):int(v) for k,v in spec['class_map'].items()}
        if name in ('rfdetr_small','rfdetr_medium','scoutra_rfdetr'):
            from rfdetr import RFDETRSmall,RFDETRMedium
            if name=='scoutra_rfdetr' and spec.get('architecture') not in ('small','medium'):raise ValueError('Specify the actual Scoutra checkpoint architecture')
            cls=RFDETRSmall if name=='rfdetr_small' or spec.get('architecture')=='small' else RFDETRMedium
            self.model=cls(pretrain_weights=spec['checkpoint']['path'],device='cuda',**spec.get('model_kwargs',{}))
        elif name=='rtdetr_hf':
            from transformers import AutoImageProcessor,RTDetrForObjectDetection
            self.processor=AutoImageProcessor.from_pretrained(spec['checkpoint']['path'],local_files_only=True)
            self.model=RTDetrForObjectDetection.from_pretrained(spec['checkpoint']['path'],local_files_only=True).to(device).eval()
        elif name=='dfine':
            import sys,torch
            from scoutra_lab.common import verify_repo
            verify_repo(spec['upstream'])
            sys.path.insert(0,str(Path(spec['upstream']['path']).resolve()))
            from src.core import YAMLConfig
            cfg=YAMLConfig(spec['upstream_config'],resume=spec['checkpoint']['path'])
            if 'HGNetv2' in cfg.yaml_cfg:cfg.yaml_cfg['HGNetv2']['pretrained']=False
            if not spec.get('trust_checkpoint_pickle',False):raise ValueError('Review trusted checkpoint serialization and enable trust_checkpoint_pickle explicitly')
            ckpt=torch.load(spec['checkpoint']['path'],map_location='cpu',weights_only=False)
            state=ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model']
            cfg.model.load_state_dict(state); self.model=cfg.model.deploy().to(device).eval(); self.post=cfg.postprocessor.deploy()
        else:raise ValueError(f'Unsupported detector: {name}')
        self.device=device
    def predict(self,image):
        import torch,numpy as np
        name=self.spec['method']; threshold=self.spec.get('threshold',0.05)
        with torch.inference_mode():
            if name.startswith('rfdetr') or name=='scoutra_rfdetr':
                pred=self.model.predict(image,threshold=threshold)
                raw=zip(pred.xyxy,pred.confidence,pred.class_id)
            elif name=='rtdetr_hf':
                inputs=self.processor(images=image,return_tensors='pt').to(self.device)
                output=self.model(**inputs)
                result=self.processor.post_process_object_detection(output,target_sizes=torch.tensor([[image.height,image.width]],device=self.device),threshold=threshold)[0]
                raw=zip(result['boxes'].cpu().numpy(),result['scores'].cpu().numpy(),result['labels'].cpu().numpy())
            else:
                from torchvision import transforms as T
                size=self.spec.get('resolution',640)
                preprocess=T.Compose([T.Resize((size,size)),T.ToTensor()])
                x=preprocess(image).unsqueeze(0).to(self.device)
                labels,boxes,scores=self.post(self.model(x),torch.tensor([[image.width,image.height]],device=self.device))
                raw=zip(boxes[0].cpu().numpy(),scores[0].cpu().numpy(),labels[0].cpu().numpy())
            result=[]
            for box,score,cls in raw:
                cls=int(cls); score=float(score)
                if cls not in self.class_map or score<threshold:continue
                x1,y1,x2,y2=map(float,box)
                x1=max(0,min(image.width,x1)); x2=max(0,min(image.width,x2)); y1=max(0,min(image.height,y1)); y2=max(0,min(image.height,y2))
                if x2>x1 and y2>y1:result.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'confidence':score,'class_id':self.class_map[cls]})
            return result
