import contextlib, datetime, hashlib, importlib.metadata, json, os, platform, random, subprocess, time
from pathlib import Path

SCHEMA_VERSION = 'scoutra-benchmark/1.0'
TABLES = {
 'detections': {'clip_id':'string','scene_id':'string','frame_index':'int64','timestamp_s':'float64','detection_id':'string','class_id':'int64','x1':'float64','y1':'float64','x2':'float64','y2':'float64','confidence':'float64'},
 'tracks': {'clip_id':'string','scene_id':'string','frame_index':'int64','timestamp_s':'float64','tracklet_id':'string','detection_id':'string','class_id':'int64','x1':'float64','y1':'float64','x2':'float64','y2':'float64','confidence':'float64'},
 'identities': {'clip_id':'string','scene_id':'string','frame_index':'int64','tracklet_id':'string','player_id':'string','confidence':'float64','state':'string','evidence':'string'},
 'embeddings': {'crop_id':'string','head':'string','part_index':'int64','visibility':'float64','feature_path':'string','array_index':'int64'},
 'calibration': {'clip_id':'string','scene_id':'string','frame_index':'int64','homography':'string','pitch_length_m':'float64','pitch_width_m':'float64','confidence':'float64','coordinate_system':'string'},
 'events': {'clip_id':'string','scene_id':'string','start_s':'float64','end_s':'float64','event_type':'string','actor_id':'string','confidence':'float64','evidence':'string'}
}

def dump(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n'); tmp.replace(path)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def asset_fingerprint(path):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(f'Missing local asset: {path}')
    if path.is_file(): return sha256(path)
    entries=[(str(p.relative_to(path)),sha256(p)) for p in sorted(path.rglob('*')) if p.is_file()]
    return hashlib.sha256(json.dumps(entries,separators=(',',':')).encode()).hexdigest()

def verify_asset(spec):
    if not spec.get('license_reviewed',False): raise ValueError('Asset license has not been reviewed')
    expected=spec.get('sha256','')
    if len(expected)!=64: raise ValueError('Supply an expected SHA256 (directory: sorted relative-path/digest list fingerprint)')
    actual=asset_fingerprint(spec['path'])
    if actual!=expected: raise ValueError(f'Asset digest mismatch: {actual}')
    return actual

def load_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())

def seed_all(seed):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    if 'torch' in __import__('sys').modules:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True)

def gpu_guard(config):
    # A directory alone is not proof; require Kaggle runtime's environment marker too.
    if not config.get('execute',False): raise RuntimeError('NOT EXECUTED: execute=false')
    if not Path('/kaggle/working').is_dir() or not any(k.startswith('KAGGLE_') for k in os.environ):
        raise RuntimeError('NOT EXECUTED: Kaggle runtime is not confirmed')
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('NOT EXECUTED: CUDA unavailable')
    device=int(config.get('gpu_index',0))
    if device>=torch.cuda.device_count(): raise ValueError('Requested GPU does not exist')
    torch.cuda.set_device(device)
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    return torch,device

def write_table(path, rows, kind):
    import pyarrow as pa, pyarrow.parquet as pq
    spec=TABLES[kind]; schema=pa.schema([(k,pa.type_for_alias(t)) for k,t in spec.items()])
    clean=[]
    for row in rows:
        missing=set(spec)-set(row)
        if missing: raise ValueError(f'{kind}: missing fields {missing}')
        clean.append({k:row[k] for k in spec})
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.parquet'); pq.write_table(pa.Table.from_pylist(clean,schema=schema),tmp); tmp.replace(path)

def read_table(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()

def environment():
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    result={'python':platform.python_version(),'platform':platform.platform(),'packages':dict(sorted(packages.items()))}
    try:
        import torch
        result.update(torch_cuda=torch.version.cuda,devices=[{'name':torch.cuda.get_device_name(i),'total_vram_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())])
    except ImportError: result['devices']=[]
    return result

class Run:
    def __init__(self,config,kind):
        self.config=config; self.kind=kind; self.path=Path(config['output_root'])/config['experiment_id']
        self.path.mkdir(parents=True,exist_ok=False)
        (self.path/'visualizations').mkdir()
        import yaml
        (self.path/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=True))
        self.metrics={'status':'NOT EXECUTED','metrics':None,'reason':'Run not started'}
        self.runtime={'status':'NOT EXECUTED','elapsed_s':None,'peak_allocated_vram_bytes':None,'peak_reserved_vram_bytes':None,'timing_scope':None}
        self.errors=[]; self.provenance={'schema_version':SCHEMA_VERSION,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'environment':environment()}
        dump(self.path/'metadata.json',self.provenance)
        write_table(self.path/'predictions.parquet',[],kind)
        self.finish()
    def finish(self):
        dump(self.path/'metrics.json',self.metrics); dump(self.path/'runtime.json',self.runtime); dump(self.path/'errors.json',self.errors)
        (self.path/'report.md').write_text(f"# {self.config['experiment_id']}\n\nInference: {self.runtime['status']}\n\nEvaluation: {self.metrics['status']}\n\n{self.metrics.get('reason','')}\n\nNo absent metric is a measured zero. See metadata.json and errors.json.\n")
    @contextlib.contextmanager
    def measured(self,torch=None,device=0):
        if torch:
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter()
        try:
            yield
            if torch: torch.cuda.synchronize(device)
            self.runtime.update(status='EXECUTED',elapsed_s=time.perf_counter()-start,timing_scope='model calls, decode/read, preprocessing, synchronization and prediction serialization; loading excluded; no warmup exclusion')
            if torch: self.runtime.update(peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device))
        except Exception as exc:
            self.runtime.update(status='FAILED',elapsed_s=time.perf_counter()-start)
            self.errors.append({'type':type(exc).__name__,'message':str(exc)})
            raise
        finally: self.finish()

def verify_repo(spec):
    path=Path(spec['path'])
    actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if actual!=spec['commit']: raise ValueError(f'Upstream commit mismatch: {actual}')
    dirty=subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True)
    if dirty.strip():raise ValueError('Tracked upstream files modified; record a reviewed patch as a separate experimental condition')
    return actual
