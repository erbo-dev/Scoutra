"""Isolate packages per experiment and retain Kaggle's matched Torch/CUDA wheels."""
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import hashlib
import venv
from pathlib import Path

from common import Blocked, json_write

# Explicit scientific and upstream API pins. Freeze captures the full resolved graph.
BASE = ("numpy==2.2.6", "opencv-python==4.11.0.86", "pillow==11.3.0")
PACKAGE_PROFILES = {
    "rf": BASE + ("rfdetr==1.12.1", "transformers==5.12.1", "supervision==0.29.0"),
    "hf": BASE + ("transformers==4.57.6", "huggingface-hub==0.36.2", "safetensors==0.6.2"),
    "tracker": BASE + ("ultralytics==8.3.235", "lap==0.5.12"),
    "boxmot": BASE + ("boxmot==16.0.10", "lapx==0.5.12", "gdown==5.2.0"),
    "eval": ("numpy==2.3.5", "opencv-python==4.11.0.86", "pycocotools==2.0.10", "trackeval==1.3.0"),
}
IMPORT_SMOKES = {
    "rf": "from rfdetr import RFDETRSmall, RFDETRMedium, RFDETR; import cv2",
    "hf": "from transformers import DFineForObjectDetection, RTDetrForObjectDetection, RTDetrV2ForObjectDetection, RTDetrImageProcessor; import cv2",
    "tracker": "import lap; from ultralytics.trackers.byte_tracker import BYTETracker; from ultralytics.trackers.bot_sort import BOTSORT",
    "boxmot": "from boxmot import StrongSort; from boxmot.reid.core.auto_backend import ReidAutoBackend; import cv2",
    "eval": "from pycocotools.cocoeval import COCOeval; import trackeval",
}
NO_NET_ENV = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "YOLO_AUTOINSTALL": "false",
    "YOLO_OFFLINE": "true", "HF_HUB_DISABLE_TELEMETRY": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "MPLBACKEND": "Agg",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def command(argv, log, timeout=1200, environment=None):
    with Path(log).open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {argv!r}\n")
        handle.flush()
        result = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout, env=environment)
        handle.write(f"\nexit={result.returncode}\n")
    return result.returncode


def inventory(path):
    """Run in a disposable subprocess so the notebook process has no CUDA context."""
    probe = """
import json, sys, traceback
result = {'python':sys.version, 'cuda_available':False, 'gpu_count':0, 'gpus':[]}
try:
 import torch, torchvision
 result.update(torch=torch.__version__, torchvision=torchvision.__version__, cuda_version=torch.version.cuda,
               cuda_available=torch.cuda.is_available(), gpu_count=torch.cuda.device_count())
 for i in range(result['gpu_count']):
  entry={'index':i}
  try:
   with torch.cuda.device(i):
    p=torch.cuda.get_device_properties(i)
    free,total=torch.cuda.mem_get_info(i)
    entry.update(name=p.name, uuid=str(getattr(p,'uuid','')) or None,
                 total_bytes=p.total_memory, free_bytes_before_probe=free,
                 capability=[p.major,p.minor])
    x=torch.ones(16, device=f'cuda:{i}')
    assert (x+x).sum().item()==32
    torch.cuda.synchronize(i)
    entry['functional']=True
  except Exception:
   entry.update(functional=False,error=traceback.format_exc())
  result['gpus'].append(entry)
except Exception:
 result['error']=traceback.format_exc()
print(json.dumps(result))
"""
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=90)
    if result.returncode or not result.stdout.strip():
        values = {"cuda_available": False, "gpu_count": 0, "gpus": [], "error": result.stderr}
    else:
        values = json.loads(result.stdout.strip().splitlines()[-1])
        if result.stderr:
            values["probe_stderr"] = result.stderr
    values.update(platform=platform.platform(), cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"))
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free",
                                "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=10)
        values["nvidia_smi"] = {"exit": query.returncode, "stdout": query.stdout, "stderr": query.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        values["nvidia_smi"] = {"error": str(error)}
    json_write(path, values)
    return values


def device_plan(info, dual, allow_single):
    good = [f"cuda:{p['index']}" for p in info["gpus"] if p["functional"]]
    if dual and len(good) >= 2:
        return good[:2]
    if (allow_single or not dual) and good:
        return good[:1]
    raise Blocked(f"Need two functioning CUDA GPUs for dual mode. Found {good}. Select GPU T4 x2, or explicitly set ALLOW_SINGLE_GPU=True for one GPU.")


def _share_torch(site):
    """Expose only native runtime libraries, avoiding unrelated packages in Kaggle."""
    names = {"torch", "torchvision", "triton", "pytorch-triton", "pytorch-triton-rocm", "cuda-bindings", "cuda-pathfinder"}
    packages = {"torch", "torchgen", "functorch", "torchvision", "torchvision.libs", "triton", "nvidia", "cuda"}
    linked = set()
    for dist in metadata.distributions():
        name = dist.metadata["Name"].lower().replace("_", "-")
        if name not in names and not name.startswith("nvidia-"):
            continue
        for item in dist.files or []:
            top = item.parts[0]
            if top not in packages and not top.endswith(".dist-info"):
                continue
            source = Path(dist.locate_file(top))
            if source.is_dir() and top not in linked:
                target = site / top
                if not target.exists():
                    target.symlink_to(source.resolve(), target_is_directory=True)
                linked.add(top)
    if not {"torch", "torchvision"} <= linked:
        raise Blocked("Cannot share Kaggle's Torch/TorchVision runtime. Check the native CUDA probe and standard wheel installation.")
    return {name: metadata.version(name) for name in ("torch", "torchvision")}


def prepare_profile(profile, workspace, run, online):
    if not (3, 11) <= sys.version_info[:2] <= (3, 12):
        raise Blocked(f"The pinned core profiles target Python 3.11–3.12. Found {sys.version.split()[0]}.")
    versions = {} if profile == "eval" else {name: metadata.version(name) for name in ("torch", "torchvision")}
    profile_digest = hashlib.sha256(repr((PACKAGE_PROFILES[profile], versions, sys.version)).encode()).hexdigest()[:12]
    root = workspace / "profiles" / f"{profile}-{profile_digest}"
    python = root / "bin/python"
    ready = root / ".validated"
    log = run / "environment" / "installation_log.txt"
    if not ready.is_file():
        if not online:
            raise Blocked(f"Profile {profile!r} has not been installed. Enable AUTO_ACQUIRE with Kaggle Internet for the first run.")
        root.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True).create(root)
        if versions:
            site = root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
            native = _share_torch(site)
            constraints = root / "native-constraints.txt"
            constraints.write_text("".join(f"{name}=={version}\n" for name, version in native.items()))
            options = ["-c", str(constraints), "torch", "torchvision"]
        else:
            options = []
        if command([str(python), "-m", "pip", "install", "--disable-pip-version-check", *options, *PACKAGE_PROFILES[profile]], log):
            raise RuntimeError(f"Dependency installation failed for {profile}; see {log.name}")
    yolo_dir = workspace / "ultralytics"
    if profile == "tracker":
        yolo_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **NO_NET_ENV, "YOLO_CONFIG_DIR": str(yolo_dir)}
    if command([str(python), "-m", "pip", "check"], log, environment=env):
        raise RuntimeError(f"pip check failed for {profile}; see {log.name}")
    result = command([str(python), "-c", IMPORT_SMOKES[profile]], log, timeout=120, environment=env)
    if result:
        raise RuntimeError(f"Import contract failed for {profile}; see {log.name}")
    ready.touch()
    freeze = subprocess.check_output([str(python), "-m", "pip", "list", "--format=freeze"], text=True, env=env, timeout=60)
    with (run / "environment" / "dependencies.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{profile} | {profile_digest}]\n{freeze}")
    return str(python), env
