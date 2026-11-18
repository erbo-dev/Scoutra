"""Launch genuine upstream PRTReID/BPBReID in a user-provisioned isolated Python."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from advanced import PART_COMMITS, crop_manifest, write_embeddings
from common import Blocked, json_read, verified
from environment import NO_NET_ENV


GTA_COMMIT = "e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca"
GTA_WEIGHT_HASH = "8d5b2fd8763db34c2aad69810466adf413f0426d9f8119d322227e0e639c5fbd"


def validate_checkout(entry, commit, name):
    executable = Path(entry["python"])
    repo = Path(entry["repo"]).resolve()
    if not executable.is_absolute() or not executable.is_file() or not (repo / ".git").exists():
        raise Blocked("Legacy Python and checkout must already exist in the isolated environment.")
    if os.path.abspath(executable) == os.path.abspath(sys.executable):
        raise Blocked("Use a separate Python environment for legacy Research Re-ID.")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, timeout=20).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                                   text=True, timeout=20).strip()
    if head != commit or dirty:
        raise ValueError(f"Expected clean {name} checkout at {commit}; found {head}")
    return str(executable), repo


def legacy_environment(device):
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    index = int(device.split(":")[1])
    visible = original.split(",")[index].strip() if original else str(index)
    return {**os.environ, **NO_NET_ENV, "CUDA_VISIBLE_DEVICES": visible}


def part_profile(method, configuration, device, crops, output, batch=16):
    package, commit, outputs = PART_COMMITS[method]
    entry = configuration.get(package)
    if not entry:
        raise Blocked(f"Configure LEGACY_REID[{package!r}] with the exact source, compatible interpreter, config, checkpoint and HRNet assets.")
    required = {"python", "repo", "checkpoint", "checkpoint_sha256", "config", "config_sha256", "backbone",
                "backbone_sha256", "num_classes", "rights_reviewed", "provenance"}
    if not required <= set(entry) or entry["rights_reviewed"] is not True or not entry["provenance"]:
        raise Blocked(f"Legacy {package} requires a completed source/weight review and fields {sorted(required)}.")
    executable, repo = validate_checkout(entry, commit, package)
    paths = {key: str(verified(entry[key], entry[f"{key}_sha256"])) for key in ("checkpoint", "config", "backbone")}
    config = {**entry, **paths, "package": package, "method": method, "upstream_output_count": outputs, "upstream": commit,
              "repo": str(repo), "logical_device": device, "crops": crops, "output": str(output), "batch": batch}
    return executable, config, legacy_environment(device)


def parts(spec):
    import importlib
    import numpy as np
    import torch
    from yacs.config import CfgNode

    if not torch.cuda.is_available():
        raise RuntimeError("Legacy environment has no functional CUDA device")
    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    sys.path.insert(0, spec["repo"])
    namespace = spec["package"]
    package = importlib.import_module(namespace)
    if not Path(spec["repo"]) in Path(package.__file__).resolve().parents:
        raise RuntimeError("Legacy import resolved outside the verified upstream checkout")
    extractor = importlib.import_module(f"{namespace}.tools.feature_extractor").FeatureExtractor
    constants_module = importlib.import_module(f"{namespace}.utils.constants")
    with open(spec["config"], encoding="utf-8") as f:
        cfg = CfgNode.load_cfg(f)
    cfg.defrost()
    cfg.model.bpbreid.hrnet_pretrained_path = spec["backbone"]
    kwargs = {"cfg": cfg, "image_size": (cfg.data.height, cfg.data.width), "device": device,
              "pixel_mean": cfg.data.norm_mean, "pixel_std": cfg.data.norm_std,
              "num_classes": spec["num_classes"], "verbose": False}
    model = extractor(**kwargs)
    # The original convenience loader silently skips incompatible keys. Require exact tensor coverage.
    try:
        checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        checkpoint = torch.load(spec["checkpoint"], map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.model.load_state_dict(state, strict=True)
    ckpt_metadata = {key: str(checkpoint[key]) for key in ("epoch", "arch") if key in checkpoint}
    torch.cuda.synchronize(device)
    load_s = time.perf_counter() - started
    manifest = crop_manifest(spec["crops"])
    batches = {}
    elapsed = 0.0
    for index in range(0, len(manifest), spec["batch"]):
        names = [x["path"] for x in manifest[index:index + spec["batch"]]]
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = model(names)
        torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - start
        if len(result) != spec["upstream_output_count"]:
            raise ValueError("Legacy method returned unexpected feature tuple")
        emb, visibility, masks = result[0], result[1], result[-1]
        if spec["method"].endswith("_parts"):
            key = constants_module.PARTS
            features = {"parts": emb[key], "visibility": visibility[key], "part_masks": masks[key]}
        else:
            features = {"global": emb[constants_module.GLOBAL]}
        for key, data in features.items():
            batches.setdefault(key, []).append(data.detach().cpu().numpy())
    asset = {k: spec[k] for k in ("upstream", "config_sha256", "checkpoint_sha256", "backbone_sha256", "provenance")}
    write_embeddings(spec["output"], manifest, {k: np.concatenate(v) for k, v in batches.items()},
                     {**asset, "checkpoint_metadata": ckpt_metadata},
                     {"input_size": kwargs["image_size"], "pixel_mean": kwargs["pixel_mean"], "pixel_std": kwargs["pixel_std"],
                      "torch": torch.__version__, "python": sys.version, "device": spec["logical_device"],
                      "model_load_s": load_s, "inference_s": elapsed, "crop_batch": spec["batch"],
                      "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                      "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}, source_manifest=spec["crops"])


def gta_profile(configuration, device, tracks, video, output):
    required = {"python", "repo", "rights_reviewed", "provenance"}
    if not required <= set(configuration) or configuration["rights_reviewed"] is not True or not configuration["provenance"]:
        raise Blocked("Set GTA_CONFIG for a verified isolated GTA-Link checkout and sports checkpoint rights.")
    executable, repo = validate_checkout(configuration, GTA_COMMIT, "GTA-Link")
    checkpoint = verified(repo / "reid_checkpoints" / "sports_model.pth.tar-60", GTA_WEIGHT_HASH)
    parameters = {"eps": 0.7, "max_k": 3, "min_samples": 10, "min_len": 100,
                  "spatial_factor": 1.0, "merge_dist_thres": 0.4}
    configured = configuration.get("parameters", {})
    if not set(configured) <= parameters.keys():
        raise ValueError("Unrecognized GTA-Link parameter")
    parameters.update(configured)
    return executable, {
        "method": "gta_link", "repo": str(repo), "checkpoint": str(checkpoint), "tracks": str(tracks),
        "video": video, "output": str(output), "parameters": parameters, "device": device,
        "provenance": configuration["provenance"],
    }, legacy_environment(device)


def verify_sports_encoder(encoder, path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing = [k for k, v in encoder.model.state_dict().items()
               if not k.startswith("classifier.") and (k not in state or not torch.equal(v.detach().cpu(), state[k]))]
    if missing:
        raise ValueError(f"Sports OSNet weights were not loaded on all inference layers: {missing[:10]}")


def gta_link(spec):
    import collections
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms as T

    from common import digest, json_write
    from evaluate import bbox, checked_tracks
    from jobs import capture

    if not torch.cuda.is_available():
        raise RuntimeError("GTA-Link feature extraction requires a functional CUDA device in the isolated environment")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    time_start = time.perf_counter()
    sys.path.insert(0, spec["repo"])
    from Tracklet import Tracklet
    from refine_tracklets import get_distance_matrix, get_spatial_constraints, merge_tracklets, split_tracklets
    from torchreid.utils import FeatureExtractor

    encoder = FeatureExtractor(model_name="osnet_x1_0", model_path=spec["checkpoint"], device="cuda:0")
    verify_sports_encoder(encoder, spec["checkpoint"])
    transform = T.Compose([T.Resize([256, 128]), T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    video = spec["video"]
    tracks = checked_tracks(spec["tracks"], video)
    inputs = {}
    original = collections.defaultdict(list)
    features_s = 0.0
    with capture(video["path"]) as source:
        for row in tracks:
            ok, frame = source.read()
            if not ok:
                raise RuntimeError("Original clip is shorter than its frozen track cache")
            if not row["tracks"]:
                continue
            pil = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
            crops = [transform(pil.crop(tuple(x[coord] for coord in ("x1", "y1", "x2", "y2")))) for x in row["tracks"]]
            torch.cuda.synchronize()
            started = time.perf_counter()
            vectors = [encoder(torch.stack(crops[i:i + 32])).detach().cpu().numpy()
                       for i in range(0, len(crops), 32)]
            torch.cuda.synchronize()
            features_s += time.perf_counter() - started
            for item, vector in zip(row["tracks"], np.concatenate(vectors)):
                tid, frame_id = item["track_id"], row["frame"]
                xywh = bbox(item)
                track = inputs.setdefault(tid, Tracklet(track_id=tid))
                track.append_det(frame_id, item["score"], xywh)
                norm = np.linalg.norm(vector)
                if not np.isfinite(norm) or norm <= 0:
                    raise ValueError("Invalid original GTA-Link appearance feature")
                track.append_feat(vector / norm)
                original[(frame_id, tuple(xywh))].append(tid)
    if not inputs:
        raise Blocked("GTA-Link requires actual track observations")
    settings = spec["parameters"]
    spatial_x, spatial_y = get_spatial_constraints(inputs, settings["spatial_factor"])
    split = split_tracklets(inputs, eps=settings["eps"], min_samples=settings["min_samples"],
                            max_k=settings["max_k"], len_thres=settings["min_len"])
    split_count = len(split)
    merged = merge_tracklets(split, {}, get_distance_matrix(split), seq_name=video["clip"],
                             max_x_range=spatial_x, max_y_range=spatial_y,
                             merge_dist_thres=settings["merge_dist_thres"])
    mapped = collections.defaultdict(dict)
    for identity, track in sorted(merged.items()):
        for frame_id, xywh in zip(track.times, track.bboxes):
            parents = original[(frame_id, tuple(xywh))]
            if len(parents) != 1:
                raise ValueError("Cannot recover a unique input observation for an upstream GTA-Link box")
            tid = parents[0]
            if tid in mapped[frame_id]:
                raise ValueError("Upstream GTA-Link has a duplicate output for one input observation")
            mapped[frame_id][tid] = str(identity)
    outputs = sum(map(len, mapped.values()))
    json_write(spec["output"], {"method": "gta_link", "clip_sha256": video["sha256"],
                                "tracking_sha256": digest(spec["tracks"]), "frame_groups": mapped,
                                "source_commit": GTA_COMMIT, "checkpoint_sha256": GTA_WEIGHT_HASH,
                                "provenance": spec["provenance"], "parameters": settings, "offline": True,
                                "input_tracks": len(inputs), "split_tracks": split_count, "linked_tracks": len(merged),
                                "retained_observations": outputs, "dropped_observations": sum(map(len, original.values())) - outputs,
                                "feature_seconds": features_s, "wall_s": time.perf_counter() - time_start,
                                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0), "device": spec["device"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    config = json_read(parser.parse_args().spec)
    (gta_link if config["method"] == "gta_link" else parts)(config)
