"""Adapter for the author's Deep-EIoU association on externally frozen detections."""
import argparse
import time
from pathlib import Path
from types import SimpleNamespace

from backends import HUMANS
from common import BOX, Blocked, atomic_path, canonical_box, digest, freeze_predictions, json_read, json_write, write_line
from jobs import capture, overlay
from legacy import GTA_WEIGHT_HASH, legacy_environment, validate_checkout
from common import verified

COMMIT = "2fc7d270eec8290fda075b976b5e06683c788e1e"


def deep_profile(configuration, device, cache, video, output, thresholds, overlays):
    required = {"python", "repo", "sports_checkpoint", "rights_reviewed", "provenance"}
    if not required <= set(configuration) or configuration["rights_reviewed"] is not True or not configuration["provenance"]:
        raise Blocked("Deep-EIoU needs a rights-reviewed clean checkout, legacy Python, and the verified GTA-packaged sports OSNet weights.")
    python, repo = validate_checkout(configuration, COMMIT, "Deep-EIoU")
    checkpoint = verified(configuration["sports_checkpoint"], GTA_WEIGHT_HASH)
    settings = dict(track_high_thresh=thresholds["high"], track_low_thresh=thresholds["low"],
                    new_track_thresh=thresholds["new"], track_buffer=thresholds["buffer"], match_thresh=thresholds["match"],
                    with_reid=True, proximity_thresh=configuration.get("proximity_thresh", .5),
                    appearance_thresh=configuration.get("appearance_thresh", .25))
    return python, dict(source=str(repo / "Deep-EIoU"), checkpoint=str(checkpoint), cache=str(cache), video=video,
                        output=str(output), device=device, thresholds=settings, overlays=overlays,
                        provenance=configuration["provenance"]), legacy_environment(device)


def run(spec, *, device="cuda:0"):
    import math
    import sys

    import cv2
    import numpy as np
    import torch

    gpu = torch.device(device).type == "cuda"
    if gpu and not torch.cuda.is_available():
        raise RuntimeError("Deep-EIoU appearance encoder requires CUDA in its isolated profile")
    if np.lib.NumpyVersion(np.__version__) >= "1.24.0":
        raise Blocked(f"The original tracker uses np.float; install NumPy <1.24 in the isolated profile (found {np.__version__}).")
    cv2.setNumThreads(1)
    if gpu:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    sys.path.insert(0, spec["source"])
    from tracker.Deep_EIoU import Deep_EIoU
    from legacy import sports_encoder

    video = spec["video"]
    extractor = sports_encoder(spec["checkpoint"], device)
    tracker = Deep_EIoU(SimpleNamespace(**spec["thresholds"]), frame_rate=max(1, round(video["nominal_fps"])))
    out = Path(spec["output"])
    tracks = out / "tracks" / f"{video['clip']}.jsonl"
    mot = tracks.with_suffix(".txt")
    annotated_path = out / "annotated_videos" / f"{video['clip']}.mp4"
    durations, count, boxes = 0., 0, 0
    with capture(video["path"]) as source, atomic_path(tracks) as temporary, temporary.open("w") as rows, \
            atomic_path(mot) as mot_tmp, mot_tmp.open("w") as mot_rows, overlay(annotated_path, video, spec["overlays"]) as annotated:
        for frame in freeze_predictions(spec["cache"], video):
            ok, image = source.read()
            if not ok:
                raise ValueError("Frozen detection cache outlasts the clip")
            people = [row for row in frame["detections"] if row["class"] in HUMANS]
            detections = np.asarray([[p[k] for k in (*BOX, "score")] for p in people], dtype=np.float32).reshape(-1, 5)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            crops = [rgb[max(0, math.floor(p["y1"])):math.ceil(p["y2"]),
                           max(0, math.floor(p["x1"])):math.ceil(p["x2"])] for p in people]
            if gpu:
                torch.cuda.synchronize(device)
            ts = time.perf_counter()
            if crops:
                embs = np.concatenate([extractor(crops[i:i + 32]).detach().cpu().numpy() for i in range(0, len(crops), 32)])
            else:
                embs = np.empty((0, 512), dtype=np.float32)
            if embs.shape != (len(detections), 512) or not np.isfinite(embs).all():
                raise ValueError("Deep-EIoU expects one finite sports OSNet vector per frozen detection")
            predicted = tracker.update(detections, embs.copy())
            if gpu:
                torch.cuda.synchronize(device)
            durations += time.perf_counter() - ts
            actual = []
            for current in predicted:
                if current.frame_id != frame["frame"]:
                    continue
                x, y, w, h = map(float, current.last_tlwh)
                vals = [x, y, x + w, y + h]
                matches = np.flatnonzero(np.max(np.abs(detections[:, :4] - vals), axis=1) < 1e-3)
                matches = [i for i in matches if abs(float(detections[i, 4]) - current.score) < 1e-7]
                if len(matches) != 1:
                    raise ValueError("Deep-EIoU observation cannot be mapped uniquely to a frozen detection")
                original = people[matches[0]]
                actual.append(dict(canonical_box(vals, (video["width"], video["height"])), track_id=int(current.track_id),
                                   score=float(current.score), detection_id=original["detection_id"], **{"class": original["class"]}))
                mot_rows.write(f"{frame['frame']},{current.track_id},{x+1:.3f},{y+1:.3f},{w:.3f},{h:.3f},"
                               f"{current.score:.7f},-1,-1,-1\n")
            count = frame["frame"]
            boxes += len(actual)
            write_line(rows, {"frame": count, "tracks": actual})
            annotated.draw(image, actual, True)
    json_write(tracks.with_suffix(".json"), {
        "clip": video["clip"], "clip_sha256": video["sha256"], "track_sha256": digest(tracks), "frames": count,
        "detection_sha256": digest(spec["cache"]), "method": "deep_eiou",
        "condition": "Original Deep-EIoU association with frozen detector and GTA-packaged sports OSNet x1.0",
        "source_commit": COMMIT, "checkpoint_sha256": GTA_WEIGHT_HASH, "provenance": spec["provenance"],
        "parameters": spec["thresholds"], "inference_s": durations, "wall_s": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if gpu else 0, "device": spec["device"] if gpu else str(device), "boxes": boxes,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    run(json_read(parser.parse_args().spec))
