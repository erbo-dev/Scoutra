"""Opt-in independent research methods and methods with recorded prerequisites."""
import csv
import importlib.util
import itertools
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

from common import Blocked, atomic_path, digest, fetch, freeze_predictions, json_read, json_write, verified
from evaluate import checked_tracks

AGENTCOACH_COMMIT = "eef7b0c5d757844fbbb1eeafa6a8050a863a6623"
AGENTCOACH_HASH = "2a9bd53fb03db7557991d1fa30bb387ee3f9fe086f52f13113f6235fb024f148"
PART_COMMITS = {
    "prtreid_global": ("prtreid", "30617a75967e84d5d516959c4b84cbeea6f56493", 8),
    "prtreid_parts": ("prtreid", "30617a75967e84d5d516959c4b84cbeea6f56493", 8),
    "bpbreid_parts": ("torchreid", "a2dc4304284784d8b2c061764512f0698aa38c21", 6),
}
ADVANCED = {
    "detection": {
        "specialized_soccer_ball": "No licensed, source-pinned ball checkpoint with verified class map was selected. Generic COCO sports ball is available in the core detectors.",
    },
    "tracking": {
        "deep_eiou": None,
        "mcbyteplusplus": "The pinned upstream tracker requires EdgeTAM masks, the sports Re-ID checkpoint and a validated frozen-detection bridge in its separate Torch/CUDA environment.",
    },
    "reid": dict.fromkeys(("prepare_crops", "osnet", *PART_COMMITS), None),
    "identity": {
        "no_link": None, "agentcoach_reid_only": None, "cosine_tracklet_link": None,
        "gta_link": None,
        "gtatrack": "Separate GTATrack source, checkpoints and exact detector-feature contract have not been validated.",
        "mcbyte_identity_bank": "Requires real McByte++ association and EdgeTAM mask streams, bank-state export and the upstream bank-on/off switch.",
        "cross_multiplication_reid": "No verified standalone original checkpoint and inference/preprocessing contract is wired.",
    },
    "jersey": {
        "parseq": "The PARSeq STR runtime and jersey-specific weights/tokenizer have not been validated in an isolated profile.",
        "koshkina": "Needs compatible legibility, pose, torso-crop and PARSeq weights plus the upstream tracklet workflow across its environments.",
        "temporal_ocr_vote": None,
    },
    "calibration": {
        "opencv_homography": None, "tvcalib": "Needs pinned pitch-segment network/checkpoint, pitch model and calibration runner.",
        "pnlcalib": "Needs pinned line/point detectors, checkpoint hashes and verified correspondence format.",
    },
    "ball": {
        "temporal_ball_baseline": None, "ballradar": "Needs verified upstream expected ball/player trajectories and checkpoint input conversion.",
        "pathcrf": "Exact source/checkpoint and candidate-association contract are unverified.",
    },
    "events": {
        "t_deed": "Needs checkpoint-specific frame rate, backbone/features and SoccerNet event alignment.",
        "opensportslib": "No committed real event backend and usable checkpoint resolved for this name.",
        "footpass": "Requires verified checkpoint, pass taxonomy and continuous time/actor labels.",
    },
    "vlm": {
        "qwen3_vl": "Needs standalone model revision, video sampling, T4-compatible precision/kernels and measured memory fit.",
        "qwen3_5": "No verified 27B–35B T4 fit, quantization or multi-device execution plan.",
        "soccerchat": "Requires exact base VLM, adapters, video input contract and checkpoint/license verification.",
        "football_qwen": "No selected auditable football-specific model revision with a verified T4-compatible runtime.",
    },
}


def crop_manifest(path):
    path = Path(path)
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            image = verified(path.parent / item["path"], item["sha256"])
            records.append({**item, "path": str(image)})
    ids = [row["crop_id"] for row in records]
    if not ids or any(not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Crop manifest needs nonempty unique crop_id and verified crop bytes")
    return records


def osnet_backend(asset, device):
    import torch
    from boxmot.reid.core.auto_backend import ReidAutoBackend

    path = Path(asset["path"])
    backend = ReidAutoBackend(weights=path, device=torch.device(device), half=False).model
    verify_osnet_backbone(backend, path, device)
    return backend


def verify_osnet_backbone(backend, path, device):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    weights = {k.removeprefix("module."): v for k, v in checkpoint.get("state_dict", checkpoint).items()}
    loaded = backend.model.state_dict()
    missing = [k for k, v in loaded.items() if not k.startswith("classifier.") and
               (k not in weights or not torch.equal(v.detach().cpu(), weights[k]))]
    if missing:
        raise ValueError(f"OSNet backbone tensors do not match the supplied checkpoint: {missing[:10]}")
    if next(backend.model.parameters()).device != torch.device(device):
        raise RuntimeError("OSNet model is not on the assigned device")


def extract_osnet(task, device):
    import cv2
    import numpy as np
    import torch

    crops = crop_manifest(task["crops"])
    start = time.perf_counter()
    model = osnet_backend(task["asset"], device)
    sync = lambda: torch.cuda.synchronize(device) if device.startswith("cuda") else None
    sync()
    load_s = time.perf_counter() - start
    all_vectors = []
    inference = 0.0
    batch = task["batch"]
    for offset in range(0, len(crops), batch):
        blocks = []
        for crop in crops[offset:offset + batch]:
            image = cv2.imread(crop["path"])
            if image is None:
                raise ValueError(f"Cannot open crop {crop['path']}")
            box = np.asarray([[0, 0, image.shape[1], image.shape[0]]])
            blocks.append(model.get_crops(box, image))
        sync()
        start = time.perf_counter()
        with torch.inference_mode():
            vector = model.inference_postprocess(model.forward(model.inference_preprocess(torch.cat(blocks))))
        sync()
        inference += time.perf_counter() - start
        all_vectors.append(np.asarray(vector))
    array = np.concatenate(all_vectors)
    write_embeddings(task["output"], crops, {"global": array}, task["asset"], {"input_size": model.input_shape,
                     "color": "RGB", "resize": "OpenCV INTER_LINEAR via BoxMOT", "batch": batch},
                     source_manifest=task["crops"])
    return {"crops_processed": len(crops), "model_load_s": load_s, "inference_s": inference,
            "prediction_path": str(task["output"])}


def write_embeddings(path, crops, features, asset, preprocessing, source_manifest):
    import numpy as np

    path = Path(path)
    count = len(crops)
    if any(x.shape[0] != count or not np.isfinite(x).all() for x in features.values()):
        raise ValueError("Nonfinite embeddings or a misaligned crop batch")
    if "parts" in features:
        if features["parts"].ndim != 3 or features["visibility"].shape != features["parts"].shape[:2]:
            raise ValueError("Expected full parts [N,P,D] and visibility [N,P]")
        if features["part_masks"].shape[:2] != features["parts"].shape[:2] or features["part_masks"].ndim != 4:
            raise ValueError("Expected full spatial part_masks [N,P,H,W]")
    if "global" in features and (features["global"].ndim != 2 or not np.linalg.norm(features["global"], axis=1).all()):
        raise ValueError("Expected nonzero global vectors [N,D]")
    with atomic_path(path) as temporary:
        np.savez_compressed(temporary, crop_id=np.asarray([r["crop_id"] for r in crops]), **features)
    json_write(path.with_suffix(".json"), {"asset": asset, "preprocessing": preprocessing,
                                           "crop_manifest_sha256": digest(source_manifest),
                                           "embedding_sha256": digest(path), "crops": crops,
                                           "shapes": {k: list(v.shape) for k, v in features.items()}})


def vectors(path):
    import numpy as np

    path = Path(path)
    meta = json_read(path.with_suffix(".json"))
    if meta["embedding_sha256"] != digest(path):
        raise ValueError(f"Embedding SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    if [x["crop_id"] for x in meta["crops"]] != data["crop_id"].tolist():
        raise ValueError("Embedding crop provenance mismatch")
    return data, meta


def pairwise_distance(data, query, gallery, visibility_min):
    import numpy as np

    if "parts" not in data:
        array = data["global"]
        left, right = array[query], array[gallery]
        left = left / np.linalg.norm(left, axis=-1, keepdims=True)
        right = right / np.linalg.norm(right, axis=-1, keepdims=True)
        return 1 - np.clip(left @ right.T, -1, 1)

    # Named analysis condition, separate from the upstream training/gallery evaluator.
    # Average cosine distance over jointly visible parts. No common visible part is unscorable.
    parts, visibility = data["parts"], data["visibility"]
    score = np.zeros((len(query), len(gallery)))
    count = np.zeros(score.shape, dtype=int)
    for index in range(parts.shape[1]):
        array = parts[:, index]
        norm = np.linalg.norm(array, axis=-1)
        normalized = array / np.maximum(norm[:, None], 1e-12)
        visible = (visibility[:, index] >= visibility_min) & (norm > 0)
        valid = visible[query, None] & visible[gallery][None, :]
        score += (1 - np.clip(normalized[query] @ normalized[gallery].T, -1, 1)) * valid
        count += valid
    return np.divide(score, count, out=np.full(score.shape, np.nan), where=count > 0)


def compare_embeddings(path, predictions, reviewed_labels, max_crops, visibility_min, threshold):
    import numpy as np

    data, meta = vectors(predictions)
    count = len(meta["crops"])
    rng = np.random.default_rng(1729)
    sample = np.sort(rng.choice(count, min(count, max_crops), replace=False))
    matrix = pairwise_distance(data, sample, sample, visibility_min)
    upper = matrix[np.triu_indices(len(matrix), 1)]
    valid = upper[np.isfinite(upper)]
    output = {
        "embedding_sha256": digest(predictions), "sample_crops": [meta["crops"][i]["crop_id"] for i in sample],
        "distance": "joint_visible_parts_mean_cosine" if "parts" in data else "global_cosine",
        "visibility_min": visibility_min, "sampled_pairs": int(len(upper)), "scorable_pairs": int(len(valid)),
        "histogram": np.histogram(np.clip(valid, 0, 2), bins=20, range=(0, 2))[0].tolist(),
        "histogram_edges": np.linspace(0, 2, 21).tolist(),
    }
    if reviewed_labels:
        output.update(gt_sha256=digest(reviewed_labels), metrics=retrieval(data, meta["crops"], reviewed_labels,
                                                                         visibility_min, threshold))
    json_write(path, output)
    return output


def retrieval(data, crops, labels, visibility_min, threshold):
    import numpy as np

    labels = json_read(labels)
    if labels.get("human_reviewed") is not True:
        raise ValueError("Re-ID labels must be human reviewed")
    index = {row["crop_id"]: i for i, row in enumerate(crops)}
    roles = {"query": [], "gallery": []}
    seen = set()
    for row in labels["crops"]:
        key = row["crop_id"]
        if key not in index or key in seen or crops[index[key]]["sha256"] != row["sha256"]:
            raise ValueError("Re-ID labels must identify unique supplied crops by ID and SHA-256")
        if any(not row.get(k) for k in ("match_id", "player_id", "team_id", "tracklet_id")):
            raise ValueError("Re-ID labels require match, physical player, team and original tracklet IDs")
        seen.add(key)
        roles[row["split"]].append(row)
    if not all(roles.values()):
        raise ValueError("Both query and gallery are required")

    gallery = roles["gallery"]
    gallery_indices = [index[x["crop_id"]] for x in gallery]
    all_metrics, team_metrics = [], []
    false = negatives = 0
    for query in roles["query"]:
        qi = index[query["crop_id"]]
        distance = pairwise_distance(data, [qi], gallery_indices, visibility_min)[0]
        eligible = np.asarray([p["match_id"] == query["match_id"] and p["tracklet_id"] != query["tracklet_id"]
                               and p["sha256"] != query["sha256"] for p in gallery]) & np.isfinite(distance)
        same_id = np.asarray([p["player_id"] == query["player_id"] for p in gallery])
        same_team = np.asarray([p["team_id"] == query["team_id"] for p in gallery])
        if threshold is not None:
            negative = eligible & ~same_id
            negatives += int(negative.sum())
            false += int((negative & (distance <= threshold)).sum())
        ordering = np.argsort(distance, kind="stable")
        for scope, target in ((eligible, all_metrics), (eligible & same_team, team_metrics)):
            relevant = same_id[ordering[scope[ordering]]]
            if relevant.any():
                ranks = np.flatnonzero(relevant) + 1
                target.append((float(relevant[0]), float((np.arange(1, len(ranks) + 1) / ranks).mean())))

    def metric(rows, column):
        return float(np.mean(rows, axis=0)[column]) if rows else None

    return dict(rank1=metric(all_metrics, 0), mAP=metric(all_metrics, 1),
                same_team_rank1=metric(team_metrics, 0), same_team_mAP=metric(team_metrics, 1),
                false_match_rate=false / negatives if negatives else None, valid_queries=len(all_metrics),
                valid_same_team_queries=len(team_metrics), negative_pairs=negatives, distance_threshold=threshold)


def identity_output(method, tracks, video, output, features=None, module=None, threshold=None):
    import numpy as np

    rows = checked_tracks(tracks, video)
    track_hash = digest(tracks)
    ids = sorted({entry["track_id"] for row in rows for entry in row["tracks"]})
    if not ids:
        raise Blocked(f"No local tracks for {video['clip']}")
    groups = {i: str(i) for i in ids}
    evidence = []
    if method != "no_link":
        if not features:
            raise Blocked("Identity linking requires global embeddings for the same frozen tracklets")
        data, meta = vectors(features)
        if "global" not in data:
            raise Blocked("This comparison requires global embeddings")
        lookup = defaultdict(list)
        for vector, crop in zip(data["global"], meta["crops"]):
            if crop.get("clip") == video["clip"] and crop.get("tracking_sha256") == track_hash:
                lookup[int(crop["track_id"])].append(vector)
        if not lookup:
            raise Blocked("No global crops are tied to this exact clip and track-prediction SHA-256")
        by_frame = [{x["track_id"] for x in r["tracks"]} for r in rows]
        cooccurring = {pair for group in by_frame for pair in itertools.combinations(sorted(group), 2)}
        if method == "agentcoach_reid_only":
            metadata = [module.TrackMeta(track_id=i, team="unknown", role="player", jersey_number=None, jersey_confidence=0,
                                         frame_count=sum(i in frame for frame in by_frame),
                                         reid_centroid=module.compute_centroid(lookup[i]) if lookup[i] else None)
                        for i in ids]
            mapped, clusters = module.consolidate(metadata, same_person_threshold=threshold,
                                                  orphan_absorb_threshold=max(threshold, 0.92), cooccur_pairs=cooccurring)
            groups = {i: str(mapped[i]) for i in ids}
            evidence = [row.to_dict() for row in clusters]
        else:
            centroid = {i: np.mean(v, axis=0) for i, v in lookup.items() if v}
            score = []
            for a, b in itertools.combinations(sorted(centroid), 2):
                x, y = centroid[a], centroid[b]
                norm = np.linalg.norm(x) * np.linalg.norm(y)
                if norm > 0:
                    score.append((float(np.clip(np.dot(x, y) / norm, -1, 1)), a, b))
            for similarity, a, b in sorted(score, reverse=True):
                left = [i for i in ids if groups[i] == groups[a]]
                right = [i for i in ids if groups[i] == groups[b]]
                decision = groups[a] != groups[b] and similarity >= threshold and not any(
                    tuple(sorted((i, j))) in cooccurring for i in left for j in right)
                evidence.append(dict(track_a=a, track_b=b, cosine=similarity, linked=decision))
                if decision:
                    label = groups[a]
                    groups.update(dict.fromkeys(right, label))
    json_write(output, {"method": method, "clip_sha256": video["sha256"], "tracking_sha256": track_hash,
                        "source_commit": AGENTCOACH_COMMIT if module else None, "threshold": threshold,
                        "offline": True, "groups": groups, "evidence": evidence})
    return len(ids)


def load_agentcoach(cache, online):
    base = f"https://raw.githubusercontent.com/outstandingcandy/AgentCoach/{AGENTCOACH_COMMIT}"
    path = fetch(f"{base}/goalinsight/track_consolidation/_consolidator.py",
                 cache / "agentcoach_consolidator.py", AGENTCOACH_HASH, online)
    spec = importlib.util.spec_from_file_location("scoutra_agentcoach_upstream", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def homography(video, spec, output):
    import cv2
    import numpy as np

    labels = json_read(spec)
    if labels.get("human_reviewed") is not True:
        raise ValueError("Pitch landmarks require human review")
    item = labels["clips"].get(video["clip"])
    if not item:
        raise Blocked(f"No pitch landmarks for {video['clip']}")
    if item["clip_sha256"] != video["sha256"]:
        raise ValueError("Pitch landmark video hash mismatch")
    results = []
    for row in item["frames"]:
        image, world = [np.asarray(row[field], dtype=np.float64) for field in ("pixel_xy", "pitch_xy_m")]
        count = len(image)
        if image.shape != world.shape or image.shape != (count, 2) or count < 4 or not 1 <= row["frame"] <= video["frames"]:
            raise ValueError("Supply four or more pixel/meter correspondences for each valid frame")
        mat, mask = cv2.findHomography(image, world, cv2.RANSAC, 0.5)
        if mat is None or np.linalg.matrix_rank(mat) != 3 or int(mask.sum()) < 4:
            raise ValueError("Degenerate landmarks or insufficient inliers")
        meters = cv2.perspectiveTransform(image.reshape(-1, 1, 2), mat)[:, 0]
        error = np.linalg.norm(meters - world, axis=1)
        results.append(dict(frame=row["frame"], H_image_to_pitch=mat.tolist(), inliers=mask[:, 0].tolist(),
                            fit_rmse_m=float(np.sqrt(np.mean(error[mask[:, 0] == 1] ** 2)))))
    if not results:
        raise Blocked("No frame-specific pitch correspondences")
    json_write(output, {"clip_sha256": video["sha256"], "source_sha256": digest(spec), "frames": results})
    return len(results)


def temporal_ocr(input_path, output):
    rows = json_read(input_path)
    by_track = defaultdict(lambda: defaultdict(float))
    for row in rows:
        value = row["digits"]
        confidence = float(row["confidence"])
        if value.isascii() and value.isdigit() and 1 <= len(value) <= 2 and 0 <= confidence <= 1:
            by_track[(row["clip"], int(row["track_id"]))][value] += confidence
    if not by_track:
        raise Blocked("No valid jersey OCR candidates to aggregate")
    result = []
    for (clip, track_id), scores in sorted(by_track.items()):
        ranked = sorted(scores, key=lambda k: (-scores[k], k))
        top = ranked[0]
        if scores[top] > 0:
            result.append({"clip": clip, "track_id": track_id, "digits": top,
                           "vote_share": scores[top] / sum(scores.values()), "ties": sum(x == scores[top] for x in scores.values()),
                           "votes": dict(scores)})
    if not result:
        raise Blocked("All candidate votes have zero weight")
    json_write(output, {"source_sha256": digest(input_path), "votes": result})
    return len(result)


def temporal_ball(cache, video, output, min_score, gate_fraction, max_gap):
    records = []
    previous = newer = None
    gate = gate_fraction * math.hypot(video["width"], video["height"])
    for row in freeze_predictions(cache, video):
        idx = row["frame"]
        all_balls = [d for d in row["detections"] if d["class"] == "ball" and d["score"] >= min_score]
        if newer and idx - newer[0] > max_gap:
            previous = newer = None

        def center(box):
            return ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)

        prediction = newer[1] if newer else None
        if previous:
            elapsed = (idx - newer[0]) / (newer[0] - previous[0])
            prediction = tuple(a + elapsed * (a - b) for a, b in zip(newer[1], previous[1]))
        candidates = [b for b in all_balls if prediction is None or math.dist(center(b), prediction) <= gate]
        chosen = (min(candidates, key=lambda x: math.dist(center(x), prediction) / gate + (1 - x["score"]))
                  if candidates and prediction else max(candidates, key=lambda x: x["score"], default=None))
        if chosen:
            previous, newer = newer, (idx, center(chosen))
        records.append({"frame": idx, "state": "observed" if chosen else "missing", "detection": chosen})
    json_write(output, {"clip_sha256": video["sha256"], "detection_sha256": digest(cache),
                        "minimum_confidence": min_score, "gate_pixels": gate, "max_gap_frames": max_gap, "frames": records})
    return len(records)


def prepare_crops(cache_lookup, videos, output, stride, per_track):
    import cv2

    from common import table
    from jobs import capture

    records = []
    for video in videos:
        clip = video["clip"]
        tracks = cache_lookup(clip)
        tracking_hash = digest(tracks)
        seen, kept = defaultdict(int), defaultdict(int)
        with capture(video["path"]) as source:
            for row in checked_tracks(tracks, video):
                success, frame = source.read()
                if not success:
                    raise ValueError(f"Source ended before track cache for {clip}")
                for person in row["tracks"]:
                    track = person["track_id"]
                    seen[track] += 1
                    if (seen[track] - 1) % stride or kept[track] >= per_track:
                        continue
                    x1, y1 = (int(math.floor(person[k])) for k in ("x1", "y1"))
                    x2, y2 = (int(math.ceil(person[k])) for k in ("x2", "y2"))
                    if x2 - x1 < 8 or y2 - y1 < 16:
                        continue
                    name = f"{clip}_{row['frame']:06}_{track}"
                    relative = Path("images") / f"{name}.png"
                    with atomic_path(output.parent / relative) as temporary:
                        if not cv2.imwrite(str(temporary), frame[y1:y2, x1:x2]):
                            raise RuntimeError(f"Crop serialization failed: {name}")
                    kept[track] += 1
                    records.append(dict(crop_id=name, path=str(relative), sha256=digest(output.parent / relative),
                                        clip=clip, frame=row["frame"], track_id=track, tracking_sha256=tracking_hash,
                                        clip_sha256=video["sha256"], source_detection_id=person["detection_id"]))
    if not records:
        raise Blocked("No crops met the immutable selection rules (track observations, stride, minimum 8×16 pixels)")
    table(output, records, tuple(records[0]))
    return len(records)


def identity_overlay(tracks, mapping_path, video, path):
    from jobs import capture, overlay

    mapping = json_read(mapping_path)
    groups = mapping.get("groups")
    frame_groups = mapping.get("frame_groups", {})
    with capture(video["path"]) as source, overlay(path, video, True) as annotated:
        for row in checked_tracks(tracks, video):
            ok, image = source.read()
            if not ok:
                raise ValueError("Clip ended before the identity output")
            labels = groups if groups is not None else frame_groups.get(str(row["frame"]), {})
            boxes = []
            for track in row["tracks"]:
                group = labels.get(str(track["track_id"]))
                value = group if group is not None else f"unassigned/{track['track_id']}"
                boxes.append({**track, "track_id": value})
            annotated.draw(image, boxes, True)
