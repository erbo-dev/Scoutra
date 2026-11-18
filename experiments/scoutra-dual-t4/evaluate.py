"""Evaluation receives reviewed annotations only after predictions are frozen."""
import argparse
from collections import defaultdict
from pathlib import Path

from common import BOX, Blocked, digest, freeze_predictions, json_lines, json_read, json_write

DETECTION_METRICS = ("precision50", "recall50", "mAP50", "mAP50_95", "ball_recall50", "small_player_recall50")
TRACKING_METRICS = ("HOTA", "IDF1", "ID_switches", "fragmentation")
REID_METRICS = ("rank1", "mAP", "false_match_rate", "same_team_rank1", "same_team_mAP")
IDENTITY_METRICS = ("reentry_accuracy", "false_merges", "false_splits", "identity_precision",
                    "identity_coverage", "recovery_delay_frames")


def bbox(row):
    x, y, right, bottom = [float(row[key]) for key in BOX]
    return [x, y, right - x, bottom - y]


def parse_reviewed(spec, input_by_clip, dense=False):
    annotations = json_read(spec)
    if annotations.get("schema") != "scoutra-reviewed/1" or annotations.get("human_reviewed") is not True:
        raise ValueError("GT requires schema scoutra-reviewed/1 and human_reviewed: true")
    scoped = {}
    for clip, item in annotations["clips"].items():
        if clip not in input_by_clip:
            continue
        video = input_by_clip[clip]
        if item["clip_sha256"] != video["sha256"]:
            raise ValueError(f"GT clip SHA-256 mismatch: {clip}")
        if dense and item.get("continuous_scene") is not True:
            raise ValueError(f"Dense identity metrics require one reviewed continuous scene per clip: {clip}")
        selected = {}
        for frame in item["frames"]:
            index = frame["frame"]
            if not isinstance(index, int) or index in selected or not 1 <= index <= video["frames"]:
                raise ValueError(f"Invalid or duplicated annotation frame {clip}/{index}")
            ids = set()
            for truth in frame["objects"]:
                x, y, w, h = bbox(truth)
                if not (x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= video["width"] and y + h <= video["height"]):
                    raise ValueError(f"Invalid annotated box at {clip}/{index}")
                if truth["class"] not in {"person", "player", "goalkeeper", "referee", "ball"}:
                    raise ValueError(f"Invalid class: {truth['class']!r}")
                if dense:
                    identity = truth.get("player_id")
                    if truth["class"] == "ball" or not identity or identity in ids:
                        raise ValueError(f"MOT labels must be people with unique physical IDs within frame {clip}/{index}")
                    ids.add(identity)
            selected[index] = frame["objects"]
        if dense and set(selected) != set(range(1, video["frames"] + 1)):
            raise ValueError(f"Tracking GT must exhaustively list each frame, including empty frames: {clip}")
        if selected:
            scoped[clip] = selected
    if not scoped:
        raise Blocked("No reviewed GT frames overlap with this benchmark workload")
    return scoped


def detection_metrics(predictions, reviewed, inputs, operating_point, min_score):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, annotations, detections = [], [], []
    for clip, frames in reviewed.items():
        if clip not in predictions:
            continue
        video = inputs[clip]
        by_frame = {}
        for pred in freeze_predictions(predictions[clip], video):
            if pred["frame"] in frames:
                by_frame[pred["frame"]] = pred["detections"]
        if set(by_frame) != set(frames):
            raise ValueError(f"Benchmark predictions do not cover all annotated frames for {clip}")
        for frame, objects in frames.items():
            image_id = len(images) + 1
            images.append({"id": image_id, "width": video["width"], "height": video["height"]})
            for obj in objects:
                box = bbox(obj)
                annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                    "category_id": 2 if obj["class"] == "ball" else 1, "bbox": box,
                                    "area": box[2] * box[3], "iscrowd": 0, "role": obj["class"]})
            for pred in by_frame[frame]:
                detections.append({"image_id": image_id, "category_id": 2 if pred["class"] == "ball" else 1,
                                   "bbox": bbox(pred), "score": pred["score"]})
    if not annotations:
        raise Blocked("No positive reviewed GT boxes overlap with complete predictions")
    if operating_point < min_score:
        raise ValueError("Evaluation score threshold must be >= the detector cache score floor")
    categories = [{"id": 1, "name": "person"}, {"id": 2, "name": "ball"}]
    ground = COCO()
    ground.dataset = {"images": images, "annotations": annotations, "categories": categories, "info": {}}
    ground.createIndex()
    if detections:
        result = ground.loadRes(detections)
    else:
        result = COCO()
        result.dataset = {"images": images, "annotations": [], "categories": categories}
        result.createIndex()
    evaluation = COCOeval(ground, result, "bbox")
    evaluation.evaluate()
    evaluation.accumulate()
    confusion = defaultdict(lambda: [0, 0, 0])
    player_gt = {item["id"] for item in annotations if item["role"] == "player"}
    small_hits = small_eligible = 0
    all_area, small_area = map(tuple, evaluation.params.areaRng[:2])

    # COCO matches are confidence-descending. Lower scores cannot affect these TP/FP decisions.
    for entry in evaluation.evalImgs:
        if entry is None:
            continue
        mask = np.asarray(entry["dtScores"]) >= operating_point
        valid = mask & ~np.asarray(entry["dtIgnore"][0], dtype=bool)
        true = valid & (entry["dtMatches"][0] > 0)
        valid_gt = int(np.sum(~np.asarray(entry["gtIgnore"], dtype=bool)))
        area = tuple(entry["aRng"])
        counts = confusion[(entry["category_id"], area)]
        if entry["category_id"] == 1 and area == small_area:
            small_hits += sum(match in player_gt for match in entry["dtMatches"][0][true])
            small_eligible += sum(gid in player_gt and not ignore for gid, ignore in zip(entry["gtIds"], entry["gtIgnore"]))
        counts[0] += int(true.sum())
        counts[1] += int(valid.sum())
        counts[2] += valid_gt
    all_counts = np.sum([confusion[(c, all_area)] for c in (1, 2)], axis=0)

    def ratio(a, b):
        return float(a / b) if b else None

    def average(array):
        return float(array[array >= 0].mean()) if (array >= 0).any() else None

    ball = confusion[(2, all_area)]
    precision = evaluation.eval["precision"]
    return dict(precision50=ratio(all_counts[0], all_counts[1]), recall50=ratio(all_counts[0], all_counts[2]),
                mAP50=average(precision[0, :, :, 0, -1]), mAP50_95=average(precision[:, :, :, 0, -1]),
                ball_recall50=ratio(ball[0], ball[2]), small_player_recall50=ratio(small_hits, small_eligible))


def iou_matrix(left, right):
    import numpy as np

    a = np.array([[row[k] for k in BOX] for row in left], dtype=float).reshape(-1, 4)
    b = np.array([[row[k] for k in BOX] for row in right], dtype=float).reshape(-1, 4)
    overlap = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0)
    intersection = np.prod(overlap, axis=2)
    union = np.prod(a[:, 2:] - a[:, :2], axis=1)[:, None] + np.prod(b[:, 2:] - b[:, :2], axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def checked_tracks(path, video):
    path = Path(path)
    sidecar = json_read(path.with_suffix(".json"))
    if sidecar["clip_sha256"] != video["sha256"] or sidecar["track_sha256"] != digest(path):
        raise ValueError("Track file fingerprint disagrees with its sidecar or source video")
    rows = list(json_lines(path))
    if len(rows) != sidecar["frames"] or any(i != r["frame"] for i, r in enumerate(rows, 1)):
        raise ValueError("Invalid dense tracking frame sequence")
    return rows


def tracking_metrics(predictions, reviewed, inputs):
    import numpy as np
    from trackeval.metrics import CLEAR, HOTA, Identity

    evaluators = [HOTA(), Identity({"THRESHOLD": .5, "PRINT_CONFIG": False}), CLEAR({"THRESHOLD": .5, "PRINT_CONFIG": False})]
    results = [dict() for _ in evaluators]
    for clip, frames in reviewed.items():
        if clip not in predictions:
            continue
        rows = checked_tracks(predictions[clip], inputs[clip])
        if len(rows) != len(frames):
            raise ValueError(f"MOT scoring requires predictions for every reviewed frame of {clip}")
        true = [frames[r["frame"]] for r in rows]
        predicted = [r["tracks"] for r in rows]

        def ids(frame_objects, field):
            mapping = {}
            arrays = [np.asarray([mapping.setdefault(obj[field], len(mapping)) for obj in row], dtype=int)
                      for row in frame_objects]
            return mapping, arrays

        gt_ids, truth = ids(true, "player_id")
        tracker_ids, estimates = ids(predicted, "track_id")
        data = dict(num_gt_ids=len(gt_ids), num_tracker_ids=len(tracker_ids), gt_ids=truth, tracker_ids=estimates,
                    num_gt_dets=sum(map(len, truth)), num_tracker_dets=sum(map(len, estimates)), num_timesteps=len(rows),
                    similarity_scores=[iou_matrix(g, p) for g, p in zip(true, predicted)])
        for metric, by_clip in zip(evaluators, results):
            by_clip[clip] = metric.eval_sequence(data)
    if not results[0]:
        raise Blocked("No complete tracking predictions overlap with the reviewed GT")
    hota, identity, clear = [metric.combine_sequences(by_clip) for metric, by_clip in zip(evaluators, results)]
    return dict(HOTA=float(hota["HOTA"].mean()), IDF1=float(identity["IDF1"]),
                ID_switches=int(clear["IDSW"]), fragmentation=int(clear["Frag"]))


def qualified(metrics):
    return {k: {"state": "EXECUTED" if value is not None else "NOT_EVALUATED", "value": value}
            for k, value in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    work = json_read(args.spec)
    dense = work["kind"] == "tracking"
    reviewed = parse_reviewed(work["annotations"], work["inputs"], dense)
    missing = set(reviewed) - set(work["predictions"])
    if missing:
        raise ValueError(f"One or more reviewed clips have no completed prediction: {sorted(missing)}")
    options = [] if dense else [work["operating_point"], work["cache_floor"]]
    handler = tracking_metrics if dense else detection_metrics
    result = handler(work["predictions"], reviewed, work["inputs"], *options)
    json_write(args.output, {"metrics": qualified(result), "gt_sha256": digest(work["annotations"]),
                            "reviewed_clips": sorted(set(reviewed) & set(work["predictions"]))})


def identity_metrics(tracks, mapping, video, annotations, reviewed):
    """Evaluate clip-local grouping using frozen predictions, reviewed anchors and re-entry episodes."""
    import numpy as np

    from scipy.optimize import linear_sum_assignment

    clip = video["clip"]
    rows = checked_tracks(tracks, video)
    if len(rows) != len(reviewed) or len(rows) != video["frames"]:
        raise ValueError(f"Identity scoring requires complete reviewed tracks: {clip}")
    group = mapping.get("groups")
    frame_groups = mapping.get("frame_groups", {})
    if mapping["tracking_sha256"] != digest(tracks) or mapping["clip_sha256"] != video["sha256"]:
        raise ValueError("Identity output does not match the tracking and clip hashes")

    alignment = {}
    groups_by_gt, gts_by_group = defaultdict(set), defaultdict(set)
    for row in rows:
        frame = row["frame"]
        gt = reviewed[frame]
        ts = row["tracks"]
        if gt and ts:
            matrix = iou_matrix(gt, ts)
            permitted = matrix >= 0.5
            # Maximize the number of threshold-valid edges before optimizing IoU.
            left, right = linear_sum_assignment(-(permitted * (matrix + 1 + min(matrix.shape))))
            for i, j in zip(left, right):
                if permitted[i, j]:
                    label = gt[i]["player_id"]
                    key = str(ts[j]["track_id"])
                    cluster = group.get(key) if group is not None else frame_groups.get(str(frame), {}).get(key)
                    if cluster is None:
                        continue
                    alignment[(frame, label)] = cluster
                    groups_by_gt[label].add(cluster)
                    gts_by_group[cluster].add(label)
    anchors = {}
    used_players = set()
    for row in annotations["anchors"]:
        frame, player = row["frame"], row["player_id"]
        if player in used_players or frame not in reviewed or player not in {x["player_id"] for x in reviewed[frame]}:
            raise ValueError("Each identity anchor must name a unique reviewed visible player")
        used_players.add(player)
        cluster = alignment.get((frame, player))
        if cluster is None:
            continue
        if cluster in anchors and anchors[cluster] != player:
            # Ambiguous anchors can arise after an incorrect upstream merge.
            anchors[cluster] = None
        else:
            anchors[cluster] = player

    visible = sum(len(rows) for rows in reviewed.values())
    assigned = sum(anchors.get(cluster) is not None for cluster in alignment.values())
    correct = sum(anchors.get(cluster) == player for (_, player), cluster in alignment.items())
    delays = []
    episodes = annotations.get("reentries", [])
    for item in episodes:
        player = item["player_id"]
        first, last = item["return_frame"], item["end_frame"]
        if player not in used_players or not 1 <= first <= last <= len(rows):
            raise ValueError("Each re-entry needs an earlier anchor and an in-bounds visible episode")
        origin = next(x["frame"] for x in annotations["anchors"] if x["player_id"] == player)
        before_return = {x["player_id"] for x in reviewed[first - 1]} if first > 1 else set()
        if origin >= first or player in before_return or player not in {x["player_id"] for x in reviewed[first]}:
            raise ValueError("Re-entry must follow a reviewed absent frame and begin with visible GT")
        for frame in range(first, last + 1):
            cluster = alignment.get((frame, player))
            if cluster is not None and anchors.get(cluster) == player:
                delays.append(frame - first)
                break
    return {
        "metrics": {
            "reentry_accuracy": len(delays) / len(episodes) if episodes else None,
            "recovery_delay_frames": float(np.median(delays)) if delays else None,
            "identity_precision": correct / assigned if assigned else None,
            "identity_coverage": assigned / visible if visible else None,
            "false_merges": sum(math_comb(len(ids)) for ids in gts_by_group.values()) if visible else None,
            "false_splits": sum(math_comb(len(ids)) for ids in groups_by_gt.values()) if visible else None,
        },
        "counts": {"anchors": len(annotations["anchors"]), "ambiguous_anchors": sum(x is None for x in anchors.values()),
                   "gt_person_frames": visible, "iou_matched_person_frames": len(alignment), "assigned_person_frames": assigned,
                   "correct_person_frames": correct, "return_episodes": len(episodes), "recoveries": len(delays)},
        "recovery_delays_frames": delays,
        "contaminated_groups": {str(group): sorted(map(str, ids)) for group, ids in gts_by_group.items() if len(ids) > 1},
        "split_identities": {str(player): sorted(ids) for player, ids in groups_by_gt.items() if len(ids) > 1},
    }


def math_comb(n):
    return n * (n - 1) // 2


if __name__ == "__main__":
    main()
