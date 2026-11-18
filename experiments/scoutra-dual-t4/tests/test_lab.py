"""CPU contract tests. All synthetic inputs stay in temporary directories."""
import contextlib
import copy
import csv
import io
import os
import shutil
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from advanced import compare_embeddings, homography, temporal_ocr, write_embeddings
from common import Blocked, NAMES, atomic_path, canonical_box, digest, export_zip, freeze_predictions, json_read, json_write, resolve_dataset
from environment import device_plan, inventory
from evaluate import detection_metrics, identity_metrics, parse_reviewed, tracking_metrics
from jobs import dataset_audit, probe_video
from lab import launch
from notebook_config import CONFIG
from schedule import execute


class FixtureRunner:
    def __init__(self, device):
        self.device = device

    def __call__(self, job):
        if job.get("crash"):
            os._exit(73)
        if job.get("invalid"):
            raise ValueError("fixture exception")
        time.sleep(job["delay"])
        return {"frames": 4, "inference_s": job["delay"]}


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = dict(clip="a", sha256="1" * 64, frames=3, width=100, height=50, nominal_fps=25)
        self.box = dict(x1=2.0, y1=3.0, x2=16.0, y2=23.0, score=.9, **{"class": "person"})

    def paths(self, directory):
        directory.mkdir(parents=True)
        for name in NAMES:
            (directory / name).touch()

    def predictions(self, frames=3):
        path = self.root / "pred.jsonl"
        import json
        rows = [{"frame": i, "detections": [dict(self.box, detection_id=f"{i}:0", raw_id=1)]}
                for i in range(1, frames + 1)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        json_write(path.with_suffix(".json"), dict(clip="a", clip_sha256=self.video["sha256"],
                                                   prediction_sha256=digest(path), frames=frames))
        return path

    def tracks(self, ids):
        import json
        path = self.root / "tracks.jsonl"
        rows = [{"frame": i, "tracks": [] if track is None else [dict(self.box, track_id=track)]} for i, track in enumerate(ids, 1)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        json_write(path.with_suffix(".json"), dict(clip_sha256=self.video["sha256"], track_sha256=digest(path), frames=len(ids)))
        return path

    def make_video(self, path):
        import cv2
        import numpy as np
        video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        self.assertTrue(video.isOpened())
        try:
            for color in (0, 80, 160):
                video.write(np.full((48, 64, 3), color, np.uint8))
        finally:
            video.release()
        return path

    def test_exact_mount_then_nested(self):
        direct = self.root / "datasets/erbosynshaimardan/zhanto-highlights"
        alternate = self.root / "datasets/random-owner/alternate"
        self.paths(direct)
        self.paths(alternate)
        self.assertEqual(resolve_dataset("auto", self.root), direct)
        (direct / NAMES[-1]).unlink()
        self.assertEqual(resolve_dataset("auto", self.root), alternate)
        with self.assertRaisesRegex(Blocked, NAMES[-1]):
            resolve_dataset(str(direct), self.root)

    def test_ambiguous_exact_dataset(self):
        for subdir in ("one", "two"):
            self.paths(self.root / subdir)
        with self.assertRaisesRegex(Blocked, "Multiple complete datasets"):
            resolve_dataset("auto", self.root)

    def test_missing_dataset_does_not_choose_other_videos(self):
        (self.root / "unrelated.mp4").touch()
        with self.assertRaisesRegex(Blocked, "No directory with all 12"):
            resolve_dataset("auto", self.root)

    def test_real_decode_synthetic_files(self):
        clip = self.make_video(self.root / NAMES[0])
        for name in NAMES[1:]:
            shutil.copy(clip, self.root / name)
        good, bad = dataset_audit(self.root)
        self.assertEqual(len(good), 12)
        self.assertFalse(bad)
        self.assertEqual(good[0]["frames"], 3)
        (self.root / NAMES[1]).write_bytes(b"corrupt")
        good, bad = dataset_audit(self.root)
        self.assertEqual(len(good), 11)
        self.assertEqual(bad[0]["file"], NAMES[1])

    def test_gpu_plan_requires_explicit_single(self):
        info = {"gpus": [{"index": 0, "functional": True}, {"index": 1, "functional": False}]}
        with self.assertRaises(Blocked):
            device_plan(info, True, False)
        self.assertEqual(device_plan(info, True, True), ["cuda:0"])
        info["gpus"][1]["functional"] = True
        self.assertEqual(device_plan(info, True, False), ["cuda:0", "cuda:1"])

    def test_spawn_scheduler_overlaps_and_preserves_failure(self):
        jobs = [dict(id=str(i), method="fixture", phase="smoke", cost=8-i, delay=.12, invalid=(i == 3)) for i in range(7)]
        report = execute(jobs, ["cpu-test0", "cpu-test1"], self.root, 7, FixtureRunner, telemetry=False)
        rows = report["results"]
        self.assertEqual(len(rows), 7)
        self.assertEqual(len({row["pid"] for row in rows}), 2)
        self.assertEqual({row["id"] for row in rows if row["state"] == "FAILED"}, {"3"})
        self.assertIn("fixture exception", rows[3]["traceback"])
        self.assertLess(max(rows[0]["start_monotonic_s"], rows[1]["start_monotonic_s"]),
                        min(rows[0]["end_monotonic_s"], rows[1]["end_monotonic_s"]))
        self.assertEqual(sum(w["frames"] for w in report["workers"]), 24)

    def test_spawn_survives_abrupt_worker_exit(self):
        jobs = [dict(id=str(i), method="die", phase="smoke", cost=5-i, delay=.02, crash=(i == 0)) for i in range(5)]
        report = execute(jobs, ["cpu-test0", "cpu-test1"], self.root, 7, FixtureRunner, telemetry=False)
        self.assertEqual(report["results"][0]["state"], "FAILED")
        self.assertEqual([r["state"] for r in report["results"]].count("EXECUTED"), 4)

    def test_box_and_cache_schema(self):
        p = self.predictions()
        self.assertEqual([x["frame"] for x in freeze_predictions(p, self.video)], [1, 2, 3])
        self.assertEqual(canonical_box([-10, 0, 92, 70], (100, 50))["y2"], 50)
        with self.assertRaises(ValueError):
            canonical_box([float("nan"), 3, 15, 23], (100, 50))
        p.write_text(p.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            list(freeze_predictions(p, self.video))

    def test_coco_metrics_with_known_labels(self):
        p = self.predictions()
        g = self.root / "gt.json"
        json_write(g, dict(schema="scoutra-reviewed/1", human_reviewed=True,
                           clips={"a": dict(clip_sha256=self.video["sha256"], frames=[{"frame": 1, "objects": [{**self.box, "class": "player"}]}])}))
        inputs = {"a": self.video}
        gt = parse_reviewed(g, inputs)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = detection_metrics({"a": p}, gt, inputs, .25, .05)
        self.assertEqual(metrics["precision50"], 1)
        self.assertAlmostEqual(metrics["mAP50_95"], 1)
        self.assertEqual(metrics["small_player_recall50"], 1)
        self.assertIsNone(metrics["ball_recall50"])
        with self.assertRaisesRegex(ValueError, "continuous scene"):
            parse_reviewed(g, inputs, dense=True)

    def test_trackeval_with_known_ids(self):
        p = self.tracks([10, 10, 10])
        truth = {"a": {i: [dict(self.box, player_id="verified")] for i in range(1, 4)}}
        result = tracking_metrics({"a": p}, truth, {"a": self.video})
        self.assertEqual(result, {"HOTA": 1.0, "IDF1": 1.0, "ID_switches": 0, "fragmentation": 0})

    def test_identity_scoring_detects_split_and_failed_reentry(self):
        p = self.tracks([10, None, 20])
        mapping = {"groups": {"10": "10", "20": "20"}, "tracking_sha256": digest(p), "clip_sha256": self.video["sha256"]}
        truth = {1: [dict(self.box, player_id="p1")], 2: [], 3: [dict(self.box, player_id="p1")]}
        labels = {"anchors": [{"frame": 1, "player_id": "p1"}],
                  "reentries": [{"return_frame": 3, "end_frame": 3, "player_id": "p1"}]}
        result = identity_metrics(p, mapping, self.video, labels, truth)
        self.assertEqual(result["metrics"]["false_splits"], 1)
        self.assertEqual(result["metrics"]["reentry_accuracy"], 0)
        self.assertEqual(result["metrics"]["identity_coverage"], 0.5)

    def test_part_exports_preserve_visibility_masks(self):
        import numpy as np
        path = self.root / "parts.npz"
        crops = [{"crop_id": "one", "path": str(path), "sha256": "a"*64},
                 {"crop_id": "two", "path": str(path), "sha256": "b"*64}]
        config = self.root / "crops.csv"
        config.write_text("fixture")
        features = dict(parts=np.ones((2, 3, 4)), visibility=np.asarray([[0, 1, 0], [0, 0, 1]]),
                        part_masks=np.ones((2, 3, 8, 4)))
        write_embeddings(path, crops, features, {}, {}, config)
        result = compare_embeddings(self.root / "pairs.json", path, None, 10, .5, None)
        self.assertEqual(result["sampled_pairs"], 1)
        self.assertEqual(result["scorable_pairs"], 0)
        self.assertNotIn("metrics", result)

    def test_homography_and_temporal_ocr(self):
        p = self.root / "landmarks.json"
        points = [[0, 0], [0, 40], [80, 0], [80, 40], [40, 20]]
        json_write(p, {"human_reviewed": True, "clips": {"a": {"clip_sha256": self.video["sha256"],
                   "frames": [{"frame": 1, "pixel_xy": points, "pitch_xy_m": [[x/2, y/2] for x, y in points]}]}}})
        out = self.root / "homography.json"
        self.assertEqual(homography(self.video, p, out), 1)
        self.assertLess(json_read(out)["frames"][0]["fit_rmse_m"], 1e-6)
        ocr = self.root / "ocr.json"
        json_write(ocr, [{"clip": "a", "track_id": 10, "digits": "12", "confidence": x} for x in [.8, .9]])
        number = self.root / "number.json"
        self.assertEqual(temporal_ocr(ocr, number), 1)
        self.assertEqual(json_read(number)["votes"][0]["digits"], "12")

    def test_atomic_export_excludes_intermediate_artifacts(self):
        output = self.root / "results.txt"
        output.write_text("original\n")
        with self.assertRaises(RuntimeError):
            with atomic_path(output) as temp:
                temp.write_text("unfinished")
                raise RuntimeError("failed")
        self.assertEqual(output.read_text(), "original\n")
        (self.root / ".scratch").write_text("transient")
        zip_path = export_zip(self.root, self.root / "export" / "actual.zip")
        with zipfile.ZipFile(zip_path) as z:
            self.assertEqual(z.namelist(), ["results.txt"])
            self.assertIsNone(z.testzip())

    def test_preflight_reports_and_report_keeps_saved_inputs(self):
        data = self.root / "inputs"
        data.mkdir()
        self.make_video(data / NAMES[0])
        for x in NAMES[1:]:
            shutil.copy(data / NAMES[0], data / x)
        config = copy.deepcopy(CONFIG)
        config.update(mode="PREFLIGHT", dataset_root=str(data))
        out = self.root / "run"
        with contextlib.redirect_stdout(io.StringIO()):
            artifact = launch(out, config)
        with zipfile.ZipFile(artifact) as z:
            self.assertIn("dataset/manifest.csv", z.namelist())
            statuses = list(csv.DictReader(io.StringIO(z.read("benchmarks/experiment_status.csv").decode())))
            self.assertTrue(any(x["group"] == "dataset" and x["state"] == "EXECUTED" for x in statuses))
            self.assertTrue(any(x["method"] == "rfdetr_small" and x["stage"] == "execution" and x["state"] == "SKIPPED" for x in statuses))
            self.assertFalse(any(x.endswith(".mp4") for x in z.namelist()))
        config.update(mode="REPORT")
        shutil.rmtree(data)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launch(out, config), artifact)

    def test_dataset_blocker_still_exports_failure(self):
        config = {**CONFIG, "mode": "SMOKE", "dataset_root": str(self.root / "missing")}
        with contextlib.redirect_stdout(io.StringIO()):
            artifact = launch(self.root / "results", config)
        with zipfile.ZipFile(artifact) as archive:
            statuses = archive.read("benchmarks/experiment_status.csv").decode()
            self.assertIn("BLOCKED", statuses)
            self.assertIn("Explicit DATASET_ROOT", archive.read("reports/failures.md").decode())
            self.assertFalse(any("/predictions/" in entry for entry in archive.namelist()))


if __name__ == "__main__":
    unittest.main()
