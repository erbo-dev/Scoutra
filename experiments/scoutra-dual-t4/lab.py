"""Standalone notebook coordinator. The notebook materializes this exact source."""
import os
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from advanced import ADVANCED, PART_COMMITS, compare_embeddings, homography, identity_output, identity_overlay, load_agentcoach, prepare_crops, temporal_ball, temporal_ocr
from backends import DETECTORS, HF, RF, TRACKERS, detector_asset, osnet_asset
from common import Blocked, GROUPS, NAMES, STATES, digest, export_zip, json_read, json_write, resolve_dataset, table
from environment import PACKAGE_PROFILES, command, device_plan, inventory, prepare_profile
from evaluate import (DETECTION_METRICS, IDENTITY_METRICS, REID_METRICS, TRACKING_METRICS,
                      identity_metrics, parse_reviewed)
from jobs import dataset_audit
from schedule import SAMPLE_FIELDS

STATUS_FIELDS = ("group", "method", "stage", "state", "reason", "clip", "device", "pid", "traceback")
RUNTIME_FIELDS = ("method", "phase", "clip", "device", "pid", "state", "frames", "crops_processed", "boxes", "clip_seconds",
                  "wall_s", "inference_s", "model_load_s", "cache_sha256", "prediction_sha256", "prediction_path",
                  "start_monotonic_s", "end_monotonic_s",
                  "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "cuda_allocated_bytes", "cuda_reserved_bytes")
METRIC_FIELDS = ("group", "method", "metric", "state", "value", "reason", "gt_sha256")
MODES = {"PREFLIGHT", "SMOKE", "BENCHMARK", "ADVANCED", "REPORT"}


def show_failure(error):
    return ("BLOCKED" if isinstance(error, Blocked) else "FAILED",
            f"{type(error).__name__}: {error}", traceback.format_exc())


def validate(config):
    if config["mode"] not in MODES:
        raise ValueError(f"RUN_MODE must be one of {sorted(MODES)}")
    if not set(config["groups"]) <= GROUPS:
        raise ValueError(f"Unknown MODEL_GROUPS, available groups: {sorted(GROUPS)}")
    if config["mode"] == "ADVANCED" and not config["advanced_methods"]:
        raise ValueError("ADVANCED requires at least one explicit ADVANCED_METHODS selection")
    if config["mode"] == "ADVANCED" and config["advanced_methods"] and not set(config["advanced_methods"]) <= set().union(*ADVANCED.values()):
        raise ValueError("Unknown ADVANCED_METHODS, see the method matrix in the notebook")
    if not set(config["detectors"]) <= set(DETECTORS) or not set(config["trackers"]) <= set(TRACKERS):
        raise ValueError(f"Unknown detector/tracker. Options: {list(DETECTORS)} / {list(TRACKERS)}")
    if any(len(config[key]) != len(set(config[key])) for key in ("detectors", "trackers", "advanced_methods")):
        raise ValueError("Method selections must be unique")
    if config["track_detector"] not in DETECTORS or config["identity_tracker"] not in {*TRACKERS, "deep_eiou"}:
        raise ValueError("Invalid TRACK_DETECTOR or IDENTITY_TRACKER")
    if config["precision"] not in {"fp32", "fp16"} or not 1 <= config["max_videos"] <= len(NAMES):
        raise ValueError("Use fp32/fp16 and MAX_VIDEOS between 1 and 12")
    if not 1 <= config["batch"] <= 32 or config["smoke_frames"] < 1 or config["timeout"] < 30:
        raise ValueError("Set batch 1–32, positive smoke frames and JOB_TIMEOUT_S >= 30")
    if not 0 <= config["score"] <= config["operating_score"] <= 1:
        raise ValueError("Require 0 <= detection score floor <= evaluation operating point <= 1")
    settings = config["tracker_settings"]
    if not 0 <= config["score"] <= settings["low"] < settings["high"] <= 1:
        raise ValueError("Detector floor must retain the low-score stage of ByteTrack and BoT-SORT")
    if not all(0 <= settings[key] <= 1 for key in ("new", "match")) or settings["buffer"] < 1:
        raise ValueError("Invalid tracker scores or buffer")
    if not 0 <= config["visibility_min"] <= 1 or not -1 <= config["identity_threshold"] <= 1:
        raise ValueError("Invalid visibility or identity threshold")
    if any(not -1 <= x <= 1 for x in config["identity_sweep"]):
        raise ValueError("Identity sweep thresholds must be within [-1, 1]")
    if min(config["crop_stride"], config["crops_per_track"], config["distance_sample"]) < 1:
        raise ValueError("Crop limits and sample size must be positive")
    if not set(config["scaling_models"]) <= set(DETECTORS):
        raise ValueError("Unknown detector in SCALING_MODELS")


def new_run(root):
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = root / "runs" / name
    path.mkdir(parents=True)
    return path


def find_run(root, name):
    if not name:
        raise Blocked("Supply CACHE_RUN with a saved run ID or extracted result directory")
    path = Path(name)
    path = path if path.is_absolute() else root / "runs" / path
    if not (path / "configuration.json").is_file():
        raise Blocked(f"No saved Scoutra run at {path}")
    return path


def groups_for(method):
    for group, entries in ADVANCED.items():
        if method in entries:
            return group
    raise ValueError(method)


class Laboratory:
    def __init__(self, root, config, run):
        self.root, self.config, self.run = root, config, run
        self.workspace = root / ".runtime"
        self.assets = Path(config["asset_cache"]) if config["asset_cache"] else self.workspace / "assets"
        self.status, self.runtime, self.metrics, self.samples = [], [], [], []
        self.schedules, self.comparisons = [], []
        self.profiles, self.video_by_clip, self.selected = {}, {}, []
        self.devices, self.device_failure = None, None
        self.cache_run = None
        self.generated_crops = None
        environment = run / "environment"
        environment.mkdir()
        (environment / "installation_log.txt").write_text("Package installation occurs only for selected, enabled methods.\n")
        (environment / "dependencies.txt").write_text(f"Host Python {sys.version}\n")
        json_write(run / "configuration.json", config)

    def log(self, group, method, stage, state, reason="", *, brief=True, **details):
        if state not in STATES:
            raise ValueError(f"Invalid status {state}")
        self.status.append(dict(group=group, method=method, stage=stage, state=state, reason=reason, **details))
        if brief:
            print(f"{state:13} {group}/{method} [{stage}] {reason}", flush=True)

    def record_error(self, group, method, stage, error):
        state, reason, trace = show_failure(error)
        self.log(group, method, stage, state, reason, traceback=trace)
        return state, reason

    def hardware(self):
        if self.devices is not None:
            return self.devices
        if self.device_failure:
            raise Blocked(self.device_failure)
        try:
            self.devices = device_plan(self.gpu, self.config["use_both"], self.config["allow_single"])
        except Blocked as error:
            self.device_failure = str(error)
            raise
        print(f"Assigned logical devices: {self.devices}", flush=True)
        if len(self.devices) < 2:
            self.log("environment", "gpu", "device_plan", "EXECUTED", f"Explicit single-GPU mode on {self.devices[0]}")
        elif any("T4" not in x.get("name", "") for x in self.gpu["gpus"] if f"cuda:{x['index']}" in self.devices):
            self.log("environment", "gpu", "device_plan", "EXECUTED", "Two functioning CUDA devices; at least one is a different model from Tesla T4.")
        else:
            self.log("environment", "gpu", "device_plan", "EXECUTED", "Two independently validated T4 devices.")
        return self.devices

    def profile(self, name):
        if name not in self.profiles:
            try:
                self.profiles[name] = prepare_profile(name, self.workspace, self.run, self.config["auto_acquire"])
                self.log("environment", name, "profile", "EXECUTED", "Isolated import test and pip check passed")
            except Exception as error:
                self.profiles[name] = error
                self.record_error("environment", name, "profile", error)
        result = self.profiles[name]
        if isinstance(result, Exception):
            raise result
        return result

    def audit(self):
        self.gpu = inventory(self.run / "environment" / "gpu_inventory.json")
        self.log("environment", "gpu", "inventory", "EXECUTED", f"{self.gpu['gpu_count']} CUDA device(s) observed. See kernel test results.")
        json_write(self.run / "environment" / "profiles.json", PACKAGE_PROFILES)
        root = resolve_dataset(self.config["dataset_root"])
        records, errors = dataset_audit(root)
        if records:
            table(self.run / "dataset" / "manifest.csv", records, tuple(records[0]))
        for e in errors:
            self.log("dataset", "video_decode", "audit", "FAILED", e["error"], clip=e["file"])
        if errors:
            raise Blocked(f"{len(errors)} of 12 exact clips failed validation. The dataset manifest contains only verified clips.")
        self.videos = records
        self.video_by_clip = {video["clip"]: video for video in records}
        self.selected = records[:self.config["max_videos"]]
        self.log("dataset", "zhanto_highlights", "audit", "EXECUTED",
                 f"{len(records)} of 12 clips decoded and hashed. {len(self.selected)} selected for this run.")
        if self.config["cache_run"]:
            self.cache_run = find_run(self.root, self.config["cache_run"])

    def preflight(self):
        for name in self.config["detectors"]:
            paths = []
            if name in RF:
                paths = [self.assets / f"rf-detr-{RF[name][0]}.pth"]
            elif name in HF:
                paths = [self.assets / name / f for f in ("config.json", "preprocessor_config.json", "model.safetensors")]
            elif self.config["scoutra_checkpoint"]:
                paths = [Path(self.config["scoutra_checkpoint"]["path"])]
            present = bool(paths) and all(path.is_file() for path in paths)
            self.log("detection", name, "asset_preflight", "EXECUTED" if present else "BLOCKED",
                     "Cache paths exist. Hashes and architecture are verified on load." if present else
                     "Asset missing from cache. SMOKE can acquire the default public assets with AUTO_ACQUIRE=True.")
        try:
            self.hardware()
        except Exception as error:
            self.record_error("environment", "gpu", "device_plan", error)

    def schedule(self, group, method, phase, tasks, devices, profile):
        python, env = self.profile(profile)
        logs = self.run / "environment" / "workers" / group / method / phase
        logs.mkdir(parents=True, exist_ok=True)
        request = logs / "schedule_input.json"
        response = logs / "schedule_output.json"
        active = devices[:len(tasks)]
        telemetry = any(device.startswith("cuda") for device in active)
        json_write(request, {"tasks": tasks, "devices": active, "logdir": str(logs),
                             "timeout": self.config["timeout"], "telemetry": telemetry})
        with (logs / "scheduler.log").open("w") as log:
            proc = subprocess.Popen([python, "-u", str(Path(__file__).with_name("schedule.py")), str(request), str(response)],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            try:
                for line in proc.stdout:
                    log.write(line)
                    print(line.rstrip(), flush=True)
                code = proc.wait()
            finally:
                if proc.poll() is None:
                    import signal
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait()
            if code or not response.is_file():
                raise RuntimeError(f"Scheduler failed with exit code {code}. See {logs / 'scheduler.log'}.")
        report = json_read(response)
        rows = report["results"]
        for row in rows:
            if row["state"] == "EXECUTED":
                reason = f"{row.get('frames', row.get('crops_processed', 0))} items processed"
            else:
                reason = row["reason"]
            self.log(group, method, phase, row["state"], reason, brief=row["state"] != "EXECUTED",
                     clip=row["clip"], device=row.get("device", ""), pid=row.get("pid", ""), traceback=row.get("traceback", ""))
        successes = [row for row in rows if row["state"] == "EXECUTED"]
        total_frames = sum(row.get("frames", 0) for row in successes)
        wall = report["wall_s"]
        used = sorted({row["device"] for row in successes})
        performance = {
            "group": group, "method": method, "phase": phase, "planned_devices": active, "working_devices": used,
            "jobs_succeeded": len(successes), "jobs_expected": len(tasks), "frames": total_frames,
            "wall_s": wall, "frames_per_s": total_frames / wall,
            "video_seconds_per_wall_s": sum(row.get("clip_seconds", 0) for row in successes) / wall,
            "device_wall_hours": wall * len(active) / 3600 if telemetry else 0,
            "samples": len(report["telemetry_samples"]), "telemetry_diagnostics": report["telemetry_diagnostics"],
            "workers": report["workers"],
        }
        if len(used) < len(active):
            self.log(group, method, f"{phase}_coverage", "FAILED", f"Completed work on {used}; expected {active}. Inspect worker failures.")
        print(f"{method} {phase}: {len(successes)}/{len(tasks)} tasks, {total_frames} frames, "
              f"{wall:.2f}s, devices {used}", flush=True)
        self.runtime.extend(rows)
        self.schedules.append(performance)
        self.samples.extend({**row, "method": method, "phase": phase} for row in report["telemetry_samples"])
        return successes, performance

    def detection_tasks(self, method, asset, phase, videos, limit, overlays):
        return [{"id": v["clip"], "method": method, "phase": phase, "kind": "detect", "video": v,
                 "cost": min(v["frames"], limit or v["frames"]), "limit": limit, "batch": self.config["batch"],
                 "asset": asset, "score": self.config["score"], "precision": self.config["precision"], "overlays": overlays,
                 "output": str(self.run / "detections" / method / phase)} for v in videos]

    def detect(self, method):
        devices = self.hardware()
        asset = detector_asset(method, self.config, self.assets)
        smoke_count = len(devices)
        videos = sorted(self.selected, key=lambda row: -row["frames"])[:smoke_count]
        profile = DETECTORS[method]
        tasks = self.detection_tasks(method, asset, "smoke", videos, self.config["smoke_frames"], self.config["save_videos"])
        smoke, _ = self.schedule("detection", method, "smoke", tasks, devices, profile)
        if self.config["mode"] == "SMOKE":
            return
        if len(smoke) != len(tasks):
            self.log("detection", method, "benchmark", "BLOCKED", "The real model smoke test failed on at least one assigned device.")
            return
        tasks = self.detection_tasks(method, asset, "benchmark", self.selected, None, self.config["save_videos"])
        predictions, dual = self.schedule("detection", method, "benchmark", tasks, devices, profile)
        self.evaluate("detection", method, predictions)
        if method in self.config["scaling_models"] and len(devices) == 2 and len(tasks) >= 2:
            if len(predictions) != len(tasks):
                self.log("detection", method, "scaling", "BLOCKED", "Two-GPU workload did not fully complete.")
                return
            trial = self.detection_tasks(method, asset, "scaling_single", self.selected, None, self.config["save_videos"])
            results, single = self.schedule("detection", method, "scaling_single", trial, devices[:1], profile)
            frames_dual = {row["clip"]: row["frames"] for row in predictions}
            frames_single = {row["clip"]: row["frames"] for row in results}
            if frames_dual != frames_single:
                self.log("detection", method, "scaling", "NOT_EVALUATED", "Single- and dual-GPU runs processed different workloads.")
                return
            speedup = single["wall_s"] / dual["wall_s"]
            inputs = {row["clip"]: row["sha256"] for row in self.selected}
            self.comparisons.append({"method": method, "clips": len(inputs), "frames": dual["frames"],
                                     "dual_wall_s": dual["wall_s"], "single_wall_s": single["wall_s"],
                                     "speedup_single_over_dual": speedup, "clip_hashes": inputs,
                                     "order": "dual then single", "devices": devices, "batch": self.config["batch"],
                                     "precision": self.config["precision"], "overlays": self.config["save_videos"]})
            self.log("detection", method, "scaling", "EXECUTED", f"Measured speedup {speedup:.3f}× on identical inputs.")
        elif method in self.config["scaling_models"]:
            self.log("detection", method, "scaling", "NOT_EVALUATED", "A scaling measurement needs two GPUs and at least two clips.")

    def cached(self, kind, method, clip, phase="benchmark"):
        video = self.video_by_clip[clip]
        directory = "predictions" if kind == "detections" else "tracks"
        for source in (self.run, self.cache_run):
            if source:
                path = source / kind / method / phase / directory / f"{clip}.jsonl"
                if not path.exists():
                    continue
                meta = json_read(path.with_suffix(".json"))
                key = "prediction_sha256" if kind == "detections" else "track_sha256"
                if meta["clip_sha256"] != video["sha256"] or digest(path) != meta[key]:
                    raise ValueError(f"Cached artifacts fail input/output fingerprint validation: {path}")
                if phase == "benchmark" and meta["frames"] != video["frames"]:
                    raise ValueError(f"Saved benchmark incomplete: {path}")
                return path
        raise Blocked(f"Missing {phase} cache for {kind}/{method}/{clip}. Run the source method or set CACHE_RUN.")

    def track(self, method):
        devices = self.hardware() if method == "strongsort" else ["cpu:0", "cpu:1"]
        asset = osnet_asset(self.config) if method == "strongsort" else None
        phase = "smoke" if self.config["mode"] == "SMOKE" else "benchmark"
        videos = (sorted(self.selected, key=lambda item: -item["frames"])[:(len(self.devices) if self.devices else 1)]
                  if phase == "smoke" else self.selected)
        tasks = []
        for video in videos:
            try:
                raw = self.cached("detections", self.config["track_detector"], video["clip"], phase)
                if json_read(raw.with_suffix(".json"))["score_floor"] > self.config["tracker_settings"]["low"]:
                    raise ValueError("Selected detector cache removed scores needed by the low-score tracking stage")
                tasks.append({"id": video["clip"], "method": method, "phase": phase, "kind": "track", "video": video,
                              "cost": video["frames"], "cache": str(raw), "thresholds": self.config["tracker_settings"],
                              "asset": asset, "overlays": self.config["save_videos"],
                              "output": str(self.run / "tracking" / method / phase)})
            except Exception as error:
                state, reason, trace = show_failure(error)
                self.log("tracking", method, phase, state, reason, clip=video["clip"], traceback=trace)
        if not tasks:
            return
        rows, _ = self.schedule("tracking", method, phase, tasks, devices, TRACKERS[method])
        if phase == "benchmark":
            self.evaluate("tracking", method, rows)

    def capture_metrics(self, group, method, names, values=None, reason="", fingerprint=""):
        for name in names:
            result = (values or {}).get(name)
            state = "EXECUTED" if result is not None else "NOT_EVALUATED"
            self.metrics.append(dict(group=group, method=method, metric=name, state=state,
                                     reason=("" if result is not None else reason or "No eligible annotated positives or denominator"),
                                     value=result, gt_sha256=fingerprint))
        if any(row["group"] == group and row["method"] == method and row["state"] == "EXECUTED" for row in self.metrics):
            self.log(group, method, "evaluation", "EXECUTED", "Reviewed annotations evaluated; see metrics.csv.")
        else:
            self.log(group, method, "evaluation", "NOT_EVALUATED", reason or "No metric has an eligible labeled denominator.")

    def evaluate(self, group, method, predictions):
        names = DETECTION_METRICS if group == "detection" else TRACKING_METRICS
        key = "detection_annotations" if group == "detection" else "tracking_annotations"
        gt = self.config[key]
        if not predictions or not gt:
            reason = "No successful full-clip predictions" if not predictions else "No human-reviewed annotation file selected."
            self.capture_metrics(group, method, names, reason=reason)
            return
        try:
            python, env = self.profile("eval")
            folder = self.run / "benchmarks" / "evaluation" / group / method
            spec, result = folder / "input.json", folder / "metrics.json"
            data = dict(kind=group, annotations=gt, predictions={p["clip"]: p["prediction_path"] for p in predictions},
                        inputs={v["clip"]: v for v in self.selected}, operating_point=self.config["operating_score"], cache_floor=self.config["score"])
            json_write(spec, data)
            code = command([python, str(Path(__file__).with_name("evaluate.py")), str(spec), str(result)],
                           folder / "calculation.log", environment=env)
            if code:
                raise RuntimeError(f"Evaluation exit={code}; see {folder / 'calculation.log'}")
            output = json_read(result)
            self.capture_metrics(group, method, names, {name: v["value"] for name, v in output["metrics"].items()},
                                 fingerprint=output["gt_sha256"])
        except Exception as error:
            _, reason = self.record_error(group, method, "evaluation_error", error)
            self.capture_metrics(group, method, names, reason=reason)

    def reid(self, method):
        if method == "prepare_crops":
            output = self.run / "reid" / "crops" / "manifest.csv"
            count = prepare_crops(lambda clip: self.cached("tracking", self.config["identity_tracker"], clip),
                                  self.selected, output, self.config["crop_stride"], self.config["crops_per_track"])
            self.generated_crops = str(output)
            self.log("reid", method, "advanced", "EXECUTED", f"{count} immutable crops stored")
            return
        crops = self.config["crop_manifest"] or self.generated_crops
        if not crops:
            raise Blocked("Set CROP_MANIFEST to a set of frozen image crops with crop_id,path,sha256. See annotation guide.")
        output = self.run / "reid" / "embeddings" / f"{method}.npz"
        if method == "osnet":
            devices = self.hardware()
            asset = osnet_asset(self.config)
            task = dict(id="crop_manifest", method=method, phase="advanced", kind="osnet", batch=self.config["batch"],
                        crops=crops, asset=asset, output=str(output))
            rows, _ = self.schedule("reid", method, "advanced", [task], devices[:1], "boxmot")
            if not rows:
                return
        else:
            self.part_reid(method, crops, output)
        comparisons = self.run / "reid" / "comparisons" / f"{method}.json"
        reviewed = self.config["reid_annotations"]
        result = compare_embeddings(comparisons, output, reviewed, self.config["distance_sample"],
                                    self.config["visibility_min"], self.config["reid_threshold"])
        self.capture_metrics("reid", method, REID_METRICS, result.get("metrics"), fingerprint=result.get("gt_sha256", ""),
                             reason="" if reviewed else "No human-reviewed query/gallery split selected.")

    def part_reid(self, method, crops, output):
        from legacy import part_profile
        python, spec, env = part_profile(method, self.config["legacy_reid"], self.hardware()[0], crops, output, self.config["batch"])
        request = self.run / "reid" / f"{method}_legacy_input.json"
        json_write(request, spec)
        folder = self.run / "environment"
        logfile = folder / f"{method}_legacy.log"
        if command([python, str(Path(__file__).with_name("legacy.py")), str(request)], logfile,
                   timeout=self.config["timeout"], environment=env) or not output.is_file():
            raise RuntimeError(f"Isolated upstream extraction failed. See {logfile}.")
        self.log("reid", method, "advanced", "EXECUTED", "Full upstream feature extraction completed. See embedding sidecar.")

    def advanced(self, group, method):
        blocker = ADVANCED[group][method]
        if blocker:
            raise Blocked(blocker)
        if group == "reid":
            self.reid(method)
            return
        if method == "deep_eiou":
            self.deep_eiou()
            return
        if method == "temporal_ocr_vote":
            source = self.config["ocr_predictions"]
            if not source:
                raise Blocked("Set OCR_PREDICTIONS to saved frame-level digit predictions.")
            count = temporal_ocr(source, self.run / "jersey" / "temporal_ocr_vote.json")
            self.log(group, method, "advanced", "EXECUTED", f"{count} tracklets aggregated")
            return
        if group == "calibration" and not self.config["landmarks"]:
            raise Blocked("Set PITCH_LANDMARKS with reviewed framewise pixel-to-meter correspondences.")
        module = load_agentcoach(self.assets, self.config["auto_acquire"]) if method == "agentcoach_reid_only" else None
        for video in self.selected:
            clip = video["clip"]
            try:
                if group == "identity":
                    raw = self.cached("tracking", self.config["identity_tracker"], clip)
                    output = self.run / "identity" / "results" / method / f"{clip}.json"
                    if method == "gta_link":
                        from legacy import gta_profile

                        python, request, env = gta_profile(self.config["gta_config"], self.hardware()[0], raw, video, output)
                        spec = self.run / "identity" / "legacy" / f"{clip}.json"
                        logfile = self.run / "environment" / f"gta_{clip}.log"
                        json_write(spec, request)
                        if command([python, str(Path(__file__).with_name("legacy.py")), str(spec)], logfile,
                                   timeout=self.config["timeout"], environment=env) or not output.is_file():
                            raise RuntimeError(f"Original GTA-Link runner failed. See {logfile}.")
                        count = json_read(output)["retained_observations"]
                    else:
                        count = identity_output(method, raw, video, output, self.config["identity_features"],
                                                module, self.config["identity_threshold"] if method != "no_link" else None)
                    if self.config["save_videos"]:
                        identity_overlay(raw, output, video, self.run / "identity" / "visualizations" / method / f"{clip}.mp4")
                elif group == "calibration":
                    output = self.run / group / method / f"{clip}.json"
                    count = homography(video, self.config["landmarks"], output)
                else:
                    raw = self.cached("detections", self.config["track_detector"], clip)
                    output = self.run / "ball" / method / f"{clip}.json"
                    count = temporal_ball(raw, video, output, self.config["operating_score"], .06, 10)
                self.log(group, method, "advanced", "EXECUTED", f"{count} items processed", clip=clip)
            except Exception as error:
                state, reason, trace = show_failure(error)
                self.log(group, method, "advanced", state, reason, clip=clip, traceback=trace)
        if group == "identity":
            self.evaluate_identity(method)
            if method == "cosine_tracklet_link":
                for threshold in self.config["identity_sweep"]:
                    scenario = f"{method}@{threshold:g}"
                    for video in self.selected:
                        clip = video["clip"]
                        try:
                            raw = self.cached("tracking", self.config["identity_tracker"], clip)
                            output = self.run / "identity" / "results" / scenario / f"{clip}.json"
                            count = identity_output(method, raw, video, output, self.config["identity_features"], threshold=threshold)
                            self.log(group, scenario, "advanced", "EXECUTED", f"{count} local tracklets", clip=clip)
                        except Exception as error:
                            state, reason, trace = show_failure(error)
                            self.log(group, scenario, "advanced", state, reason, clip=clip, traceback=trace)
                    self.evaluate_identity(scenario)

    def deep_eiou(self):
        from upstream_tracking import deep_profile

        name = "deep_eiou"
        if not self.config["deep_eiou_config"]:
            raise Blocked("Set DEEP_EIOU_CONFIG with the rights-reviewed source, isolated NumPy <1.24 environment and verified sports OSNet weights.")
        device = self.hardware()[0]
        predictions = []
        for video in self.selected:
            clip = video["clip"]
            try:
                cache = self.cached("detections", self.config["track_detector"], clip)
                output = self.run / "tracking" / name / "benchmark"
                python, spec, env = deep_profile(self.config["deep_eiou_config"], device, cache, video, output,
                                                 self.config["tracker_settings"], self.config["save_videos"])
                request = self.run / "environment" / f"{name}_{clip}.json"
                logfile = request.with_suffix(".log")
                json_write(request, spec)
                tracker = Path(__file__).with_name("upstream_tracking.py")
                code = command([python, str(tracker), str(request)], logfile, timeout=self.config["timeout"], environment=env)
                result = output / "tracks" / f"{clip}.jsonl"
                if code or not result.is_file():
                    raise RuntimeError(f"Isolated Deep-EIoU process failed. See {logfile}.")
                info = json_read(result.with_suffix(".json"))
                self.log("tracking", name, "advanced", "EXECUTED", f"{info['frames']} frames", clip=clip, device=device)
                row = dict(method=name, phase="advanced", clip=clip, state="EXECUTED", device=device,
                           frames=info["frames"], boxes=info["boxes"], wall_s=info["wall_s"], inference_s=info["inference_s"],
                           cuda_peak_allocated_bytes=info["peak_cuda_allocated_bytes"],
                           prediction_path=str(result), cache_sha256=info["detection_sha256"])
                self.runtime.append(row)
                predictions.append(row)
            except Exception as error:
                state, reason, trace = show_failure(error)
                self.log("tracking", name, "advanced", state, reason, clip=clip, traceback=trace)
        self.evaluate("tracking", name, predictions)

    def evaluate_identity(self, method):
        spec = self.config["identity_annotations"]
        if not spec:
            self.capture_metrics("identity", method, IDENTITY_METRICS,
                                 reason="No human-reviewed continuous scenes, anchors or re-entry episodes selected.")
            return
        per_clip = {}
        try:
            reviewed = parse_reviewed(spec, self.video_by_clip, dense=True)
            annotations = json_read(spec)["clips"]
            for clip, item in reviewed.items():
                output = self.run / "identity" / "results" / method / f"{clip}.json"
                if not output.is_file():
                    continue
                mapped = json_read(output)
                raw = self.cached("tracking", self.config["identity_tracker"], clip)
                per_clip[clip] = identity_metrics(raw, mapped, self.video_by_clip[clip], annotations[clip], item)
            if not per_clip:
                raise Blocked("No reviewed clip overlaps with successful identity output")
            stats = Counter()
            for x in per_clip.values():
                stats.update(x["counts"])
            delays = sorted(delay for x in per_clip.values() for delay in x["recovery_delays_frames"])

            def fraction(key, denom):
                return stats[key] / stats[denom] if stats[denom] else None

            totals = dict(reentry_accuracy=fraction("recoveries", "return_episodes"),
                          identity_precision=fraction("correct_person_frames", "assigned_person_frames"),
                          identity_coverage=fraction("assigned_person_frames", "gt_person_frames"),
                          recovery_delay_frames=((delays[(len(delays) - 1) // 2] + delays[len(delays) // 2]) / 2
                                                 if delays else None))
            totals.update({key: sum(x["metrics"][key] or 0 for x in per_clip.values()) if stats["gt_person_frames"] else None
                           for key in ("false_merges", "false_splits")})
            fingerprint = digest(spec)
            json_write(self.run / "identity" / "evaluations" / f"{method}.json",
                       {"gt_sha256": fingerprint, "metrics": totals, "per_clip": per_clip})
            self.capture_metrics("identity", method, IDENTITY_METRICS, totals, fingerprint=fingerprint)
        except Exception as error:
            _, reason = self.record_error("identity", method, "evaluation_error", error)
            self.capture_metrics("identity", method, IDENTITY_METRICS, reason=reason)

    def record_unrun(self):
        advertised = {group: list(methods) for group, methods in ADVANCED.items()}
        advertised["detection"][:0] = DETECTORS
        advertised["tracking"][:0] = TRACKERS
        for group, methods in advertised.items():
            for method in methods:
                if not any(r["group"] == group and r["method"] == method and r["stage"] != "asset_preflight" for r in self.status):
                    reason = "PREFLIGHT inventories assets and hardware only." if self.config["mode"] == "PREFLIGHT" else "Method not selected in this run."
                    blocker = ADVANCED[group].get(method)
                    if blocker:
                        reason = f"{reason} {blocker}"
                    self.log(group, method, "execution", "SKIPPED", reason, brief=False)
                names = {"detection": DETECTION_METRICS, "tracking": TRACKING_METRICS,
                         "reid": REID_METRICS, "identity": IDENTITY_METRICS}.get(group)
                if names and not any(r["group"] == group and r["method"] == method for r in self.metrics):
                    reason = "No eligible reviewed benchmark executed."
                    self.metrics.extend(dict(group=group, method=method, metric=n, state="NOT_EVALUATED", reason=reason) for n in names)

    def core(self, group, methods, handler):
        if group not in self.config["groups"]:
            return
        for method in methods:
            try:
                handler(method)
            except Exception as error:
                self.record_error(group, method, "execution", error)

    def reports(self):
        self.record_unrun()
        bench = self.run / "benchmarks"
        table(bench / "experiment_status.csv", self.status, STATUS_FIELDS)
        table(bench / "metrics.csv", self.metrics, METRIC_FIELDS)
        if self.runtime:
            table(bench / "runtime.csv", self.runtime, RUNTIME_FIELDS)
        if self.samples:
            table(bench / "gpu_usage.csv", self.samples, ("method", "phase", *SAMPLE_FIELDS))
        json_write(bench / "execution_summary.json", {"schedules": self.schedules, "scaling": self.comparisons})
        report = self.run / "reports"
        report.mkdir(exist_ok=True)
        counts = Counter(row["state"] for row in self.status)
        (report / "summary.md").write_text(
            f"# Scoutra run {self.run.name}\n\nMode: {self.config['mode']}\n\n"
            + "\n".join(f"- {name}: {count}" for name, count in sorted(counts.items()))
            + "\n\nSee benchmarks/experiment_status.csv for each method and clip. A hardware or asset check is distinct from inference.\n"
            + ("\nGPU samples are device-wide nvidia-smi utilization. Match by UUID in environment/gpu_inventory.json.\n" if self.samples else
               "\nNo device-wide utilization samples were recorded. See execution_summary.json for the sampler diagnostics and run scope.\n")
            + "\nInference time includes model preprocessing/postprocessing, with CUDA synchronization for GPU workers. "
              "Wall time includes cold worker/model startup, decode, inference, serialization and enabled overlays. "
              "Device-wall hours use planned device count; they are a scheduling cost estimate.\n", encoding="utf-8")
        failures = [row for row in self.status if row["state"] in {"FAILED", "BLOCKED"}]
        details = [f"## {row['group']}/{row['method']} · {row['stage']} · {row.get('clip', '')}\n\n"
                   f"{row['state']}: {row['reason']}" + (f"\n\n```text\n{row['traceback']}```" if row.get("traceback") else "")
                   for row in failures]
        (report / "failures.md").write_text("# Failures and blockers\n\n" + ("\n\n".join(details) if details else
                                             "No failed or blocked stages recorded. See experiment_status.csv for execution scope.") + "\n")
        mapping = ["# Method coverage", "", "| Group | Method | Stage | States |", "|---|---|---|---|"]
        by_key = {}
        for row in self.status:
            key = (row["group"], row["method"], row["stage"])
            by_key.setdefault(key, Counter())[row["state"]] += 1
        for key, counts in sorted(by_key.items()):
            state = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
            mapping.append(f"| {' | '.join((*key, state))} |")
        mapping.extend(["", "## Same-workload GPU scaling", ""])
        mapping.extend([f"- {x['method']}: {x['frames']} frames, single {x['single_wall_s']:.2f}s, dual {x['dual_wall_s']:.2f}s, "
                        f"speedup {x['speedup_single_over_dual']:.3f}×." for x in self.comparisons] or ["No complete scaling measurement."])
        mapping.extend(["", "Tracking uses one frozen detector and clip-local IDs. BoT-SORT uses camera compensation and has appearance disabled.",
                        "Inspect cache fingerprints before comparing runs. Accuracy metrics use only reviewed labels.",
                        "The dataset consists of independent short highlights. Full-match identity reliability is outside this measurement."])
        (report / "method_comparison.md").write_text("\n".join(mapping) + "\n", encoding="utf-8")

    def run_selected(self):
        self.audit()
        mode = self.config["mode"]
        if mode == "PREFLIGHT":
            self.preflight()
        elif mode in {"SMOKE", "BENCHMARK"}:
            self.core("detection", self.config["detectors"], self.detect)
            self.core("tracking", self.config["trackers"], self.track)
        else:
            for method in self.config["advanced_methods"]:
                group = groups_for(method)
                try:
                    self.advanced(group, method)
                except Exception as error:
                    self.record_error(group, method, "advanced", error)


def launch(root, config):
    validate(config)
    root = Path(root)
    release = root / "export" / "scoutra_experiments_results.zip"
    pointer = root / "latest_run.json"
    if config["mode"] == "REPORT":
        selected = config["report_run"] or (json_read(pointer)["run_id"] if pointer.exists() else None)
        path = find_run(root, selected)
        for name in ("summary.md", "method_comparison.md"):
            file = path / "reports" / name
            if file.is_file():
                print(file.read_text(encoding="utf-8"))
        return export_zip(path, release)
    directory = new_run(root)
    lab = Laboratory(root, config, directory)
    try:
        lab.run_selected()
    except Exception as error:
        lab.record_error("run", "orchestrator", "execution", error)
    finally:
        lab.reports()
        release = export_zip(directory, release)
        json_write(pointer, {"run_id": directory.name, "path": str(directory), "zip": str(release)})
    print((directory / "reports" / "summary.md").read_text(encoding="utf-8"))
    print(f"Run folder: {directory}\nZIP: {release}", flush=True)
    if any(row["state"] in {"FAILED", "BLOCKED"} and row["group"] != "environment" for row in lab.status):
        print("This run contains FAILED or BLOCKED stages. Open reports/failures.md before comparing results.")
    return release


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: lab.py OUTPUT_ROOT CONFIG_JSON EMBEDDED_SOURCE_SHA256")
    config = json_read(sys.argv[2])
    config["embedded_source_sha256"] = sys.argv[3]
    launch(Path(sys.argv[1]), config)
