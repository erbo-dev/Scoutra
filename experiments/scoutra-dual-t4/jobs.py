"""Video decoding, native prediction export and worker task dispatch."""
import itertools
import math
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from backends import Detector, Tracker
from common import BOX, NAMES, atomic_path, digest, freeze_predictions, json_write, write_line


@contextmanager
def capture(path):
    import cv2
    video = cv2.VideoCapture(str(path))
    try:
        if not video.isOpened():
            raise ValueError(f"OpenCV could not open {path}")
        yield video
    finally:
        video.release()


def frames(video, maximum=None):
    for _ in itertools.count(1) if maximum is None else range(maximum):
        ok, frame = video.read()
        if not ok:
            break
        yield frame


def probe_video(path):
    import cv2
    with capture(path) as video:
        fps = float(video.get(cv2.CAP_PROP_FPS))
        nominal_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        count, size = 0, None
        for count, frame in enumerate(frames(video), 1):
            current = (frame.shape[1], frame.shape[0])
            if size is None:
                size = current
            if size != current:
                raise ValueError("Variable frame dimensions require separate research handling")
    if count < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Cannot decode valid frames and FPS for {path}")
    if nominal_frames > 0 and count != nominal_frames:
        raise ValueError(f"Inconsistent decode: container {nominal_frames}, decoded {count}")
    return dict(clip=path.stem, file=path.name, path=str(path), sha256=digest(path), bytes=path.stat().st_size,
                width=size[0], height=size[1], frames=count, nominal_fps=fps,
                duration_s=count / fps, duration_basis="decoded frame count / nominal FPS")


def dataset_audit(root):
    records = []
    failures = []
    for name in NAMES:
        try:
            records.append(probe_video(root / name))
        except Exception as error:
            failures.append({"file": name, "error": f"{type(error).__name__}: {error}"})
    return records, failures


class Overlay:
    def __init__(self, process):
        self.process = process

    def draw(self, frame, objects, tracks):
        if not self.process:
            return
        import cv2

        for entity in objects:
            x1, y1, x2, y2 = [round(entity[key]) for key in BOX]
            color = (70, 120, 255) if entity["class"] == "ball" else (80, 230, 100)
            label = f"ID {entity['track_id']}" if tracks else f"{entity['class']} {entity['score']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
        self.process.stdin.write(frame.tobytes())


@contextmanager
def overlay(path, video, enabled):
    if not enabled:
        yield Overlay(None)
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg with libx264 is required for annotated video. Install it or disable SAVE_ANNOTATED_VIDEOS.")
    with atomic_path(path) as temporary, (path.parent / f".{path.stem}.ffmpeg.log").open("w+") as log:
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{video['width']}x{video['height']}", "-r", str(video["nominal_fps"]), "-i", "pipe:0",
                   "-an", "-c:v", "libx264", "-preset", "veryfast", "-threads", "1",
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            yield Overlay(proc)
            proc.stdin.close()
            code = proc.wait(timeout=90)
            if code or temporary.stat().st_size == 0:
                log.seek(0)
                raise RuntimeError(f"Overlay encoder exit={code}: {log.read()}")
        except (BrokenPipeError, OSError) as error:
            log.seek(0)
            raise RuntimeError(f"Overlay encoder error: {log.read()}") from error
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdin.close()
    (path.parent / f".{path.stem}.ffmpeg.log").unlink(missing_ok=True)


class JobRunner:
    def __init__(self, device):
        import cv2
        cv2.setNumThreads(1)
        self.device = "cpu" if device.startswith("cpu:") else device
        self.detector = None
        self.detector_error = None
        self.torch = None
        self.gpu_uuid = None
        if device.startswith("cuda:"):
            import torch
            torch.set_num_threads(1)
            torch.cuda.set_device(device)
            from environment import cuda_uuid
            self.torch = torch
            self.gpu_uuid = cuda_uuid(torch, torch.cuda.current_device())

    def sync(self):
        if self.torch:
            self.torch.cuda.synchronize(self.device)

    def memory(self):
        if not self.torch:
            return {}
        cuda = self.torch.cuda
        return dict(cuda_peak_allocated_bytes=cuda.max_memory_allocated(self.device),
                    cuda_peak_reserved_bytes=cuda.max_memory_reserved(self.device),
                    cuda_allocated_bytes=cuda.memory_allocated(self.device), cuda_reserved_bytes=cuda.memory_reserved(self.device))

    def __call__(self, task):
        operation = task["kind"]
        if operation == "detect":
            if self.detector_error:
                raise RuntimeError(f"Model initialization previously failed in this worker: {self.detector_error}")
            start = time.perf_counter()
            load_s = 0
            if self.detector is None:
                try:
                    self.detector = Detector(task["method"], task["asset"], self.device, task["precision"], task["score"])
                    self.sync()
                    load_s = time.perf_counter() - start
                except Exception as error:
                    self.detector_error = repr(error)
                    raise
            if self.torch:
                self.torch.cuda.reset_peak_memory_stats(self.device)
            return {**self.detect(task), **self.memory(), "model_load_s": load_s}
        if operation == "track":
            if self.torch:
                self.torch.cuda.reset_peak_memory_stats(self.device)
            return {**self.track(task), **self.memory()}
        if operation == "osnet":
            if self.torch:
                self.torch.cuda.reset_peak_memory_stats(self.device)
            from advanced import extract_osnet
            return {**extract_osnet(task, self.device), **self.memory()}
        raise ValueError(f"Unknown job kind: {operation}")

    def detect(self, task):
        import cv2
        video = task["video"]
        output = Path(task["output"])
        pred = output / "predictions" / f"{video['clip']}.jsonl"
        overlay_path = output / "annotated_videos" / f"{video['clip']}.mp4"
        expected = min(task["limit"] or video["frames"], video["frames"])
        count = boxes = batches = 0
        inference = 0.0
        with capture(video["path"]) as source, atomic_path(pred) as temporary, temporary.open("w") as rows, \
                overlay(overlay_path, video, task["overlays"]) as annotated:
            decoded = iter(frames(source, expected))
            while batch := list(itertools.islice(decoded, task["batch"])):
                rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in batch]
                self.sync()
                start = time.perf_counter()
                results = self.detector.predict(rgb)
                self.sync()
                batches += 1
                inference += time.perf_counter() - start
                for frame, detections in zip(batch, results, strict=True):
                    count += 1
                    for i, d in enumerate(detections):
                        d["detection_id"] = f"{count}:{i}"
                    boxes += len(detections)
                    write_line(rows, {"frame": count, "detections": detections})
                    annotated.draw(frame, detections, False)
            if count != expected:
                raise ValueError(f"Decoded {count} frames during inference, expected {expected}")
        fingerprint = digest(pred)
        json_write(pred.with_suffix(".json"), {
            "clip": video["clip"], "clip_sha256": video["sha256"], "prediction_sha256": fingerprint,
            "frames": count, "complete_clip": count == video["frames"], "method": task["method"],
            "score_floor": task["score"], "asset": task["asset"], "model_contract": self.detector.load_info,
            "batch_size": task["batch"], "precision": task["precision"],
            "device": task.get("assigned_device", self.device), "worker_cuda_device": self.device,
            "gpu_uuid": self.gpu_uuid, "inference_batches": batches, "cuda_frames": count if self.torch else 0,
        })
        return {"frames": count, "boxes": boxes, "inference_s": inference, "inference_batches": batches,
                "cuda_frames": count if self.torch else 0,
                "clip_seconds": count / video["nominal_fps"], "prediction_path": str(pred),
                "prediction_sha256": fingerprint}

    def track(self, task):
        video = task["video"]
        output = Path(task["output"])
        raw = Path(task["cache"])
        pred = output / "tracks" / f"{video['clip']}.jsonl"
        mot = output / "tracks" / f"{video['clip']}.txt"
        overlay_path = output / "annotated_videos" / f"{video['clip']}.mp4"
        count = boxes = 0
        infer = 0.0
        start = time.perf_counter()
        tracker = Tracker(task["method"], video["nominal_fps"], task["thresholds"], task.get("asset"), self.device)
        self.sync()
        load_s = time.perf_counter() - start
        with capture(video["path"]) as source, atomic_path(pred) as json_tmp, json_tmp.open("w") as rows, \
                atomic_path(mot) as mot_tmp, mot_tmp.open("w") as mot_rows, \
                overlay(overlay_path, video, task["overlays"]) as annotated:
            for line in freeze_predictions(raw, video):
                ok, frame = source.read()
                if not ok:
                    raise ValueError("Frozen predictions outlast the decoded video")
                self.sync()
                start = time.perf_counter()
                tracks = tracker.update(line["detections"], frame)
                self.sync()
                infer += time.perf_counter() - start
                count = line["frame"]
                boxes += len(tracks)
                write_line(rows, {"frame": count, "tracks": tracks})
                for r in tracks:
                    mot_rows.write(f"{count},{r['track_id']},{r['x1'] + 1:.3f},{r['y1'] + 1:.3f},"
                                   f"{r['x2'] - r['x1']:.3f},{r['y2'] - r['y1']:.3f},{r['score']:.7f},-1,-1,-1\n")
                annotated.draw(frame, tracks, True)
        fingerprint = digest(raw)
        json_write(pred.with_suffix(".json"), {
            "clip": video["clip"], "clip_sha256": video["sha256"], "detection_sha256": fingerprint,
            "track_sha256": digest(pred), "frames": count, "method": task["method"], "parameters": tracker.parameters,
            "cache": str(raw), "asset": task.get("asset"),
        })
        return {"frames": count, "boxes": boxes, "inference_s": infer, "model_load_s": load_s,
                "clip_seconds": count / video["nominal_fps"], "prediction_path": str(pred), "cache_sha256": fingerprint}
