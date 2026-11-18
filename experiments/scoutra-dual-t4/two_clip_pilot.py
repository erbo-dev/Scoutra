"""Run the two original Scoutra clips with a shared detector cache."""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import Blocked, export_zip, json_read, json_write, table, verified
from environment import command
from jobs import probe_video
from lab import Laboratory, new_run, validate
from legacy import GTA_COMMIT, GTA_WEIGHT_HASH
from upstream_tracking import COMMIT as DEEP_COMMIT

NAMES = tuple(f"document_544086323054304{clip}.mp4" for clip in ("5483", "5485"))
DEPENDENCIES = (
    "numpy==1.23.5", "scipy==1.11.4", "opencv-python-headless==4.11.0.86",
    "lap==0.5.12", "scikit-learn==1.4.2", "matplotlib==3.9.4", "seaborn==0.13.2",
    "loguru==0.7.3", "tqdm==4.67.1", "gdown==5.2.0", "h5py==3.12.1",
    "yacs==0.1.8", "tensorboard==2.19.0", "six==1.17.0", "imageio==2.37.0",
    "tabulate==0.9.0", "Cython==0.29.37", "setuptools==75.8.2", "wheel==0.45.1",
)
GTA_SETTINGS = dict(eps=.65, max_k=2, min_samples=15, min_len=50,
                    spatial_factor=1.5, merge_dist_thres=.35)


def selected_paths(request, mount=Path("/kaggle/input")):
    if request != "auto":
        candidates = [Path(request).resolve()]
    else:
        preferred = mount / "datasets/erbosynshaimardan/zhanto-highlights"
        if all((preferred / name).is_file() for name in NAMES):
            return [preferred / name for name in NAMES]
        candidates = sorted({path.parent.resolve() for path in mount.rglob(NAMES[0])})
    complete = [root for root in candidates if all((root / name).is_file() for name in NAMES)]
    if len(complete) != 1:
        raise Blocked(f"Need one directory containing both exact clips {NAMES}; found {complete}. Set DATASET_ROOT or attach zhanto-highlights.")
    return [complete[0] / name for name in NAMES]


def recorded_command(args, log, timeout=900, env=None):
    if command([str(arg) for arg in args], log, timeout=timeout, environment=env):
        raise RuntimeError(f"Setup command failed. See {log.name} in the results ZIP.")


def checkout(owner, name, sha, work, log):
    path = work / name
    if not (path / ".git").exists():
        path.mkdir(parents=True, exist_ok=True)
        recorded_command(["git", "init", path], log)
    head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode or head.stdout.strip() != sha:
        url = f"https://github.com/{owner}/{name}.git"
        recorded_command(["git", "-C", path, "fetch", "--depth=1", url, sha], log)
        recorded_command(["git", "-C", path, "checkout", "--detach", sha], log)
    return path


def install_legacy(working, logs):
    working.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    gta = checkout("sjc042", "gta-link", GTA_COMMIT, working, logs / "gta-source.log")
    deep = checkout("hsiangwei0903", "Deep-EIoU", DEEP_COMMIT, working, logs / "deep-source.log")
    checkpoint = verified(gta / "reid_checkpoints" / "sports_model.pth.tar-60", GTA_WEIGHT_HASH)
    env = {**os.environ, "MPLBACKEND": "Agg", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    env.pop("PYTHONPATH", None)
    uv = shutil.which("uv")
    if not uv:
        bootstrap = working / "bootstrap"
        recorded_command([sys.executable, "-m", "pip", "install", "--target", bootstrap, "uv==0.10.4"],
                         logs / "uv.log", env=env)
        uv = str(bootstrap / "bin" / "uv")
    deps_key = hashlib.sha256("\n".join(DEPENDENCIES).encode()).hexdigest()[:12]
    venv = working / f"legacy-cu118-{deps_key}"
    python = venv / "bin" / "python"
    ready = venv / ".packages-installed"
    if not ready.is_file():
        recorded_command([uv, "venv", "--python", "3.11", venv], logs / "python.log", env=env)
        pip = [uv, "pip", "install", "--python", python]
        recorded_command([*pip, "--index-url", "https://download.pytorch.org/whl/cu118",
                          "torch==2.7.1", "torchvision==0.22.1"], logs / "torch.log", env=env)
        recorded_command([*pip, *DEPENDENCIES], logs / "packages.log", env=env)
        recorded_command([*pip, "--no-build-isolation", "cython-bbox==0.1.5"],
                         logs / "cython-bbox.log", env=env)
        ready.touch()
    env["PYTHONPATH"] = str(gta / "reid")
    preflight = (
        "import torch,numpy,cv2,lap,cython_bbox,seaborn; "
        "from torchreid.utils import FeatureExtractor; "
        "assert numpy.__version__ == '1.23.5'; "
        "assert torch.cuda.is_available() and torch.version.cuda == '11.8'; "
        "print(torch.__version__, numpy.__version__, cv2.__version__, torch.cuda.device_count()); "
        "print([(torch.ones(2,device=f'cuda:{i}') + 1).tolist() for i in range(torch.cuda.device_count())])"
    )
    recorded_command([python, "-c", preflight], logs / "gpu-imports.log", env=env)
    provenance = "Pinned author repositories and SHA-256-verified sports OSNet; research comparison"
    shared = dict(python=str(python), rights_reviewed=True, provenance=provenance)
    return ({**shared, "repo": str(deep), "sports_checkpoint": str(checkpoint)},
            {**shared, "repo": str(gta), "parameters": GTA_SETTINGS}, env)


class TwoClipLaboratory(Laboratory):
    def audit_dataset(self, root):
        records = [probe_video(path) for path in root]
        table(self.run / "dataset" / "manifest.csv", records, tuple(records[0]))
        self.videos = records
        self.video_by_clip = {video["clip"]: video for video in records}
        self.selected = records
        self.log("dataset", "two_clips", "audit", "EXECUTED", f"{len(records)} original clips decoded and hashed")

    def run_selected(self):
        paths = selected_paths(self.config["dataset_root"])
        self.environment_preflight()
        self.preflight()
        if not any(self.readiness.values()):
            raise Blocked("Preflight blocked every requested detector and tracker.")
        self.audit_dataset(paths)
        self.core("detection", self.config["detectors"], self.detect)
        self.core("tracking", self.config["trackers"], self.track)
        self.compare_identity()
        self.report_other_tools()

    def compare_identity(self):
        try:
            self.hardware()
            for video in self.selected:
                self.cached("detections", self.config["track_detector"], video["clip"])
            if not self.config["auto_acquire"]:
                raise Blocked("Legacy setup requires Kaggle Internet to acquire two verified source revisions and dependencies.")
            setup = self.run / "environment" / "legacy_setup"
            deep, gta, env = install_legacy(self.workspace / "two_clip_sources", setup)
            self.config["deep_eiou_config"], self.config["gta_config"] = deep, gta
        except Exception as error:
            self.record_error("tracking", "deep_eiou", "advanced", error)
            self.log("identity", "gta_link", "advanced", "BLOCKED", "Deep-EIoU/OSNet setup is unavailable")
            return

        previous = {key: os.environ.get(key) for key in ("PYTHONPATH", "MPLBACKEND")}
        try:
            os.environ.update({key: env[key] for key in previous})
            self.deep_eiou()
            self.config["identity_tracker"] = "deep_eiou"
            self.advanced("identity", "gta_link")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def report_other_tools(self):
        self.log("detection", "gtatrack_competition_yolov11x", "pilot", "SKIPPED",
                 "This controlled comparison uses frozen RF-DETR, so it does not reproduce the competition detector.")
        if self.config["landmarks"]:
            self.advanced("calibration", "opencv_homography")
        else:
            self.log("calibration", "opencv_homography", "advanced", "SKIPPED",
                     "Provide reviewed PITCH_LANDMARKS for this optional calibration control.")
        self.log("calibration", "roboflow_sports_auto", "pilot", "SKIPPED",
                 "A licensed pitch keypoint model and independent pitch validation are not configured.")
        self.log("jersey", "koshkina_pipeline", "pilot", "SKIPPED",
                 "Requires its isolated original legibility, pose and SoccerNet PARSeq checkpoints.")
        if self.config["landmarks"]:
            self.render_pitch_previews()
        else:
            self.log("report", "mplsoccer", "pilot", "SKIPPED",
                     "Supply PITCH_LANDMARKS with pitch_size_m to render a preview.")
        self.log("report", "socceraction", "pilot", "SKIPPED", "Requires labeled soccer event streams.")


    def render_pitch_previews(self):
        setup = self.run / "environment" / "legacy_setup"
        setup.mkdir(parents=True, exist_ok=True)
        python = self.config.get("gta_config", {}).get("python")
        if not python:
            self.log("report", "mplsoccer", "pilot", "BLOCKED", "The optional plotting environment is unavailable.")
            return
        for video in self.selected:
            clip = video["clip"]
            try:
                mapping = self.run / "calibration" / "opencv_homography" / f"{clip}.json"
                if not mapping.is_file():
                    raise Blocked("No framewise homography output for this clip.")
                tracks = self.cached("tracking", "botsort_cmc", clip)
                dimensions = json_read(self.config["landmarks"])["clips"][clip].get("pitch_size_m")
                if not dimensions:
                    raise Blocked("Specify pitch_size_m [length, width] for this clip in the reviewed landmarks.")
                uv = shutil.which("uv") or str(self.workspace / "two_clip_sources" / "bootstrap" / "bin" / "uv")
                recorded_command([uv, "pip", "install", "--python", python, "mplsoccer==1.5.1"], setup / "mplsoccer.log")
                root = self.run / "calibration" / "pitch_previews" / clip
                tracks_info = json_read(tracks.with_suffix(".json"))
                request = {"map": str(mapping), "tracks": str(tracks), "pitch_size_m": dimensions,
                           "directory": str(root), "clip_sha256": tracks_info["clip_sha256"]}
                request_path = setup / f"{clip}.pitch.json"
                json_write(request_path, request)
                recorded_command([python, str(Path(__file__)), "--pitch-preview", request_path],
                                 setup / f"{clip}.pitch.log", env={**os.environ, "MPLBACKEND": "Agg"})
                self.log("report", "mplsoccer", "pilot", "EXECUTED", "Projected points and a top-down preview", clip=clip)
            except Exception as error:
                self.record_error("report", "mplsoccer", clip, error)


def render_pitch(request):
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from mplsoccer import Pitch

    from common import json_lines

    length, width = map(float, request["pitch_size_m"])
    if not all(np.isfinite((length, width))) or min(length, width) <= 0:
        raise ValueError("Pitch length and width must be positive finite meters.")
    source = json_read(request["map"])
    if source["clip_sha256"] != request["clip_sha256"]:
        raise ValueError("Calibration and tracking clips do not match.")
    matrices = {row["frame"]: np.asarray(row["H_image_to_pitch"], dtype=np.float64) for row in source["frames"]}
    rows = []
    for frame in json_lines(request["tracks"]):
        matrix = matrices.get(frame["frame"])
        tracks = frame["tracks"]
        if matrix is None or not tracks:
            continue
        points = [[(track["x1"] + track["x2"]) / 2, track["y2"]] for track in tracks]
        projected = cv2.perspectiveTransform(np.asarray(points, dtype=np.float64).reshape(-1, 1, 2), matrix)[:, 0]
        for track, (x, y) in zip(tracks, projected, strict=True):
            if np.isfinite(x) and np.isfinite(y):
                rows.append({"frame": frame["frame"], "track_id": track["track_id"], "x_m": float(x), "y_m": float(y),
                             "on_pitch": bool(0 <= x <= length and 0 <= y <= width)})
    if not rows:
        raise Blocked("No valid calibrated frame and track observations overlap.")
    directory = Path(request["directory"])
    table(directory / "positions.csv", rows, tuple(rows[0]))
    frame = rows[0]["frame"]
    visible = [row for row in rows if row["frame"] == frame and row["on_pitch"]]
    pitch = Pitch(pitch_type="custom", pitch_length=length, pitch_width=width, pitch_color="#173a2a",
                  line_color="#e1efe4")
    fig, ax = pitch.draw(figsize=(8, 5))
    pitch.scatter([row["x_m"] for row in visible], [row["y_m"] for row in visible],
                  ax=ax, s=40, color="#ffc857")
    ax.set_title(f"Frame {frame} · visual calibration check")
    fig.savefig(directory / "pitch_preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def launch(root, config):
    validate(config)
    if config["mode"] != "BENCHMARK" or config["max_videos"] != 2:
        raise ValueError("This pilot runs BENCHMARK on two named clips. Use the main lab for other modes.")
    directory = new_run(root)
    lab = TwoClipLaboratory(root, config, directory)
    try:
        lab.run_selected()
    except Exception as error:
        lab.record_error("run", "orchestrator", "execution", error)
    finally:
        lab.reports()
        result = json_read(directory / "RUN_RESULT.json")
        release = export_zip(directory, root / "export" / "scoutra_two_clip_results.zip")
        json_write(root / "latest_run.json", {**result, "path": str(directory), "zip": str(release)})
    return result["exit_code"]


if __name__ == "__main__":
    if sys.argv[1] == "--pitch-preview":
        render_pitch(json_read(sys.argv[2]))
        raise SystemExit(0)
    configuration = json_read(sys.argv[2])
    configuration["embedded_source_sha256"] = sys.argv[3]
    raise SystemExit(launch(Path(sys.argv[1]), configuration))
