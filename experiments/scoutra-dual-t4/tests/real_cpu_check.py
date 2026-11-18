"""Run the actual pretrained benchmark on CPU in a validation environment.

Use a Linux CPython 3.13.16 host with the CPU Torch/TorchVision pair and cv2.
The Kaggle notebook has no CPU detector option.
"""
import argparse
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from common import digest, json_read
from lab import Laboratory, launch
from notebook_config import CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-blocked-model", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    detectors = ["rfdetr_small"]
    if args.include_blocked_model:
        detectors.append("scoutra_rfdetr")
    config = {**CONFIG, "mode": "BENCHMARK", "auto_acquire": False, "wheelhouse": str(args.wheels.resolve()),
              "asset_cache": str(args.assets.resolve()), "dataset_root": str(args.videos.resolve()),
              "scaling_models": [], "batch": 2, "smoke_frames": 1, "detectors": detectors}

    def cpu_profile(lab, name):
        lab.profile(name)

    # Only the hardware gate changes in this test. These are two actual CPU workers
    # with CUDA metrics kept empty, calling the same shipped RF-DETR and tracker code.
    with patch.object(Laboratory, "hardware", return_value=["cpu:0", "cpu:1"]), \
            patch.object(Laboratory, "cuda_profile", cpu_profile):
        release = launch(output, config)

    pointer = json_read(output / "latest_run.json")
    run = Path(pointer["path"])
    expected_state = "BLOCKED" if args.include_blocked_model else "EXECUTED"
    assert pointer["state"] == expected_state
    assert pointer["cuda_detection_frames"] == 0
    assert pointer["successful_detection_jobs"] == 14
    with (run / "dataset" / "manifest.csv").open() as handle:
        frame_counts = {row["clip"]: int(row["frames"]) for row in csv.DictReader(handle)}
    assert pointer["detection_frames"] == sum(frame_counts.values()) + 2
    for tracker in ("bytetrack", "botsort_cmc"):
        tracks = list((run / "tracking" / tracker / "benchmark" / "tracks").glob("*.jsonl"))
        assert len(tracks) == 12
        for path in tracks:
            meta = json_read(path.with_suffix(".json"))
            predictions = run / "detections" / "rfdetr_small" / "benchmark" / "predictions" / path.name
            assert meta["detection_sha256"] == digest(predictions)
    with (run / "benchmarks" / "device_frames.csv").open() as handle:
        smoke = [row for row in csv.DictReader(handle) if row["method"] == "rfdetr_small" and row["phase"] == "smoke"]
    assert sorted((r["device"], int(r["frames"])) for r in smoke) == [("cpu:0", 1), ("cpu:1", 1)]
    overlays = [p for p in run.rglob("*.mp4") if "/benchmark/" in str(p)]
    assert len(overlays) == 36
    for filename in overlays:
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                         "-show_entries", "stream=codec_name,nb_frames", "-of", "json", str(filename)])
        assert json.loads(probe)["streams"][0] == {"codec_name": "h264", "nb_frames": str(frame_counts[filename.stem])}
    with zipfile.ZipFile(release) as z:
        assert z.testzip() is None
        assert {"RUN_RESULT.json", "benchmarks/device_frames.csv"} <= set(z.namelist())
    print(f"Verified 12 CPU clips, {sum(frame_counts.values())} benchmark frames, shared detection hashes for both trackers, "
          f"36 H.264 benchmark overlays and a valid ZIP. Run state {expected_state}.", flush=True)


if __name__ == "__main__":
    main()
