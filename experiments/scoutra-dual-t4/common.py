"""Shared contracts. Importing this file never initializes CUDA."""
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path

CLIP_SUFFIXES = (5479, 5480, 5483, 5485, 5487, 5491, 5495, 5497, 5503, 5504, 5505, 5513)
NAMES = tuple(f"document_544086323054304{s}.mp4" for s in CLIP_SUFFIXES)
BOX = ("x1", "y1", "x2", "y2")
STATES = {"EXECUTED", "FAILED", "SKIPPED", "BLOCKED", "NOT_EVALUATED"}
GROUPS = {"detection", "tracking", "reid", "identity", "jersey", "calibration", "ball", "events", "vlm"}


class Blocked(RuntimeError):
    """An explicit prerequisite is missing."""


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as handle:
        value = hashlib.new(algorithm)
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
        return value.hexdigest()


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def json_write(path, data):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(path, records, fields):
    with atomic_path(path) as temporary, temporary.open("w", newline="", encoding="utf-8") as handle:
        out = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        out.writeheader()
        out.writerows(records)


def verified(path, sha256):
    path = Path(path)
    if not re.fullmatch(r"[a-f0-9]{64}", sha256 or ""):
        raise Blocked(f"Expected SHA-256 is required for {path.name}")
    if not path.is_file():
        raise Blocked(f"Required file missing: {path}")
    actual = digest(path)
    if actual != sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {sha256}, got {actual}")
    return path.resolve()


def fetch(url, path, sha256, online):
    path = Path(path)
    if not path.exists():
        if not online:
            raise Blocked(f"Missing verified asset {path.name}. Enable AUTO_ACQUIRE with Kaggle Internet, or provide the verified cache.")
        with atomic_path(path) as temporary:
            with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            verified(temporary, sha256)
    return verified(path, sha256)


def resolve_dataset(request, mount=Path("/kaggle/input")):
    mount = Path(mount)
    if request != "auto":
        root = Path(request).resolve()
        missing = [name for name in NAMES if not (root / name).is_file()]
        if missing:
            raise Blocked(f"Explicit DATASET_ROOT {root} is missing: {', '.join(missing)}")
        return root
    direct = mount / "datasets/erbosynshaimardan/zhanto-highlights"
    if all((direct / name).is_file() for name in NAMES):
        return direct
    candidates = {path.parent.resolve() for path in mount.rglob(NAMES[0])}
    complete = sorted(path for path in candidates if all((path / name).is_file() for name in NAMES))
    if len(complete) == 1:
        return complete[0]
    if complete:
        raise Blocked(f"Multiple complete datasets: {complete}. Set DATASET_ROOT explicitly.")
    partial = {str(path): [name for name in NAMES if not (path / name).is_file()] for path in sorted(candidates)}
    raise Blocked(f"No directory with all 12 exact clips in {mount}. Attach zhanto-highlights. Partial candidates: {partial}")


def canonical_box(values, size):
    w, h = size
    x1, y1, x2, y2 = map(float, values)
    if not all(map(math.isfinite, (x1, y1, x2, y2))):
        raise ValueError("Nonfinite bounding box")
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return dict(zip(BOX, (x1, y1, x2, y2)))


def valid_score(score):
    score = float(score)
    if not 0 <= score <= 1:
        raise ValueError(f"Invalid confidence {score}")
    return score


def json_lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def write_line(handle, row):
    handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def freeze_predictions(path, video):
    """Read consecutive, validated frames and verify physical clip provenance."""
    path = Path(path)
    meta = json_read(path.with_suffix(".json"))
    if meta["clip_sha256"] != video["sha256"] or meta["clip"] != video["clip"]:
        raise ValueError("Frozen detection cache belongs to another clip")
    if digest(path) != meta["prediction_sha256"]:
        raise ValueError("Frozen detection cache failed SHA-256 check")
    count = 0
    for count, row in enumerate(json_lines(path), 1):
        if row["frame"] != count:
            raise ValueError("Detection cache has missing or nonconsecutive frames")
        for index, box in enumerate(row["detections"]):
            if box["detection_id"] != f"{count}:{index}":
                raise ValueError("Detection provenance ID disagrees with row ordering")
            if box["class"] not in {"person", "player", "goalkeeper", "referee", "ball"}:
                raise ValueError("Unexpected canonical class in detection cache")
            valid_score(box["score"])
            coordinates = {key: box[key] for key in BOX}
            raw_id = box.get("raw_id")
            if canonical_box(coordinates.values(), (video["width"], video["height"])) != coordinates:
                raise ValueError("Invalid native-pixel box in frozen predictions")
            if type(raw_id) is not int:
                raise ValueError("Invalid raw class ID in frozen predictions")
        yield row
    if count != meta["frames"] or count < 1:
        raise ValueError("Detection cache is empty or row count disagrees with metadata")


def export_zip(run, output):
    run, output = Path(run), Path(output)
    with atomic_path(output) as temporary, zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(run.rglob("*")):
            relative = path.relative_to(run)
            if path.is_file() and not any(part.startswith(".") or part == "export" for part in relative.parts):
                archive.write(path, relative.as_posix())
    return output
