"""Generate the standalone Scoutra two-clip experiment from the shared lab."""
import argparse
import hashlib
import json

from build_notebook import NOTEBOOK as BASE_NOTEBOOK, cells


NOTEBOOK = BASE_NOTEBOOK.with_name("Scoutra_Two_Clip_Pilot_T4x2.ipynb")
INTRO = """# Scoutra · Two-clip pilot

Attach **zhanto-highlights**, choose **GPU T4 ×2**, enable **Internet**, and **Run All**. The pilot finds and decodes only the original clips ending in **5483** and **5485**. Keep the original resolution and use the same source frames for all methods.

1. RF-DETR Small and D-FINE Small both receive the same clips. Both GPU workers must pass the model-specific smoke test.
2. BoT-SORT with CMC and original Deep-EIoU with sports OSNet use the same **RF-DETR Small** cache.
3. GTA-Link applies independent offline split/connect to the Deep-EIoU output. The identity JSON records per-frame source-to-refined IDs. Compare the three tracking overlays.
4. Optional `PITCH_LANDMARKS` enables the OpenCV manual-correspondence control and a top-down mplsoccer preview when each clip entry also has a measured `pitch_size_m` pair. Automatic Roboflow pitch keypoints, Koshkina OCR and socceraction report explicit SKIPPED statuses.
5. `scoutra_two_clip_results.zip` includes predictions, MOT tracks, video overlays, statuses and GPU evidence. Failed or blocked methods keep their logs and any completed outputs. Metrics need reviewed annotations and otherwise remain NOT_EVALUATED.

This is a controlled test with frozen RF-DETR detections. It does not use the original competition YOLOv11x checkpoint. The legacy research stage uses a separate Python 3.11 / PyTorch 2.7.1 CUDA 11.8 environment and verifies the sports OSNet checkpoint hash. The selected Kaggle session can download packages and weights; original football frames stay in the notebook environment.
"""


def document():
    result = list(cells())
    result[0]["source"] = INTRO.splitlines(keepends=True)
    configuration = "".join(result[1]["source"])
    settings = {
        "MAX_VIDEOS": "2",
        "DETECTOR_METHODS": '["rfdetr_small", "dfine_small"]',
        "TRACKER_METHODS": '["botsort_cmc"]',
        "SCALING_MODELS": "[]",
    }
    config_lines = []
    for line in configuration.splitlines(keepends=True):
        name = line.split(" = ", 1)[0]
        config_lines.append(f"{name} = {settings[name]}\n" if name in settings else line)
    result[1]["source"] = config_lines
    source = BASE_NOTEBOOK.parent / "scoutra-dual-t4" / "two_clip_pilot.py"
    code = source.read_text(encoding="utf-8")
    assert "'''" not in code
    result.insert(-1, {"cell_type": "code", "id": "two-clip-pilot", "metadata": {},
                       "execution_count": None, "outputs": [],
                       "source": f"SOURCES['two_clip_pilot.py'] = r'''{code}'''\n".splitlines(keepends=True)})
    runner = "".join(result[-1]["source"])
    runner = runner.replace('WORK / "scoutra_experiments"', 'WORK / "scoutra_two_clip_pilot"')
    runner = runner.replace('str(runtime / "lab.py")', 'str(runtime / "two_clip_pilot.py")')
    result[-1]["source"] = runner.splitlines(keepends=True)
    for cell in result:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    return {"cells": result, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.16"},
    }, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(document(), ensure_ascii=False, indent=1) + "\n"
    if args.check:
        if content != NOTEBOOK.read_text(encoding="utf-8"):
            raise SystemExit("Two-clip notebook is stale.")
    else:
        NOTEBOOK.write_text(content, encoding="utf-8")
    print(NOTEBOOK.name, hashlib.sha256(content.encode()).hexdigest())
