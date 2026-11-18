"""Build an independently runnable ipynb embedding audited, readable module sources."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT.parent / "Scoutra_All_Experiments_T4x2.ipynb"
MODULES = ("common", "backends", "environment", "jobs", "schedule", "evaluate", "advanced", "legacy", "upstream_tracking", "lab")
INTRO = """# Scoutra AI experiments on Kaggle 2×T4

Attach the **zhanto-highlights** Kaggle Dataset and pick **GPU T4 x2**. To acquire the first detector and packages, enable **Kaggle Internet** and set `AUTO_ACQUIRE=True`. Run All with `RUN_MODE="SMOKE"`. The default enabled methods are RF-DETR Small, ByteTrack and BoT-SORT with camera compensation. Smoke runs the detector on up to eight real frames per worker and reports the actual number decoded.

After the smoke, set `RUN_MODE="BENCHMARK"`. For the full COCO comparison set `DETECTOR_METHODS = ["rfdetr_small", "rfdetr_medium", "dfine_small", "rtdetr_r18vd", "rtdetrv2_r18vd"]`. Each enabled detector runs a per-device smoke before full clips. ByteTrack and BoT-SORT use one shared, SHA-256-verified detector cache. `SCALING_MODELS` selects the detectors for an equivalent one-GPU run. Remove a model from the list to save Kaggle time.

- **PREFLIGHT** decodes and hashes all twelve exact clips and checks CUDA and local assets.
- **SMOKE** runs selected checkpoints on short real frame sequences.
- **BENCHMARK** runs consecutive frames of the selected clips, with predictions, overlays and measured wall times.
- **ADVANCED** runs only named methods. Use `CACHE_RUN` for a previous run ID or extracted result directory.
- **REPORT** regenerates the download ZIP from a saved run without calling any model.

First-run installation and weights require opt-in. For a repeat in the same session, the verified cache and isolated environments can be reused with `AUTO_ACQUIRE=False`. Set `ALLOW_SINGLE_GPU=True` explicitly for one GPU. Prediction rows and track IDs are clip-local. Generic COCO detectors expose **person** and **ball**, without football role predictions. CUDA status, method-level results and any accuracy labeled `NOT_EVALUATED` are in the report. Codex Cloud did not run Kaggle T4 inference.
"""
WORKER_CELLS = """## Embedded implementation

Run the following cells as part of **Run All**. They create plain Python files under the notebook's working directory so CUDA workers can start in independent Python processes. No second source Dataset or GitHub checkout is required.
"""
RUN = r"""# Materialize exactly the embedded files, then start the selected run.
import hashlib
import json
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
if not WORK.is_dir():
    raise RuntimeError("Run this notebook in Kaggle. Codex Cloud CPU tests use the source modules separately.")

SOURCE_ID = hashlib.sha256(
    json.dumps(SOURCES, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
runtime = WORK / ".scoutra_notebook" / SOURCE_ID
runtime.mkdir(parents=True, exist_ok=True)
for filename, code in SOURCES.items():
    compile(code, filename, "exec")
    target = runtime / filename
    target.write_text(code, encoding="utf-8")
results = WORK / "scoutra_experiments"
launch_config = runtime / "launch.json"
launch_config.write_text(json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8")
subprocess.run([sys.executable, "-u", str(runtime / "lab.py"), str(results), str(launch_config), SOURCE_ID],
               cwd=runtime, check=True)
"""
REPORT = """# Open the Kaggle output files for this run.
from IPython.display import FileLink, Markdown, display

output = Path("/kaggle/working/scoutra_experiments")
link = output / "export" / "scoutra_experiments_results.zip"
if link.is_file():
    display(FileLink(str(link.relative_to(Path.cwd())) if link.is_relative_to(Path.cwd()) else str(link)))
latest = output / "latest_run.json"
if latest.is_file() and RUN_MODE != "REPORT":
    selected_run = Path(json.loads(latest.read_text())["path"])
    display(Markdown((selected_run / "reports" / "summary.md").read_text()))
"""


def cells():
    def cell(kind, slug, source):
        data = {"cell_type": kind, "id": slug, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            data.update(execution_count=None, outputs=[])
        return data

    yield cell("markdown", "read-first", INTRO)
    yield cell("code", "configuration", (ROOT / "notebook_config.py").read_text(encoding="utf-8"))
    yield cell("markdown", "embedded-source", WORKER_CELLS)
    yield cell("code", "source-registry", "SOURCES = {}\n")
    for name in MODULES:
        source = (ROOT / f"{name}.py").read_text(encoding="utf-8")
        compile(source, f"{name}.py", "exec")
        if "'''" in source:
            raise ValueError(f"Raw embedded source delimiter occurs in {name}")
        yield cell("code", f"module-{name}", f"SOURCES[{name + '.py'!r}] = r'''{source}'''\n")
    yield cell("code", "execute-laboratory", RUN)
    yield cell("code", "download-results", REPORT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = list(cells())
    for cell in content:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    notebook = {"cells": content, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }, "nbformat": 4, "nbformat_minor": 5}
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    if args.check:
        if NOTEBOOK.read_text(encoding="utf-8") != text:
            raise SystemExit("Notebook is stale. Run build_notebook.py.")
    else:
        NOTEBOOK.write_text(text, encoding="utf-8")
    print(NOTEBOOK.name, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
