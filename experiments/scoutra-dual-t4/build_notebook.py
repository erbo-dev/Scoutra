"""Build an independently runnable ipynb embedding audited, readable module sources."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT.parent / "Scoutra_All_Experiments_T4x2.ipynb"
MODULES = ("common", "backends", "environment", "jobs", "schedule", "evaluate", "advanced", "legacy", "upstream_tracking", "lab")
INTRO = """# Scoutra AI experiments on Kaggle 2×T4

Attach **zhanto-highlights**, select **GPU T4 x2**, turn on **Internet**, and choose **Run All**. The default `RUN_MODE="BENCHMARK"` checks prerequisites, runs a small RF-DETR Small smoke test on each physical GPU, and then automatically starts the full 12-clip benchmark. ByteTrack and BoT-SORT share the same frozen predictions. `RUN_MODE="SMOKE"` stops after the short test.

- **PREFLIGHT** checks CUDA in the isolated profile, verifies checkpoints, prepares dependencies, and audits videos when at least one method is ready.
- To run with Internet off, set `AUTO_ACQUIRE=False` and point `ASSET_CACHE` to an attached directory holding the verified `rf-detr-small.pth`. For a fresh session also point `WHEELHOUSE` to the dependency wheels. A valid profile and checkpoint already cached this session can be reused as is.
- **ADVANCED** runs selected optional methods. **REPORT** re-exports a saved run.
- If preflight or a requested method fails, the run saves its ZIP and reports **BLOCKED** or **FAILED**, and the execution cell raises an error. Open `RUN_RESULT.json`, `reports/failures.md` and `environment/network.json` in the ZIP.
- `benchmarks/device_frames.csv` shows the physical GPU UUID, worker PID and detector frames per device. GPU inference validation requires actual CUDA batches on every assigned device. The CPU validation in Codex Cloud does not establish Kaggle T4 performance.

For Python 3.13 setup, offline wheels, checkpoint URLs and the SHA-256, see the [launch and repair guide](https://github.com/erbo-dev/Scoutra/blob/research/scoutra-kaggle-runtime-repair/experiments/scoutra-dual-t4/README.md).
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
from IPython.display import FileLink, Markdown, display

latest = results / "latest_run.json"
previous = latest.read_bytes() if latest.is_file() else None
completed = subprocess.run(
    [sys.executable, "-u", str(runtime / "lab.py"), str(results), str(launch_config), SOURCE_ID],
    cwd=runtime,
)
if latest.is_file() and latest.read_bytes() != previous:
    result = json.loads(latest.read_text())
    link = Path(result["zip"])
    display(FileLink(str(link.relative_to(Path.cwd())) if link.is_relative_to(Path.cwd()) else str(link)))
    display(Markdown((Path(result["path"]) / "reports" / "summary.md").read_text()))
elif RUN_MODE == "REPORT" and completed.returncode == 0:
    link = results / "export" / "scoutra_experiments_results.zip"
    display(FileLink(str(link.relative_to(Path.cwd())) if link.is_relative_to(Path.cwd()) else str(link)))
if completed.returncode:
    state = {1: "FAILED", 2: "BLOCKED"}.get(completed.returncode, "FAILED")
    raise RuntimeError(f"Scoutra run {state} (exit {completed.returncode}). See the run log and the ZIP above if produced.")

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
        "language_info": {"name": "python", "version": "3.13.16"},
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
