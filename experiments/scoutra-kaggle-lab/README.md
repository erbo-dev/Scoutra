# Scoutra experimental laboratory — Phase 2

Prepared 17 November 2026. **All football/model experiments: NOT EXECUTED. No benchmark winner is known.**

This isolated research package prepares Kaggle experiments; it does not alter Scoutra. The accessible Scoutra material and prior audit did not supply an ML backend, trained RF-DETR checkpoint, videos or reviewed labels. The user has now supplied the Kaggle `erbosynshaimardan/zhanto-highlights` dataset location and twelve filenames; the current cloud has not opened those videos. Their FPS, duration, resolution, source-match grouping and annotation status are unverified.

Start with [Kaggle instructions](reports/KAGGLE_EXECUTION.md), [data/annotations](reports/DATASET_AND_ANNOTATIONS.md), [model inventory](reports/MODEL_INVENTORY.md), and [experiment plan](reports/EXPERIMENT_PLAN.md). [Validation status](reports/VALIDATION.md) distinguishes CPU infrastructure checks from model execution. [Risks](reports/RISKS_AND_BLOCKERS.md) lists incomplete bridges and assets.

## Execution order
1. `notebooks/01_dataset_audit.ipynb`: CPU inventory → human source-match/scene-cut review → explicit short dense frame cache → annotations.
2. `02_detector_benchmark.ipynb`: one detector/config per fresh environment → canonical detections → quality/runtime when labels exist.
3. Freeze one low-threshold detection cache. `03_tracker_benchmark.ipynb`: local trackers on exactly that cache → tracklets, supplemental identity metrics, official MOT export when dense GT exists.
4. `04_reid_benchmark.ipynb`: one shared reviewed crop manifest → OSNet; then independently provision PRTReID global/parts and BPBReID isolated upstream environments.
5. `05_identity_recovery.ipynb`: no-link baseline → upstream AgentCoach ReID-only consolidation → GTA splitting/linking and McByte++ identity memory only after bridge validation.

Every install/run flag is false. Paths are editable. An inference runner requires explicit execution opt-in plus actual Kaggle runtime/CUDA checks. Both T4s are separate 16-GB devices. Nothing automatically downloads model weights or runs a whole uploaded dataset.

## What's implemented here
Shared schemas, CPU inventory/native frame caching, human annotation export, thin detection/tracking/OSNet/AgentCoach wrappers, supplemental evaluation, official MOT export, overlays, immutable run records, five unexecuted notebooks, configs and research protocols. Upstream neural/research algorithms are not reimplemented. External methods without compatible environment/assets have explicit blockers and export contracts.

Profiles pin direct dependencies but are **not verified transitive lockfiles**. After a fresh Kaggle install, run `pip check`, record `pip freeze`, and preserve the image/Python/CUDA metadata. Model adapters have undergone static/API inspection only; real checkpoint smoke tests remain unexecuted. See the inventory for exact provenance.

## Files
`configs/` method/profile templates; `schemas/` empty annotation/schema templates only; no datasets or video inventory; `adapters/` thin interfaces; `scoutra_lab/` common I/O, caching and runners; `evaluation/` metrics/MOT export; `results/` unexecuted status records; `reports/` protocols and prior audit; `tests/` synthetic CPU checks. Empty annotation lists are templates, never ground truth.

Run CPU infrastructure checks (not model experiments):
```bash
python -m unittest discover -s tests -v
```
Base serialization dependencies are required. No production checkout or external code is copied into this package.

## GitHub delivery scope
Only research source, five unexecuted notebooks, configuration/dependency templates, blank schemas, tests and documentation are versioned. The original 202-file workspace included generated empty Parquet/result containers; these are excluded, along with the old transfer archive and supplied video-filename list. Blank templates were moved to `schemas/` so no dataset directory is committed. Actual videos, annotations, weights and runtime artifacts must be attached separately in Kaggle.

After downloading this repository, use **only** `experiments/scoutra-kaggle-lab/` as the laboratory folder when creating the Kaggle lab Dataset. Bootstrap locates `scoutra_lab/common.py`, so the outer folder name can differ from the original archive. No production application files are changed.

Read the post-delivery [readiness audit](reports/READINESS_AUDIT.md) before running notebooks. It documents path overrides, freshly checked checkpoint endpoints, unverified Kaggle dependencies and incomplete advanced-method bridges.
