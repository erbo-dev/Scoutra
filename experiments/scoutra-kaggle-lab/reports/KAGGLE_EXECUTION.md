# Manual Kaggle execution

**No Kaggle session/API/GPU is connected here. Do not execute models in this cloud.**

1. In Kaggle create a private Notebook and add the existing `erbosynshaimardan/zhanto-highlights` Dataset through **Add Input**. No re-upload of the twelve videos is needed if it is accessible to your account. The supplied candidate root is `/kaggle/input/schemas/erbosynshaimardan/zhanto-highlights/`; confirm the actual mount in the notebook. An older-style `/kaggle/input/zhanto-highlights/` mount is possible, but is not assumed.
2. Download the research branch and upload **only** its `experiments/scoutra-kaggle-lab/` folder as a separate private Kaggle Dataset. Keep all subfolders. Import `01_dataset_audit.ipynb` as a Kaggle Notebook and attach the lab Dataset as input. Its bootstrap discovers the lab from `scoutra_lab/common.py`; explicitly set `LAB_ROOT` if more than one copy is attached. No generated ZIP or runtime outputs are committed. The folder name is flexible because bootstrap finds `scoutra_lab/common.py`.
3. Leave accelerator off for the audit. Leave `INSTALL=False` if Kaggle already supplies the base utilities; otherwise install only `base.txt`. Set `RUN_AUDIT=True` after checking `DATASET_ROOT`. This probes metadata and computes video SHA256; it neither labels footage nor runs models.
4. Download `/kaggle/working/scoutra/dataset/manifest.csv`. Fill actual `source_match`, camera/challenge, selected-player field and split after watching the clips. Do not split filenames randomly. If match provenance is unknown, retain `unassigned`; inference can be exploratory, but held-out claims are blocked. Upload the reviewed manifest as part of a separate annotations Dataset; copy it to the working manifest path.
5. Explicitly choose a 20–60 s interval supported by the audited duration. Review all scene cuts in this interval. Set `segments`, scene cut **source frame indices**, `scene_boundaries_reviewed=True`; set `RUN_CACHE=True`. Do not cache all videos. Use a new cache folder for different selections. Cache writes dense native PNG frames; check free disk space. Highlight replays/cuts are not offscreen recovery events.
6. Generate the lightweight annotation HTML on sampled frames for detection; for MOT choose a small continuous interval and label every frame. Download HTML, draw/review boxes, export `detections.json`. For collaborative/large labeling, import the same frame files into CVAT and convert its export to the documented schema; no interpolation is automatically accepted as reviewed GT. Create `selected_players.json` anchors and crop labels. Upload all labels as a private Kaggle Dataset. Copy to `/kaggle/working/scoutra/annotations/`. The lab has no supplied annotations.
7. Obtain only the chosen initial checkpoint from the inventory's exact official source after reviewing terms. Store as a separate private weights Dataset along with source URL/revision, license text and hashes. Preserve any required backbone/config/tokenizer files. Do not download every candidate at once. Fill local paths and expected SHA256 in a copied method YAML; `license_reviewed` starts false. Dataset rights and model/code grants are separate.
8. Open notebook 02 in a **fresh** Kaggle GPU session; choose **GPU T4 ×2** if available. Install the selected detector profile only, run `pip check`, restart kernel if required by Kaggle, rerun bootstrap, save `pip freeze`/device metadata. A second GPU does not enlarge `cuda:0` memory. Never proceed if installation or the asset/class-map check fails. Exact Scoutra model is absent: mark that baseline unavailable rather than substitute generic weights under its name.
9. Set a unique experiment ID, `RUN_INFERENCE=True` only for one selected method. Observe device inventory and do a short checkpoint/output-contract validation before the full Stage 1 interval. Save failures verbatim. Model loading is excluded from the main timing; cold-start timings need a separate recorded measurement.
10. Repeat detector methods using the same cache/GT/evaluation settings. Select/freeze a detection cache on development evidence only. Attach its output Dataset to tracker sessions, preserving frame PNGs, `frames.json` and detection hashes. Copy all artifacts to the **same documented working paths** or regenerate/remap frames.json with new paths and a new manifest hash; its paths are absolute. Do not edit pixel data silently.
11. Execute 03, 04, then 05, one method/config per new ID. No notebook advances stages on its own. For repeated sessions, reattach and restore input caches, detections, annotations and earlier results. Kaggle input is read-only; all new outputs belong under `/kaggle/working/scoutra/`.
12. Save a notebook version and download the full working result bundle. Preserve `results/`, configs, environment freeze, dataset manifest/annotations, frame/crop cache hashes and source lock. For a new run, change its experiment ID; old results cannot be overwritten. For an interrupted run, mark FAILED and use a new run ID; Stage 1 does not pretend to support resumable inference.

## Uploading a future curated benchmark
```
scoutra-benchmark-v1/
  videos/                 # original clips; unique filename stems
  annotations/
    selected_players.json # human anchor IDs; no inferred labels
    detections.json       # native boxes + reviewed completeness + physical IDs
    events.json           # human event/actor/timing labels
  metadata/
    manifest.csv          # one row per actual clip
```
Create a private Kaggle Dataset from this folder; attach it, set `dataset_root` to its observed mount and `manifest` to the reviewed manifest. `relative_path` is relative to this root (e.g. `videos/actual-file.mp4`). Existing flat highlights are supported by recursive discovery without reorganizing their Dataset.

## Expected result tree
```
results/experiment_id/
  config.yaml
  metadata.json       # environment, input hashes, configuration provenance
  metrics.json        # null when not executed/evaluable
  predictions.parquet # typed empty container until actual predictions exist
  runtime.json        # scope, wall time, selected-device allocated/reserved peaks
  visualizations/
  errors.json
  report.md
```
Re-ID additionally saves features, crop order and optional distance arrays; AgentCoach saves cluster membership. No notebook contains historical benchmark outputs.
