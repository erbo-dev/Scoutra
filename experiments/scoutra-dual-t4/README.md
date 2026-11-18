# Scoutra dual-T4 research laboratory

[Download the standalone notebook](../Scoutra_All_Experiments_T4x2.ipynb). The Python files in this directory are the readable source for the embedded cells. Kaggle needs the notebook and its video Dataset. The notebook materializes its own modules and records a source fingerprint, an environment manifest and a per-run configuration.

## Run on Kaggle

1. Download `experiments/Scoutra_All_Experiments_T4x2.ipynb` using **Download raw file**. Import it into a fresh Kaggle notebook and attach `erbosynshaimardan/zhanto-highlights`.
2. Select **GPU T4 x2** and enable **Internet**. Leave `AUTO_ACQUIRE=True` and `RUN_MODE="BENCHMARK"`, then select **Run All**. The release checks dependencies, checks the physical GPUs in both the Kaggle and isolated worker environments, and verifies the checkpoint hash. It runs up to eight smoke frames on each GPU. A passing detector smoke automatically starts all 12 clips.
3. Look for `RUN_RESULT=EXECUTED` and positive `cuda_frames` from two distinct UUIDs and PIDs in `benchmarks/device_frames.csv` for `rfdetr_small` in both the `smoke` and `benchmark` phases. ByteTrack and BoT-SORT then consume the same verified detection files. `SCALING_MODELS` defaults to RF-DETR Small for the matched one-GPU workload. Set `RUN_MODE="SMOKE"` for only the short test or choose `"PREFLIGHT"` to prepare and check inputs without inference.
4. Download `/kaggle/working/scoutra_experiments/export/scoutra_experiments_results.zip`. The usual output includes detections, MOT tracks, H.264 overlays, runtime, GPU telemetry when available, and logs. A blocked run prints `RUN_RESULT=BLOCKED`, exits 2, and provides the ZIP before raising a notebook error. A failed run exits 1. Independent methods can still finish and their artifacts remain in the ZIP. Open `RUN_RESULT.json`, `reports/failures.md` and `environment/network.json` for the exact cause.

The default paths support standard Linux Python 3.11–3.13. The repair was tested with CPython **3.13.16**. Dependencies install into an isolated virtual environment through host `pip --python` and use the host Torch/TorchVision pair. The preflight checks `pip check`, worker imports and CUDA execution. `lap==0.5.13` has a Linux CPython 3.13 wheel. Optional BoxMOT 16.0.10 for StrongSORT/OSNet still requires Python 3.11–3.12 in this notebook. `ffmpeg` with `libx264` is required for overlays. [pip target interpreter docs](https://pip.pypa.io/en/stable/topics/python-option/) · [LAP wheels](https://pypi.org/project/lap/0.5.13/) · [BoxMOT Python requirement](https://pypi.org/project/boxmot/16.0.10/)

### Offline first run

Attach a separate Kaggle Dataset with these two folders, then set `AUTO_ACQUIRE=False`, `ASSET_CACHE="/kaggle/input/<offline-dataset>/assets"` and `WHEELHOUSE="/kaggle/input/<offline-dataset>/wheels"`. Both directories are read-only inputs. A same-session validated profile and checkpoint can also be reused without Internet.

- Put the official RF-DETR Small COCO checkpoint in `assets/rf-detr-small.pth`. Its SHA-256 is `d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33`. The exact upstream download is Apache-2.0 and 386,045,550 bytes. [Roboflow download](https://repo.roboflow.com/rfdetr/small_coco/checkpoint_best_regular.pth) · [GCS source](https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth) · [RF-DETR licensing](https://rfdetr.roboflow.com/latest/learn/pretrained/)
- On a connected **Linux x86-64 CPython 3.13** environment, prepare the tested default package set with `python -m pip download --only-binary=:all: --no-deps -r offline-rfdetr-tracking-cp313.txt -d wheels`. Use the [wheel lock](offline-rfdetr-tracking-cp313.txt) next to this guide. The notebook installs from that folder with `--no-index`. The tested wheel set uses the Kaggle image's Torch/CUDA runtime and contains the Torch Python dependencies required by the tested Torch build. If the target CUDA image requires another version of a Torch dependency, add its compatible wheel as directed by `environment/installation_log.txt`. Extra model profiles require their own dependency wheels and checkpoint assets.

The session stores each run under `runs/<run_id>`. `REPORT` re-exports a saved run, optionally chosen via `REPORT_RUN`. To share a detection cache across sessions, attach an extracted run ZIP and set `CACHE_RUN` to the folder with its `configuration.json`.

## Method support and provenance

| Method | Implementation | Inputs or current blocker |
|---|---|---|
| RF-DETR Small and Medium | Roboflow 1.12.1, source `f29e2a8bbb7cdcb8920bf74a335062f88a35046d`; exact local SHA-256, sparse class IDs 1 and 37 | Official Apache-2.0 COCO weights fetched with explicit notebook opt-in. Both models returned predictions in CPU smoke tests. |
| D-FINE Small, RT-DETR R18 and RT-DETRv2 R18 | Transformers 4.57.6, source `753d61104116eefc8ffc977327b441ee0c8d599f`; snapshot commit and all three file hashes checked | Apache-2.0 COCO weights. These exports use class IDs 0 and 32. All three models returned predictions in CPU smoke tests. |
| Scoutra RF-DETR | RF-DETR checkpoint loader and architecture, resolution and label-space validation | Supply the actual `best_model.pth` with SHA-256, exact architecture/config, training class map and provenance. Optional `trust_checkpoint` applies only to a known local legacy PyTorch checkpoint. |
| ByteTrack, BoT-SORT with CMC | Ultralytics 8.3.235 at `4b884f4a44685f6ea45f4676c1bca50a2d72d443`, AGPL-3.0 | Same native frozen detection stream. BoT-SORT uses sparse optical-flow motion compensation and appearance matching is off. |
| StrongSORT and OSNet | BoxMOT 16.0.10 at `c877349157a35df6095e703e002c60418d2b34ab`, AGPL-3.0 | Supply reviewed `osnet_x0_25_msmt17.pt`. SHA-256 `6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99`. Inference layers are verified against the weights. The file is MSMT17-pretrained, a control for football transfer. |
| PRTReID Global, PRTReID Parts, BPBReID Parts | Separate process with the respective upstream FeatureExtractor, exact tensor coverage and full part outputs | Requires a compatible pre-provisioned environment, training config, identity count, Re-ID checkpoint, HRNet asset and source/data/weight rights review. GPU execution pending. |
| No-link, AgentCoach ReID-only, cosine tracklet link | Clip-local controls, with the original AgentCoach consolidator pinned at `eef7b0c5d757844fbbb1eeafa6a8050a863a6623` | Frozen tracks. Linkers require global embeddings fingerprinted to those same tracks. AgentCoach runs its association code with OCR/team evidence omitted. |
| GTA-Link | Original split, distance, spatial-constraint and merge functions in a separate interpreter, original sports OSNet weights | Requires the pinned upstream checkout, compatible legacy interpreter and asset rights. Frozen tracks are converted to upstream Tracklet objects; per-observation splits, dropped observations and links are exported. GPU execution pending. |
| OpenCV homography, temporal jersey voting, temporal ball baseline | Independent methods | Supply framewise pitch correspondences, frame-level OCR candidates or a frozen detector cache respectively. |
| Deep-EIoU | Author's tracker at `2fc7d270eec8290fda075b976b5e06683c788e1e` with verified GTA-packaged sports OSNet x1.0 and frozen detections | Requires rights-reviewed checkout and isolated NumPy <1.24, LAP/Cython and CUDA runtime; method not GPU-tested. |
| McByte++, McByte++ bank | Dependency and interface investigation | Needs the original EdgeTAM masks, exact sports Re-ID asset, frozen-detection bridge and bank telemetry in its separate environment. |
| GTATrack, Cross-Multiplication Matrix Re-ID, specialized soccer-ball checkpoint, PARSeq, Koshkina, TVCalib, PnLCalib, BallRadar, PathCRF, T-DEED, OpenSportsLib, FOOTPASS, Qwen3-VL, Qwen3.5, SoccerChat and football Qwen | Explicit blockers in the method registry | Method-specific checkpoint, preprocessing, license or isolated runtime work is pending. Select a candidate in `ADVANCED_METHODS` to record its blocker in the Kaggle run. |

References for source and licenses are the [Roboflow package and weight documentation](https://github.com/roboflow/rf-detr/tree/f29e2a8bbb7cdcb8920bf74a335062f88a35046d), [D-FINE Small model card](https://huggingface.co/ustc-community/dfine-small-coco/tree/f79e65b5fbb33ceb9d3ebba042955d7410c608f8), [RT-DETR model card](https://huggingface.co/PekingU/rtdetr_r18vd/tree/ac77a11ff0170a41b771c03264987f8ce2b0d753), [RT-DETRv2 model card](https://huggingface.co/PekingU/rtdetr_v2_r18vd/tree/5650961749fa93567c0d46fc7f43ea4f9e914107), [Ultralytics tracker source](https://github.com/ultralytics/ultralytics/tree/4b884f4a44685f6ea45f4676c1bca50a2d72d443/ultralytics/trackers), [BoxMOT source](https://github.com/mikel-brostrom/boxmot/tree/c877349157a35df6095e703e002c60418d2b34ab) and [AgentCoach](https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/LICENSE). The repository contains no external weights, data, vendored tracker or GPU outputs.

## Advanced launch

- Frozen tracker ablations work with `RUN_MODE="BENCHMARK"`, `MODEL_GROUPS=["tracking"]`, a verified benchmark `CACHE_RUN` and any selected core tracker. `strongsort` additionally requires `OSNET_CHECKPOINT = {"path": ".../osnet_x0_25_msmt17.pt", "sha256": "...", "provenance": "..."}`. Use the checksum above and record the checkpoint rights and provenance. The registered weight URL is in the [BoxMOT checkpoint registry](https://github.com/mikel-brostrom/boxmot/blob/c877349157a35df6095e703e002c60418d2b34ab/boxmot/reid/core/config.py).
- For Re-ID run `ADVANCED_METHODS=["prepare_crops", "osnet"]` with a benchmark tracking cache and that checkpoint. The same ordered manifest is reused across models in one invocation. To reuse crop pixels in another run, set `CROP_MANIFEST` to the exported CSV alongside its `images/`. Minimum columns are `crop_id,path,sha256`. Linking additionally needs `clip,frame,track_id,tracking_sha256` for exact observation provenance. An external human-reviewed query/gallery file is optional.
- For parts set `LEGACY_REID["prtreid"]` or `LEGACY_REID["torchreid"]` to a dict with absolute `python`, clean `repo`, the `checkpoint`, fully resolved YAML `config`, HRNet `backbone` and matching `checkpoint_sha256`, `config_sha256`, `backbone_sha256`, training `num_classes`, `rights_reviewed=True`, and `provenance`. Use separate Python environments and the configs that produced the checkpoint. See [PRTReID](https://github.com/VlSomers/prtreid/tree/30617a75967e84d5d516959c4b84cbeea6f56493), [BPBReID](https://github.com/VlSomers/bpbreid/tree/a2dc4304284784d8b2c061764512f0698aa38c21) and the [previous license and environment audit](../scoutra-kaggle-lab/reports/READINESS_AUDIT.md). Global and full-part modes are separate. The exported tensors include raw part embeddings, visibility values and spatial part masks. The notebook's joint-visible-part cosine comparison is its own declared analysis rule.
- For GTA-Link set `GTA_CONFIG` with a clean [GTA-Link checkout at e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca](https://github.com/sjc042/gta-link/tree/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca), a separate executable `python`, `rights_reviewed=True`, and `provenance`. Install the upstream environment with `torchreid`, original numerical packages and CUDA, and preserve the original `reid_checkpoints/sports_model.pth.tar-60` in the checkout. Default split/merge thresholds come from that commit and are recorded. Review `min_len=100` against short tracklets. The adapter has yet to run with the isolated CUDA environment.
- For Deep-EIoU set `ADVANCED_METHODS=["deep_eiou"]`, select the frozen detector cache and configure `DEEP_EIOU_CONFIG` with absolute `python`, rights-reviewed `repo`, `sports_checkpoint`, `rights_reviewed=True` and `provenance`. The sports model hash is the one from the pinned GTA-Link checkout. The original [Deep-EIoU tracker](https://github.com/hsiangwei0903/Deep-EIoU/tree/2fc7d270eec8290fda075b976b5e06683c788e1e) uses legacy NumPy aliases and compiled LAP/Cython. Supply its own environment with NumPy 1.23.x, matching cv2/SciPy, torchreid and CUDA.
- For identity use `ADVANCED_METHODS=["no_link", "agentcoach_reid_only", "cosine_tracklet_link"]`, a tracking `CACHE_RUN` and an `IDENTITY_FEATURES` global NPZ with sidecar. `IDENTITY_SWEEP` runs the cosine linker across additional thresholds with the same tracks and embeddings. The identity methods run offline within each clip.
- Calibration accepts a reviewed JSON with `clips[clip].clip_sha256` and `frames=[{"frame":1,"pixel_xy":[...],"pitch_xy_m":[...]}]`. Supply at least four non-collinear points per frame. Output fit error applies to the correspondence set.
- Jersey voting accepts saved JSON predictions `[{"clip":"...","track_id":1,"digits":"12","confidence":0.8}]`. Ball temporal tracking accepts a detector cache and uses consecutive frames, confidence and a constant-velocity pixel gate.

## Measurements

- The scheduler seeds one job per GPU and pulls remaining clips from a longest-first queue. It masks each spawned process by physical UUID before Torch initializes and verifies that the CUDA runtime sees that same GPU. Logical Kaggle device `cuda:1` therefore has worker-local name `cuda:0` on the second worker. Each process loads its own checkpoint once per detector schedule. It reports PIDs, physical UUIDs, CUDA batches and per-device frame counts in `benchmarks/device_frames.csv` and runtime and NVML samples through `nvidia-smi` when present.
- End-to-end wall time includes worker start, model load, decoding, inference and enabled overlays. Inference includes preprocessing and postprocessing but excludes video decode and overlay encoding. The comparison runs dual then single, with the same video fingerprints, scores, batching, precision and overlay settings. Startup, OS cache and thermals can affect a single comparison. No speedup is recorded when coverage differs. Device-wall GPU-hours count planned device allocation during the timed interval.
- Each detection JSONL has one row per decoded source frame, including empty detections. Native coordinates and confidence accompany canonical and raw labels. MOT rows use one-based frames and top-left coordinates with 10 fields. The tracking sidecar records the input prediction fingerprint. Overlay H.264 files use decoded frame order at nominal FPS, with no audio. Preserve original files for variable-rate timing analysis.
- Paths inside the ZIP include `configuration.json`, `environment/`, `dataset/manifest.csv`, `detections/<method>/<phase>/`, `tracking/<method>/<phase>/`, optional `reid/` and `identity/`, plus `benchmarks/` and `reports/`. Only executed stages create result payloads. `runtime.csv` and `gpu_usage.csv` are created when runtime rows and NVML samples exist.

The twelve highlights support local tracking, ball detection and small re-entry tests. They do not establish 90-minute continuity, cross-camera association, speed without calibration, or a final technology choice.

## Human-reviewed annotation workflow

1. Mark shot cuts, replays, physical match IDs and a match-disjoint validation/test split. Preserve the original video SHA-256 and native dimensions. Keep thresholds selected on validation data, with test clips untouched until evaluation.
2. For detection, annotate an exhaustively reviewed frame subset including empty frames. Mark every visible soccer person (player, goalie, referee and other in-scope people) and visible ball in native `xyxy`. Decide how sidelines and tiny balls are labeled before annotation. The implemented COCO control pools all human categories to `person`; stratify roles in a later experiment. Review small targets and scene difficulty. `mAP50` and `mAP50_95` use COCOeval with maxDet=100 and the saved score floor; fixed-score precision/recall use the selected operating score at IoU .5. Small-player recall uses the COCO `small` area range and GT objects explicitly labeled `player`.
3. For tracking, exhaustively label every frame of a continuous clip, including empty frames. Assign the same physical ID through disappearances and include role where known. Supply a separate complete GT JSON to `TRACKING_ANNOTATIONS`. HOTA, IDF1, ID switches and fragmentation use the named [`trackeval==1.3.0` distribution](https://pypi.org/project/trackeval/1.3.0/) from the [kovalp fork](https://github.com/kovalp/TrackEval) and one-class preprocessed inputs. The range for HOTA/IDF1 is 0–1. There is no MOTChallenge distractor suppression.
4. For Re-ID review immutable crop hashes and export a disjoint query/gallery split. Include match, player, team and source tracklet IDs, and prefer actual re-entry, same-team negatives and back views. An eligible comparison is within the same match, with different tracklets and crop hashes. FMR needs a distance threshold selected on held-out data. Parts without common visible evidence are unscorable.
5. For identity use the dense tracking format with `anchors` and `reentries`. Review an anchor per player and verify the real absent interval and visible return window. Metrics are derived after inference. Re-entry accuracy is the fraction of reviewed episodes with a correct assignment by the episode end. Delay is the median frame delay among recovered episodes. False-merge and split counts use pair conflicts across IoU-aligned observations. Ambiguous anchors are excluded from identity assignment and recorded. Report disagreements across annotators before treating measurements as definitive.

Detection, tracking and identity use this format. Actual clips, coordinates and SHA-256 values must come from your labeling export.

```json
{
  "schema": "scoutra-reviewed/1",
  "human_reviewed": true,
  "clips": {
    "clip_stem": {
      "clip_sha256": "<actual video SHA-256>",
      "continuous_scene": true,
      "frames": [
        {"frame": 1, "objects": [{"class": "player", "player_id": "p17", "x1": 10, "y1": 20, "x2": 29, "y2": 60}]},
        {"frame": 2, "objects": []},
        {"frame": 3, "objects": [{"class": "player", "player_id": "p17", "x1": 22, "y1": 20, "x2": 42, "y2": 60}]}
      ],
      "anchors": [{"frame": 1, "player_id": "p17"}],
      "reentries": [{"player_id": "p17", "return_frame": 3, "end_frame": 3}]
    }
  }
}
```

Re-ID labels use `{"human_reviewed":true,"crops":[{"crop_id":"...","sha256":"...","split":"query","match_id":"m1","player_id":"p17","team_id":"a","tracklet_id":"a17"}]}`. Supply both splits. Use only `query` and `gallery` as split names. No accuracy is published for unavailable annotations.

## Rebuild and validate

Run `python experiments/scoutra-dual-t4/build_notebook.py` after changing the embedded source. `--check` ensures generated cells match the checked-in files. `tests/test_lab.py` uses generated CPU fixtures and the evaluation profile. `tests/real_cpu_check.py` runs the actual pretrained detector and trackers end to end over 12 local clips using the offline install. See [validation record](VALIDATION.md) for the fixture and measured scope. Kaggle T4 results remain pending.
