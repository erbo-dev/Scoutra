# Readiness audit after GitHub delivery

Audited 17 November 2026. **Model inference, training and GPU benchmarks: NOT EXECUTED.** This is an audit of the delivered source, not a claim that all candidate methods run. No model implementation was changed in this delivery/audit.

## Accessible delivery and provenance

- Repository: [erbo-dev/Scoutra](https://github.com/erbo-dev/Scoutra).
- Branch: [research/scoutra-kaggle-lab](https://github.com/erbo-dev/Scoutra/tree/research/scoutra-kaggle-lab/experiments/scoutra-kaggle-lab).
- [Draft PR #1](https://github.com/erbo-dev/Scoutra/pull/1), unmerged; initial delivery commit `ffb86dd42eac3378a1ebdfd082afbd1edf023091`. The audit is a documentation follow-up on that branch.
- Original directory exists; complete 202-file inventory and archive verification are in [FILE_INVENTORY](FILE_INVENTORY.md). Original ZIP SHA256: `daae05300c49c87eb6b2de98a78446667362602b03564690fad77d2e137d38a8`; ZIP integrity check passed.
- The newly created checkout is writable and its origin is `https://github.com/erbo-dev/Scoutra.git`. The GitHub connection reported push permission; the actual branch push and subsequent remote file read succeeded. This replaces the previous delivery failure.
- All changes are additions under `experiments/scoutra-kaggle-lab/`. Production files and `main` are unchanged. No videos, actual datasets/annotations, weights, secrets, caches, ZIPs, NumPy arrays, Parquet files or other generated binary outputs are committed. `schemas/` contains blank templates, not datasets. `results/STATUS.json` is a text-only readiness inventory.
- Existing Python adapters/utilities/tests, five notebooks and dependency profiles are preserved from the original laboratory. Packaging relocates blank templates from `datasets/` to `schemas/`, omits runtime artifacts and adjusts delivery documentation. The old reports' output-directory permission blocker is historical and has been superseded by GitHub delivery.

## What is ready

| Item | Audit result |
|---|---|
| Notebook files | All five exist in the pushed source; valid nbformat4, code cells parse, execution counts null, outputs empty |
| Safety defaults | INSTALL, RUN_AUDIT, RUN_CACHE, RUN_PREPARE_CROPS and RUN_INFERENCE default false where present; all 17 experiment configs have execute=false |
| No invented results | All 17 STATUS entries are NOT EXECUTED and metrics=null; no real prediction files are shipped. Prior-audit paper/author results are historical literature evidence, not Scoutra experiment results |
| CPU infrastructure | Twelve synthetic unit tests pass in the existing isolated research test environment; this is not model accuracy or a Kaggle environment validation |
| Notebook01 inventory | Runnable candidate on Kaggle CPU after attaching source/videos, confirming OpenCV/PyYAML, and setting consistent paths; no checkpoints or annotations required for inventory |
| Detection/local tracking/OSNet/AgentCoach | Thin adapters exist; no real model/load/API or T4 smoke validation. Missing assets/configurations still block actual experiments |
| PRT/BPB/GTA/McByte++ | Config/specification and generic export validation exist, but no executable method-specific bridge in the core runners. Not ready for an automatic notebook comparison |
| Official HOTA/IDF1 | Dense MOT export and official TrackEval launcher exist; scores remain null until complete GT and its separate environment are supplied |

The CPU tests verify synthetic IoU/class gating, assignment cardinality, anchor-only selected identity scoring/false merges, missing-label handling, retrieval leakage exclusion, match split leakage, small-sample bootstrap guard, empty Arrow schemas, disabled runs, GPU opt-in, cached-frame GT scope/resolution and fixed-threshold TPR/FMR. They do not exercise pretrained models or decode these football clips.

## Path audit: editable, but only partially centralized

Source: [notebooks](../notebooks/), [configs](../configs/), [dataset.py](../scoutra_lab/dataset.py), [pipelines.py](../scoutra_lab/pipelines.py).

1. The supplied dataset path is a **default**, not a confirmed Kaggle mount: `/kaggle/input/datasets/erbosynshaimardan/zhanto-highlights/`. Recursive discovery handles actual filenames; the code does not require the twelve reported files to be present.
2. `configs/dataset.yaml` exposes dataset_root, manifest and cache_root. Notebooks02–05 expose their frame/detection/crop/track/annotation/asset/output paths through a mutable CFG loaded from YAML. No local checkpoint path is assumed valid: placeholders and blank SHA256 values block loading.
3. Notebook bootstrap searches **Kaggle input** for a lab README next to `scoutra_lab/common.py`. LAB_ROOT can be set manually; WORK defaults to `/kaggle/working/scoutra` in every notebook. This is Kaggle-specific bootstrap, not an environment-agnostic path resolver.
4. **Notebook01 mismatch:** DATASET_ROOT is copied from CFG. Editing DATASET_ROOT alone affects inventory discovery, while cache_segments later reads CFG['dataset_root']. Set both. Audit output is explicitly WORK/'dataset'; changing CFG['manifest'] alone does not redirect inventory output. Align that field with the actual output before caching.
5. **Notebook04 preprocessing defaults:** its optional crop-preparation cell explicitly uses WORK/'cache/stage1-v1/frames.json', WORK/'results/B-bytetrack-001/predictions.parquet' and WORK/'crops/reid-v1'. Editing CFG['crops'] alone does not change those preprocessing inputs/outputs. Edit that cell or retain the documented default paths.
6. Changing WORK does not automatically rebase absolute paths loaded from YAML. Update all relevant CFG entries. Cached frame/crop manifests store absolute paths, so moving artifacts between sessions requires consistent restore paths or an explicit new remapped manifest/hash.

Conclusion: dataset/input paths are configurable through YAML and notebook edits, but **not all paths are controlled by one setting**. This audit documents that limitation; it does not silently rewrite the notebooks.

## Checkpoint source availability: verified metadata, not loaded weights

The HTTP checks below were performed on17November2026 using **HEAD or metadata JSON only**. See [READINESS_EVIDENCE.json](READINESS_EVIDENCE.json) for the exact requests, status codes, listings and package metadata. No checkpoint body was downloaded; storage size is not VRAM.

| Required asset | Fresh availability check | Still missing |
|---|---|---|
| RF-DETR Small | [official endpoint](https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth): HEAD200, octet-stream,386045550bytes advertised | Trusted local copy, expectedSHA256, verified raw class map, backbone/cache inventory and load test |
| RF-DETR Medium | [official endpoint](https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth): HEAD200,404992918bytes advertised | Same blockers; actual VRAM unmeasured |
| D-FINE S/N | [release API](https://api.github.com/repos/Peterande/storage/releases/tags/dfinev1.0):200; dfine_s_coco.pth41841422bytes, dfine_n_coco.pth15489558bytes listed | Matching upstream YAML/commit, local hash, class map, reviewed serialization and dependency/load validation |
| RT-DETR R50-vd | [pinned snapshot API](https://huggingface.co/api/models/PekingU/rtdetr_r50vd/revision/df939e661d8c52e80608d1ec566561aabd25a4e7):200; config, processor and model.safetensors listed | Full local snapshot/fingerprint, raw class map and inference smoke test |
| OSNet x0.25MSMT17 | [registry asset](https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF): HEAD200, octet-stream,3057863bytes advertised | SHA256/architecture-compatible filename, checkpoint/data license review and load test; no semantic verification from HEAD |
| PRTReID SoccerNet | [Zenodo10653453](https://zenodo.org/api/records/10653453):200; prtreid-soccernet-baseline.pth.tar396287605bytes and MD5 listed | Local checkpoint+HRNet, SHA256, matching config/heads, legacy environment and executable extraction bridge |
| BPBReID / HRNet | [Zenodo10604211](https://zenodo.org/api/records/10604211):200; bpbreid_market1501_hrnet32_10642.pth457750696bytes and hrnetv2_w32_imagenet_pretrained.pth165587602bytes listed with MD5 | Local assets/hashes, matching architecture/config, independent environment and full-part bridge |
| Existing Scoutra RF-DETR | No asset accessible in workspace or connected frontend repository | Actual checkpoint, architecture, class map, config/provenance and rights; baseline remains N/A |
| GTA / McByte++ extras | Pinned installation/source references exist; their complete asset chain was not independently fetched/load-tested | Exact sports Re-ID, EdgeTAM/mask and optional YOLOX assets plus licenses/hashes/environment; see blockers below |

Archive metadata labels PRT/BPB assets CC-BY4.0 while source code uses Hippocratic terms. Endpoint existence does not settle the code/weight/data license chain. BoxMOT ports are AGPL and must not be mislabeled with original BoT-SORT's MIT grant. See [license audit](LICENSE_AND_REPRODUCIBILITY.md).

Offline flags in gpu_guard apply to Hugging Face libraries; they are **not a universal network firewall**. Upstream TorchHub/backbone imports may have their own acquisition behavior. Before later model smoke execution, inventory all assets and use Kaggle Internet-off after installation if a strict no-fetch run is required. No model loading was attempted in this audit.

## Kaggle T4 dependency assumptions

These are metadata/source checks, not a successful Kaggle pip solve or GPU validation. [Versioned PyPI metadata](READINESS_EVIDENCE.json) and [dependency profiles](../configs/profiles/) are the evidence.

- torch2.6.0 / torchvision0.21.0 are a declared matching pair: torchvision metadata requires torch==2.6.0.
- RF-DETR1.4.3 declares Python>=3.10, torch>=1.13<=2.8, torchvision>=0.14 and Transformers>4<5. The proposed torch2.6/Transformers4.57.1 satisfy those inspected direct ranges. This does not resolve unpinned transitive requirements or test runtime APIs.
- BoxMOT16.0.10 declares Python>=3.9,<3.13, torch>=2.2.1,<3 and torchvision>=0.17.1,<1. The chosen direct torch/vision pins satisfy these ranges.
- **Do not assume Python3.13 is compatible.** BoxMOT excludes it, and supervision0.25.1 selects NumPy>=2.1 on Python>=3.13, conflicting with the pinned NumPy1.26.4. Use a separately verified Python3.10–3.12 image/profile; exact wheel resolution, Kaggle-preinstalled conflicts and CUDA driver compatibility remain untested.
- No installed NVIDIA driver, CUDA runtime, T4 device inventory, available VRAM or kernel support was verified on Kaggle. The current cloud is not a Kaggle GPU. A PyTorch CUDA wheel is not proof that a selected model fits or has supported kernels.
- Two16GB T4s remain independent devices. Current core runners select one explicit gpu_index. There is no automatic multi-GPU sharding or32GB pooled-memory implementation.
- Original TrackEval needs the separate NumPy1.23.5 evaluation profile because its pinned source uses removed aliases. Do not combine that profile with modern inference pins or silently substitute the independently maintained PyPI trackeval fork.
- PRT/BPB/GSR and GTA require their legacy ecosystems. McByte++ documents a different modern CUDA/torch stack; it is **not** part of the torch2.6 tracking environment. H100 availability would not fix these Python/API issues.

Required later validation: fresh Kaggle session per profile → check Python → install the intended profile → pip check → kernel restart as needed → freeze/environment inventory → asset/hash/config preflight → separately authorized small load/inference check. None of those GPU steps was executed here. No32B/35B VLM is enabled or certified for twoT4s.

## Missing annotation and data inputs

No actual footage, reviewed labels, crop corpus, detector output, saved tracklets or prior Scoutra model results are accessible here. The user supplied a Kaggle dataset location; it has not been opened in a Kaggle session.

- Detection: reviewed exhaustive native-pixel person/ball boxes on sampled frames, roles for player/keeper/referee, camera/size conditions and consistent evaluation policy.
- Local tracking: dense reviewed boxes and physical IDs for every frame of short continuous scenes; cuts, occlusion/visibility and selected-player anchors. Sparse box samples cannot produce valid HOTA/IDF1.
- Re-ID: immutable crop pixels/hashes, verified match+physical/team/tracklet labels, disjoint query/gallery roles, viewpoint/occlusion/size/time-gap metadata. Keep GT crops and predicted-track crops as distinct conditions.
- Recovery: anchors plus true absence/return episodes and identity-contamination/split review. Existing supplemental evaluator does not implement all episode-delay, split-correctness or false-confidence curves.
- Provenance: source-match grouping, replay/cut review and match-disjoint validation/test. Unknown match grouping remains unassigned; highlights cannot establish90-minute continuity or halftime/substitution handling.
- Events/calibration later: actual pitch dimensions/landmarks and event actor/timing labels; none are supplied. These do not block CPU inventory notebook01.

## Exact advanced-method blockers

| Method | Concrete missing pieces; no surrogate is bundled |
|---|---|
| PRTReID global / parts | Core reid() supports OSNet only and explicitly refuses other methods. Need a method-specific extraction runner against commit30617a7, matching checkpoint/config and HRNet assets, exact feature heads and compatible legacy environment. GSR's pinned package requires Python3.9/torch1.13.1; it cannot be pip-installed indiscriminately in a modern kernel. Global `globl` and fullpart `[N,P,D]` plus `[N,P]` visibility must remain separate. Validate upstream distance and no-common-visible-part behavior, preserve crop order/hashes and review missing/unexpected weight keys. Checkpoint/code/data rights and SHA256 remain unresolved. |
| BPBReID | Same core-runner limitation, but BPB's torchreid namespace/API/output tuple differs from PRT. Need exact BPB config/HRNet+Market1501 checkpoint, legacy environment, part+visibility extraction/distance runner and export importer. PRT code cannot be assumed interchangeable. Pedestrian-trained checkpoint is a control, not proven football transfer. |
| GTA | recover() accepts no_link and agentcoach_reid_only only. Need upstream generate_tracklets/refine_tracklets conversion around frozen MOT boxes and per-detection features, then validated split/link output import and runtime accounting. Upstream recommends Python3.8, tested torch2.3/CUDA11.8, torchreid1.4.0 and setuptools59.8.0. Exact sports_model.pth.tar-60/hashes/license and native crop/detection alignment still required if reproducing its original features. Alternative frozen features must be named as an ablation. DBSCAN thresholds and pairwise CPU memory require validation. No exact compatible Kaggle environment or bridge is bundled. |
| McByte++ / bank | Tracker() only supports supervision ByteTrack and BoxMOT ports; no McByte++ update wrapper or bank export/import exists. Need pinned upstream tracker+YOLOX-style dependencies, EdgeTAM mask creation/propagation, sports Re-ID asset and validated same-detection interface. Official installation specifies Python3.10, CUDA12.8/GCC10.5, torch2.9.1/torchvision0.24.1, NumPy1.24.4 and legacy setuptools for Re-ID—different from current profile. It warns about resolver errors; this lab does not accept ignoring them as proof of reproducibility. Check all native builds/device kernels/T4 fit, mask compute/time, bank state/association logs and bank-on/off controls. No generic ByteTrack alias or zero-cost mask assumption is acceptable. |

Pinned sources: [PRT FeatureExtractor](https://github.com/VlSomers/prtreid/blob/30617a75967e84d5d516959c4b84cbeea6f56493/prtreid/tools/feature_extractor.py), [BPB FeatureExtractor](https://github.com/VlSomers/bpbreid/blob/a2dc4304284784d8b2c061764512f0698aa38c21/torchreid/tools/feature_extractor.py), [GSR dependencies](https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/pyproject.toml), [GTA README](https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/README.md), [McByte installation](https://github.com/tstanczyk95/McBytePlusPlus/blob/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9/INSTALLATION.md). Generic external_bridge.load_export checks supplied manifests/hashes; it neither runs these methods nor automatically converts/loads their outputs into the benchmark.

## Minimal notebook01 steps on Kaggle

1. Open the research branch/PR. Download the repository ZIP using GitHub **Code → Download ZIP**, or clone with `git clone --branch research/scoutra-kaggle-lab --single-branch https://github.com/erbo-dev/Scoutra.git`. Extract and use **only** `experiments/scoutra-kaggle-lab/` as a private Kaggle lab Dataset. No app code or supplied video data needs to be uploaded with it.
2. Create/import `notebooks/01_dataset_audit.ipynb` in Kaggle. Attach the lab Dataset and existing `erbosynshaimardan/zhanto-highlights` Dataset. Leave accelerator **None/CPU**. No model weights or annotations are needed for inventory.
3. Run bootstrap. If discovery is ambiguous, set LAB_ROOT to the actual attached directory containing both README.md and scoutra_lab/. WORK may stay `/kaggle/working/scoutra`. Confirm OpenCV and PyYAML are installed. INSTALL=False uses the native environment; if missing, use the base installation cell only after checking its Python compatibility and preserve any installation error. Inventory itself needs cv2/PyYAML and the standard library, not torch.
4. In the configuration/inventory cell, before RUN_AUDIT's if-block, use the **observed** dataset mount and align both values:
```python
DATASET_ROOT = Path('/kaggle/input/datasets/erbosynshaimardan/zhanto-highlights')
# Replace the candidate above if Kaggle mounts it differently.
assert DATASET_ROOT.is_dir(), list(Path('/kaggle/input').iterdir())
CFG['dataset_root'] = str(DATASET_ROOT)
CFG['manifest'] = str(WORK / 'dataset' / 'manifest.csv')
```
5. Set RUN_AUDIT=True, leave RUN_CACHE=False, and run the inventory cell. This scans all videos recursively, reads container metadata and hashes full files on CPU. It does not run perception models. It can take disk-I/O time for many large videos. A prior manifest is not overwritten: use a fresh working session/output folder for a new audit.
6. Download `WORK/dataset/manifest.csv`, `audit_errors.json` and `audit_status.json` from notebook outputs. Verify that videos opened and review errors. Match/camera/challenge/annotation fields remain unknown until human review; container duration/FPS may be estimates. Stop here for the first successful inventory run. Select/review an explicit20–60s continuous interval and label it before enabling RUN_CACHE or any later model notebook.

Audit outcome: **the source laboratory is delivered and inspectable; notebook01's CPU inventory is the immediate next action. The overall multi-method GPU benchmark is not yet ready to run end to end.**
