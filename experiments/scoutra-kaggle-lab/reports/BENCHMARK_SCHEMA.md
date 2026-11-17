# Common benchmark schema 1.0

Authoritative typed table definitions are `scoutra_lab.common.TABLES`; all Parquet writes use an explicit Arrow schema. Common IDs are strings to prevent collisions across clip/scene/chunk. Frame indices are zero-based source indices, not sampled-array offsets. Image boxes are xyxy native pixels; timestamps are seconds with a recorded provenance. Model resizing stays inside adapters.

| Table | Required content / semantics |
|---|---|
| detections | clip, scene, frame, time, unique detection_id, canonical class, native xyxy, detector score |
| tracks | same frame/box fields, detection_id provenance, namespaced tracklet_id, tracker score |
| embeddings | crop_id, feature head, part index, visibility/null, feature file and array index; actual arrays saved separately with order/hashes |
| identities | clip, scene, frame, source tracklet, proposed player_id/null, confidence/null, assigned/unknown state, evidence description |
| calibration | clip/scene/frame, serialized 3×3 homography, pitch dimensions/metres, coordinate-system string, quality/null |
| events | clip/scene, start/end seconds, event class, actor_id/null, confidence/null, evidence |

Confidence values remain method-specific scores. A cosine threshold is not a probability. Missing confidence stays null. Ground truth IDs/teams/numbers must never be fed to deployment association, except an explicitly declared user anchor or separately named oracle experiment. Prediction IDs are not physical GT identifiers.

A feature file may store global `[N,D]` or full part `[N,P,D]` arrays plus `[N,P]` visibility and optional part masks. Preserve original dtype/head names, normalization, crop order, combination strategy and the upstream distance routine. Never average full parts to call the result PRTReID part-aware. Store query/gallery index lists and complete distance metadata. Nonfinite/sentinel distances for no common visible parts must be treated as unmatchable, with unknown coverage recorded.

Each result config captures seed, GPU index, method identity/port, thresholds, resolution, batch/precision, dependencies, source commit, assets and class map. `metadata.json` includes environment/package/device inventory and input cache hashes. User-supplied expected checkpoint hashes remain in config. Runtime records status, wall time, allocated/reserved memory and scope; peaks are PyTorch-selected-device metrics, not total process GPU memory. Add NVML measurements explicitly later if needed.

Statuses distinguish **NOT EXECUTED**, **FAILED**, **EXECUTED**, **NOT EVALUATED**, **EVALUATED**. The inference status is in runtime.json and evaluation status in metrics.json. A successful prediction job without GT is EXECUTED / NOT EVALUATED. Unexecuted metrics are null; empty Parquet is a schema container, not measured all-missed predictions. Do not rank unrun methods.

Atomic JSON/Parquet writes prevent partial final files; they do not provide model-job resumption. Reuse immutable detector/crop artifacts rather than rerun expensive preprocessing. Changed inputs/configurations get a new cache/run ID and hash. No Stage 1 runner auto-resumes partial tracks or restores an identity bank; long-video checkpointing is a later experiment.

## External upstream export contract
`adapters.external_bridge.load_export` validates a supplied `export.json`: EXECUTED status, exact method, kind, input cache SHA256, upstream commit, checkpoint SHA256, license review, environment/config/runtime files and a name→SHA256 file map. Exports must preserve detection provenance and empty frames, feature heads/visibility, scene resets and unknown assignments. The contract checker is not an implementation of McByte++, GTA, PRTReID or BPBReID. A bridge still needs method-specific schema/output verification and runtime accounting before comparable evaluation.

## Official tracking metrics
`evaluation.mot_export.export` refuses incomplete/gapped/ignore-region GT. It exports each continuous clip scene as a separate MOT sequence, canonical person only; resets define the comparison scope. Store separate full-clip identity metrics for cross-scene linking. Official TrackEval at commit `12c8791b303e0a0b50f753af204249e622d0281a` computes HOTA/CLEAR/Identity. Its source uses deprecated NumPy aliases: use a separate NumPy 1.23.5 evaluation environment, not the inference profile or silent monkeypatching. [Source](https://github.com/JonathonLuiten/TrackEval/blob/12c8791b303e0a0b50f753af204249e622d0281a/trackeval/datasets/mot_challenge_2d_box.py).

After source-only acquisition and export, the unexecuted command template is:
```bash
python TRACK_EVAL_CHECKOUT/scripts/run_mot_challenge.py --GT_FOLDER EXPORT_ROOT/gt --TRACKERS_FOLDER EXPORT_ROOT/trackers --TRACKERS_TO_EVAL method --SEQMAP_FILE EXPORT_ROOT/seqmap.txt --SKIP_SPLIT_FOL True --DO_PREPROC False --BENCHMARK MOT17 --SPLIT_TO_EVAL test --METRICS HOTA CLEAR Identity --USE_PARALLEL False --PLOT_CURVES False
```
MOT17 here selects the loader convention; it does not mean these football videos are MOT17 data. Source is MIT. The `trackeval==1.0.0` PyPI metadata inspected points to a separate fork, so this lab does not silently substitute that distribution for the pinned original source.
