# C–D. Technical Deep Dives and Algorithm Comparison
**Read-only source inspection, 17 November 2026.** V establishes code structure; A denotes reported evaluation; I/H denotes interpretation/proposed testing.

## 1. Scoutra baseline and comparison boundary
The accessible repository is a frontend: `src/App.tsx` composes presentation sections, `package.json` defines Vite/TypeScript build tools, and the README explicitly distinguishes the site from an assessment product. The supplied product-context document describes future academy perception and evidence-linked Player Passport metrics. No current detector, tracker, inference worker, match database, or model-quality baseline was inspectable. [Scoutra-readme] [Scoutra-package] [Scoutra-context]

Consequently, for **every component below**, the current-equivalent answer is: **none found in this snapshot; unknown in any separate backend**. “Potentially better” means stronger problem fit or stronger external evidence than a simple reference method, never measured superiority to Scoutra.

## 2. AgentCoach / GoalInsight — priority deep dive
### Actual data flow
`goalinsight/pipeline/_pipeline.py` orders registered stages, supports `skip_existing`, handles cancellation between stages, and writes run metadata. Calibration precedes tracking; consolidation then produces track-to-player mappings for downstream consumers. This is genuine stage separation, not merely a notebook. Reusing a completed stage is different from serializing every tracker's state mid-stage. [A-pipeline] [A-runner]

`tracking/orchestrator.py` reads with OpenCV and batch/prefetch helpers, loads calibration poses/homographies, filters detections by size and pitch, extracts crops, runs Re-ID, then associates tracks. In the inspected batch path, `_all_crops` collects crops across the selected frame range before Re-ID chunks of 2,048 are processed. Batching limits each forward call; it does **not** bound retained crop memory over the video. Detections, embeddings, track histories, and final JSON outputs also accumulate. **V/I: important 90-minute scaling risk.** [A-tracking]

### Tracking
`strongsort/tracker.py::StrongSORTTracker` uses Kalman motion, appearance costs, IoU/pitch-related fallback stages, and gates. `matching.py` isolates association; `gates.py::MotionPitchGate` combines image motion and pitch feasibility; `lifecycle.py::TrackLifecycle.cleanup` removes tracks after `max_age`. This is a locally customized StrongSORT-style implementation, not proof of the complete upstream StrongSORT++ algorithm. Track fields for team/jersey do not prove those signals participate in every online matching stage; the decisive team/OCR reasoning occurs later. [A-tracker] [A-gates] [A-lifecycle] [A-runner]

The pitch gate can prevent implausible same-team jumps **if calibration is good**. It can also reject correct matches when geometry drifts. The code's fallbacks for missing calibration are sensible but do not detect every erroneous calibration. These are architectural observations, not an accuracy result. [A-gates]

### Identity consolidation
The top docstring still describes greedy clustering, but the current body of `_consolidator.py::consolidate` builds similarity edges, sorts them, and unions components subject to team, co-occurrence, and strong conflicting jersey constraints. It then votes jersey labels, performs a jersey-aware second merge, resolves label collisions, and attempts orphan absorption. The source body is more authoritative than the stale summary. It is an offline identity hypothesis generator with global access to selected tracklets. [A-consolidate]

The `_runner.py` gathers co-occurrence pairs, track features, OCR votes/candidates, and team/role information; writes `track_to_player.json`, `players.json`, and consolidated trajectories. One raw track ID is generally mapped to one player ID: a tracklet already containing an identity switch is therefore a serious problem. A separate split-before-merge stage is not established by this mapping. [A-runner]

### Re-ID fidelity
`PRTReIDExtractor._extract_prtreid` selects `bn_globl` or `globl`, normalizes it, and emits a global vector plus role/visibility information. It does not preserve the full body-part distance computation used by TrackLab. This simplification might save complexity but needs a direct ablation on occluded teammates. The adapter also reconstructs checkpoint configuration, loads with `strict=False`, and contains compatibility handling for changed Albumentations imports. **V: reproducibility/maintenance risk, not demonstrated wrong inference.** [A-reid]

### Calibration, events, outputs, reuse
`fixed_camera_runner.py` replays one annotated camera solution for a truly static rig. Other named calibration backends address different camera models; a fixed solution cannot follow pan/zoom. `events/_orchestrator.py` orchestrates detectors; `pass_detector.py` and `shot.py` derive events from trajectories/possession/geometry and attach available actor identities. These are implemented rules; no held-out event-accuracy evidence was verified. [A-calib] [A-events] [A-pass] [A-shot]

**Adaptation:** define Scoutra's timestamp, coordinate, observation-quality, identity-version, and event contracts; isolate each stage; validate camera model selection; replace full-range memory accumulation before long jobs. The offline Docker build requires local CLIP-ReID weights and demo annotation/video inputs. Root Apache licensing does not resolve vendored calibration and detector/Re-ID licenses. **Reuse of ideas: useful. Drop-in whole system: high effort and unproven.** [A-offline] [A-notice]

## 3. jersey-tracker — priority deep dive
`detect_track.py::Config` resolves camera layout, detector and sampled FPS. `iterate_frames` aligns sampling to global source frame indices so independently decoded overlap chunks refer to the same frames. `iterate_frames_per_lens` handles the two physical inputs; source image coordinates and per-camera boxes must remain distinct. `run` loads YOLO and uses ByteTrack per camera, writing detections/ball observations. This is local MOT, not appearance-based re-entry recognition. [J-detect]

`jersey_ocr.py` has EasyOCR and PARSeq backends. `back_of_jersey_crop` extracts an upper-back region and enhances contrast; `build_plan` selects track samples; `vote_number` aggregates confidence-weighted full-number reads. It deliberately avoids inventing a two-digit number by concatenating separate single-digit reads. A legibility or pose model is not equivalent to this fixed crop heuristic. Repeated correlated OCR errors can still win a vote. [J-ocr]

`run_parallel.py::plan_chunks` creates overlaps; `run_chunks_parallel` launches bounded subprocess workers with per-chunk logs; `merge_chunks.py::build_track_map` matches boxes by IoU in overlap frames, and `merge` namespaces IDs, remaps links transitively, and removes duplicate overlap rows. It cannot match a player absent from the shared window using appearance. The comment calling the resolution “one-to-one” deserves validation: the inspected final mapping chooses the best earlier ID per later ID without an explicit reverse-uniqueness set across all later IDs. **I: potential many-to-one boundary error, not reproduced.** [J-chunks] [J-merge]

`cross_lens_merge.py` handles an additional camera-fusion problem; it is not a general roster recognizer. `project_to_pitch.py` and `calibrate_match.py` make lens dimensions, distortion, homography, periods, and pitch geometry part of the data contract. This is directly relevant for raw Veo lenses; portability to an already auto-panned export needs separate validation. [J-cross] [J-project] [J-calib]

`trackio.py::StreamingRowWriter` supports JSONL/Parquet and progress sidecars with atomic progress-file replacement. The existence of this helper does not mean all stages use it. Detection/chunk merge and OCR have their own storage paths; merge and OCR load large row sets. `find_player.py` raises a stub error. **Best reuse:** chunk timestamps, camera-aware schemas, OCR voting, bounded row writers. **Likely adaptation:** generic camera layouts, missing-chunk validation, true tracker-state restart, global identity association. [J-io] [J-detect] [J-merge] [J-ocr] [J-find]

## 4. acato/soccer-video-pipeline — priority deep dive
### Implemented path versus older documents
`DualPassDetector` implements multiple modes: an 8B triage/observation → 32B text classification path and a direct 32B frame-classification path. Do not describe every mode as “32B reads video”: `_classify_window` in the older dual-pass path uses an observation generated upstream. The direct single-pass path receives sampled images. Config names and historical architecture notes describe several revisions. [C-dual] [C-config]

`FrameSampler.sample` tries batch FFmpeg extraction and falls back to single-frame extraction. `_extract_batch` uses `subprocess.run(capture_output=True)` and splits the complete JPEG byte stream. `TriageScanner.scan` pre-extracts a full-range sparse frame collection. That is economical compared with dense frames, but not fully bounded streaming. [C-frames] [C-triage]

VLM outputs become typed events; time-window deduplication, geometric/ball evidence gates, and diagnostic “canaries” catch implausible label rates, empty outputs, model-swap problems, and latency anomalies. Confidence fields are model/rule scores, not calibrated probabilities. The event schema permits `player_track_id`, but a field is not a verified actor-identification pipeline; no equivalent of GSR or AgentCoach identity consolidation was found in the current detector path. [C-models] [C-dual]

`scripts/combined_pipeline.py` and the architecture describe script-driven cache/ensemble/reel processing. Historical GPU documentation mentions API/Redis/Celery and `chunk_tagger.py`, but these are absent from the audited tree. Treat those older orchestration descriptions as **unverified for this commit**. The current reusable architecture is the visible detector/scripts/artifact chain. [C-combined] [C-arch] [C-gpu]

### Evidence and limitations
The public v11 merged FP8 model has actual safetensors shards in Hugging Face metadata; the adapter and ball detector files also exist. Existence does not prove clean-load reproduction. Training scripts require an internal v8 BF16 predecessor; the adapter card confirms this. The merged checkpoint is therefore the closer reproducible inference artifact; attaching the adapter to the original public base is a different model. [C-weights] [C-adapter] [C-train]

Reported four-game metrics use broad coverage tolerances, same-game tuning, and event proxies. Missing goalmouth views and new venues are documented failure cases. Extracting a watchable event reel is a narrower, easier target than counting every touch and assigning it to the correct player. [C-status]

**Adaptation:** preserve proposal/verification distinction, cache model responses, ground actors through verified tracks, use held-out venues and tight temporal metrics, and audit schema/weights/licenses. Reusing selective VLM scheduling is simpler than building a general video-language stack from scratch; reproducing the author's precise trained model from public training assets is not currently established.

## 5. SoccerNet GSR and TrackLab — priority deep dive
`sn_gamestate/configs/soccernet.yaml` is the executable architecture declaration: detector → Re-ID → tracker → pitch/calibration → jersey detection/voting → team → side labeling. Current defaults include YOLO Ultralytics, PRTReID, BPBReID StrongSORT, MMOCR, and NBJW calibration. **TVCalib is available, but is not the current default.** [G-config]

`PRTReId(DetectionLevelModule)` crops boxes, lazily constructs a feature extractor, and returns embeddings, body masks, visibility, role, and role confidence. It downloads specific PRTReID/HRNet assets with checksums. `VotingTrackletJerseyNumber.process` groups by existing `track_id` and votes one number. `TrackletTeamClustering`-related code aggregates embeddings for clustering teams. Voting labels a tracklet; it does not itself stitch fragmented IDs. [G-reid] [G-jersey] [G-team]

TrackLab's `OfflineTrackingEngine.video_loop` executes modules over an entire video and resets models between videos. `TrackerState` stores per-video outputs in ZIP/pickled tables and supports selective reload. The part-aware distance function compares visible components, rather than only a global vector. The inspected tracker config sets `max_age: 300`, `nn_budget: 100`, `ecc: False`, and a Kalman prediction cutoff. Configured optional camera compensation must not be described as enabled by default. [L-engine] [L-state] [L-parts] [L-config]

**Key distinction:** the PRT-Track paper includes an offline TrackMerge method. The current GSR pipeline list contains attribute aggregation and team modules, but no named general TrackMerge/global association stage. The paper's full pipeline result cannot be automatically attributed to the inspected GSR config. **Best reuse:** evaluation, module contracts, part/visibility interface, and controlled baseline comparisons. **Risks:** GPL/Hippocratic license chain, older research dependencies, calibration assumptions, short sequences, and match-wide state size. [P-paper] [G-config] [G-pyproject]

## 6. SportVision — priority deep dive
`SportsDetector` wraps standard detector backends; COCO person and sports-ball labels are mapped into a sports schema. This does not create learned goalkeeper/referee discrimination. `SportsTracker` wraps `trackers.ByteTrackTracker` and excludes balls from person association, retaining ball detections with ID -1. It explicitly supports only ByteTrack. No Re-ID, jersey OCR, or global identity bank appears here. [S-detect] [S-track]

`SportVisionPipeline` combines detection, team inference, annotations, possession and review statistics. Its detection cadence distinguishes fresh observations from reused display boxes. Separate speed/distance utilities require metric inputs and correct frame time; their existence is not evidence that the default pipeline produces validated player kinematics. The current project documents a clip-review scope and provisional evaluation. [S-pipeline] [S-speed] [S-eval]

**Useful idea:** expose evidence coverage, unknown teams and ambiguous possession; replay corrections from cached observations without rerunning perception. **Not a suitable replacement for persistent player identity.** Its small components may be easier to adapt than an entire large football pipeline, provided actual detector/model terms are compatible. [S-readme] [S-poss]

## 7. Remaining six Group A repositories
| Repository | Reconstructed implementation | What to study | Main barrier |
|---|---|---|---|
| Soccer-Tracking | `TrackingPipeline.run/write_files` writes frames, executes several detector passes, merges text detections, runs imported HMReid tracker, clusters teams, renders frames/video. Other variants implement jersey-constrained matching. | Deep-EIoU appearance fusion and the risks of hard OCR constraints | Local checkpoint paths; duplicated variants; no whole-match identity layer/license verified |
| budun-ov | `src/runner.py` batches YOLO/ByteTrack from a generator, rereads frames for team assignment, invokes TVCalib, kinematics and renderers | TVCalib integration, SigLIP team features, radar/Voronoi | Local weights; accumulated tracks; AGPL |
| samuehuang | `run_pipeline.py` starts independent stages with run records and resume; `src/events/engine.py` consumes possession/motion/air-event CSVs; analytics can correct team assignments | Separating possession, controller, last touch and motion; evidence QA; report contracts | One-sample validation; weights excluded; “identity-aware” is team correction |
| footyviz | `fv.job` builds a short clip, detector/BoT-SORT/OSNet tracks, team clusters, optional jersey/fingerprint fusion, stats and overlay | Average-link constrained association and honest failure reporting | 180s default cap; residual fragmentation; estimated metres without calibrated pitch |
| matchtrace-cv | `MotionAwareTracker.update` greedily assigns distance+IoU matches, smooths velocity, expires old tracks; `AnalysisPipeline` streams frames; `evaluate_model` evaluates team CNN | Bounded decoding, seed/provenance separation, simple interfaces | Synthetic-first evidence; not a strong long-term identity algorithm |
| soccer360 | `Pipeline.run` stages detection, ball tracking/stabilization, center-of-play and virtual camera/export; `WatcherDaemon` uses ingest fingerprint state | 360° ingest, duplicate-job prevention, ball-follow camera | Its “player cluster” is a centroid for camera control, not player identity clustering |

Direct source: [T-main] [T-hm] [T-jid] [B-runner] [B-track] [B-team] [H-run] [H-identity] [H-events] [F-job] [F-identity] [F-stats] [M-pipeline] [M-track] [M-eval] [360-pipeline] [360-cluster] [360-watch].

An additional code-review concern in budun-ov: `interpolate_ball_positions` applies limited interpolation and then unconditional backfill. Backfill can propagate a future ball location through a long missing interval beyond the intended short interpolation horizon. Treat this as a source-level risk requiring a targeted test, not an observed failure. [B-track]

## D. Algorithm comparison
| Layer | Mechanism and evidence | Football-specific decision |
|---|---|---|
| Detection: YOLO reference | Convenient pretrained/local wrappers across Group A; specialized weights may distinguish football roles | Keep as a baseline; establish small-ball recall and dataset/weight rights |
| RF-DETR | Transformer-based detector API and pretrained asset resolver; current core has Apache models and separately restricted “plus” variants | Plausible permissive alternative for selected sizes; generic COCO evidence is not football superiority [RF-code] [RF-readme] |
| D-FINE | `DFINETransformer`, decoder refinement and `Integral/LQE` model distributional box localization | Localization improvements may help feet/crops; tiny-ball gain unproven [D-code] |
| RT-DETR | Efficient encoder/decoder family; official PyTorch/Paddle implementations, pretrained releases and sliced inference support | Reasonable detector baseline; its license and results must be tied to the chosen implementation [RT-readme] |
| ByteTrack | Two confidence levels associate strong then weak detections with track motion/overlap | Good cheap local baseline; no long-term appearance identity by itself [BYTE-code] |
| BoT-SORT | Kalman geometry, high/low-score association, optional Re-ID and camera-motion compensation | Relevant moving-camera baseline; confirm Re-ID and GMC flags rather than assuming both are active [BOT-code] [BOT-gmc] |
| StrongSORT/++ | Appearance/motion tracking; AFLink and Gaussian-smoothed interpolation are distinct optional postprocessing | Pedestrian evidence alone insufficient; long gaps and offscreen events must not become fabricated motion [STR-readme] [STR-af] [STR-gsi] |
| PRTReID/BPBReID | Part-sensitive appearance with visibility; PRT adds team/role objectives | Strong football research rationale; preserve part information and audit license [G-reid] [L-parts] [P-license] |
| McByte++ | Masks, conditional camera compensation, good-crop feature memory, old-ID reconnection | Test actual long-absence mechanism and tiny-player exclusion gates [MC-code] |
| GTA | Split contaminated tracklets, then offline link by feature similarity with temporal constraints | Useful complement to online methods; no jersey/roster truth and pairwise costs can grow [GTA-code] [GTA-batch] |
| Calibration | Fixed rig/manual homography; TVCalib optimizes pitch-to-image alignment using a camera model and field evidence | Choose by camera/lens; wrong pitch or airborne-ball homography creates wrong metres [TV-code] [J-project] |
| Events | Rule states from tracks (AgentCoach/samuehuang), specialized temporal networks (FOOTPASS/OSL), VLM clip proposals (acato) | Evaluate actor+event+time together; event proposals cannot establish personal counts [A-events] [H-events] [FP-taad] [C-dual] |
| Analytics | socceraction consumes structured events; unravelsports consumes tracking/graph-ready data | Useful downstream once Scoutra's perception produces trustworthy schemas; neither extracts players from pixels [SO-action] [UN-readme] |

Detector comparison must hold annotation split, resolution, sampling, and postprocessing constant. Tracker comparison must hold detections constant before assessing the complete detection+tracking stack. These are experiment-design recommendations.

## 8. Specialist event systems: source reconstruction
**FOOTPASS:** `X3D_TAAD_Baseline` extracts multiscale 3D video features, applies ROIAlign to each tracked player's regions, masks missing observations, and predicts nine logits per player/time step (action classes plus background). `TAAD_GNN` adds EdgeConv and temporal context over player features, with a recurrent event head. `DST_Logits2Events` is an encoder-decoder Transformer that turns noisy action sequences into action/role and timestamp outputs; it is not a general language model or a player Re-ID network. [FP-taad] [FP-gnn] [FP-dst]

`run_TAAD_on_matches.py` reads half-match tracking arrays with frame, player, team direction, jersey, tactical role, position, speed and ROI columns. It builds 26 role slots (13 per team), samples clips, resizes video to 640×352, and rescales ROIs from assumed 1920×1080 coordinates. Porting arbitrary Scoutra input therefore requires explicit coordinate, role, missing-player and substitution adapters. The implementation includes unconditional CUDA allocations inside model helpers, so a nominal CPU fallback in a runner is not proof of complete CPU compatibility. No ready custom TAAD checkpoint download was established from the inspected tree; pretrained X3D initialization is not a trained FOOTPASS head. [FP-run] [FP-taad] [FP-dst]

The repository's evaluation uses player/team/jersey/time information. This is closer to Scoutra's actor-linked objective than QA about a supplied clip, but it benefits from rich upstream inputs that Scoutra must first produce. A GNN modeling tactical context does not repair incorrect physical identities automatically. [FP-eval] [FP-gnn]

**OpenSportsLib:** `QwenVLNativeModel` backend code samples and normalizes video frames, constructs multimodal messages, loads the visual/text model and optional PEFT adapter, then generates answers. This is actual implementation beyond a notebook. Its model zoo separately lists classification, spotting and foul-QA assets; these are different output tasks, not one universal football model. Reported model-card scores are author evidence, not certification by the library. [OSL-model] [OSL-zoo]

## Direct sources

[Scoutra-readme]: https://github.com/erbo-dev/Scoutra/blob/a6a02f176f55dd044112951506fa63a919ca7e5b/README.md "erbo-dev/Scoutra — README.md"
[Scoutra-package]: https://github.com/erbo-dev/Scoutra/blob/a6a02f176f55dd044112951506fa63a919ca7e5b/package.json "erbo-dev/Scoutra — package.json"
[Scoutra-context]: https://github.com/erbo-dev/Scoutra/blob/a6a02f176f55dd044112951506fa63a919ca7e5b/docs/scoutra-product-context.md "erbo-dev/Scoutra — docs/scoutra-product-context.md"
[A-pipeline]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/pipeline/_pipeline.py "outstandingcandy/AgentCoach — goalinsight/pipeline/_pipeline.py"
[A-runner]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/track_consolidation/_runner.py "outstandingcandy/AgentCoach — goalinsight/track_consolidation/_runner.py"
[A-tracking]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/orchestrator.py "outstandingcandy/AgentCoach — goalinsight/tracking/orchestrator.py"
[A-tracker]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/strongsort/tracker.py "outstandingcandy/AgentCoach — goalinsight/tracking/strongsort/tracker.py"
[A-gates]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/strongsort/gates.py "outstandingcandy/AgentCoach — goalinsight/tracking/strongsort/gates.py"
[A-lifecycle]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/strongsort/lifecycle.py "outstandingcandy/AgentCoach — goalinsight/tracking/strongsort/lifecycle.py"
[A-consolidate]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/track_consolidation/_consolidator.py "outstandingcandy/AgentCoach — goalinsight/track_consolidation/_consolidator.py"
[A-reid]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/reid/prtreid_extractor.py "outstandingcandy/AgentCoach — goalinsight/tracking/reid/prtreid_extractor.py"
[A-calib]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/field_registration/fixed_camera_runner.py "outstandingcandy/AgentCoach — goalinsight/field_registration/fixed_camera_runner.py"
[A-events]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/events/_orchestrator.py "outstandingcandy/AgentCoach — goalinsight/events/_orchestrator.py"
[A-pass]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/events/detectors/pass_detector.py "outstandingcandy/AgentCoach — goalinsight/events/detectors/pass_detector.py"
[A-shot]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/events/detectors/shot.py "outstandingcandy/AgentCoach — goalinsight/events/detectors/shot.py"
[A-offline]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/deploy/offline/README.md "outstandingcandy/AgentCoach — deploy/offline/README.md"
[A-notice]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/NOTICE "outstandingcandy/AgentCoach — NOTICE"
[J-detect]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/detect_track.py "spindle79/jersey-tracker — scripts/detect_track.py"
[J-ocr]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/jersey_ocr.py "spindle79/jersey-tracker — scripts/jersey_ocr.py"
[J-chunks]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/run_parallel.py "spindle79/jersey-tracker — scripts/run_parallel.py"
[J-merge]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/merge_chunks.py "spindle79/jersey-tracker — scripts/merge_chunks.py"
[J-cross]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/cross_lens_merge.py "spindle79/jersey-tracker — scripts/cross_lens_merge.py"
[J-project]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/project_to_pitch.py "spindle79/jersey-tracker — scripts/project_to_pitch.py"
[J-calib]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/calibrate_match.py "spindle79/jersey-tracker — scripts/calibrate_match.py"
[J-io]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/trackio.py "spindle79/jersey-tracker — scripts/trackio.py"
[J-find]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/find_player.py "spindle79/jersey-tracker — scripts/find_player.py"
[C-dual]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/dual_pass_detector.py "acato/soccer-video-pipeline — src/detection/dual_pass_detector.py"
[C-config]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/config.py "acato/soccer-video-pipeline — src/config.py"
[C-frames]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/frame_sampler.py "acato/soccer-video-pipeline — src/detection/frame_sampler.py"
[C-triage]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/triage_scanner.py "acato/soccer-video-pipeline — src/detection/triage_scanner.py"
[C-models]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/models.py "acato/soccer-video-pipeline — src/detection/models.py"
[C-combined]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/scripts/combined_pipeline.py "acato/soccer-video-pipeline — scripts/combined_pipeline.py"
[C-arch]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/docs/architecture.md "acato/soccer-video-pipeline — docs/architecture.md"
[C-gpu]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/docs/vllm-gpu-orchestration.md "acato/soccer-video-pipeline — docs/vllm-gpu-orchestration.md"
[C-weights]: https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-fp8/blob/ab56ac6743fc4f5ec299ed951f989d37f5874c56/README.md "acato v11 FP8 checkpoint card"
[C-adapter]: https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-lora/blob/5aca84c3c96f42d582346710fd22322ae0ed6d6b/README.md "acato v11 LoRA card"
[C-train]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/scripts/train_lora_v11.sh "acato/soccer-video-pipeline — scripts/train_lora_v11.sh"
[C-status]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/docs/pipeline_status.md "acato/soccer-video-pipeline — docs/pipeline_status.md"
[G-config]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/configs/soccernet.yaml "SoccerNet/sn-gamestate — sn_gamestate/configs/soccernet.yaml"
[G-reid]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/reid/prtreid_api.py "SoccerNet/sn-gamestate — sn_gamestate/reid/prtreid_api.py"
[G-jersey]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/jersey/voting_tracklet_jn_api.py "SoccerNet/sn-gamestate — sn_gamestate/jersey/voting_tracklet_jn_api.py"
[G-team]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/team/tracklet_team_clustering_api.py "SoccerNet/sn-gamestate — sn_gamestate/team/tracklet_team_clustering_api.py"
[L-engine]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/tracklab/engine/offline.py "TrackingLaboratory/tracklab — tracklab/engine/offline.py"
[L-state]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/tracklab/datastruct/tracker_state.py "TrackingLaboratory/tracklab — tracklab/datastruct/tracker_state.py"
[L-parts]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/plugins/track/bpbreid_strong_sort/sort/nn_matching.py "TrackingLaboratory/tracklab — plugins/track/bpbreid_strong_sort/sort/nn_matching.py"
[L-config]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/tracklab/configs/modules/track/bpbreid_strong_sort.yaml "TrackingLaboratory/tracklab — tracklab/configs/modules/track/bpbreid_strong_sort.yaml"
[P-paper]: https://arxiv.org/html/2401.09942v1 "PRTReID paper, Tables 3–6 and §3.2"
[G-pyproject]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/pyproject.toml "SoccerNet/sn-gamestate — pyproject.toml"
[S-detect]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/src/sportvision/detection.py "MohibShaikh/sportvision — src/sportvision/detection.py"
[S-track]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/src/sportvision/tracking.py "MohibShaikh/sportvision — src/sportvision/tracking.py"
[S-pipeline]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/src/sportvision/pipeline.py "MohibShaikh/sportvision — src/sportvision/pipeline.py"
[S-speed]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/src/sportvision/analytics/speed.py "MohibShaikh/sportvision — src/sportvision/analytics/speed.py"
[S-eval]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/docs/evaluation/README.md "MohibShaikh/sportvision — docs/evaluation/README.md"
[S-readme]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/README.md "MohibShaikh/sportvision — README.md"
[S-poss]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/src/sportvision/analytics/possession.py "MohibShaikh/sportvision — src/sportvision/analytics/possession.py"
[T-main]: https://github.com/Streakfull/Soccer-Tracking/blob/9d3488fc88390cbc64eb444eb08e481b79df0aef/src/TrackingPipeline/main.py "Streakfull/Soccer-Tracking — src/TrackingPipeline/main.py"
[T-hm]: https://github.com/Streakfull/Soccer-Tracking/blob/9d3488fc88390cbc64eb444eb08e481b79df0aef/src/pytorchmodels/DeepIOUTracker/HMReidTracker.py "Streakfull/Soccer-Tracking — src/pytorchmodels/DeepIOUTracker/HMReidTracker.py"
[T-jid]: https://github.com/Streakfull/Soccer-Tracking/blob/9d3488fc88390cbc64eb444eb08e481b79df0aef/src/hm_sort_jid/tracker.py "Streakfull/Soccer-Tracking — src/hm_sort_jid/tracker.py"
[B-runner]: https://github.com/budun-ov/football-analytics/blob/cd2c43fce7f54100500058ac0ff5aba7a10fce94/src/runner.py "budun-ov/football-analytics — src/runner.py"
[B-track]: https://github.com/budun-ov/football-analytics/blob/cd2c43fce7f54100500058ac0ff5aba7a10fce94/src/detection/tracker.py "budun-ov/football-analytics — src/detection/tracker.py"
[B-team]: https://github.com/budun-ov/football-analytics/blob/cd2c43fce7f54100500058ac0ff5aba7a10fce94/src/team/classifier.py "budun-ov/football-analytics — src/team/classifier.py"
[H-run]: https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/run_pipeline.py "samuehuang/football-cv-analytics — run_pipeline.py"
[H-identity]: https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/analytics/identity_qa.py "samuehuang/football-cv-analytics — src/analytics/identity_qa.py"
[H-events]: https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/events/engine.py "samuehuang/football-cv-analytics — src/events/engine.py"
[F-job]: https://github.com/dan-slater/footyviz/blob/4493f88d554a80dc3d9858c276173a17786caf8d/src/fv/job.py "dan-slater/footyviz — src/fv/job.py"
[F-identity]: https://github.com/dan-slater/footyviz/blob/4493f88d554a80dc3d9858c276173a17786caf8d/src/fv/identity.py "dan-slater/footyviz — src/fv/identity.py"
[F-stats]: https://github.com/dan-slater/footyviz/blob/4493f88d554a80dc3d9858c276173a17786caf8d/src/fv/stats.py "dan-slater/footyviz — src/fv/stats.py"
[M-pipeline]: https://github.com/a6rahamjr/matchtrace-cv/blob/aa61cd6c301238875387df51b0467f401bc0b880/src/matchtrace/inference/pipeline.py "a6rahamjr/matchtrace-cv — src/matchtrace/inference/pipeline.py"
[M-track]: https://github.com/a6rahamjr/matchtrace-cv/blob/aa61cd6c301238875387df51b0467f401bc0b880/src/matchtrace/inference/tracker.py "a6rahamjr/matchtrace-cv — src/matchtrace/inference/tracker.py"
[M-eval]: https://github.com/a6rahamjr/matchtrace-cv/blob/aa61cd6c301238875387df51b0467f401bc0b880/src/matchtrace/evaluation/evaluate.py "a6rahamjr/matchtrace-cv — src/matchtrace/evaluation/evaluate.py"
[360-pipeline]: https://github.com/humac/soccer360/blob/7b5546c8d0a648dd4c5037dfc70a9d0b09ee3fbb/src/pipeline.py "humac/soccer360 — src/pipeline.py"
[360-cluster]: https://github.com/humac/soccer360/blob/7b5546c8d0a648dd4c5037dfc70a9d0b09ee3fbb/src/player_cluster.py "humac/soccer360 — src/player_cluster.py"
[360-watch]: https://github.com/humac/soccer360/blob/7b5546c8d0a648dd4c5037dfc70a9d0b09ee3fbb/src/watcher.py "humac/soccer360 — src/watcher.py"
[RF-code]: https://github.com/roboflow/rf-detr/blob/e56ec6152b0eb1ac8bcca8690aa7ba59761d861f/src/rfdetr/detr.py "roboflow/rf-detr — src/rfdetr/detr.py"
[RF-readme]: https://github.com/roboflow/rf-detr/blob/e56ec6152b0eb1ac8bcca8690aa7ba59761d861f/README.md "roboflow/rf-detr — README.md"
[D-code]: https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/zoo/dfine/dfine_decoder.py "Peterande/D-FINE — src/zoo/dfine/dfine_decoder.py"
[RT-readme]: https://github.com/lyuwenyu/RT-DETR/blob/df2a96c8c3a098567a1149bf0248da6efe6d495d/README.md "lyuwenyu/RT-DETR — README.md"
[BYTE-code]: https://github.com/FoundationVision/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/byte_tracker.py "FoundationVision/ByteTrack — yolox/tracker/byte_tracker.py"
[BOT-code]: https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/tracker/bot_sort.py "NirAharon/BoT-SORT — tracker/bot_sort.py"
[BOT-gmc]: https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/tracker/gmc.py "NirAharon/BoT-SORT — tracker/gmc.py"
[STR-readme]: https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/README.md "dyhBUPT/StrongSORT — README.md"
[STR-af]: https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/AFLink/AppFreeLink.py "dyhBUPT/StrongSORT — AFLink/AppFreeLink.py"
[STR-gsi]: https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/GSI.py "dyhBUPT/StrongSORT — GSI.py"
[P-license]: https://github.com/VlSomers/prtreid/blob/30617a75967e84d5d516959c4b84cbeea6f56493/LICENSE "VlSomers/prtreid — LICENSE"
[MC-code]: https://github.com/tstanczyk95/McBytePlusPlus/blob/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9/yolox/tracker/mcbyteplusplus_tracker__with_reid.py "tstanczyk95/McBytePlusPlus — yolox/tracker/mcbyteplusplus_tracker__with_reid.py"
[GTA-code]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/refine_tracklets.py "sjc042/gta-link — refine_tracklets.py"
[GTA-batch]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/refine_tracklets_batched.py "sjc042/gta-link — refine_tracklets_batched.py"
[TV-code]: https://github.com/MM4SPA/tvcalib/blob/1222c5230af2742395d74918ed6f34eb2b9bf7f9/tvcalib/module.py "MM4SPA/tvcalib — tvcalib/module.py"
[FP-taad]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/models/model_TAAD_baseline.py "JeremieOchin/FOOTPASS — models/model_TAAD_baseline.py"
[SO-action]: https://github.com/ML-KULeuven/socceraction/blob/93a1242d46c104889205753accaabadb00c45c6d/README.md "ML-KULeuven/socceraction — README.md"
[UN-readme]: https://github.com/UnravelSports/unravelsports/blob/2bd24d7e2b3cc19cbdba6486c7abb70270a732ad/README.md "UnravelSports/unravelsports — README.md"
[FP-gnn]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/models/model_GNN.py "JeremieOchin/FOOTPASS — models/model_GNN.py"
[FP-dst]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/models/model_DST.py "JeremieOchin/FOOTPASS — models/model_DST.py"
[FP-run]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/run_TAAD_on_matches.py "JeremieOchin/FOOTPASS — run_TAAD_on_matches.py"
[FP-eval]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/evaluation.py "JeremieOchin/FOOTPASS — evaluation.py"
[OSL-model]: https://github.com/OpenSportsLab/opensportslib/blob/b2fe8f16a35336d93b1b0eabe1741bbbbc36cd68/opensportslib/models/base/qwen_vl_native.py "OpenSportsLab/opensportslib — opensportslib/models/base/qwen_vl_native.py"
[OSL-zoo]: https://github.com/OpenSportsLab/opensportslib/blob/b2fe8f16a35336d93b1b0eabe1741bbbbc36cd68/docs/model-zoo.md "OpenSportsLab/opensportslib — docs/model-zoo.md"
