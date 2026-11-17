# B. Repository Comparison and Football Compatibility
**Snapshot: 17 November 2026.** V = source inspection; A = author report; I/H = interpretation or compatibility hypothesis. “Implemented” never means “executed successfully by this audit.” All scores are engineering judgments, not model accuracy.

## Repository comparison matrix

| Project | Main capabilities | Actual implementation | Weights | Source license / caveat | Long-video support | Scoutra fit |
|---|---|---|---|---|---|---|
| AgentCoach / GoalInsight | Calibration, tracking, identity consolidation, events, profiles | V: stage runner, custom StrongSORT, constraints-based consolidation, event detectors | PRTReID URL and standard detector models; offline build requires local CLIP-ReID/demo assets | Apache-2.0 root; vendored and model/dependency terms differ | Stage reuse; sequential frame decoding but match-wide crops/results retained | Closest broad reference; promising, requires validation |
| jersey-tracker | Youth/Veo tracking, teams, OCR, calibration, ball/possession | V: detection, OCR voting, chunks, merge, projection; target finder stub | Standard YOLO/PARSeq/EasyOCR fetch paths; custom asset portability incomplete | MIT root; Ultralytics and weights separate | Explicit overlap chunks and logs; merge/OCR load full detection sets | Useful amateur and storage modules |
| soccer-video-pipeline | GK/highlight events, VLM/YOLO evidence fusion | V: single/dual-pass event code and script-based reel stages; not full identity | Public v11 FP8, v11 LoRA, ball PT/ONNX file listings verified | Apache-2.0 code; model/data licensing issues | Full-game sparse windows and caches; no verified transactional resume | Promising event proposals |
| SoccerNet GSR | Player identification and minimap reconstruction | V: TrackLab config, PRTReID adapter, OCR voting, team clustering, calibration | Explicit Zenodo PRTReID/HRNet checksummed URLs; other modules download assets | GPL-3.0; PRT/BPB Hippocratic dependencies | Saved states and offline modules; benchmark is short clips | Strong research baseline; license-sensitive |
| SportVision | Clip review, team/possession evidence | V: detector wrapper, ByteTrack, unknown states, correction/review; speed/distance utilities separate | Standard COCO weights; no verified football-specialized model | Apache-2.0 root; backend terms separate | Frame loop; clip-oriented UI/evaluation; no cross-chunk identity | Individual modules, especially evidence UX |
| Soccer-Tracking | Detector + Deep-EIoU/HMSort, Re-ID/OCR variants | V: disk frame pipeline, several tracker variants; hardcoded local checkpoints | README links project assets on Drive; inspected paths not packaged | No root license in inspected tree; vendored licenses do not license whole repo | Extracts full frame directories; no verified long-job recovery | Research reference only |
| budun-ov/football-analytics | Tracking, TVCalib, teams, radar, kinematics | V: generator/batch detection, ByteTrack, SigLIP team features, TVCalib wrapper | Local detection file required; calibration external setup | AGPL-3.0 | Sequential decoding; full track collections and caches | Calibration/team/visualization modules |
| samuehuang/football-cv-analytics | Ball/possession states, events, tactics, reports | V: independent CLI stages, resume, CSV/event reasoning, team identity QA | Model/video files excluded; caller supplies assets | MIT root; external detector/assets separate | Stage reuse and explicit run records; demonstrated sample is short | Strong module reference, no physical identity solution |
| footyviz | Amateur overlays, identity fusion, roster, proxy stats | V: BoT-SORT/OSNet, constrained clustering, optional VLM; actual clip cap | YOLO/OSNet; optional OCR/SR assets; backend-dependent | MIT root; BoxMOT/YOLO and model terms separate | `fv.job` defaults to max 180-second clips | Useful identity failure case and fusion design |
| matchtrace-cv | Synthetic demo, streaming pipeline, tracking, team CNN | V: motion+IoU greedy matching, synthetic/classifier evaluation, optional real detector | No verified real-football specialized checkpoint; default generated data | MIT root | Bounded frame loop; no demonstrated full-match persistent identity | Engineering reference, not validated football perception |
| soccer360 | 360° ball-following, auto-camera, reframe/export | V: streaming detector call, ball tracker/stabilizer, center-of-play, watcher dedupe | Custom model resolution/bootstrap; full asset portability unverified | No root license found | Job phases, streaming ingest and dedupe; downstream lists grow | Specialist 360° acquisition/reframing only |

Sources by row: [A-pipeline] [A-tracking] [A-offline]; [J-detect] [J-merge] [J-find]; [C-dual] [C-frames] [C-weights]; [G-config] [G-reid]; [S-pipeline] [S-readme]; [T-main] [T-readme]; [B-runner] [B-track] [B-license]; [H-run] [H-readme]; [F-job] [F-identity]; [M-pipeline] [M-eval]; [360-pipeline] [360-track] [360-watch].

## Football compatibility matrix
These are **I/H assessments from implementation requirements**, not observed results on Scoutra footage. P = plausible with explicit adaptation; R = major risk; S = source explicitly supports relevant mode; U = insufficient evidence. S describes a mechanism, not accuracy.

| Project | Amateur/youth | Fixed camera | Pan/zoom | 720p/1080p small players | Occlusion/re-entry | Halftime/subs | 90-minute reliability |
|---|---|---|---|---|---|---|---|
| AgentCoach | P; author youth use | S: fixed pose backend | P: dynamic calibration backends; gates depend on quality | R: size filtering/Re-ID crops can lose distant players | S offline merging; R unseen/twin-like players | U roster/substitution continuity | U; match-wide data structures |
| jersey-tracker | P: Veo youth design | S per-lens calibration | R for arbitrary auto-pan/zoom; raw fixed lens assumptions | R number visibility; 5-fps sample tradeoff | R ByteTrack; stitching only common overlap | Period-aware calibration; U physical identity continuity | S chunks; U validated completeness/restart semantics |
| soccer-video-pipeline | P: amateur sideline target | P | R field/landmark assumptions and venue shift | R tiny ball/sparse sample; context events plausible | No general player identity implementation | Some game-clock offset handling; no identity memory | Full-game intended; U interrupted-job guarantees |
| SoccerNet GSR | R domain transfer | P | S calibrated moving broadcast benchmark | R resolution/geometry shift | S part-aware association; finite track lifetime | U full-match roster logic | U; short-clip GSR validation |
| SportVision | R football validation | P manual homography | R static homography invalidates | R COCO ball recall | R ByteTrack; no Re-ID | U | U; clip app |
| Soccer-Tracking | R broadcast-trained/local assets | P | R no verified robust calibrated-camera path | R resolution/assets | S appearance association within lost buffer; U global memory | U | R all-frame disk materialization |
| budun-ov | R broadcast priors | P | P TVCalib frame calibration | R detector/field lines | R ByteTrack | U | U stored all tracks |
| samuehuang | U camera/domain generalization | P manual/geometric assumptions | R fixed mapping requires validation | R ball/identity-dependent outputs | R local IDs; team correction not Re-ID | U | U stage resume is not full-match proof |
| footyviz | P: night amateur reference | P | P BoT-SORT/GMC; metric position still unavailable | R small players; author reports overfragmentation | S constrained fusion; R residual mistakes | U | R default clip cap |
| matchtrace-cv | U real football accuracy | P | P translation estimate; R zoom/perspective | U | S short occlusion only | U | U streaming helps but no match validation |
| soccer360 | P only suitable 360° recording | S equirectangular geometry | S virtual camera, not arbitrary TV camera calibration | R ball pixels/model | No player Re-ID pipeline | U | U deployment mechanisms, not identity proof |

Sources: fixed/dynamic modes [A-calib] [A-gates]; Veo sampling/projection [J-detect] [J-project] [J-calib]; VLM failures [C-status]; GSR benchmark [G-config] [G-paper]; SportVision calibration limits [S-readme]; other source paths in the preceding matrix.

### Conditions shared by every candidate
- **Pitch dimensions:** define actual youth/futsal dimensions and units. A default FIFA pitch can yield plausible-looking but wrong distances. GSR's projection and AgentCoach's pitch variants are references, not automatic measurement of an arbitrary field. [G-project] [A-calib]
- **Similar jerseys:** team classification reduces the candidate set but cannot distinguish teammates; OCR must aggregate readable evidence and retain “unknown.” [G-team] [J-ocr]
- **Goalkeepers/referees:** COCO's person category does not provide these roles. Dedicated role models or reviewed labeling are necessary. [S-detect] [G-reid]
- **Ball visibility:** a missing ball is missing evidence. Ground-plane projection of an airborne ball is geometrically wrong. No system can recover exact contacts outside the image as verified observations. [S-readme] [G-project]
- **Weather/lighting:** no inspected Group A evaluation establishes robust performance across rain, glare, changing floodlights, and all kit types. Treat all such compatibility conclusions as H.
- **Halftime/substitution:** preserve match identity while resetting short-horizon motion and side priors. Youth rolling substitutions and duplicate/missing numbers invalidate a simple “22 fixed players” assumption. This is a proposed requirement, not a verified capability.

## Comparative scores
Scale: 1 poor, 2 limited, 3 usable reference, 4 substantial, 5 strong documented suitability. Integration effort is scored positively: 5 = easiest. Dimensions score **architecture/evidence**, not execution or accuracy. Weights: relevance R 20%, completeness C 20%, football F 15%, reproducibility P 15%, long-video L 10%, maintainability M 10%, ease E 5%, license S 5%.

**N/A is retained for unverified whole-stack licensing.** “Known subtotal” is the contribution of scored dimensions to a 100-point total. Interval assigns unknown dimensions 1–5; it is not an estimated performance confidence interval. No re-normalization conceals missing evidence.

| Project | R | C | F | P | L | M | E | S | Known subtotal / coverage | Possible total | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| AgentCoach | 5 | 4 | 4 | 2 | 2 | 3 | 2 | N/A | 66 / 95 | 67–71 | Promising but requires validation |
| jersey-tracker | 4 | 3 | 4 | 2 | 4 | 2 | 3 | N/A | 61 / 95 | 62–66 | Useful for individual modules |
| soccer-video-pipeline | 4 | 3 | 4 | 3 | 4 | 2 | 2 | N/A | 63 / 95 | 64–68 | Promising but requires validation |
| SoccerNet GSR | 5 | 4 | 5 | 4 | 2 | 4 | 2 | 1 | 78 / 100 | 78 | Strong candidate for research |
| SportVision | 2 | 3 | 2 | 3 | 2 | 4 | 4 | N/A | 51 / 95 | 52–56 | Useful for individual modules |
| Soccer-Tracking | 4 | 3 | 4 | 1 | 1 | 1 | 1 | N/A | 48 / 95 | 49–53 | Research reference only |
| budun-ov | 3 | 3 | 4 | 2 | 2 | 3 | 2 | 1 | 55 / 100 | 55 | Useful for individual modules |
| samuehuang | 4 | 4 | 4 | 2 | 3 | 3 | 3 | N/A | 65 / 95 | 66–70 | Useful for individual modules |
| footyviz | 4 | 3 | 4 | 3 | 1 | 3 | 2 | N/A | 59 / 95 | 60–64 | Useful for individual modules |
| matchtrace-cv | 2 | 3 | 1 | 3 | 3 | 3 | 3 | N/A | 47 / 95 | 48–52 | Research reference only |
| soccer360 | 2 | 3 | 3 | 2 | 3 | 3 | 1 | N/A | 48 / 95 | 49–53 | Useful for individual modules |

These labels apply to the stated role. “Strong candidate for research” is not commercial integration clearance.

### Concrete score rationale, in R/C/F/P/L/M/E/S order
- **AgentCoach:** closest scope; connected stages; amateur author use but no independent benchmark; offline build assets missing; in-memory crop/result growth; separated interfaces plus custom stack; several adapters needed; mixed license chain unresolved. [A-pipeline] [A-tracking] [A-offline] [A-notice]
- **jersey-tracker:** relevant amateur engineering; missing finder; youth lens design; private calibration/input assets and no public held-out metrics; explicit chunks; many camera-specific scripts; useful row contracts need normalization; full dependency/model clearance absent. [J-find] [J-chunks] [J-io] [J-project]
- **acato:** relevant event subsystem; substantial detector code but fragmented script hot path; actual amateur model; public checkpoint with private training base/data caveats; full-game sparse windows; hardcoded operational paths and stale docs; schema/venue adaptation substantial; model-license contradiction. [C-dual] [C-train] [C-adapter] [C-gpu]
- **GSR:** core perception fit; implemented modular baseline; football benchmark; explicit weights/configs; not full-match identity; TrackLab interfaces; plugin and dataset adaptation; GPL/Hippocratic conditions reduce closed-product suitability. [G-config] [G-reid] [L-engine] [P-license]
- **SportVision:** not persistent-identity focused; review pipeline works at source level; current initial evaluation is clip/basketball-oriented; reproducible provisional diagnostic, not football validation; no long-video orchestration; package/tests/error handling; small modules accessible; backend licenses unresolved. [S-readme] [S-eval] [S-track]
- **Soccer-Tracking:** relevant algorithms; multiple partial/hardcoded variants; football source; absent local checkpoints prevent self-contained reproduction; disk frames and finite buffer; duplicated vendored stacks; significant cleanup; no root license. [T-main] [T-hm] [T-jid]
- **budun-ov:** useful visualization/calibration; coherent runner; football inputs; local model and TVCalib setup; accumulates tracks; readable modules; not global ID; AGPL confirmed. [B-runner] [B-track] [B-calib] [B-license]
- **samuehuang:** many target downstream functions; separated runnable stages; explicit football event logic; weights excluded and one sample; resume plus match-wide CSVs; understandable stage separation despite large files; adapters needed; MIT source does not clear outside models. [H-run] [H-events] [H-readme]
- **footyviz:** amateur failure case valuable; implemented fusion/review pipeline; real amateur author example; public recipe but limited testing; 180-second cap; modular but multiple roster variants; not metric tracking replacement; backend terms unresolved. [F-job] [F-identity] [F-stats]
- **matchtrace:** general reference; implemented framework; synthetic-first, not football evidence; synthetic reproducibility only; bounded frames; clear simple modules; adapt detector/calibration; real model licensing unknown. [M-pipeline] [M-track] [M-eval]
- **soccer360:** specialized capture use; detection/tracking/reframe stages; soccer ball context; custom assets unverified; ingestion phases but full-list downstream data; extensive test tree and operational structure; incompatible input geometry without new adapter; no root license. [360-pipeline] [360-watch] [360-cluster]

## Specialist shortlist and score boundaries
TrackLab, RF-DETR's Apache models, and GTA are **strong candidates to study/test**; PRTReID and McByte++ are **promising but require validation**; TVCalib, Roboflow Sports, socceraction, and unravelsports are **useful individual modules**; FOOTPASS and OpenSportsLib are **research references pending rights and domain checks**. SoccerNet Re-ID/Tracking/Jersey are evaluation resources, not complete platforms. Scoring a dataset kit as though it were a full analytics product would be misleading. Specialist module scores and interfaces are in REUSABLE_COMPONENTS.md. [L-readme] [RF-readme] [GTA-code] [MC-code] [SO-action] [UN-readme]

## Additional edge-condition compatibility
P = plausible but unvalidated; R = important failure risk; U = no adequate evidence. These are hypotheses inferred from the cited implementations.

| Project | Youth pitch dimensions | Referee / goalkeeper handling | Ball invisibility / airborne ball | Similar kits, weather and changing light |
|---|---|---|---|---|
| AgentCoach | P configurable pitch templates | P roles and consolidation; referee fragments may be grouped | P two-pass/trajectory logic; true contact accuracy U | R appearance/OCR/calibration; varied-weather evidence U |
| jersey-tracker | P configured calibration | P manual/team/period metadata | P pitch-space ball smoothing; airborne plane error R | R numbers/color; broad weather evidence U |
| acato | R venue/landmark normalization | P goalkeeper event focus, not persistent roster | R tiny or offscreen ball; proxies yield candidates | R framing/lighting shift; broad weather evidence U |
| GSR | P explicit field model adaptation | P role head and attribute voting | No demonstrated full ball-control engine in default GSR path | R broadcast-to-youth shift; weather evidence U |
| SportVision | P manual plane/corners | R generic COCO persons; manual ignore/review | R observation coverage; ambiguity explicitly possible | R color clustering; weather evidence U |
| Soccer-Tracking | U metric calibration | P classifier variants; actual active path matters | P detector/spatial filters; contact validity U | R learned appearances/checkpoints; weather evidence U |
| budun-ov | P TVCalib configuration | P football detector classes depend on supplied model | R interpolation/backfill and planar projection | R team embeddings/calibration; weather evidence U |
| samuehuang | P configured mapping | P role/team handling in local tracking | P explicit motion/controller/air states; accuracy U | R one-sample evidence; weather evidence U |
| footyviz | R metres estimated without homography | P other/keeper/roster heuristics | R nearest-player proxy with missed balls | R author night-case fragmentation; all-weather U |
| matchtrace-cv | P configured geometric projector | R real model supplied by caller | R synthetic ball/control behavior not validated on real footage | U real-football generalization |
| soccer360 | U metric pitch analytics | Player centroid for camera, no roster solution | P ball stabilizer/center-of-play fallback; unseen ball remains uncertain | R detector domain/360° quality; weather evidence U |

Sources per row are the same pinned source paths used in the main matrix. These symbols do not certify on-pitch performance.

**Professional-only claims:** GSR/PRTReID, TVCalib and FOOTPASS have professional-broadcast formulations/evidence. That supports caution about amateur transfer, not a categorical claim that their code can only operate on professional footage. Conversely, author testing on one amateur venue does not establish arbitrary-camera robustness. [G-paper] [P-readme] [TV-readme] [FP-readme]

## Direct sources

[A-pipeline]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/pipeline/_pipeline.py "outstandingcandy/AgentCoach — goalinsight/pipeline/_pipeline.py"
[A-tracking]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/orchestrator.py "outstandingcandy/AgentCoach — goalinsight/tracking/orchestrator.py"
[A-offline]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/deploy/offline/README.md "outstandingcandy/AgentCoach — deploy/offline/README.md"
[J-detect]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/detect_track.py "spindle79/jersey-tracker — scripts/detect_track.py"
[J-merge]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/merge_chunks.py "spindle79/jersey-tracker — scripts/merge_chunks.py"
[J-find]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/find_player.py "spindle79/jersey-tracker — scripts/find_player.py"
[C-dual]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/dual_pass_detector.py "acato/soccer-video-pipeline — src/detection/dual_pass_detector.py"
[C-frames]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/frame_sampler.py "acato/soccer-video-pipeline — src/detection/frame_sampler.py"
[C-weights]: https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-fp8/blob/ab56ac6743fc4f5ec299ed951f989d37f5874c56/README.md "acato v11 FP8 checkpoint card"
[G-config]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/configs/soccernet.yaml "SoccerNet/sn-gamestate — sn_gamestate/configs/soccernet.yaml"
[G-reid]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/reid/prtreid_api.py "SoccerNet/sn-gamestate — sn_gamestate/reid/prtreid_api.py"
[S-pipeline]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/src/sportvision/pipeline.py "MohibShaikh/sportvision — src/sportvision/pipeline.py"
[S-readme]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/README.md "MohibShaikh/sportvision — README.md"
[T-main]: https://github.com/Streakfull/Soccer-Tracking/blob/9d3488fc88390cbc64eb444eb08e481b79df0aef/src/TrackingPipeline/main.py "Streakfull/Soccer-Tracking — src/TrackingPipeline/main.py"
[T-readme]: https://github.com/Streakfull/Soccer-Tracking/blob/9d3488fc88390cbc64eb444eb08e481b79df0aef/README.md "Streakfull/Soccer-Tracking — README.md"
[B-runner]: https://github.com/budun-ov/football-analytics/blob/cd2c43fce7f54100500058ac0ff5aba7a10fce94/src/runner.py "budun-ov/football-analytics — src/runner.py"
[B-track]: https://github.com/budun-ov/football-analytics/blob/cd2c43fce7f54100500058ac0ff5aba7a10fce94/src/detection/tracker.py "budun-ov/football-analytics — src/detection/tracker.py"
[B-license]: https://github.com/budun-ov/football-analytics/blob/cd2c43fce7f54100500058ac0ff5aba7a10fce94/LICENSE "budun-ov/football-analytics — LICENSE"
[H-run]: https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/run_pipeline.py "samuehuang/football-cv-analytics — run_pipeline.py"
[H-readme]: https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/README.md "samuehuang/football-cv-analytics — README.md"
[F-job]: https://github.com/dan-slater/footyviz/blob/4493f88d554a80dc3d9858c276173a17786caf8d/src/fv/job.py "dan-slater/footyviz — src/fv/job.py"
[F-identity]: https://github.com/dan-slater/footyviz/blob/4493f88d554a80dc3d9858c276173a17786caf8d/src/fv/identity.py "dan-slater/footyviz — src/fv/identity.py"
[M-pipeline]: https://github.com/a6rahamjr/matchtrace-cv/blob/aa61cd6c301238875387df51b0467f401bc0b880/src/matchtrace/inference/pipeline.py "a6rahamjr/matchtrace-cv — src/matchtrace/inference/pipeline.py"
[M-eval]: https://github.com/a6rahamjr/matchtrace-cv/blob/aa61cd6c301238875387df51b0467f401bc0b880/src/matchtrace/evaluation/evaluate.py "a6rahamjr/matchtrace-cv — src/matchtrace/evaluation/evaluate.py"
[360-pipeline]: https://github.com/humac/soccer360/blob/7b5546c8d0a648dd4c5037dfc70a9d0b09ee3fbb/src/pipeline.py "humac/soccer360 — src/pipeline.py"
[360-track]: https://github.com/humac/soccer360/blob/7b5546c8d0a648dd4c5037dfc70a9d0b09ee3fbb/src/tracker.py "humac/soccer360 — src/tracker.py"
[360-watch]: https://github.com/humac/soccer360/blob/7b5546c8d0a648dd4c5037dfc70a9d0b09ee3fbb/src/watcher.py "humac/soccer360 — src/watcher.py"
[A-calib]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/field_registration/fixed_camera_runner.py "outstandingcandy/AgentCoach — goalinsight/field_registration/fixed_camera_runner.py"
[A-gates]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/strongsort/gates.py "outstandingcandy/AgentCoach — goalinsight/tracking/strongsort/gates.py"
[J-project]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/project_to_pitch.py "spindle79/jersey-tracker — scripts/project_to_pitch.py"
[J-calib]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/calibrate_match.py "spindle79/jersey-tracker — scripts/calibrate_match.py"
[C-status]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/docs/pipeline_status.md "acato/soccer-video-pipeline — docs/pipeline_status.md"
[G-paper]: https://arxiv.org/abs/2404.11335 "SoccerNet GSR paper"
[G-project]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/calibration/bbox2pitch.py "SoccerNet/sn-gamestate — sn_gamestate/calibration/bbox2pitch.py"
[G-team]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/team/tracklet_team_clustering_api.py "SoccerNet/sn-gamestate — sn_gamestate/team/tracklet_team_clustering_api.py"
[J-ocr]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/jersey_ocr.py "spindle79/jersey-tracker — scripts/jersey_ocr.py"
[S-detect]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/src/sportvision/detection.py "MohibShaikh/sportvision — src/sportvision/detection.py"
[A-notice]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/NOTICE "outstandingcandy/AgentCoach — NOTICE"
[J-chunks]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/run_parallel.py "spindle79/jersey-tracker — scripts/run_parallel.py"
[J-io]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/trackio.py "spindle79/jersey-tracker — scripts/trackio.py"
[C-train]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/scripts/train_lora_v11.sh "acato/soccer-video-pipeline — scripts/train_lora_v11.sh"
[C-adapter]: https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-lora/blob/5aca84c3c96f42d582346710fd22322ae0ed6d6b/README.md "acato v11 LoRA card"
[C-gpu]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/docs/vllm-gpu-orchestration.md "acato/soccer-video-pipeline — docs/vllm-gpu-orchestration.md"
[L-engine]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/tracklab/engine/offline.py "TrackingLaboratory/tracklab — tracklab/engine/offline.py"
[P-license]: https://github.com/VlSomers/prtreid/blob/30617a75967e84d5d516959c4b84cbeea6f56493/LICENSE "VlSomers/prtreid — LICENSE"
[S-eval]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/docs/evaluation/README.md "MohibShaikh/sportvision — docs/evaluation/README.md"
[S-track]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/src/sportvision/tracking.py "MohibShaikh/sportvision — src/sportvision/tracking.py"
[T-hm]: https://github.com/Streakfull/Soccer-Tracking/blob/9d3488fc88390cbc64eb444eb08e481b79df0aef/src/pytorchmodels/DeepIOUTracker/HMReidTracker.py "Streakfull/Soccer-Tracking — src/pytorchmodels/DeepIOUTracker/HMReidTracker.py"
[T-jid]: https://github.com/Streakfull/Soccer-Tracking/blob/9d3488fc88390cbc64eb444eb08e481b79df0aef/src/hm_sort_jid/tracker.py "Streakfull/Soccer-Tracking — src/hm_sort_jid/tracker.py"
[B-calib]: https://github.com/budun-ov/football-analytics/blob/cd2c43fce7f54100500058ac0ff5aba7a10fce94/src/calibration/tvcalib_runner.py "budun-ov/football-analytics — src/calibration/tvcalib_runner.py"
[H-events]: https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/events/engine.py "samuehuang/football-cv-analytics — src/events/engine.py"
[F-stats]: https://github.com/dan-slater/footyviz/blob/4493f88d554a80dc3d9858c276173a17786caf8d/src/fv/stats.py "dan-slater/footyviz — src/fv/stats.py"
[M-track]: https://github.com/a6rahamjr/matchtrace-cv/blob/aa61cd6c301238875387df51b0467f401bc0b880/src/matchtrace/inference/tracker.py "a6rahamjr/matchtrace-cv — src/matchtrace/inference/tracker.py"
[360-cluster]: https://github.com/humac/soccer360/blob/7b5546c8d0a648dd4c5037dfc70a9d0b09ee3fbb/src/player_cluster.py "humac/soccer360 — src/player_cluster.py"
[L-readme]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/README.md "TrackingLaboratory/tracklab — README.md"
[RF-readme]: https://github.com/roboflow/rf-detr/blob/e56ec6152b0eb1ac8bcca8690aa7ba59761d861f/README.md "roboflow/rf-detr — README.md"
[GTA-code]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/refine_tracklets.py "sjc042/gta-link — refine_tracklets.py"
[MC-code]: https://github.com/tstanczyk95/McBytePlusPlus/blob/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9/yolox/tracker/mcbyteplusplus_tracker__with_reid.py "tstanczyk95/McBytePlusPlus — yolox/tracker/mcbyteplusplus_tracker__with_reid.py"
[SO-action]: https://github.com/ML-KULeuven/socceraction/blob/93a1242d46c104889205753accaabadb00c45c6d/README.md "ML-KULeuven/socceraction — README.md"
[UN-readme]: https://github.com/UnravelSports/unravelsports/blob/2bd24d7e2b3cc19cbdba6486c7abb70270a732ad/README.md "UnravelSports/unravelsports — README.md"
[P-readme]: https://github.com/VlSomers/prtreid/blob/30617a75967e84d5d516959c4b84cbeea6f56493/README.md "VlSomers/prtreid — README.md"
[TV-readme]: https://github.com/MM4SPA/tvcalib/blob/1222c5230af2742395d74918ed6f34eb2b9bf7f9/README.md "MM4SPA/tvcalib — README.md"
[FP-readme]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/README.md "JeremieOchin/FOOTPASS — README.md"
