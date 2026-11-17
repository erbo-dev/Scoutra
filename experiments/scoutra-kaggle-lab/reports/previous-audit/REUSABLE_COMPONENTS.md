# H. Best Reusable Components
**These are research candidates, not code imported into Scoutra.** “Reuse” may mean adopting an algorithm or data contract rather than copying code. Source licenses are not whole-stack clearance.

No equivalent ML components were found in the accessible Scoutra frontend. Equivalence in an unseen backend is unknown. Integration estimates below are relative scope judgments; no engineering-day estimates or performance gains are asserted. [Scoutra-package]

## Component inventory
| Candidate and relevant source | Algorithm / expected input → output | Dependencies and license | Adaptation / likely breakage | Why useful; reuse vs self-implementation |
|---|---|---|---|---|
| AgentCoach stage framework: `pipeline/_pipeline.py` | Stage DAG/order with artifact reuse; video/config/prior artifact → stage outputs/run metadata | Python stage framework; Apache root | Define Scoutra schema, invalidation and completion semantics; partial files may pass naive existence checks | Study explicit boundaries; implementing a small equivalent may be simpler than adopting its full web/cloud stack [A-pipeline] |
| AgentCoach consolidation: `_consolidator.py`, `_runner.py` | Similarity graph + cannot-links + jersey votes; track features/co-occurrence/OCR → track-to-player mapping | NumPy and crop/OCR/Re-ID producers; Apache root, upstream assets separate | Split impure tracks first, preserve uncertainty and substitutions; wrong team/OCR can force false splits | Clear match-identity layer; algorithm study saves design effort, full runner includes substantial application coupling [A-consolidate] [A-runner] |
| TrackLab interfaces, offline engine and state | Module-level stages and table outputs; dataset/detections → cached enriched detections/evaluation | Hydra, pandas/PyTorch, selected plugins; MIT framework | Timestamp/dataset adapters; large per-video state; avoid equating saved output with live-state checkpoint | Strong experiment harness likely easier than rebuilding every evaluation adapter [L-engine] [L-state] |
| PRTReID + part-aware metric: GSR `prtreid_api.py`, TrackLab `nn_matching.py` | RGB crop → part embeddings, visibility, role; visible-parts distances → association costs | PyTorch, HRNet, PRT/BPB; Hippocratic code; Zenodo weight metadata CC-BY | Crop scale, visibility model, color order, training domain, checkpoint config; global-only adapter loses method | Strong football-specific experimental candidate; model training from scratch is far harder, but code/weight rights and validation first [G-reid] [L-parts] [P-license] [Z-prt] |
| McByte++ `mcbyteplusplus_tracker__with_reid.py` | Online motion/mask-assisted association + feature bank; detections/frames/masks → tracks/export IDs | YOLOX-style stack, torchreid, mask producer; Apache root, asset terms separate | Quality thresholds, optional Re-ID activation, no roster/number memory, no restart bank persistence established | Direct model of re-entry memory; likely cheaper to benchmark existing pipeline than reproduce all internals blindly [MC-code] [MC-install] |
| GTA `Tracklet.py`, `refine_tracklets*.py` | DBSCAN split + offline feature linking; MOT tracklets/features → refined tracks | NumPy/SciPy/sklearn/PyTorch, OSNet features; MIT | Domain thresholds, tiny crops, temporal validity, pairwise memory; no proven optimal global solution | Especially valuable split-before-merge comparator; pure association logic more reusable than its legacy training stack [GTA-code] [GTA-batch] [GTA-gen] |
| jersey-tracker `run_parallel.py`, `merge_chunks.py`, `trackio.py` | Overlap job plan, IoU boundary links, row writers; source/time spans → merged observations/manifest | OpenCV/FFmpeg, subprocess, optional PyArrow; MIT scripts, detector terms separate | Worker GPU scheduling, PTS alignment, failed intervals, one-to-one links, live-state restart | Concrete chunk/storage pattern; adapt small utilities or independently implement contracts instead of importing camera-specific scripts [J-chunks] [J-merge] [J-io] |
| Jersey OCR: `jersey_ocr.py::vote_number`; jersey-number-pipeline `main.py` | Track crops → legibility/pose/number evidence → aggregated number/unknown | PARSeq, EasyOCR; MIT voting wrapper vs noncommercial full specialist pipeline | Dataset fonts, back-facing crops, correlated reads, duplicate numbers, unreadable frames | Weighted temporal voting is small enough to implement locally; specialist crop/legibility research avoids naive full-box OCR [J-ocr] [JR-main] [JR-license] |
| TVCalib + camera-specific adapters | Pitch markings/camera priors → calibrated mapping; feet pixels → field coordinates | PyTorch/camera geometry/segmentation assets; MIT core | Youth dimensions, fisheye, static/PTZ model, visible lines, pitch symmetry, calibration confidence | Optimizer is substantial; adapt existing math after camera-model validation rather than reinventing it [TV-code] [J-project] |
| Roboflow Sports `team.py`, `view.py`, soccer example | SigLIP → UMAP → KMeans teams; keypoint correspondences → homography | Transformers, supervision, sklearn/UMAP; MIT package, hosted detector assets separate | Sample quotas, team-label permutation, refs/GKs, unknowns; example accumulates crops | Useful team/projection baseline; compare simple color clustering first, since embeddings may cost more without benefit [R-team] [R-view] [R-main] |
| samuehuang event and QA stages | CSV controller/possession/motion/air evidence → passes/receptions/dribbles/episodes | NumPy/pandas; MIT code, models separate | Definitions, actor identity, sample-rate/coordinate schema; team QA must not become person Re-ID | Clear football state semantics save conceptual effort; substantial Python files favor porting specifications over whole-stack adoption [H-events] [H-identity] [H-run] |
| acato `DualPassDetector`, `FrameSampler` | Selected frame windows + optional ball context → event proposals/diagnostics | FFmpeg, vLLM API, Qwen weights; Apache source, checkpoint grant U | Exact model/mode/quantization, venue calibration, broad timestamps, missing actors, private old assets | Reuse proposal scheduling/diagnostics; do not transplant reported precision as a guarantee [C-dual] [C-frames] [C-weights] |
| FOOTPASS TAAD/GNN/DST | Video features+track ROIs/masks+tactics → per-player action logits; long sequence → denoised events | PyTorchVideo X3D, ROIAlign, Decord, h5py, PyG; conflicting Apache/NC statements | Needs upstream identity/roles/trajectories, coordinate scaling, class imbalance; professional-to-amateur shift | Best actor-centric event research specification; not minimal-adaptation raw-video pipeline [FP-taad] [FP-gnn] [FP-dst] [FP-run] |
| OpenSportsLib native Qwen / event models | Clip/question or spotting manifest → answer/event predictions | Transformers/PEFT/bitsandbytes or specialized backbones; AGPL/commercial option | Config/schema/version, frame/pixel budget, dataset and model grants | Saves model-serving/evaluation machinery for research; may be too heavy/license-sensitive for immediate product integration [OSL-model] [OSL-zoo] |
| socceraction / unravelsports | Structured events → SPADL/VAEP/xT; calibrated tracking → graphs/pressing/formation analysis | pandas/ML libraries or Polars/Kloppy/graph stack; MIT / MPL-2.0 | Missing offscreen players, noisy coordinates, event ontology, identity/time units, training domain | Large downstream engineering saving once reliable inputs exist; does not solve perception [SO-action] [UN-readme] |
| SportVision review concepts | Cached observations + manual corrections → evidence-aware report/render | NumPy/OpenCV and UI extras; Apache root | Correction scope after ID switches, cached-model provenance; possession remains a proxy | Useful product-quality pattern: uncertainty visible to users, original and edited evidence distinguishable [S-readme] [S-pipeline] |
| soccer360 watcher and reframer architecture | Stable input file/fingerprint → queued job/artifacts and virtual-camera output | Watchdog/FFmpeg/360° tools; root license absent | Input geometry incompatible with ordinary videos; custom assets and rights | Study dedupe/operational design; do not copy source into product without a grant [360-watch] [360-pipeline] |

## Additional detector choices
RF-DETR, D-FINE, and official RT-DETR all have genuine source and pretrained releases. Their primary input/output contract is images → boxes/classes/confidences, which makes controlled replacement experiments fairly contained. The harder work is dataset labeling, correct resolution, class mapping, metric calibration, and latency measurement. A more accurate COCO detector does not automatically detect 3-pixel footballs or preserve teammate identity. [RF-code] [D-code] [RT-readme]

ByteTrack is an inexpensive local reference; BoT-SORT is a stronger mechanistic comparator when moving-camera compensation and appearance are enabled; StrongSORT++ postprocessing is a separate algorithm family. Keep observations fixed when comparing these trackers. [BYTE-code] [BOT-code] [STR-readme]

## Specialist scores within the requested decision framework
These scores describe **module suitability**, so do not rank them directly against whole platforms. Same 1–5 scales/weights as REPOSITORY_COMPARISON.md: R/C/F/P/L/M/E/S = 20/20/15/15/10/10/5/5%. Unknown model/data license suitability remains N/A.

| Module/project | R | C | F | P | L | M | E | S | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TrackLab framework | 5 | 4 | 4 | 4 | 3 | 4 | 3 | 4 | Strong candidate |
| PRTReID/BPB research stack | 5 | 4 | 5 | 3 | 3 | 2 | 2 | 1 | Promising but requires validation |
| McByte++ | 5 | 4 | 4 | 3 | 3 | 2 | 2 | N/A | Promising but requires validation |
| GTA | 5 | 4 | 4 | 3 | 3 | 2 | 3 | N/A | Strong candidate for offline experiments |
| RF-DETR Apache models | 4 | 5 | 2 | 4 | 4 | 4 | 4 | 4 | Strong candidate for detector tests |
| D-FINE | 4 | 4 | 2 | 4 | 4 | 3 | 3 | N/A | Useful for individual modules |
| Official RT-DETR | 4 | 4 | 2 | 4 | 4 | 4 | 3 | N/A | Useful for individual modules |
| ByteTrack | 4 | 5 | 2 | 4 | 3 | 3 | 4 | N/A | Useful local baseline |
| BoT-SORT | 4 | 4 | 2 | 3 | 3 | 3 | 3 | N/A | Useful comparison baseline |
| StrongSORT/++ | 4 | 4 | 2 | 3 | 3 | 2 | 2 | 1 | Research reference only |
| TVCalib | 4 | 4 | 5 | 3 | 3 | 3 | 2 | N/A | Useful calibration module |
| Roboflow Sports | 3 | 3 | 4 | 3 | 2 | 3 | 4 | N/A | Useful for individual modules |
| Jersey-number-pipeline | 4 | 4 | 5 | 2 | 2 | 2 | 2 | 1 | Research reference only |
| FOOTPASS | 5 | 4 | 5 | 2 | 4 | 2 | 1 | N/A | Research reference pending rights/assets |
| OpenSportsLib | 4 | 4 | 5 | 3 | 3 | 3 | 2 | 1 | Research reference / licensed option |
| socceraction | 4 | 5 | 5 | 4 | 4 | 4 | 3 | 4 | Strong downstream candidate |
| unravelsports | 4 | 4 | 5 | 3 | 4 | 4 | 3 | 3 | Useful downstream modules |

Rationale follows the inventory row for each component. High C means source coverage of its own narrow task, not full Scoutra coverage. High F for calibration/events refers to football formulation, not amateur generalization. P is reduced for missing private assets or older environments; L reflects stageability/storage structure, not measured 90-minute performance. E is lower when rich tracking/role schemas or multiple model stacks are prerequisites. Source-only license scores for framework/library rows are higher where grants are clear; they still do not license external data.

SoccerNet Re-ID, Tracking and Jersey task kits have **N/A weighted product scores** because ingestion, full pipeline completeness and integration effort are not their claimed purpose. Their value is evaluation data/protocol, subject to access terms. General VLM families likewise require checkpoint-specific trials rather than a popularity score. [SN-reid] [SN-track] [SN-jersey]

## Integration risks shared by all components
Coordinate conventions (xyxy versus xywh, pixel versus normalized versus metre), BGR/RGB order, effective FPS, original PTS, team-ID permutation, period-relative time, model class IDs, and the meaning of “unknown” must be normalized. Otherwise apparently valid arrays can yield systematically wrong analytics. Explicit adapters and a small reference fixture may be more valuable than choosing a marginally stronger neural architecture.

For long-video compatibility, prefer stateless/batched perception with explicit artifact contracts; keep local tracker state sequential; perform global identity over compact tracklets; invoke expensive models only on selected evidence. These are proposed architecture principles, not measured wins.

### Weighted specialist decision aid
Known subtotal uses the scored share of the original 100-point weighting; unknown-license intervals retain the missing 5%.

| Module | Known subtotal / coverage | Possible total |
|---|---|---|
| TrackLab framework | 81 / 100 | 81 |
| PRTReID/BPB research stack | 73 / 100 | 73 |
| McByte++ | 69 / 95 | 70–74 |
| GTA | 70 / 95 | 71–75 |
| RF-DETR Apache models | 78 / 100 | 78 |
| D-FINE | 67 / 95 | 68–72 |
| Official RT-DETR | 69 / 95 | 70–74 |
| ByteTrack | 70 / 95 | 71–75 |
| BoT-SORT | 62 / 95 | 63–67 |
| StrongSORT/++ | 60 / 100 | 60 |
| TVCalib | 70 / 95 | 71–75 |
| Roboflow Sports | 59 / 95 | 60–64 |
| Jersey-number-pipeline | 64 / 100 | 64 |
| FOOTPASS | 70 / 95 | 71–75 |
| OpenSportsLib | 71 / 100 | 71 |
| socceraction | 86 / 100 | 86 |
| unravelsports | 78 / 100 | 78 |

## Direct sources

[Scoutra-package]: https://github.com/erbo-dev/Scoutra/blob/a6a02f176f55dd044112951506fa63a919ca7e5b/package.json "erbo-dev/Scoutra — package.json"
[A-pipeline]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/pipeline/_pipeline.py "outstandingcandy/AgentCoach — goalinsight/pipeline/_pipeline.py"
[A-consolidate]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/track_consolidation/_consolidator.py "outstandingcandy/AgentCoach — goalinsight/track_consolidation/_consolidator.py"
[A-runner]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/track_consolidation/_runner.py "outstandingcandy/AgentCoach — goalinsight/track_consolidation/_runner.py"
[L-engine]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/tracklab/engine/offline.py "TrackingLaboratory/tracklab — tracklab/engine/offline.py"
[L-state]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/tracklab/datastruct/tracker_state.py "TrackingLaboratory/tracklab — tracklab/datastruct/tracker_state.py"
[G-reid]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/reid/prtreid_api.py "SoccerNet/sn-gamestate — sn_gamestate/reid/prtreid_api.py"
[L-parts]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/plugins/track/bpbreid_strong_sort/sort/nn_matching.py "TrackingLaboratory/tracklab — plugins/track/bpbreid_strong_sort/sort/nn_matching.py"
[P-license]: https://github.com/VlSomers/prtreid/blob/30617a75967e84d5d516959c4b84cbeea6f56493/LICENSE "VlSomers/prtreid — LICENSE"
[Z-prt]: https://zenodo.org/records/10653453 "PRTReID checkpoint archive and license metadata"
[MC-code]: https://github.com/tstanczyk95/McBytePlusPlus/blob/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9/yolox/tracker/mcbyteplusplus_tracker__with_reid.py "tstanczyk95/McBytePlusPlus — yolox/tracker/mcbyteplusplus_tracker__with_reid.py"
[MC-install]: https://github.com/tstanczyk95/McBytePlusPlus/blob/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9/INSTALLATION.md "tstanczyk95/McBytePlusPlus — INSTALLATION.md"
[GTA-code]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/refine_tracklets.py "sjc042/gta-link — refine_tracklets.py"
[GTA-batch]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/refine_tracklets_batched.py "sjc042/gta-link — refine_tracklets_batched.py"
[GTA-gen]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/generate_tracklets.py "sjc042/gta-link — generate_tracklets.py"
[J-chunks]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/run_parallel.py "spindle79/jersey-tracker — scripts/run_parallel.py"
[J-merge]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/merge_chunks.py "spindle79/jersey-tracker — scripts/merge_chunks.py"
[J-io]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/trackio.py "spindle79/jersey-tracker — scripts/trackio.py"
[J-ocr]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/jersey_ocr.py "spindle79/jersey-tracker — scripts/jersey_ocr.py"
[JR-main]: https://github.com/mkoshkina/jersey-number-pipeline/blob/007d54e5530a66616ed5081ca35e0028b36aadb5/main.py "mkoshkina/jersey-number-pipeline — main.py"
[JR-license]: https://github.com/mkoshkina/jersey-number-pipeline/blob/007d54e5530a66616ed5081ca35e0028b36aadb5/license.txt "mkoshkina/jersey-number-pipeline — license.txt"
[TV-code]: https://github.com/MM4SPA/tvcalib/blob/1222c5230af2742395d74918ed6f34eb2b9bf7f9/tvcalib/module.py "MM4SPA/tvcalib — tvcalib/module.py"
[J-project]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/project_to_pitch.py "spindle79/jersey-tracker — scripts/project_to_pitch.py"
[R-team]: https://github.com/roboflow/sports/blob/42c80c06b6b65a7f89455b89fe31cdf4c38ba227/sports/common/team.py "roboflow/sports — sports/common/team.py"
[R-view]: https://github.com/roboflow/sports/blob/42c80c06b6b65a7f89455b89fe31cdf4c38ba227/sports/common/view.py "roboflow/sports — sports/common/view.py"
[R-main]: https://github.com/roboflow/sports/blob/42c80c06b6b65a7f89455b89fe31cdf4c38ba227/examples/soccer/main.py "roboflow/sports — examples/soccer/main.py"
[H-events]: https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/events/engine.py "samuehuang/football-cv-analytics — src/events/engine.py"
[H-identity]: https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/analytics/identity_qa.py "samuehuang/football-cv-analytics — src/analytics/identity_qa.py"
[H-run]: https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/run_pipeline.py "samuehuang/football-cv-analytics — run_pipeline.py"
[C-dual]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/dual_pass_detector.py "acato/soccer-video-pipeline — src/detection/dual_pass_detector.py"
[C-frames]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/frame_sampler.py "acato/soccer-video-pipeline — src/detection/frame_sampler.py"
[C-weights]: https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-fp8/blob/ab56ac6743fc4f5ec299ed951f989d37f5874c56/README.md "acato v11 FP8 checkpoint card"
[FP-taad]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/models/model_TAAD_baseline.py "JeremieOchin/FOOTPASS — models/model_TAAD_baseline.py"
[FP-gnn]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/models/model_GNN.py "JeremieOchin/FOOTPASS — models/model_GNN.py"
[FP-dst]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/models/model_DST.py "JeremieOchin/FOOTPASS — models/model_DST.py"
[FP-run]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/run_TAAD_on_matches.py "JeremieOchin/FOOTPASS — run_TAAD_on_matches.py"
[OSL-model]: https://github.com/OpenSportsLab/opensportslib/blob/b2fe8f16a35336d93b1b0eabe1741bbbbc36cd68/opensportslib/models/base/qwen_vl_native.py "OpenSportsLab/opensportslib — opensportslib/models/base/qwen_vl_native.py"
[OSL-zoo]: https://github.com/OpenSportsLab/opensportslib/blob/b2fe8f16a35336d93b1b0eabe1741bbbbc36cd68/docs/model-zoo.md "OpenSportsLab/opensportslib — docs/model-zoo.md"
[SO-action]: https://github.com/ML-KULeuven/socceraction/blob/93a1242d46c104889205753accaabadb00c45c6d/README.md "ML-KULeuven/socceraction — README.md"
[UN-readme]: https://github.com/UnravelSports/unravelsports/blob/2bd24d7e2b3cc19cbdba6486c7abb70270a732ad/README.md "UnravelSports/unravelsports — README.md"
[S-readme]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/README.md "MohibShaikh/sportvision — README.md"
[S-pipeline]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/src/sportvision/pipeline.py "MohibShaikh/sportvision — src/sportvision/pipeline.py"
[360-watch]: https://github.com/humac/soccer360/blob/7b5546c8d0a648dd4c5037dfc70a9d0b09ee3fbb/src/watcher.py "humac/soccer360 — src/watcher.py"
[360-pipeline]: https://github.com/humac/soccer360/blob/7b5546c8d0a648dd4c5037dfc70a9d0b09ee3fbb/src/pipeline.py "humac/soccer360 — src/pipeline.py"
[RF-code]: https://github.com/roboflow/rf-detr/blob/e56ec6152b0eb1ac8bcca8690aa7ba59761d861f/src/rfdetr/detr.py "roboflow/rf-detr — src/rfdetr/detr.py"
[D-code]: https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/zoo/dfine/dfine_decoder.py "Peterande/D-FINE — src/zoo/dfine/dfine_decoder.py"
[RT-readme]: https://github.com/lyuwenyu/RT-DETR/blob/df2a96c8c3a098567a1149bf0248da6efe6d495d/README.md "lyuwenyu/RT-DETR — README.md"
[BYTE-code]: https://github.com/FoundationVision/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/byte_tracker.py "FoundationVision/ByteTrack — yolox/tracker/byte_tracker.py"
[BOT-code]: https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/tracker/bot_sort.py "NirAharon/BoT-SORT — tracker/bot_sort.py"
[STR-readme]: https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/README.md "dyhBUPT/StrongSORT — README.md"
[SN-reid]: https://github.com/SoccerNet/sn-reid/blob/621e2b0f2d2a7a3e207b8dd747542b6608bf72db/README.md "SoccerNet/sn-reid — README.md"
[SN-track]: https://github.com/SoccerNet/sn-tracking/blob/b0bbba35e07ff58010b6313ef8aa59ef663ad392/README.md "SoccerNet/sn-tracking — README.md"
[SN-jersey]: https://github.com/SoccerNet/sn-jersey/blob/2f43b48c59eefe0bb5d948888db07f55f51208ad/README.md "SoccerNet/sn-jersey — README.md"
