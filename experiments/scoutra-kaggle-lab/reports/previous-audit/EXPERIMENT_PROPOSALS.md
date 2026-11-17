# I–J. Research Gaps and Recommended Experiments
**Status: proposals only. None executed. No training, inference, GPU work or checkpoint download was performed for this audit.**

## I. Questions the evidence cannot answer
1. Scoutra's actual ML baseline and private experiments are unavailable. The accessible code is a frontend. [Scoutra-readme]
2. Which detector preserves tiny-player and ball recall under Scoutra's camera heights, 720p/1080p compression and weather?
3. Which representation distinguishes same-kit youth players after minutes offscreen, rather than only adjacent frames?
4. How often local tracks already contain multiple people, and whether splitting helps more than it harms.
5. Whether body-part visibility remains meaningful at small crop sizes.
6. Whether number recognition works often enough to justify a hard identity constraint.
7. How calibration errors propagate into gates, distance, sprint speeds and event attribution.
8. Whether any reported event result transfers to unseen amateur venues with tight actor/time evaluation.
9. Which quantized VLM/engine configuration fits T4 and preserves output quality.
10. Whether a complete match can be resumed, reprocessed and audited without silent holes or inconsistent identities.
11. Whether professional-broadcast tactical/event priors generalize to youth formations, rolling substitutions and nonstandard pitches.
12. Exact grants for conflicting/custom model licenses and missing private assets.

These are not solved by README popularity, generic video benchmarks, or a convincing minimap animation.

## Shared evaluation design
**Required Scoutra materials:** original match files with decode metadata; camera/lens information where available; actual pitch dimensions; roster/number/period/substitution notes if known; a small hand-labeled track/event set. None was accessible here.

Build a **match-disjoint** development/test split, also holding out at least one venue/camera setup. Split by whole source match, not random neighboring frames. Include static and moving cameras, similar kits, small players, glare/night lighting, occlusions, re-entry, keeper/referee views, halftime and substitutions. Resizing a high-quality clip is useful controlled analysis but does not replace native low-quality footage.

For identity, annotate physical IDs, ambiguous/unknown regions and track contamination; for events, annotate class, actor, start/contact timestamp, visible/not visible, and reviewer agreement. Retain visibility masks when a player/ball is offscreen. Public SoccerNet Tracking/GSR and FOOTPASS can supplement, subject to terms; do not label their professional data as amateur validation. [SN-track] [G-paper] [DS-fp]

All numerical success gates below are **proposed decision thresholds**, not existing results or validated product requirements. Adjust them before opening the held-out test set. Estimate uncertainty by match-level resampling where enough independent matches exist; tiny samples require a plainly exploratory conclusion.

## Experiment 1 — Detection and spatial sampling
**Hypothesis:** a football-relevant resolution/crop policy matters at least as much as a generic detector ranking.

**Dataset:** labeled player/keeper/referee/ball frames sampled across held-out match conditions, with dense short sequences around contacts. Record object pixel height/diameter and visibility.

**Methods:** one documented football YOLO reference (if rights permit); selected Apache RF-DETR size; D-FINE; official RT-DETR. First compare available weights; if training adaptation is later authorized, use equal training data and budget in a separate experiment. Compare full image versus sliced/high-resolution processing with identical merging rules. [RF-code] [D-code] [RT-readme]

**Metrics:** AP50/AP50–95; player and ball recall at a fixed false-positive rate; pixel-size bins; feet-point error; decode/preprocess/inference/postprocess wall time; peak CPU RAM and GPU allocated/reserved memory.

**Hardware:** one T4 at a time, second T4 for independent repeats; H100 later for larger-resolution/batch sweeps. Do not infer TensorRT speed from PyTorch.

**Proposed success:** improve tiny-ball recall by >=5 percentage points at matched FP rate, without >2-point player-recall regression, and fit the stated memory envelope. **Failure:** no held-out gain, duplicate slice detections, or unusable memory/latency. Keep the simpler baseline if results overlap.

## Experiment 2 — Online tracking at fixed detections
**Hypothesis:** appearance and camera-motion handling improve association on moving/occluded football footage, but may not help tiny fixed-camera crops.

**Dataset:** physical-ID annotated sequences with crossings, camera pans, small boxes and offscreen transitions. Include public football association evaluation separately.

**Methods:** ByteTrack, BoT-SORT with explicit Re-ID/GMC ablations, GSR part-aware StrongSORT, AgentCoach-style local tracker, McByte++ with/without its re-entry module. Freeze detections, class mapping and processed timestamps. [BYTE-code] [BOT-code] [L-config] [A-tracker] [MC-code]

**Metrics:** HOTA/AssA/DetA, IDF1, switches per player-minute, fragmentation, tracklet purity; error by camera regime; throughput including feature/mask extraction.

**Hardware:** sequential candidates on T4, reducing batch/window only if logged; McByte++ mask+feature memory feasibility is itself a measurement. H100 later for equal-settings comparison.

**Proposed success:** >=3-point AssA or IDF1 gain with no material purity loss and acceptable measured compute. **Failure:** apparent gain caused by skipped hard frames, lower recall, or forced merges.

## Experiment 3 — Re-ID representation and quality
**Hypothesis:** part-aware football features outperform global-only features on partially occluded same-team players, if enough pixels remain.

**Dataset:** manually verified same-player/different-teammate crop pairs across time and viewpoints, grouped by match. Separate 30-second retrieval from minute-scale re-entry.

**Methods:** OSNet; AgentCoach global PRT embedding; GSR full part+visibility PRT route; a licensed CLIP-ReID comparator only if exact weights are obtainable. Keep crop/time sampling fixed. [A-reid] [G-reid] [L-parts]

**Metrics:** same-team retrieval mAP/Rank-1, false match rate at selected recall, feature-distance calibration, cross-gap ROC/PR, degradation by crop height/occlusion. Retain ambiguous labels.

**Hardware:** batched crops on a T4; second T4 independent feature runs; H100 optional for throughput only.

**Proposed success:** >=5-point minute-gap Rank-1 gain or lower false-match rate at the same recall, with consistent held-out-camera results. **Failure:** part-aware advantage exists only for large broadcast crops or vanishes on amateur data.

## Experiment 4 — Offline identity: split, then link
**Hypothesis:** detecting contaminated tracklets before global association reduces false player-stat aggregation.

**Dataset:** fixed tracklets from Experiment 2, annotated switches and match identities; include clean controls that should not split.

**Methods:** no postprocessing; AgentCoach constrained union; footyviz constrained average linkage; GTA connect-only; GTA split+connect. A min-cost-flow prototype is a later research comparator, not something implemented now. [A-consolidate] [F-identity] [GTA-code] [GTA-batch]

**Metrics:** pairwise link precision/recall, IDF1, purity, false merges, unnecessary splits, identity coverage at a fixed link precision, CPU/GPU/RAM growth versus tracklet count.

**Hardware:** CPU association plus T4 feature extraction; H100 only if larger feature comparisons warrant it.

**Proposed success:** >=5-point IDF1 improvement while accepted-link precision stays >=99% on the selected test, with fewer contaminated player-minutes. **Failure:** better IDF1 hides dangerous mergers or fragmentation rises sharply. A 99% observed precision on a small set is not a statistical guarantee of 99% deployment precision.

## Experiment 5 — Jersey numbers and confidence aggregation
**Hypothesis:** crop selection/legibility gating plus temporal voting beats full-box OCR and reduces false confident identity links.

**Dataset:** native player track crops with readable-number labels, unreadable frames, stylized fonts, repeated team numbers and opposing identical numbers.

**Methods:** EasyOCR; PARSeq fixed torso crop; pose/legibility-gated specialist pipeline as licensed research; small Qwen OCR on best crops; frame prediction versus temporally weighted vote. [J-ocr] [JR-main] [G-jersey]

**Metrics:** full-number accuracy, accepted-read precision/coverage, wrong-confident-number rate, calibration, inference per tracklet; downstream identity false merges. Separate many adjacent frames from independent views.

**Hardware:** T4 stage-separated models; 4-bit small Qwen if supported; H100 for larger-model comparison later.

**Proposed success:** >=98% precision on accepted tracklet numbers at useful prespecified coverage, with reduced identity error. **Failure:** coverage improves by accepting wrong digits or confidence repeats correlated mistakes. Unknown is a valid output.

## Experiment 6 — Calibration and trustworthy kinematics
**Hypothesis:** camera-specific calibration with uncertainty yields more reliable feet coordinates and identity gates than a fixed arbitrary homography.

**Dataset:** static and pan/zoom segments, known pitch dimensions, manually marked correspondences, control points not used to fit; reference trajectories/known-distance tests if available.

**Methods:** manual fixed homography; lens-corrected fixed mapping; TVCalib; GSR NBJW/PnL alternatives where assets/rights permit; no metric output when calibration fails. [J-project] [TV-code] [G-config]

**Metrics:** held-out landmark reprojection, pitch-point metre error, temporal jitter, invalid projections, propagated distance/speed error, false identity-gate rejection. For airborne balls, evaluate ground-projection error separately.

**Hardware:** CPU/OpenCV baseline; one T4 learned calibration; H100 larger batching later.

**Proposed success:** median held-out feet error <=1m in valid regions and prespecified p95 bound, with lower kinematic/gating error than the manual baseline. **Failure:** pretty lines but large field-point errors, drift or symmetry flips. Thresholds must match Scoutra's actual metric tolerances.

## Experiment 7 — Ball control and player-attributed events
**Hypothesis:** explicitly separating control, possession, last touch and airborne/transit state improves event attribution over nearest-player proximity.

**Dataset:** dense ball/actor/contact annotations with unknown/offscreen intervals, passes, receptions, losses, tackles, shots and saves.

**Methods:** nearest-player baseline; samuehuang state formulation; AgentCoach rule detectors; FOOTPASS TAAD as an offline research comparator only when required ROI/identity inputs and rights exist. Evaluate oracle tracks/ball first, predicted inputs second. [S-poss] [H-events] [A-events] [FP-taad] [FP-run]

**Metrics:** control-state macro-F1 including unknown; event one-to-one precision/recall/F1 by class; actor+event+time F1 at ±0.5s, ±1s, ±2s; actor accuracy conditional on a matched event; duplicate event count; event-count bias per player.

**Hardware:** CPU cached rules + T4; batch-1 specialized temporal model if it fits; H100 later. No throughput assumed.

**Proposed success:** >=5-point actor-event F1 gain without increasing false player counts beyond the prespecified tolerance. **Failure:** only team/event coverage improves, or oracle gains collapse with predicted tracks.

## Experiment 8 — Football VLM versus newer general VLM
**Hypothesis:** domain adaptation and evidence selection can improve event verification, but model size alone will not solve missing-ball/actor ambiguity.

**Dataset:** held-out camera/venue clips with positive events, hard negatives, no-visible-ball clips, offscreen actors and time/actor annotations. Exclude training matches from custom checkpoints as far as provenance allows; unknown overlap limits conclusions.

**Methods:** Qwen3-VL small baseline; Qwen3.5-9B; SoccerChat; OSL Qwen XFoul for foul-only tasks; acato v11 32B versus public base on H100; Qwen3.6-35B-A3B and one InternVL/InternVideo control later. Compare equal frame/pixel budgets before scaling budgets. [Q35-card] [CHAT-card] [OSL-card] [C-weights] [Q36-card] [IV-card]

**Metrics:** class macro-F1, actor+time F1, abstention quality, hallucinations on negative/occluded clips, JSON schema validity, confidence calibration; per-window latency and memory. Foul QA accuracy is a separate task from event spotting.

**Hardware:** T4 for small 4-bit checkpoints with supported kernels; proposed H100 80GB for 32B/35B modes, using explicit context limits. Quantization mode is not guaranteed to load. [V-quant]

**Proposed success:** >=5-point held-out macro-F1 over the matched baseline with no actor/hallucination regression. **Failure:** gains only on same-venue/tuned clips, explanation fluency without factual correctness, or actor IDs copied from leading prompts.

## Experiment 9 — Expensive-model scheduling and quantization
**Hypothesis:** candidate-driven VLM inference can retain most useful event recall while reducing expensive visual calls.

**Dataset:** complete matches with all target events and candidate-window coverage annotations, plus cached perception.

**Methods:** uniform sparse windows; detector/ball/control-triggered proposals; small-model triage → large verification; large-model direct windows; 4-bit/8-bit/FP16 where exact formats are supported. Keep result matching one-to-one; report ±60s clip coverage separately if relevant. [C-triage] [C-dual]

**Metrics:** candidate recall ceiling, end-to-end actor-event F1, windows and visual tokens per match, duplicated decoding, actual VRAM, stage wall time and confidence drift.

**Hardware:** 2×T4 separate workers/small VLM; H100 large VLM stage; TP versus layer sharding measured separately.

**Proposed success:** >=50% fewer large-model windows with <=2-point event-recall loss and no material actor degradation. **Failure:** rare events are systematically filtered out or quantization changes labels unpredictably.

## Experiment 10 — Complete-match continuity and restart
**Hypothesis:** bounded chunks plus a match identity layer can process/recover a full match without changing accepted observations or silently losing intervals.

**Dataset:** at least two complete 90+ minute matches with halftime and substitution/re-entry annotations; retain full originals. Also test native VFR or deliberately interrupted decoding if representative.

**Methods:** sequential reference; overlapping chunks; interrupted worker restart; rerun one failed partition; change a model/config and verify dependency invalidation. Compare explicit identity memory with independent chunk IDs. Use source-inspired patterns from jersey-tracker, TrackLab, AgentCoach and soccer360, not their unsupported guarantees. [J-chunks] [J-merge] [L-state] [A-pipeline] [360-watch]

**Metrics:** missing/duplicate observations, boundary association errors, identity consistency across periods, exact deterministic output differences or documented floating tolerance, peak RAM/VRAM, artifact bytes/minute, completion and reprocessing time, human corrections.

**Hardware:** Kaggle 2×T4 with session/storage constraints recorded; H100 repeat later using same artifact contracts.

**Proposed success:** zero unmarked gaps/duplicates; all expected partitions complete; resumed deterministic observations match reference; boundary identity no worse than sequential within the prespecified error margin; memory stays within measured limits. **Failure:** hidden holes, duplicate IDs/events, stale mixed-model outputs, or memory increasing without a planned bound.

## Execution order and stopping rules
First obtain representative evidence and clear asset terms. Run detection, fixed-detection tracking and Re-ID studies before optimizing VLM throughput. Evaluate offline identity and OCR next, then calibration/event attribution, then full-match recovery and large-model comparisons. Test perception and long-video robustness early enough to avoid polishing reports over wrong identities.

Stop adding complexity when a simpler baseline matches a method on the prespecified metrics. Failure is useful evidence: if team color/number visibility is inadequate, define supported footage and human-review needs before creating an automated per-player promise.

## Direct sources

[Scoutra-readme]: https://github.com/erbo-dev/Scoutra/blob/a6a02f176f55dd044112951506fa63a919ca7e5b/README.md "erbo-dev/Scoutra — README.md"
[SN-track]: https://github.com/SoccerNet/sn-tracking/blob/b0bbba35e07ff58010b6313ef8aa59ef663ad392/README.md "SoccerNet/sn-tracking — README.md"
[G-paper]: https://arxiv.org/abs/2404.11335 "SoccerNet GSR paper"
[DS-fp]: https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026 "FOOTPASS dataset card and access terms"
[RF-code]: https://github.com/roboflow/rf-detr/blob/e56ec6152b0eb1ac8bcca8690aa7ba59761d861f/src/rfdetr/detr.py "roboflow/rf-detr — src/rfdetr/detr.py"
[D-code]: https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/zoo/dfine/dfine_decoder.py "Peterande/D-FINE — src/zoo/dfine/dfine_decoder.py"
[RT-readme]: https://github.com/lyuwenyu/RT-DETR/blob/df2a96c8c3a098567a1149bf0248da6efe6d495d/README.md "lyuwenyu/RT-DETR — README.md"
[BYTE-code]: https://github.com/FoundationVision/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/byte_tracker.py "FoundationVision/ByteTrack — yolox/tracker/byte_tracker.py"
[BOT-code]: https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/tracker/bot_sort.py "NirAharon/BoT-SORT — tracker/bot_sort.py"
[L-config]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/tracklab/configs/modules/track/bpbreid_strong_sort.yaml "TrackingLaboratory/tracklab — tracklab/configs/modules/track/bpbreid_strong_sort.yaml"
[A-tracker]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/strongsort/tracker.py "outstandingcandy/AgentCoach — goalinsight/tracking/strongsort/tracker.py"
[MC-code]: https://github.com/tstanczyk95/McBytePlusPlus/blob/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9/yolox/tracker/mcbyteplusplus_tracker__with_reid.py "tstanczyk95/McBytePlusPlus — yolox/tracker/mcbyteplusplus_tracker__with_reid.py"
[A-reid]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/reid/prtreid_extractor.py "outstandingcandy/AgentCoach — goalinsight/tracking/reid/prtreid_extractor.py"
[G-reid]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/reid/prtreid_api.py "SoccerNet/sn-gamestate — sn_gamestate/reid/prtreid_api.py"
[L-parts]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/plugins/track/bpbreid_strong_sort/sort/nn_matching.py "TrackingLaboratory/tracklab — plugins/track/bpbreid_strong_sort/sort/nn_matching.py"
[A-consolidate]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/track_consolidation/_consolidator.py "outstandingcandy/AgentCoach — goalinsight/track_consolidation/_consolidator.py"
[F-identity]: https://github.com/dan-slater/footyviz/blob/4493f88d554a80dc3d9858c276173a17786caf8d/src/fv/identity.py "dan-slater/footyviz — src/fv/identity.py"
[GTA-code]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/refine_tracklets.py "sjc042/gta-link — refine_tracklets.py"
[GTA-batch]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/refine_tracklets_batched.py "sjc042/gta-link — refine_tracklets_batched.py"
[J-ocr]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/jersey_ocr.py "spindle79/jersey-tracker — scripts/jersey_ocr.py"
[JR-main]: https://github.com/mkoshkina/jersey-number-pipeline/blob/007d54e5530a66616ed5081ca35e0028b36aadb5/main.py "mkoshkina/jersey-number-pipeline — main.py"
[G-jersey]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/jersey/voting_tracklet_jn_api.py "SoccerNet/sn-gamestate — sn_gamestate/jersey/voting_tracklet_jn_api.py"
[J-project]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/project_to_pitch.py "spindle79/jersey-tracker — scripts/project_to_pitch.py"
[TV-code]: https://github.com/MM4SPA/tvcalib/blob/1222c5230af2742395d74918ed6f34eb2b9bf7f9/tvcalib/module.py "MM4SPA/tvcalib — tvcalib/module.py"
[G-config]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/configs/soccernet.yaml "SoccerNet/sn-gamestate — sn_gamestate/configs/soccernet.yaml"
[S-poss]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/src/sportvision/analytics/possession.py "MohibShaikh/sportvision — src/sportvision/analytics/possession.py"
[H-events]: https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/events/engine.py "samuehuang/football-cv-analytics — src/events/engine.py"
[A-events]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/events/_orchestrator.py "outstandingcandy/AgentCoach — goalinsight/events/_orchestrator.py"
[FP-taad]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/models/model_TAAD_baseline.py "JeremieOchin/FOOTPASS — models/model_TAAD_baseline.py"
[FP-run]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/run_TAAD_on_matches.py "JeremieOchin/FOOTPASS — run_TAAD_on_matches.py"
[Q35-card]: https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md "Official Qwen3.5-9B model card"
[CHAT-card]: https://huggingface.co/SimulaMet/SoccerChat-qwen2-vl-7b/blob/29871536004c0ac788af09cbb87969ab9f6e1410/README.md "SoccerChat checkpoint card"
[OSL-card]: https://huggingface.co/OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora/blob/8c7721ff5fd74a1ad27f64203aa3218c5f9f5db1/README.md "OSL Qwen3-VL XFoul checkpoint card"
[C-weights]: https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-fp8/blob/ab56ac6743fc4f5ec299ed951f989d37f5874c56/README.md "acato v11 FP8 checkpoint card"
[Q36-card]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/345d467b117d1fd5eda9b879a5988de9ae5f47a7/README.md "Official Qwen3.6-35B-A3B model card"
[IV-card]: https://huggingface.co/OpenGVLab/InternVL3_5-8B "InternVL3.5-8B official checkpoint"
[V-quant]: https://docs.vllm.ai/en/v0.19.1/features/quantization/#supported-hardware "vLLM 0.19.1 quantization hardware matrix"
[C-triage]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/triage_scanner.py "acato/soccer-video-pipeline — src/detection/triage_scanner.py"
[C-dual]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/dual_pass_detector.py "acato/soccer-video-pipeline — src/detection/dual_pass_detector.py"
[J-chunks]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/run_parallel.py "spindle79/jersey-tracker — scripts/run_parallel.py"
[J-merge]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/merge_chunks.py "spindle79/jersey-tracker — scripts/merge_chunks.py"
[L-state]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/tracklab/datastruct/tracker_state.py "TrackingLaboratory/tracklab — tracklab/datastruct/tracker_state.py"
[A-pipeline]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/pipeline/_pipeline.py "outstandingcandy/AgentCoach — goalinsight/pipeline/_pipeline.py"
[360-watch]: https://github.com/humac/soccer360/blob/7b5546c8d0a648dd4c5037dfc70a9d0b09ee3fbb/src/watcher.py "humac/soccer360 — src/watcher.py"
