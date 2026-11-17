# K. Final Research Conclusions
**Research-only conclusions, 17 November 2026. No implementation decision is implied.**

## 1. Which existing project is closest to Scoutra?
**AgentCoach/GoalInsight**, by breadth of connected perception, identity, calibration, events and player outputs. Its stage contracts are more useful than adopting its entire application. It is research-grade, has private build inputs and retains match-wide data structures. Scoutra's accessible repository is a frontend, so this is a match to intended requirements, not a measured comparison with a current backend. [A-pipeline] [A-runner] [A-offline] [A-tracking] [Scoutra-readme]

## 2. Which has the most useful tracking architecture?
**SoccerNet GSR + TrackLab** for modular football evaluation, interchangeable perception stages and part-aware association. **McByte++** is the more directly relevant additional online long-absence comparator because it actually stores identity features and reconnects exported IDs. **GTA** is the strongest additional split-and-link reference inspected. These serve different roles; there is no evidence-backed universal winner for Scoutra's videos. [G-config] [L-parts] [MC-code] [GTA-batch]

## 3. Which has the most promising Re-ID approach?
**PRTReID with body-part visibility**, evaluated as the full representation rather than AgentCoach's global-only adapter. Football-specific research and working adapter/checkpoint paths make it more relevant than pedestrian metrics alone. License constraints and tiny-player domain transfer remain material. Compare it against simpler OSNet/appearance baselines before accepting its complexity. [G-reid] [L-parts] [A-reid] [P-license]

## 4. Which best handles long videos?
No inspected system proves reliable end-to-end 90-minute amateur analytics with persistent identity and exact interrupted-job recovery. **jersey-tracker** has the clearest overlapping perception chunks; **acato** has useful full-game sparse event caching; **TrackLab and samuehuang** provide stage reuse patterns. Select their engineering ideas separately and test a complete-match contract. [J-chunks] [J-merge] [C-triage] [L-state] [H-run]

## 5. Which has the most reusable football event pipeline?
For **actor-centric learned event research**, **FOOTPASS** most closely specifies “who did what when,” but assumes tracking/identity/tactical inputs and has rights/reproducibility barriers. For **understandable deterministic event logic**, study **samuehuang** and AgentCoach. For **amateur highlight candidate retrieval**, study **acato**. An event/reel pipeline with broad time tolerance is not a per-player statistics engine. [FP-run] [FP-gnn] [FP-readme] [H-events] [A-events] [C-status]

## 6. Which VLM implementation is most relevant?
**acato's Qwen3-VL soccer event path** is the closest to amateur event/reel requirements and publishes a merged artifact. **OpenSportsLib native Qwen** is a useful library reference for specialist QA and serving; **SoccerChat** is a smaller football QA comparison with a documented quantized example. Include general Qwen3.5/3.6 as controls instead of assuming a football fine-tune always wins. None establishes dense player identity and metric tracking. [C-dual] [C-weights] [OSL-model] [CHAT-card] [Q35-doc] [Q36-card]

## 7. Which components could save most engineering effort?
Prioritize:
- TrackLab evaluation and dataset/module interfaces.
- Established detector wrappers/checkpoints after exact rights checks.
- Calibration geometry and camera-model optimizers, matched to the recording.
- Source-observation and stage-artifact contracts from the chunked systems.
- Explicit identity mapping, cannot-links, multi-frame number evidence and split-before-link concepts.
- Structured event/trajectory analytics from socceraction/unravelsports once upstream inputs are trustworthy.

These reduce duplicated infrastructure and conceptual mistakes. Small voting/state rules may be simpler to implement independently than importing a large research package. Savings are architectural judgments, not measured development estimates. [L-engine] [RF-code] [TV-code] [J-io] [A-consolidate] [GTA-batch] [SO-action] [UN-readme]

## 8. Which components require original research or substantial adaptation?
Full-match youth-player identity under long absence; impure tracklet correction; uncertainty-aware association across rolling substitutions and halftime; reliable tiny-ball/contact observation; actor-attributed event recognition from imperfect tracks; calibration quality estimation on nonstandard pitches; and evidence-quality propagation into scouting metrics.

Most algorithms have prior art. Scoutra's original work should be in their validated combination, failure handling and domain adaptation, not assuming every module needs a novel network. In some cases the image simply lacks enough information; supported footage requirements and human review are part of the solution. [MC-code] [GTA-code] [A-gates] [FP-run]

## 9. What should be tested before architecture decisions?
Run the proposed matched-detection tracking, minute-gap Re-ID, split/link, jersey confidence, calibration, actor-event and complete-match restart studies. Measure identity precision/coverage, contamination duration and player-event errors alongside HOTA. Compare small quantized VLMs on T4 before renting larger hardware for model-scale studies. The accessible Scoutra snapshot cannot supply a baseline; obtain and freeze one first. See EXPERIMENT_PROPOSALS.md.

## 10. What should be avoided prematurely?
- Migrating the whole product around a large VLM before dense measurement/identity tests.
- Treating jersey number or a local tracker ID as a globally verified person.
- Increasing `max_age` and calling the result persistent identity.
- Merging fragments without checking whether each fragment contains more than one player.
- Computing definitive speed/distance from pixels or invalid camera calibration.
- Presenting missing-ball proximity as verified control, or event proposals as confirmed personal counts.
- Rebuilding calibration, evaluation and generic analytics already available in inspectable libraries.
- Copying a full research stack before understanding dependencies, exact model grant and private assets.
- Optimizing H100 throughput before a small held-out amateur test establishes correctness.

The evidence behind these cautions is concrete: finite tracker lifetimes, different Re-ID representations, uncalibrated proxy statistics, broad event metrics and artifact/license gaps in the inspected code. [A-lifecycle] [A-reid] [L-parts] [F-stats] [C-status] [P-license]

## What is solved, partly solved, and unresolved?
| Capability | Research conclusion |
|---|---|
| Detection frameworks and local MOT plumbing | Available and reusable; target-footage recall remains to be established |
| Football part-aware Re-ID, jersey aggregation, pitch calibration | Substantial implemented research; domain and licensing constraints |
| Clip-level and full-game event proposal generation | Available; precise actor attribution and complete counts remain conditional |
| Full-match scheduling, cached stages, artifact replay | Useful patterns exist; no complete verified Scoutra-ready reliability solution |
| Tactical geometry and event-value analytics | Established downstream modules; trustworthy calibrated input required |
| Persistent physical identity for every amateur player over 90+ minutes | Not established by this audit; central empirical research priority |
| Automated scouting conclusions | Can summarize verified evidence; no inspected work establishes definitive talent prediction |

Sources for the layer distinctions: [G-config] [J-ocr] [TV-code] [C-dual] [L-state] [SO-action] [UN-readme]. The scouting boundary also matches Scoutra's own documented evidence-led product context. [Scoutra-context]

## Evidence limits and responsible conclusion
All eleven requested Group A repositories were available. The audit verified source structure and text/metadata, not runtime behavior. Source snapshots are pinned; author experiments are identified; uncertainty remains explicit. No Scoutra match frames or annotations were available for compatibility validation, and the only connected Scoutra code found was its landing page.

The best next investment is a small, match-disjoint amateur evaluation set and a transparent identity/event baseline. The report provides a shortlist and experiments for the team to challenge before implementation, not a claim that any external system already solves Scoutra.

## Direct sources

[A-pipeline]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/pipeline/_pipeline.py "outstandingcandy/AgentCoach — goalinsight/pipeline/_pipeline.py"
[A-runner]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/track_consolidation/_runner.py "outstandingcandy/AgentCoach — goalinsight/track_consolidation/_runner.py"
[A-offline]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/deploy/offline/README.md "outstandingcandy/AgentCoach — deploy/offline/README.md"
[A-tracking]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/orchestrator.py "outstandingcandy/AgentCoach — goalinsight/tracking/orchestrator.py"
[Scoutra-readme]: https://github.com/erbo-dev/Scoutra/blob/a6a02f176f55dd044112951506fa63a919ca7e5b/README.md "erbo-dev/Scoutra — README.md"
[G-config]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/configs/soccernet.yaml "SoccerNet/sn-gamestate — sn_gamestate/configs/soccernet.yaml"
[L-parts]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/plugins/track/bpbreid_strong_sort/sort/nn_matching.py "TrackingLaboratory/tracklab — plugins/track/bpbreid_strong_sort/sort/nn_matching.py"
[MC-code]: https://github.com/tstanczyk95/McBytePlusPlus/blob/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9/yolox/tracker/mcbyteplusplus_tracker__with_reid.py "tstanczyk95/McBytePlusPlus — yolox/tracker/mcbyteplusplus_tracker__with_reid.py"
[GTA-batch]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/refine_tracklets_batched.py "sjc042/gta-link — refine_tracklets_batched.py"
[G-reid]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/reid/prtreid_api.py "SoccerNet/sn-gamestate — sn_gamestate/reid/prtreid_api.py"
[A-reid]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/reid/prtreid_extractor.py "outstandingcandy/AgentCoach — goalinsight/tracking/reid/prtreid_extractor.py"
[P-license]: https://github.com/VlSomers/prtreid/blob/30617a75967e84d5d516959c4b84cbeea6f56493/LICENSE "VlSomers/prtreid — LICENSE"
[J-chunks]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/run_parallel.py "spindle79/jersey-tracker — scripts/run_parallel.py"
[J-merge]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/merge_chunks.py "spindle79/jersey-tracker — scripts/merge_chunks.py"
[C-triage]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/triage_scanner.py "acato/soccer-video-pipeline — src/detection/triage_scanner.py"
[L-state]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/tracklab/datastruct/tracker_state.py "TrackingLaboratory/tracklab — tracklab/datastruct/tracker_state.py"
[H-run]: https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/run_pipeline.py "samuehuang/football-cv-analytics — run_pipeline.py"
[FP-run]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/run_TAAD_on_matches.py "JeremieOchin/FOOTPASS — run_TAAD_on_matches.py"
[FP-gnn]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/models/model_GNN.py "JeremieOchin/FOOTPASS — models/model_GNN.py"
[FP-readme]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/README.md "JeremieOchin/FOOTPASS — README.md"
[H-events]: https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/events/engine.py "samuehuang/football-cv-analytics — src/events/engine.py"
[A-events]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/events/_orchestrator.py "outstandingcandy/AgentCoach — goalinsight/events/_orchestrator.py"
[C-status]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/docs/pipeline_status.md "acato/soccer-video-pipeline — docs/pipeline_status.md"
[C-dual]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/dual_pass_detector.py "acato/soccer-video-pipeline — src/detection/dual_pass_detector.py"
[C-weights]: https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-fp8/blob/ab56ac6743fc4f5ec299ed951f989d37f5874c56/README.md "acato v11 FP8 checkpoint card"
[OSL-model]: https://github.com/OpenSportsLab/opensportslib/blob/b2fe8f16a35336d93b1b0eabe1741bbbbc36cd68/opensportslib/models/base/qwen_vl_native.py "OpenSportsLab/opensportslib — opensportslib/models/base/qwen_vl_native.py"
[CHAT-card]: https://huggingface.co/SimulaMet/SoccerChat-qwen2-vl-7b/blob/29871536004c0ac788af09cbb87969ab9f6e1410/README.md "SoccerChat checkpoint card"
[Q35-doc]: https://huggingface.co/docs/transformers/model_doc/qwen3_5 "Transformers Qwen3.5 documentation"
[Q36-card]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/345d467b117d1fd5eda9b879a5988de9ae5f47a7/README.md "Official Qwen3.6-35B-A3B model card"
[L-engine]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/tracklab/engine/offline.py "TrackingLaboratory/tracklab — tracklab/engine/offline.py"
[RF-code]: https://github.com/roboflow/rf-detr/blob/e56ec6152b0eb1ac8bcca8690aa7ba59761d861f/src/rfdetr/detr.py "roboflow/rf-detr — src/rfdetr/detr.py"
[TV-code]: https://github.com/MM4SPA/tvcalib/blob/1222c5230af2742395d74918ed6f34eb2b9bf7f9/tvcalib/module.py "MM4SPA/tvcalib — tvcalib/module.py"
[J-io]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/trackio.py "spindle79/jersey-tracker — scripts/trackio.py"
[A-consolidate]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/track_consolidation/_consolidator.py "outstandingcandy/AgentCoach — goalinsight/track_consolidation/_consolidator.py"
[SO-action]: https://github.com/ML-KULeuven/socceraction/blob/93a1242d46c104889205753accaabadb00c45c6d/README.md "ML-KULeuven/socceraction — README.md"
[UN-readme]: https://github.com/UnravelSports/unravelsports/blob/2bd24d7e2b3cc19cbdba6486c7abb70270a732ad/README.md "UnravelSports/unravelsports — README.md"
[GTA-code]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/refine_tracklets.py "sjc042/gta-link — refine_tracklets.py"
[A-gates]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/strongsort/gates.py "outstandingcandy/AgentCoach — goalinsight/tracking/strongsort/gates.py"
[A-lifecycle]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/strongsort/lifecycle.py "outstandingcandy/AgentCoach — goalinsight/tracking/strongsort/lifecycle.py"
[F-stats]: https://github.com/dan-slater/footyviz/blob/4493f88d554a80dc3d9858c276173a17786caf8d/src/fv/stats.py "dan-slater/footyviz — src/fv/stats.py"
[J-ocr]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/jersey_ocr.py "spindle79/jersey-tracker — scripts/jersey_ocr.py"
[Scoutra-context]: https://github.com/erbo-dev/Scoutra/blob/a6a02f176f55dd044112951506fa63a919ca7e5b/docs/scoutra-product-context.md "erbo-dev/Scoutra — docs/scoutra-product-context.md"
