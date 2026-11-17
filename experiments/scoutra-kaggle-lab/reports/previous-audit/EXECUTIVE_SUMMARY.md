# Scoutra AI — Independent Research and Architecture Audit
**Research snapshot: 17 November 2026. Research only.**

## Scope, evidence, and reading guide
All eleven requested Group A repositories were accessible. The five priority repositories were examined first, followed by the remaining six and selected specialist implementations. The audit includes 227 distinct source/document snapshots, with GitHub commit references recorded in `SOURCE_MANIFEST.json`. That count measures inspection scope, not software quality. Some files were examined in selected sections; this is an architecture audit, not exhaustive line-by-line verification.

No external repository was executed. No inference, training, dependency installation, GPU benchmark, checkpoint download, third-party integration, or application modification occurred. Lightweight GitHub/Hugging Face/Zenodo metadata and text files were read. A public landing-page video in the Scoutra tree was not treated as match-test footage.

Evidence labels throughout:
- **V — Verified by inspection:** source implements the stated operation, or an artifact/file exists. This does not establish successful execution.
- **A — Author claim/result:** a README, paper, model card, or experiment note reports it; this audit did not reproduce it.
- **I — Researcher interpretation:** engineering conclusion drawn from the cited implementation.
- **H — Hypothesis:** requires experiments on representative Scoutra footage.
- **U — Unverified:** missing, conflicting, or insufficient evidence.

The accessible Scoutra repository is `erbo-dev/Scoutra` at `a6a02f176f55dd044112951506fa63a919ca7e5b`. Its README calls it an editorial landing page; `package.json` contains React/Vite dependencies, and the tree contains no ML backend. No Scoutra match recordings, annotations, benchmark logs, or ML experiment reports were found in the available workspace. Thus “Scoutra already has an equivalent” is **unknown for any separate private/backend codebase**, and **no equivalent was found in this accessible snapshot**. No improvement over Scoutra's current algorithms can be claimed. [Scoutra-readme] [Scoutra-package]

## A. The ten most important findings

1. **There is no verified turnkey solution for Scoutra's full requirement.** Detection, local tracking, calibration, OCR, and event models exist, but reliable per-player attribution through an entire amateur match is not established by the inspected systems. Local track IDs, jersey labels, and physical player identities must remain separate concepts. AgentCoach's consolidation stage is a concrete example of this separation. **V/I.** [A-consolidate] [S-track]

2. **AgentCoach, internally named GoalInsight, is the closest overall architectural reference.** It connects calibration, detection/tracking, offline identity consolidation, rule-based events, profiles, and rendering. The useful part is its stage contracts and explicit mapping from source tracklets to players. Its research-grade status and private build inputs prevent treating it as a drop-in product. **V/I.** [A-pipeline] [A-runner] [A-offline]

3. **SoccerNet GSR/TrackLab is the strongest established football-specific modular reference.** The default config wires PRTReID, part-aware StrongSORT, calibration, jersey voting, and team clustering. Its evaluation framework is more valuable than a visual demo alone. The GSR benchmark's 30-second sequences do not establish halftime continuity. **V/A/I.** [G-config] [L-parts] [G-paper]

4. **PRTReID deserves comparison, with its complete feature representation preserved.** AgentCoach's adapter takes the global embedding, whereas the GSR/TrackLab route retains body-part embeddings and visibility. Using the same checkpoint name does not mean two systems implement the same matching algorithm. PRTReID's football results justify an experiment, not a promised gain on youth footage. **V/I/H.** [A-reid] [G-reid] [L-parts]

5. **McByte++ and GTA are high-value additional identity research candidates.** McByte++ has an explicit identity bank, quality-filtered embedding updates, and reconnection logic. GTA splits potentially contaminated tracklets before linking fragments offline. These address long absence and intra-track ID switches more directly than simply increasing ByteTrack's lost buffer. Neither establishes correct roster identity across 90 minutes. **V/I.** [MC-code] [GTA-code] [GTA-batch]

6. **jersey-tracker contributes unusually relevant amateur-video engineering, with incomplete product features.** It contains overlapping chunk processing, boundary IoU stitching, jersey-number voting, lens/pitch handling, and row-oriented storage. Its target-player finder is a stub. Boundary stitching is not general long-term Re-ID. **V.** [J-chunks] [J-merge] [J-ocr] [J-find]

7. **acato's soccer-video-pipeline is a relevant amateur event/reel system, not a replacement for player tracking.** Its reported four-game goal-union recall is 1.00, but union precision is 0.17 and matching uses ±60-second coverage on tuned games. That is highlight retrieval evidence, not frame-accurate event counting or actor attribution. **A/I.** [C-status] [C-models]

8. **A Video-VLM should initially supplement specialized perception.** The inspected football VLMs classify/explain clips or generate event proposals; none provides a validated substitute for continuous trajectories, metric calibration, full-match physical identity, and actor-linked statistics. Sparse frames also miss brief contacts. Scoutra can test VLM verification and narrative generation around grounded evidence. **V/I/H.** [C-dual] [OSL-model] [CHAT-card]

9. **License suitability is a component-level issue.** PRTReID/BPBReID code uses Hippocratic licenses; GSR is GPL-3.0; OpenSportsLib is AGPL-3.0 with a commercial option. FOOTPASS has conflicting top-level Apache and README noncommercial statements. acato's Qwen license metadata conflicts with its upstream base's Apache metadata. ONNX export does not erase a checkpoint's stated license. **V/U.** [P-license] [BP-license] [G-license] [OSL-license] [OSL-commercial] [FP-license] [FP-readme] [C-weights] [Q32-card] [C-ball]

10. **The next decision should be an evaluation plan, not a new architecture commitment.** Compare identity purity, re-entry correctness by absence length, actor-attributed event F1, calibration error, and restart behavior on held-out matches. Kaggle 2×T4 is suitable for staged baseline comparisons and small quantized VLM proposals; it is not equivalent to one 32-GB GPU. An H100 expands model/context capacity but does not restore unrecorded visual evidence. **I/H.** [V-quant] [MC-code] [L-state]

## Decisions this report supports
Study AgentCoach for end-to-end contracts, TrackLab/GSR for controlled football experiments, McByte++ for online reconnection, GTA for offline split-and-link, jersey-tracker for chunk orchestration, and FOOTPASS for actor-centric event modeling. Study samuehuang's possession/event separation and evidence-backed reporting. Prefer reusable interfaces and algorithms over importing an entire experimental stack. These are study priorities, not implementation approvals.

Read [REPOSITORY_COMPARISON.md](REPOSITORY_COMPARISON.md) for matrices and scores, [TECHNICAL_DEEP_DIVES.md](TECHNICAL_DEEP_DIVES.md) for source paths, [PERSISTENT_IDENTITY.md](PERSISTENT_IDENTITY.md) for the highest-priority problem, and [EXPERIMENT_PROPOSALS.md](EXPERIMENT_PROPOSALS.md) for unexecuted validation plans. The remaining documents cover long videos, VLMs, licenses, reusable components, and final decisions.

## Direct sources

[Scoutra-readme]: https://github.com/erbo-dev/Scoutra/blob/a6a02f176f55dd044112951506fa63a919ca7e5b/README.md "erbo-dev/Scoutra — README.md"
[Scoutra-package]: https://github.com/erbo-dev/Scoutra/blob/a6a02f176f55dd044112951506fa63a919ca7e5b/package.json "erbo-dev/Scoutra — package.json"
[A-consolidate]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/track_consolidation/_consolidator.py "outstandingcandy/AgentCoach — goalinsight/track_consolidation/_consolidator.py"
[S-track]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/src/sportvision/tracking.py "MohibShaikh/sportvision — src/sportvision/tracking.py"
[A-pipeline]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/pipeline/_pipeline.py "outstandingcandy/AgentCoach — goalinsight/pipeline/_pipeline.py"
[A-runner]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/track_consolidation/_runner.py "outstandingcandy/AgentCoach — goalinsight/track_consolidation/_runner.py"
[A-offline]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/deploy/offline/README.md "outstandingcandy/AgentCoach — deploy/offline/README.md"
[G-config]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/configs/soccernet.yaml "SoccerNet/sn-gamestate — sn_gamestate/configs/soccernet.yaml"
[L-parts]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/plugins/track/bpbreid_strong_sort/sort/nn_matching.py "TrackingLaboratory/tracklab — plugins/track/bpbreid_strong_sort/sort/nn_matching.py"
[G-paper]: https://arxiv.org/abs/2404.11335 "SoccerNet GSR paper"
[A-reid]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/reid/prtreid_extractor.py "outstandingcandy/AgentCoach — goalinsight/tracking/reid/prtreid_extractor.py"
[G-reid]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/reid/prtreid_api.py "SoccerNet/sn-gamestate — sn_gamestate/reid/prtreid_api.py"
[MC-code]: https://github.com/tstanczyk95/McBytePlusPlus/blob/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9/yolox/tracker/mcbyteplusplus_tracker__with_reid.py "tstanczyk95/McBytePlusPlus — yolox/tracker/mcbyteplusplus_tracker__with_reid.py"
[GTA-code]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/refine_tracklets.py "sjc042/gta-link — refine_tracklets.py"
[GTA-batch]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/refine_tracklets_batched.py "sjc042/gta-link — refine_tracklets_batched.py"
[J-chunks]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/run_parallel.py "spindle79/jersey-tracker — scripts/run_parallel.py"
[J-merge]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/merge_chunks.py "spindle79/jersey-tracker — scripts/merge_chunks.py"
[J-ocr]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/jersey_ocr.py "spindle79/jersey-tracker — scripts/jersey_ocr.py"
[J-find]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/find_player.py "spindle79/jersey-tracker — scripts/find_player.py"
[C-status]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/docs/pipeline_status.md "acato/soccer-video-pipeline — docs/pipeline_status.md"
[C-models]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/models.py "acato/soccer-video-pipeline — src/detection/models.py"
[C-dual]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/dual_pass_detector.py "acato/soccer-video-pipeline — src/detection/dual_pass_detector.py"
[OSL-model]: https://github.com/OpenSportsLab/opensportslib/blob/b2fe8f16a35336d93b1b0eabe1741bbbbc36cd68/opensportslib/models/base/qwen_vl_native.py "OpenSportsLab/opensportslib — opensportslib/models/base/qwen_vl_native.py"
[CHAT-card]: https://huggingface.co/SimulaMet/SoccerChat-qwen2-vl-7b/blob/29871536004c0ac788af09cbb87969ab9f6e1410/README.md "SoccerChat checkpoint card"
[P-license]: https://github.com/VlSomers/prtreid/blob/30617a75967e84d5d516959c4b84cbeea6f56493/LICENSE "VlSomers/prtreid — LICENSE"
[BP-license]: https://github.com/VlSomers/bpbreid/blob/a2dc4304284784d8b2c061764512f0698aa38c21/LICENSE "VlSomers/bpbreid — LICENSE"
[G-license]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/LICENSE "SoccerNet/sn-gamestate — LICENSE"
[OSL-license]: https://github.com/OpenSportsLab/opensportslib/blob/b2fe8f16a35336d93b1b0eabe1741bbbbc36cd68/LICENSE "OpenSportsLab/opensportslib — LICENSE"
[OSL-commercial]: https://github.com/OpenSportsLab/opensportslib/blob/b2fe8f16a35336d93b1b0eabe1741bbbbc36cd68/LICENSE-COMMERCIAL "OpenSportsLab/opensportslib — LICENSE-COMMERCIAL"
[FP-license]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/LICENSE "JeremieOchin/FOOTPASS — LICENSE"
[FP-readme]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/README.md "JeremieOchin/FOOTPASS — README.md"
[C-weights]: https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-fp8/blob/ab56ac6743fc4f5ec299ed951f989d37f5874c56/README.md "acato v11 FP8 checkpoint card"
[Q32-card]: https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/blob/0cfaf48183f594c314753d30a4c4974bc75f3ccb/README.md "Official Qwen3-VL-32B model card"
[C-ball]: https://huggingface.co/acatorcini/yolov9-soccer-ball/blob/b30df5abc9f3eda4a9d326d953be12b3541a14b4/README.md "acato soccer ball checkpoint card"
[V-quant]: https://docs.vllm.ai/en/v0.19.1/features/quantization/#supported-hardware "vLLM 0.19.1 quantization hardware matrix"
[L-state]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/tracklab/datastruct/tracker_state.py "TrackingLaboratory/tracklab — tracklab/datastruct/tracker_state.py"
