# E. Persistent Player Identity
**Highest-priority research finding:** Scoutra needs a layer that distinguishes a physical player from a temporary tracker ID. Several inspected projects implement pieces of that layer. None establishes full-match identity reliability on unseen amateur footage.

## 1. Four different identity tasks
A **detection** is one player box in one frame. A **tracklet** is a locally associated sequence of boxes. A **match identity** links fragments belonging to the same footballer across a match. A **roster identity** binds that visual identity to a known participant. Jersey recognition helps with the last two but is not an unconditional identifier: numbers can be missing, duplicated, shared across teams, or misread.

A tracker can have few short-term ID switches yet create a new ID every time a player leaves view. It can also maintain one long ID while silently switching between teammates. Therefore both **fragmentation** and **tracklet contamination** matter. AgentCoach's mapping is a concrete example of fragment consolidation; GTA explicitly addresses splitting contaminated fragments. [A-consolidate] [GTA-batch]

## 2. What the requested key systems actually implement

| Capability | AgentCoach | SoccerNet GSR/current default | jersey-tracker | Soccer-Tracking |
|---|---|---|---|---|
| Short-term tracking | Custom StrongSORT or BoT-SORT option | BPBReID StrongSORT | ByteTrack per camera | Deep-EIoU/HMSort variants |
| Re-ID appearance | Global vector from PRT/OSNet/CLIP adapter | Part embeddings, visibility, role | No general person embedding path in inspected detection/OCR stages | Learned appearance in HMReid path |
| Long absence | Offline fragment consolidation can link non-overlapping tracks | Finite online age; no named global TrackMerge in pipeline | No general long-absence matching | Finite lost buffer; no inspected match-level bank |
| Jersey aggregation | Sampled OCR/VLM votes and candidates | Confidence voting per tracklet | Full-number votes per track | Separate jersey-ID tracker variant with hard cost override |
| Team constraints | Offline cannot-link gates | Team clustering is downstream of tracker | Team color labels downstream | Team classification pipeline; activation depends on variant |
| Co-occurrence constraints | Explicit component-level veto | Not a match-wide identity merger in current config | Boundary matching / camera-specific fusion | No verified global constraint graph |
| Chunk boundary identity | No generic overlap+global-identity contract verified | Video reset boundaries in engine | Explicit overlap IoU stitching | No verified chunk reconciliation |
| Full-match roster / halftime / subs | Unverified | Unverified | Unverified | Unverified |

Sources: [A-tracker] [A-reid] [A-consolidate] [G-config] [G-reid] [G-jersey] [L-config] [J-detect] [J-ocr] [J-merge] [T-hm] [T-app] [T-jid].

**Conclusion:** AgentCoach has real offline identity consolidation beyond ordinary MOT. GSR has real football representation and attribute aggregation, but paper-level TrackMerge should not be assumed present in its current default pipeline. jersey-tracker's cross-chunk links preserve overlapping tracks, not arbitrary absent players. Soccer-Tracking implements appearance-augmented local association and jersey variants, but a persistent whole-match identity engine was not verified.

## 3. Body-part matching and football evidence
A global embedding compresses the entire crop into one vector. An occluder, teammate, or background can dominate that vector. Part-aware matching instead compares corresponding visible body regions. TrackLab's `_nn_part_based` passes both features and visibility to its distance calculation; this is materially different from AgentCoach selecting `bn_globl`. Body-part visibility does not create information when a distant player is only a few pixels tall. [L-parts] [A-reid]

**A: relevant published evidence, not Scoutra results.** PRT-Track's paper reports, on SoccerNet test with YOLOv8 detections, HOTA 59.77 and IDF1 74.44 versus StrongSORT++ 56.21 and 66.53. Its oracle-detection result is a separate setting: HOTA 90.77. Table 4 compares global and part-based TrackMerge; this does not prove the current GSR package runs that full postprocessor. The paper builds its Re-ID crops from SoccerNet Tracking; that is distinct from the cross-view SoccerNet Re-ID challenge. [P-paper]

This is stronger selection evidence than pedestrian-only benchmarks, but transfer to youth body proportions, fixed low cameras, lower crop resolution, and long absence remains H. Checkpoint identity, training split, feature head, visibility handling, and tracker parameters must match before reproducing results.

## 4. The most relevant additional mechanisms
### McByte++: online identity bank
The inspected `McBytePlusPlusTracker` includes:
- `STrack.reid_feats`: bounded recent feature samples and a normalized mean.
- `_update_reid_features_for_tracks`: quality/cadence filtering before memory updates.
- `_stash_identity_if_ready`: retains sufficiently supported identities in `reid_identity_bank`.
- `try_reconnect_new_tracks`: considers young tracklets against stored identities, applies similarity and mutual-nearest-neighbor criteria, and manages exported IDs.
- `_ensure_unique_export_ids`: protects against simultaneous duplicate exported identities. [MC-code]

This is a substantive long-absence implementation. It is not merely a long `max_age`. Important caveats: missing Re-ID dependencies/model path can disable Re-ID; its default good-crop gates include minimum width/height/area, so many tiny amateur players may never enter memory. Its exported-ID logic is not a roster, jersey, substitution, or restart database. **A:** the 2026 paper reports football/SportsMOT improvements over McByte; this audit did not reproduce them. **I:** high-priority comparator, especially if appearance updates can abstain when crops are poor. [MC-code] [MC-paper] [MC-install]

### GTA: offline split and connect
`refine_tracklets_batched.py::detect_id_switch` clusters per-track features with DBSCAN; `split_tracklets` creates cleaner fragments. `refine_tracklets.py` computes cross-track distances, checks temporal overlap and spatial conditions, and links fragments. `generate_tracklets.py` derives features from tracker outputs and images. **A:** the paper reports SoccerNet HOTA 79.41 → 83.11 in a tested configuration; that result is not comparable directly to PRT-Track's detector/setup. [GTA-batch] [GTA-code] [GTA-gen] [GTA-paper]

**I:** splitting is essential because a single centroid for a mixed-player track can contaminate an entire cluster. **H:** pose/lighting change can also create multiple feature clusters for one person, so DBSCAN clusters are not definitive proof of an ID switch. Sparse sampling, split penalties, OCR conflicts, and manual evidence review should be ablated. Pairwise feature comparisons and clustering need bounded galleries or candidate pruning for whole matches.

### footyviz: constrained appearance fusion
`identity.py::fuse` combines OSNet embedding distance, body-band color fingerprints, and aspect ratio. Different teams, overlapping frames, and confident different jersey numbers are cannot-links. Matching confident numbers reduce distance. `_agglomerate` uses average linkage and rejects a merge if any cross-member pair violates a constraint. This avoids some transitive “A resembles B, B resembles C” mistakes. [F-identity]

Its research value includes admitted failure: the author reports approximately 23 stable identities for roughly 14–16 actual players in a difficult night slice. That is a fragmentation diagnostic, not an identity-accuracy metric. The 180-second job cap means the main runner is not evidence of a complete 39-minute identity pass. [F-readme] [F-job]

## 5. How the identity signals should be understood
| Signal | What it helps | Where it fails | Evidence/reference |
|---|---|---|---|
| Short motion/Kalman | Bridge adjacent observations | Pan/zoom, abrupt cuts, irregular runs, long absence | [A-tracker] [BOT-code] |
| Camera compensation | Separate camera motion from player motion | Zoom/parallax, few static features, bad homography | [BOT-gmc] |
| Team agreement | Reject opposite-team matches | Similar kits, shadows, wrong team label, goalkeeper color | [A-consolidate] [G-team] |
| Part embeddings | Compare visible appearance | Too-small crops, uniform body parts, occluded face/boots | [L-parts] |
| Number votes | Join views with readable backs | Duplicate numbers, partial digits, correlated OCR mistakes | [J-ocr] |
| Temporal galleries | Preserve multiple views | Wrong accepted match poisons memory | [MC-code] |
| Co-occurrence veto | Reject physically impossible merges | Duplicate detections of one player; cross-camera same-person views | [A-consolidate] [J-cross] |
| Offline clustering | Use evidence before and after disappearance | Early irreversible merge; quadratic growth; mixed tracklets | [GTA-code] [F-identity] |
| Tactical/motion priors | Rule out impossible short-gap travel | Long-gap position prediction becomes weak; changing roles | [A-gates] |

“More persistent” should never mean “always force a match.” An unknown identity is preferable to confident wrong per-player statistics. That is a proposed product-quality principle.

## 6. Proposed research architecture, not an implementation prescription
Maintain immutable detections and pure local tracklets. Detect possible track contamination before linking. Represent each match identity by multiple high-quality appearance views, number candidates with evidence timestamps, team/role distributions, active periods, and explicit constraints. Keep uncertainty and alternative associations until enough evidence exists.

An interpretable candidate link could combine appearance, jersey agreement, team consistency, temporal/geometric feasibility, and crop quality. This is a **hypothesis**, not a validated universal weighted formula. Hard gates should be reserved for reliable impossibilities; noisy team/OCR labels should not become unreviewable truth.

Compare three offline association families:
1. **Constrained clustering:** AgentCoach/footyviz/GTA-style merging, relatively understandable; can be greedy and order/threshold sensitive.
2. **Directed graph/min-cost flow:** candidate edges represent plausible continuation; optimize total association cost under single-identity constraints. Requires score calibration and does not naturally solve every nonlocal roster constraint.
3. **Integer/correlation-clustering formulation:** allows nonlocal cannot-links and alternative hypotheses; optimization cost and uncertain evidence need careful control.

The last two are **research proposals**, not verified Group A implementations. A graph-shaped algorithm is not automatically globally optimal. AgentCoach's union-find construction is constrained clustering; FOOTPASS's GNN models event context, not a drop-in identity solver. [A-consolidate] [FP-gnn]

## 7. Halftime, substitutions, and irreducible ambiguity
At halftime, retain visual identity hypotheses, clear short-horizon velocity, re-establish pitch direction and camera calibration, and record any kit/camera change. Do not swap team identities simply because ends change. Substitutions need active intervals; rolling youth substitutions require re-entry without treating it as a new participant. Duplicate team numbers require an additional discriminator or manual resolution.

No inspected system proves these policies across representative 90-minute amateur matches. When two players leave view and return with indistinguishable appearance and no readable number, video evidence alone may not determine which is which. More compute cannot make that ambiguity disappear. Scoutra needs abstention and evidence review, not an unconditional persistence promise.

## 8. Evaluation that answers the actual problem
Use HOTA/AssA, IDF1, switches, and fragmentation **plus** match-level measures:
- Tracklet purity and identity contamination duration.
- Pairwise association precision/recall, emphasizing false merges.
- Correct re-entry at gaps 0–2s, 2–10s, 10–60s, 1–5min, >5min.
- Identity coverage at a specified precision, with unknown outputs counted.
- Roster consistency across halftime/substitutions.
- Correct player+event+time F1, not event F1 alone.
- Human corrections required per ten minutes.

A short-clip benchmark can evaluate local association; it cannot certify match identity. SoccerNet Tracking includes a long annotated half, unlike GSR's standard short-sequence scope, and is a useful supplementary test if access/rights permit. [SN-track] [G-paper]

## Direct sources

[A-consolidate]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/track_consolidation/_consolidator.py "outstandingcandy/AgentCoach — goalinsight/track_consolidation/_consolidator.py"
[GTA-batch]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/refine_tracklets_batched.py "sjc042/gta-link — refine_tracklets_batched.py"
[A-tracker]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/strongsort/tracker.py "outstandingcandy/AgentCoach — goalinsight/tracking/strongsort/tracker.py"
[A-reid]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/reid/prtreid_extractor.py "outstandingcandy/AgentCoach — goalinsight/tracking/reid/prtreid_extractor.py"
[G-config]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/configs/soccernet.yaml "SoccerNet/sn-gamestate — sn_gamestate/configs/soccernet.yaml"
[G-reid]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/reid/prtreid_api.py "SoccerNet/sn-gamestate — sn_gamestate/reid/prtreid_api.py"
[G-jersey]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/jersey/voting_tracklet_jn_api.py "SoccerNet/sn-gamestate — sn_gamestate/jersey/voting_tracklet_jn_api.py"
[L-config]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/tracklab/configs/modules/track/bpbreid_strong_sort.yaml "TrackingLaboratory/tracklab — tracklab/configs/modules/track/bpbreid_strong_sort.yaml"
[J-detect]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/detect_track.py "spindle79/jersey-tracker — scripts/detect_track.py"
[J-ocr]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/jersey_ocr.py "spindle79/jersey-tracker — scripts/jersey_ocr.py"
[J-merge]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/merge_chunks.py "spindle79/jersey-tracker — scripts/merge_chunks.py"
[T-hm]: https://github.com/Streakfull/Soccer-Tracking/blob/9d3488fc88390cbc64eb444eb08e481b79df0aef/src/pytorchmodels/DeepIOUTracker/HMReidTracker.py "Streakfull/Soccer-Tracking — src/pytorchmodels/DeepIOUTracker/HMReidTracker.py"
[T-app]: https://github.com/Streakfull/Soccer-Tracking/blob/9d3488fc88390cbc64eb444eb08e481b79df0aef/src/hm_sort_app/tracker.py "Streakfull/Soccer-Tracking — src/hm_sort_app/tracker.py"
[T-jid]: https://github.com/Streakfull/Soccer-Tracking/blob/9d3488fc88390cbc64eb444eb08e481b79df0aef/src/hm_sort_jid/tracker.py "Streakfull/Soccer-Tracking — src/hm_sort_jid/tracker.py"
[L-parts]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/plugins/track/bpbreid_strong_sort/sort/nn_matching.py "TrackingLaboratory/tracklab — plugins/track/bpbreid_strong_sort/sort/nn_matching.py"
[P-paper]: https://arxiv.org/html/2401.09942v1 "PRTReID paper, Tables 3–6 and §3.2"
[MC-code]: https://github.com/tstanczyk95/McBytePlusPlus/blob/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9/yolox/tracker/mcbyteplusplus_tracker__with_reid.py "tstanczyk95/McBytePlusPlus — yolox/tracker/mcbyteplusplus_tracker__with_reid.py"
[MC-paper]: https://arxiv.org/abs/2608.15688 "McByte++ paper"
[MC-install]: https://github.com/tstanczyk95/McBytePlusPlus/blob/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9/INSTALLATION.md "tstanczyk95/McBytePlusPlus — INSTALLATION.md"
[GTA-code]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/refine_tracklets.py "sjc042/gta-link — refine_tracklets.py"
[GTA-gen]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/generate_tracklets.py "sjc042/gta-link — generate_tracklets.py"
[GTA-paper]: https://arxiv.org/abs/2411.08216 "GTA paper"
[F-identity]: https://github.com/dan-slater/footyviz/blob/4493f88d554a80dc3d9858c276173a17786caf8d/src/fv/identity.py "dan-slater/footyviz — src/fv/identity.py"
[F-readme]: https://github.com/dan-slater/footyviz/blob/4493f88d554a80dc3d9858c276173a17786caf8d/README.md "dan-slater/footyviz — README.md"
[F-job]: https://github.com/dan-slater/footyviz/blob/4493f88d554a80dc3d9858c276173a17786caf8d/src/fv/job.py "dan-slater/footyviz — src/fv/job.py"
[BOT-code]: https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/tracker/bot_sort.py "NirAharon/BoT-SORT — tracker/bot_sort.py"
[BOT-gmc]: https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/tracker/gmc.py "NirAharon/BoT-SORT — tracker/gmc.py"
[G-team]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/team/tracklet_team_clustering_api.py "SoccerNet/sn-gamestate — sn_gamestate/team/tracklet_team_clustering_api.py"
[J-cross]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/scripts/cross_lens_merge.py "spindle79/jersey-tracker — scripts/cross_lens_merge.py"
[A-gates]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/strongsort/gates.py "outstandingcandy/AgentCoach — goalinsight/tracking/strongsort/gates.py"
[FP-gnn]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/models/model_GNN.py "JeremieOchin/FOOTPASS — models/model_GNN.py"
[SN-track]: https://github.com/SoccerNet/sn-tracking/blob/b0bbba35e07ff58010b6313ef8aa59ef663ad392/README.md "SoccerNet/sn-tracking — README.md"
[G-paper]: https://arxiv.org/abs/2404.11335 "SoccerNet GSR paper"
