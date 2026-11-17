# Risks, blockers and exact execution status

**All real experiments remain NOT EXECUTED.** No trained model, video inference, GPU benchmark, expensive experiment, training, production edit or integration was performed.

| Blocker | Consequence / next concrete action |
|---|---|
| No actual Kaggle session/GPU access | Notebooks are manual-upload deliverables; no remote execution or timing claim |
| Videos only named, not mounted here | Dimensions/duration/FPS/camera/source matches/scene cuts unknown until audit notebook runs |
| No reviewed annotations | No mAP/HOTA/IDF1/identity accuracy can be reported; use human annotation flow and null metrics |
| No true Scoutra detector/backend checkpoint | Baseline N/A; attach exact checkpoint+architecture/config/license/hash later |
| Highlights/replays and unknown match provenance | Cannot establish full-match memory, true re-entry or independent testing; review scene cuts, sources and obtain continuous recordings |
| Checkpoint hashes/licenses not supplied | Asset gates intentionally block loading; only official metadata/source listings inspected |
| Modern detector/BoxMOT pins not installed as GPU environments | Installation/API/ABI/load compatibility pending; direct pins are provisional until fresh Kaggle `pip check` and freeze |
| PRT/BPB/GSR legacy Python and asset chain | No executable legacy inference bridge is bundled; use the precise upstream extraction specification and independent environment; no global-vector stand-in |
| GTA and McByte++ bridge/mask requirements | No exact exports/measurements; generic bridge validator is infrastructure only, not a method implementation |
| Supplemental evaluation scope | Official MOT export/TrackEval for HOTA/IDF1; episode recovery/split/FMR/confidence and perclassAP sweeps require validated evaluation extensions. Those metrics are not prefilled |
| Frame timestamps estimated by index/FPS | VFR and event timing require future PTS-aware decode; do not claim subframe localization |
| Stage1 whole-short-segment buffering/no resume | Long-video streaming/chunk/global-bank serialization deliberately unimplemented; test later specification before Stage3 |
| Original TrackEval uses NumPy aliases | Separate legacy evaluation profile, do not use a different PyPI fork unknowingly |
| Designated chat output directory cannot be created | Unix permission denial for `/codex/.../output`; lab preserved in writable research workspace. No nonexistent download link is claimed |

Asset source-code/weights/dataset licenses must be reviewed independently. PRT/BPB Hippocratic, GSR GPL, BoxMOT/OpenSportsLib AGPL, FOOTPASS conflicting NC statements and acato model-card mismatch are recorded in the prior audit. These are specific asset issues, not generic objections to research. No external license is inferred from a wrapper's README. No third-party code was integrated into Scoutra.

Prepared adapters and notebooks are **not independently validated model reproductions**. A successful CPU contract test proves serialization/control-flow behavior on synthetic fixtures, not accuracy, hardware fit, correct pretrained loading or full research-method equivalence. Preserve unresolved blockers rather than force all candidates into a single notebook.
