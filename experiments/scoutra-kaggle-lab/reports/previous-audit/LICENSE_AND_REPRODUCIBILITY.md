# License and Reproducibility Audit
**Evidence snapshot: 17 November 2026.** This records inspected grants, metadata and conflicts. It does not provide whole-product legal clearance. “Commercial use” and “permissive” are different: GPL/AGPL permit commercial activity subject to obligations.

## 1. Code, weights, and footage are separate assets
A repository's root license does not automatically license its dependencies, vendored files, pretrained weights, training data, or match footage. A model being downloadable is not a grant to redistribute its source videos. Exporting a checkpoint to ONNX changes runtime dependencies, not its stated weight license. [C-ball] [G-license] [DS-chat]

## 2. Group A audit

| Project | Verified source license | Weight/model status | Data/API/private-asset barrier | Reproduction judgment |
|---|---|---|---|---|
| AgentCoach | Apache-2.0 root; NOTICE lists vendored PnLCalib/BroadTrack | Standard detector/Re-ID assets; PRT weights have separate archive terms; CLIP-ReID local input | Offline build requires named local weights, video and annotations; cloud chat uses Bedrock; OCR can use local Qwen or external providers | Substantial code, not self-contained from checkout; exact whole-stack licensing U |
| jersey-tracker | MIT | YOLO/PARSeq/EasyOCR model-fetch mechanisms; individual weight terms not all verified | Private Veo recordings and calibration; camera-specific assumptions | Install notes exist; not an independently reproduced 90-minute pipeline |
| acato | Apache-2.0 | Public merged/adapter/ball files; Qwen custom-license conflict; ball card AGPL | Training clips withheld; internal predecessor base; NAS/server-specific paths; local vLLM or optional external API | Merged artifact plausible to reproduce; exact training lineage unavailable |
| GSR | GPL-3.0 | PRTReID/HRNet linked with checksums; code and checkpoint grants differ | PRT/BPB Hippocratic code; SoccerNet video terms; research dependency stack | Strong published baseline, installation not executed |
| SportVision | Apache-2.0 | Standard COCO detector models; RF-DETR optional | Ultralytics optional app/backend; Roboflow engine optional | No API key needed for documented local path; football accuracy not established |
| Soccer-Tracking | No root license found | Hardcoded detector, classifier and Re-ID paths; Drive project links | Missing packaged checkpoints; multiple vendored libraries | Insufficient evidence for commercial source reuse or clean reproduction |
| budun-ov | AGPL-3.0 | Caller must supply local detection weights; calibration setup external | Ultralytics/TVCalib model chain, video/calibration inputs | Runnable source structure; asset/license chain incomplete |
| samuehuang | MIT | Models excluded; local paths supplied by config | Input videos excluded; one sample/reports and regression claims | Stage logic inspectable; ML inference not self-contained |
| footyviz | MIT | YOLO11/OSNet and optional OCR/SR | BoxMOT and model licenses need version-specific review; optional Gemini/OpenAI/Ollama; public-video download can fail | Short-clip recipe exists; no full-match identity reproduction |
| matchtrace-cv | MIT | Default synthetic CNN and color detector; optional real detector supplied | Synthetic evaluation does not validate real detector; actual model license U | Most self-contained synthetic demo; not strongest football reproduction |
| soccer360 | No root license found | Models directory placeholder and custom/bootstrap resolution | 360° inputs, camera tooling and model assets | No commercial clearance; operating code does not establish public trained-model reproduction |

Evidence: [A-notice] [A-offline] [J-readme] [C-train] [C-adapter] [C-ball] [G-license] [G-reid] [S-readme] [T-main] [T-readme] [B-license] [B-track] [H-readme] [F-readme] [F-track] [M-readme] [360-readme] [360-pipeline]. Root license files were also inspected where present; missing-license findings concern the complete pinned trees, not a claim about every future branch.

## 3. Important specialist licenses and conflicts

| Component | Inspected status | Commercial/reuse implication |
|---|---|---|
| PRTReID / BPBReID source | Hippocratic License 3.0 with additional ethical/use conditions | Do not call these permissive MIT/Apache or assume unrestricted open-source integration [P-license] [BP-license] |
| PRTReID and HRNet/BPB checkpoint archives | Zenodo metadata labels CC-BY-4.0 | Distinct from code; archive metadata does not settle original training-data or every bundled asset right [Z-prt] [Z-hrnet] |
| TrackLab | MIT root | Framework is a useful permissive reference; selected plugins/upstream models still need review [L-readme] |
| GTA | MIT root | Offline algorithm code is a promising reuse candidate; OSNet checkpoint grant/provenance not established by MIT source [GTA-license] [GTA-readme] |
| McByte++ | Apache-2.0 root | Validate detector, mask-propagation and Re-ID dependency/weight terms separately [MC-license] [MC-install] |
| RF-DETR | Apache core; model-specific split documented | Nano/Small/Medium/Large detection and listed Seg/Keypoint models Apache; Atto/Femto/Pico/XL/2XL detection and NAS in plus use PML. Choose exact version/model [RF-readme] |
| D-FINE / official RT-DETR | Apache-2.0 source; checkpoints linked in official repos | Per-checkpoint/model/data provenance still required; an Ultralytics port uses a different software distribution [D-readme] [RT-readme] |
| BoT-SORT / ByteTrack | MIT roots | Re-ID/detector dependencies and weights separate [BOT-code] [BYTE-code] |
| StrongSORT upstream | GPL-3.0 | Do not transfer an MIT/Apache label from a wrapper onto upstream code [STR-readme] |
| TVCalib / Roboflow Sports | MIT roots | Calibration/detection weights and dataset terms separate [TV-readme] [R-main] |
| Jersey-number-pipeline | Custom `license.txt` contains explicit noncommercial condition | Research reference; do not assume PARSeq's own license licenses the whole pipeline [JR-license] |
| FOOTPASS | Root LICENSE Apache-2.0; README says annotations **and baselines** CC-BY-NC-4.0 | Conflicting grant, not resolved. Dataset currently requires manually approved SoccerNet NDA [FP-license] [FP-readme] [DS-fp] |
| OpenSportsLib | AGPL-3.0; commercial-license option documented | AGPL obligations or separate agreement; individual model cards/data grants still apply [OSL-license] [OSL-commercial] [OSL-card] |
| socceraction | MIT | Useful structured-event analytics code, not a license for StatsBomb/Wyscout/etc. data [SO-action] |
| unravelsports | MPL-2.0 | File-level copyleft framework; provider/training-data rights remain separate [UN-license] [UN-readme] |
| SoccerNet Re-ID / Tracking / Jersey kits | Re-ID root MIT; Tracking/Jersey complete license grants not established here | Public task instructions do not settle footage or every pretrained baseline grant [SN-reid] [SN-track] [SN-jersey] |

### Qwen-related assets
Official Qwen3-VL 8B/32B, Qwen3.5 9B/27B, and Qwen3.6-35B-A3B metadata inspected here declares Apache-2.0. acato's v11 merged and LoRA cards instead declare “other”/Tongyi-Qianwen and refer to the base's license. That discrepancy is **U** for the custom artifact; do not assert a specific MAU restriction as settled for Scoutra. [Q32-card] [Q35-card] [Q36-card] [C-weights] [C-adapter]

SoccerChat's checkpoint card declares Apache-2.0. Its dataset separates MIT annotations/metadata from SoccerNet-derived video, which requires NDA and excludes commercial video use/redistribution. OSL's Qwen XFoul adapter card declares AGPL-3.0. Soccer QA 4B declares CC-BY-NC-4.0. These differences matter even if all can be downloaded through Hugging Face. [CHAT-card] [DS-chat] [OSL-card] [QA-card]

Ultralytics publishes AGPL and enterprise licensing options. The exact grant for the chosen version, model, and deployment must be evaluated before a proprietary integration. A permissively licensed wrapper or ONNX runtime does not, by itself, resolve the model's grant. [UL-license] [C-ball]

## 4. Dependency compatibility: observed barriers
- **GSR is an older isolated stack:** inspected `pyproject.toml` specifies Python >=3.9,<3.10, torch 1.13.1, TrackLab 1.3.24, MMOCR 1.0.1, older Transformers and NumPy pins. TrackLab tag `v1.3.24` resolves to the same commit inspected in this audit (`5767e86…`), so its source analysis is consistent with this dependency pin. Do not combine its environment blindly with current Qwen runtimes. [G-pyproject]
- **AgentCoach's PRT adapter has compatibility shims and partial state loading.** That is an observable maintenance seam. It is not proof the model produces the intended features across arbitrary package versions. [A-reid]
- **jersey-tracker documents NumPy/EasyOCR ABI constraints and PARSeq loading requirements.** Its environment is tuned for Apple Silicon; a CUDA reproduction needs its own record. [J-readme]
- **Soccer-Tracking's requirements list omits some libraries imported by code/advertised in the README, and local weights remain hardcoded.** A requirements file is not a complete reproducibility specification. [T-main] [T-readme]
- **acato serving is version/format-specific.** The v11 card distinguishes `compressed-tensors` from base FP8 and reports one known-good vLLM version. Earlier “latest” image recipes are not equivalent. [C-weights] [C-gpu]
- **FOOTPASS depends on Decord, h5py, PyTorchVideo, and optionally PyG, with versioned environment descriptions.** Code expects precomputed tracking/roles and data layout; it is not raw amateur-video-to-events out of the box. [FP-run] [FP-gnn] [FP-readme]
- **GTA documents older Python/torchreid/setuptools setup requirements.** A reusable algorithm can still have a fragile packaged environment. [GTA-readme]

No dependency upgrades were attempted, and no installation claim is inferred from a lockfile or passing-test claim.

## 5. Reproducibility levels used in recommendations
**Level 1:** code/architecture only. **Level 2:** source plus accessible checkpoint listings and documented inference recipe. **Level 3:** public data/protocol and experiments suitable for external reproduction. **Level 4:** independent execution verified on Scoutra footage. **No candidate reaches Level 4 in this audit.**

GSR has strong Level-3 research structure but data/dependency conditions. acato's merged v11 is a Level-2 artifact with private training provenance. AgentCoach and jersey-tracker have substantial Level-1/2 pieces with missing local assets. Synthetic MatchTrace reproduces a different problem; that reproducibility should not be credited as football perception evidence.

## 6. Unresolved checks before any implementation choice
Resolve conflicting licenses; identify exact checkpoint revisions and complete asset grants; verify gated-data terms for the intended evaluation; record a clean isolated environment; load only approved source/weights in a later authorized test; reproduce output schemas and a small held-out clip before claiming integration savings. These are future selection criteria, not a request to implement or contact maintainers in this research task.

## Direct sources

[C-ball]: https://huggingface.co/acatorcini/yolov9-soccer-ball/blob/b30df5abc9f3eda4a9d326d953be12b3541a14b4/README.md "acato soccer ball checkpoint card"
[G-license]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/LICENSE "SoccerNet/sn-gamestate — LICENSE"
[DS-chat]: https://huggingface.co/datasets/SimulaMet/SoccerChat "SoccerChat dataset terms"
[A-notice]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/NOTICE "outstandingcandy/AgentCoach — NOTICE"
[A-offline]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/deploy/offline/README.md "outstandingcandy/AgentCoach — deploy/offline/README.md"
[J-readme]: https://github.com/spindle79/jersey-tracker/blob/e8021b51a1e021da058f4acd17e39f3d4ecb5051/README.md "spindle79/jersey-tracker — README.md"
[C-train]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/scripts/train_lora_v11.sh "acato/soccer-video-pipeline — scripts/train_lora_v11.sh"
[C-adapter]: https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-lora/blob/5aca84c3c96f42d582346710fd22322ae0ed6d6b/README.md "acato v11 LoRA card"
[G-reid]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/sn_gamestate/reid/prtreid_api.py "SoccerNet/sn-gamestate — sn_gamestate/reid/prtreid_api.py"
[S-readme]: https://github.com/MohibShaikh/sportvision/blob/972a3ed29d02aa9c7298924fa2ef8aafacd16998/README.md "MohibShaikh/sportvision — README.md"
[T-main]: https://github.com/Streakfull/Soccer-Tracking/blob/9d3488fc88390cbc64eb444eb08e481b79df0aef/src/TrackingPipeline/main.py "Streakfull/Soccer-Tracking — src/TrackingPipeline/main.py"
[T-readme]: https://github.com/Streakfull/Soccer-Tracking/blob/9d3488fc88390cbc64eb444eb08e481b79df0aef/README.md "Streakfull/Soccer-Tracking — README.md"
[B-license]: https://github.com/budun-ov/football-analytics/blob/cd2c43fce7f54100500058ac0ff5aba7a10fce94/LICENSE "budun-ov/football-analytics — LICENSE"
[B-track]: https://github.com/budun-ov/football-analytics/blob/cd2c43fce7f54100500058ac0ff5aba7a10fce94/src/detection/tracker.py "budun-ov/football-analytics — src/detection/tracker.py"
[H-readme]: https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/README.md "samuehuang/football-cv-analytics — README.md"
[F-readme]: https://github.com/dan-slater/footyviz/blob/4493f88d554a80dc3d9858c276173a17786caf8d/README.md "dan-slater/footyviz — README.md"
[F-track]: https://github.com/dan-slater/footyviz/blob/4493f88d554a80dc3d9858c276173a17786caf8d/src/fv/reid_track.py "dan-slater/footyviz — src/fv/reid_track.py"
[M-readme]: https://github.com/a6rahamjr/matchtrace-cv/blob/aa61cd6c301238875387df51b0467f401bc0b880/README.md "a6rahamjr/matchtrace-cv — README.md"
[360-readme]: https://github.com/humac/soccer360/blob/7b5546c8d0a648dd4c5037dfc70a9d0b09ee3fbb/README.md "humac/soccer360 — README.md"
[360-pipeline]: https://github.com/humac/soccer360/blob/7b5546c8d0a648dd4c5037dfc70a9d0b09ee3fbb/src/pipeline.py "humac/soccer360 — src/pipeline.py"
[P-license]: https://github.com/VlSomers/prtreid/blob/30617a75967e84d5d516959c4b84cbeea6f56493/LICENSE "VlSomers/prtreid — LICENSE"
[BP-license]: https://github.com/VlSomers/bpbreid/blob/a2dc4304284784d8b2c061764512f0698aa38c21/LICENSE "VlSomers/bpbreid — LICENSE"
[Z-prt]: https://zenodo.org/records/10653453 "PRTReID checkpoint archive and license metadata"
[Z-hrnet]: https://zenodo.org/records/10604211 "HRNet/BPBReID checkpoint archive and license metadata"
[L-readme]: https://github.com/TrackingLaboratory/tracklab/blob/5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2/README.md "TrackingLaboratory/tracklab — README.md"
[GTA-license]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/LICENSE "sjc042/gta-link — LICENSE"
[GTA-readme]: https://github.com/sjc042/gta-link/blob/e4d5cc4065ceb1ec3fa9dc7478455f13a8d7f9ca/README.md "sjc042/gta-link — README.md"
[MC-license]: https://github.com/tstanczyk95/McBytePlusPlus/blob/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9/LICENSE "tstanczyk95/McBytePlusPlus — LICENSE"
[MC-install]: https://github.com/tstanczyk95/McBytePlusPlus/blob/be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9/INSTALLATION.md "tstanczyk95/McBytePlusPlus — INSTALLATION.md"
[RF-readme]: https://github.com/roboflow/rf-detr/blob/e56ec6152b0eb1ac8bcca8690aa7ba59761d861f/README.md "roboflow/rf-detr — README.md"
[D-readme]: https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/README.md "Peterande/D-FINE — README.md"
[RT-readme]: https://github.com/lyuwenyu/RT-DETR/blob/df2a96c8c3a098567a1149bf0248da6efe6d495d/README.md "lyuwenyu/RT-DETR — README.md"
[BOT-code]: https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/tracker/bot_sort.py "NirAharon/BoT-SORT — tracker/bot_sort.py"
[BYTE-code]: https://github.com/FoundationVision/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/byte_tracker.py "FoundationVision/ByteTrack — yolox/tracker/byte_tracker.py"
[STR-readme]: https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/README.md "dyhBUPT/StrongSORT — README.md"
[TV-readme]: https://github.com/MM4SPA/tvcalib/blob/1222c5230af2742395d74918ed6f34eb2b9bf7f9/README.md "MM4SPA/tvcalib — README.md"
[R-main]: https://github.com/roboflow/sports/blob/42c80c06b6b65a7f89455b89fe31cdf4c38ba227/examples/soccer/main.py "roboflow/sports — examples/soccer/main.py"
[JR-license]: https://github.com/mkoshkina/jersey-number-pipeline/blob/007d54e5530a66616ed5081ca35e0028b36aadb5/license.txt "mkoshkina/jersey-number-pipeline — license.txt"
[FP-license]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/LICENSE "JeremieOchin/FOOTPASS — LICENSE"
[FP-readme]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/README.md "JeremieOchin/FOOTPASS — README.md"
[DS-fp]: https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026 "FOOTPASS dataset card and access terms"
[OSL-license]: https://github.com/OpenSportsLab/opensportslib/blob/b2fe8f16a35336d93b1b0eabe1741bbbbc36cd68/LICENSE "OpenSportsLab/opensportslib — LICENSE"
[OSL-commercial]: https://github.com/OpenSportsLab/opensportslib/blob/b2fe8f16a35336d93b1b0eabe1741bbbbc36cd68/LICENSE-COMMERCIAL "OpenSportsLab/opensportslib — LICENSE-COMMERCIAL"
[OSL-card]: https://huggingface.co/OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora/blob/8c7721ff5fd74a1ad27f64203aa3218c5f9f5db1/README.md "OSL Qwen3-VL XFoul checkpoint card"
[SO-action]: https://github.com/ML-KULeuven/socceraction/blob/93a1242d46c104889205753accaabadb00c45c6d/README.md "ML-KULeuven/socceraction — README.md"
[UN-license]: https://github.com/UnravelSports/unravelsports/blob/2bd24d7e2b3cc19cbdba6486c7abb70270a732ad/LICENSE "UnravelSports/unravelsports — LICENSE"
[UN-readme]: https://github.com/UnravelSports/unravelsports/blob/2bd24d7e2b3cc19cbdba6486c7abb70270a732ad/README.md "UnravelSports/unravelsports — README.md"
[SN-reid]: https://github.com/SoccerNet/sn-reid/blob/621e2b0f2d2a7a3e207b8dd747542b6608bf72db/README.md "SoccerNet/sn-reid — README.md"
[SN-track]: https://github.com/SoccerNet/sn-tracking/blob/b0bbba35e07ff58010b6313ef8aa59ef663ad392/README.md "SoccerNet/sn-tracking — README.md"
[SN-jersey]: https://github.com/SoccerNet/sn-jersey/blob/2f43b48c59eefe0bb5d948888db07f55f51208ad/README.md "SoccerNet/sn-jersey — README.md"
[Q32-card]: https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/blob/0cfaf48183f594c314753d30a4c4974bc75f3ccb/README.md "Official Qwen3-VL-32B model card"
[Q35-card]: https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md "Official Qwen3.5-9B model card"
[Q36-card]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/345d467b117d1fd5eda9b879a5988de9ae5f47a7/README.md "Official Qwen3.6-35B-A3B model card"
[C-weights]: https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-fp8/blob/ab56ac6743fc4f5ec299ed951f989d37f5874c56/README.md "acato v11 FP8 checkpoint card"
[CHAT-card]: https://huggingface.co/SimulaMet/SoccerChat-qwen2-vl-7b/blob/29871536004c0ac788af09cbb87969ab9f6e1410/README.md "SoccerChat checkpoint card"
[QA-card]: https://huggingface.co/sportsvision/soccer-qa-4b/blob/e0ef6b8b795fb91c0998c089ab2dc5c291b1ce05/README.md "Soccer QA 4B checkpoint card"
[UL-license]: https://www.ultralytics.com/license "Ultralytics licensing"
[G-pyproject]: https://github.com/SoccerNet/sn-gamestate/blob/1c958345067218297d221e45e1a6405f975f83e0/pyproject.toml "SoccerNet/sn-gamestate — pyproject.toml"
[A-reid]: https://github.com/outstandingcandy/AgentCoach/blob/eef7b0c5d757844fbbb1eeafa6a8050a863a6623/goalinsight/tracking/reid/prtreid_extractor.py "outstandingcandy/AgentCoach — goalinsight/tracking/reid/prtreid_extractor.py"
[C-gpu]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/docs/vllm-gpu-orchestration.md "acato/soccer-video-pipeline — docs/vllm-gpu-orchestration.md"
[FP-run]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/run_TAAD_on_matches.py "JeremieOchin/FOOTPASS — run_TAAD_on_matches.py"
[FP-gnn]: https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/models/model_GNN.py "JeremieOchin/FOOTPASS — models/model_GNN.py"
