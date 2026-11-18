# Scoutra technology audit

Status reflects an executable implementation, prerequisites, and recorded test evidence on the available setup. CPU and fixture tests are identified as such. The failed Kaggle run executed zero detector batches, and none of these entries establishes current Kaggle T4 inference. The default notebook runs only RF-DETR Small, ByteTrack and BoT-SORT. All other technologies remain opt-in.

| Registered method | Status | Evidence or prerequisite |
|---|---|---|
| `rfdetr_small` | `IMPLEMENTED_AND_TESTED` | RF-DETR 1.12.1 official weights, pretrained forward and 12-clip CPU integration. The repaired T4 path is awaiting a run. |
| `rfdetr_medium` | `IMPLEMENTED_AND_TESTED` | Pinned official weights and real CPU image forward. Repeat the two-worker smoke for the larger 576-pixel preset. |
| `dfine_small` | `IMPLEMENTED_AND_TESTED` | Versioned Transformers model and hashed checkpoint, real CPU image forward. |
| `rtdetr_r18vd` | `IMPLEMENTED_AND_TESTED` | Versioned Transformers model and hashed checkpoint, real CPU image forward. |
| `rtdetrv2_r18vd` | `IMPLEMENTED_AND_TESTED` | Versioned Transformers model and hashed checkpoint, real CPU image forward. |
| `scoutra_rfdetr` | `REQUIRES_CHECKPOINT` | Supply the custom checkpoint, SHA-256, trained class map, input resolution and architecture. |
| `specialized_soccer_ball` | `REQUIRES_INTEGRATION` | Choose a source-pinned ball detector checkpoint with license and class map, then implement and validate its loader. |
| `bytetrack` | `IMPLEMENTED_AND_TESTED` | Ultralytics ByteTrack CPU tracking on all 12 frozen CPU detections. Kaggle run awaited. |
| `botsort_cmc` | `IMPLEMENTED_AND_TESTED` | Ultralytics BoT-SORT CPU tracking with camera-motion compensation, using the same hashes as ByteTrack. Kaggle run awaited. |
| `strongsort` | `BLOCKED` | BoxMOT upstream metadata excludes Python 3.13. Earlier Python 3.12 CPU tests used the verified OSNet weights. |
| `deep_eiou` | `REQUIRES_CHECKPOINT` | Adapter expects the verified sports OSNet checkpoint, pinned Deep-EIoU checkout and legacy NumPy runtime. |
| `mcbyteplusplus` | `REQUIRES_INTEGRATION` | Implement and test the upstream EdgeTAM mask, sports embedding and frozen-detection interfaces. |
| `prepare_crops` | `IMPLEMENTED_BUT_UNTESTED` | Clip-hash and track-hash guarded crop extraction exists. Needs a run over exported footage and tracks. |
| `osnet` | `BLOCKED` | Verified MSMT17 checkpoint produced a 512-D vector in the preceding Python 3.12 CPU test. The pinned BoxMOT profile excludes Python 3.13. |
| `prtreid_global` | `REQUIRES_CHECKPOINT` | Upstream integration requires PRTReID checkpoint, matching HRNet, resolved training config and separate environment. |
| `prtreid_parts` | `REQUIRES_CHECKPOINT` | Same prerequisites and a model output check for all part embeddings, visibility and spatial masks. |
| `bpbreid_parts` | `REQUIRES_CHECKPOINT` | Requires its own verified checkpoint, HRNet and original configuration. Its source API differs from PRTReID. |
| `no_link` | `IMPLEMENTED_BUT_UNTESTED` | Clip-local no-link baseline exists. Identity scoring has fixture tests. |
| `agentcoach_reid_only` | `IMPLEMENTED_BUT_UNTESTED` | Pinned upstream consolidation module and verified global embeddings tied to the same tracking cache. |
| `cosine_tracklet_link` | `IMPLEMENTED_BUT_UNTESTED` | Implemented co-occurrence-aware baseline. Needs verified global crop embeddings. |
| `gta_link` | `IMPLEMENTED_BUT_UNTESTED` | Pinned original GTA-Link split/merge calls exist. Needs an isolated legacy checkout, verified original sports checkpoint and a real end-to-end run. |
| `gtatrack` | `REQUIRES_INTEGRATION` | Resolve the separate upstream checkpoints and detector-feature contract. |
| `mcbyte_identity_bank` | `REQUIRES_INTEGRATION` | Run real McByte++ association first, then wire identity-bank output and the bank on/off switch. |
| `cross_multiplication_reid` | `REQUIRES_INTEGRATION` | Locate a reproducible upstream artifact and wire preprocessing and distances. |
| `parseq` | `REQUIRES_INTEGRATION` | Pin upstream STR runtime, jersey checkpoint, tokenizer and decoder. |
| `koshkina` | `REQUIRES_INTEGRATION` | Requires upstream legibility, pose-guided crops, jersey STR weights and tracking workflow. |
| `temporal_ocr_vote` | `IMPLEMENTED_AND_TESTED` | Fixture tested confidence aggregation from frame-level OCR input. |
| `opencv_homography` | `IMPLEMENTED_AND_TESTED` | Fixture tested exact image-to-meter correspondences and checked fit RMSE. Real landmark annotations awaited. |
| `tvcalib` | `REQUIRES_INTEGRATION` | Pin upstream pitch-segment network, weights and field model. |
| `pnlcalib` | `REQUIRES_INTEGRATION` | Pin upstream line/point detections, checkpoint and correspondences. |
| `temporal_ball_baseline` | `IMPLEMENTED_BUT_UNTESTED` | Implemented confidence and constant-velocity pixel gate on frozen detections. Needs footage validation. |
| `ballradar` | `REQUIRES_INTEGRATION` | Validate upstream trajectory formats and checkpoint conversion. |
| `pathcrf` | `REQUIRES_INTEGRATION` | Validate upstream checkpoint and candidate-association contract. |
| `t_deed` | `REQUIRES_INTEGRATION` | Tie backbone/features and sampling rate to the official checkpoint and SoccerNet labels. |
| `opensportslib` | `REQUIRES_INTEGRATION` | Pin a usable event backend and checkpoint. |
| `footpass` | `REQUIRES_INTEGRATION` | Pin a model checkpoint and map its pass and actor taxonomy to reviewed footage. |
| `qwen3_vl` | `REQUIRES_INTEGRATION` | Pin an official video-capable revision, processor and a compatible T4 runtime. |
| `qwen3_5` | `BLOCKED` | Registered 27B–35B condition has no verified T4 memory fit or compatible inference configuration. |
| `soccerchat` | `REQUIRES_INTEGRATION` | Verify base VLM, football adapter, video contract and respective licenses. |
| `football_qwen` | `REQUIRES_INTEGRATION` | Select a reproducible football-specific revision and validate its VLM runner. |

## Follow-up tests after the default Kaggle benchmark passes

| Method | Test and required evidence |
|---|---|
| RF-DETR Medium | Add Medium to `DETECTOR_METHODS`, keep the identical clips, and require two distinct UUIDs with CUDA forwards. Record resolution, VRAM, throughput and reviewed box quality alongside Small. |
| D-FINE and RT-DETR / RT-DETRv2 | Add one pinned model at a time. Confirm checkpoint and sparse/contiguous label handling, validate native-coordinate outputs and compare over the same clips. |
| OSNet | Use a compatible BoxMOT Python 3.12 environment with verified weights. Reuse a hashed crop manifest. Verify 512-D outputs and evaluate reviewed player retrieval across occlusions. |
| PRTReID and BPBReID | Provision the pinned source, weights, model config and HRNet before use. Validate full outputs on the same crops with visibility and spatial masks, then compare against the global embedding on a reviewed split. |
| GTA-Link | Use a pinned isolated environment and one frozen cache. Validate MOT-to-tracklet conversion and compare no-link and upstream linking on reviewed occlusions, measuring false merges, splits and recovery delay. |
| McByte++ | Confirm EdgeTAM and the sports checkpoint load in the upstream environment. Verify per-frame mask alignment and frozen detections, then compare the association bank on and off. |
| Jersey OCR | Verify the pinned PARSeq or Koshkina implementation and weights using reviewed digits, with a held-out clip split and explicit unreadable labels. Then compare per-frame STR and temporal voting. |
| Calibration | Start with independently held-out, reviewed pitch points for the OpenCV baseline. Prepare official TVCalib or PnLCalib inputs and record fit error and held-out reprojection error. |
| Football VLM | Select an upstream-video-compatible checkpoint that fits the intended T4 profile. Verify frame sampling with timestamps, measure memory, and evaluate temporal event claims on reviewed clips. |

The registry coverage test requires exactly one audit row for each registered name and one allowable status per row. Decision criteria and offline artifacts are described in [the run guide](README.md).
