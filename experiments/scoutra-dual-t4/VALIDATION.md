# Validation record

## Repository and previous runs

GitHub `main` was verified at `26cd114c0c547cc164270a5bbcee3cd95255db9b` before the rebuild. [PR #1](https://github.com/erbo-dev/Scoutra/pull/1) was closed and merged at that SHA, and `a40392fe0235cb636f4da364d5a6d6412de94ec7` was in its ancestry. The previous lab required a separately mounted code and config dataset. Its release evidence described disabled experiment flags, model assets not yet loaded and external bridges missing for part-based Re-ID, GTA and McByte++.

The supplied `notebook7ed142f34a (2).ipynb` saved a twelve-file dataset audit. Its Kaggle package cell logged incompatible environment requirements involving huggingface-hub, gradio and diffusers. Downstream model and inference cells had no saved output. The uploaded notebook also assumed contiguous RF-DETR COCO IDs, which disagree with the sparse label space verified for the selected RF-DETR Small and Medium checkpoints. This notebook validates the exact weights and each implementation's distinct label map.

## STATICALLY VERIFIED

- Jupyter nbformat 4 validation, unique notebook cell IDs, all Python cells compiled, and all ten embedded Python modules compiled and compared byte-for-byte to the readable source.
- Deterministic `build_notebook.py --check`; all repository changes are within `experiments/`.
- Source and weight checksum checks for both official RF-DETR checkpoints, the three pinned Hugging Face snapshots, the BoxMOT OSNet asset, the GTA-packaged sports OSNet and the AgentCoach module. RF-DETR downloads matched upstream's published MD5 in addition to the recorded SHA-256.

## CPU TESTED

- Python 3.12.14, Torch 2.8.0+cpu and TorchVision 0.23.0+cpu. Isolated RF-DETR, Transformers, Ultralytics, BoxMOT and evaluation profiles passed dependency checks and import contracts.
- Real pretrained detection on the repository's `public/media/football-scene-01.jpg` at FP32 with RF-DETR Small, RF-DETR Medium, D-FINE Small, RT-DETR R18 and RT-DETRv2 R18. Each returned native-coordinate predictions. Batch size two was exercised. No model was trained.
- A five-frame local video derived from that image was decoded and passed through RF-DETR Small and all three trackers. ByteTrack, BoT-SORT with camera motion compensation and BoxMOT StrongSORT read identical prediction bytes and produced tracks. FFprobe validated all four annotated MP4 outputs as five-frame H.264 streams. This fixture checks contracts with pretrained models and is unsuitable for accuracy, tracking robustness or speed claims.
- Actual BoxMOT OSNet inference returned a 512-dimensional embedding after equality checks on all inference-layer tensors.
- Two spawned CPU workers overlapped and consumed queued tasks. CPU integration also ran the real ByteTrack runner across both processes. Tests covered per-task exceptions and an abrupt worker exit.
- The 16 contract tests cover exact and relocated dataset discovery, ambiguity and missing own files, decoding twelve fixture files and rejecting corruption, GPU selection policy, multiprocessing, cache and archive integrity, COCO and TrackEval fixture scores, reviewed re-entry scoring, part/visibility export shape and no-common-parts behavior, homography, OCR aggregation and the PREFLIGHT/REPORT export paths.

## NOT TESTED

- Codex Cloud had no CUDA device or Kaggle mounts. Real Kaggle T4/T4 execution, memory, utilization, FP16 and the measured one-versus-two GPU comparison remain pending.
- The twelve Kaggle videos were neither accessible nor inferred in Codex Cloud. No accuracy or performance results on the Scoutra dataset are shipped.
- The local Scoutra checkpoint, Deep-EIoU legacy runtime, PRTReID Global, PRTReID Parts, BPBReID Parts and GTA-Link adapter require actual model smoke runs in their chosen environments.
- Human-reviewed detection, tracking, Re-ID and identity labels were not supplied. Their metrics must remain `NOT_EVALUATED` for unlabeled runs.

## BLOCKED

The [method matrix](README.md#method-support-and-provenance) records model-specific prerequisites. McByte++ and its identity bank need the genuine mask and appearance infrastructure. Other advanced candidates require method-specific checkpoints and runtime integration. The notebook keeps each candidate's blocker in its method status records.
