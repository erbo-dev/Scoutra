# Phase 2 asset review and reproducibility

This package contains laboratory utilities and research documents authored for this task. It does not vendor third-party model implementations or copy them into Scoutra. Import wrappers/source acquisition are experimental interfaces, not a declaration that every upstream asset is commercially cleared.

Review code, weights, datasets and access conditions independently. The versioned asset/source links and known grants are in MODEL_INVENTORY; the full evidence audit is preserved in `previous-audit/LICENSE_AND_REPRODUCIBILITY.md`.

- RF-DETR Small/Medium and selected D-FINE/RT-DETR source: Apache paths; choose exact distributions/models and preserve checkpoint provenance. A generic COCO model is not an available private Scoutra checkpoint.
- supervision ByteTrack: MIT distribution. BoxMOT16.0.10 ports: AGPL source, distinct from original BoT-SORT MIT / StrongSORT GPL. OSNet source licensing does not settle the downloaded MSMT17 checkpoint's complete asset chain.
- PRTReID/BPBReID: Hippocratic source; Zenodo checkpoint archives labeled CC-BY-4.0; SoccerNet/Market1501 data rights separate. They are not permissive-code drop-ins.
- GSR GPL / TrackLab MIT / original TrackEval MIT. Legacy dependency and Python requirements are not verified by a README or a source pin.
- AgentCoach Apache root and GTA MIT; chosen feature models still carry their terms. McByte++ Apache root does not license every detector/mask/Re-ID model it uses.
- OpenSportsLib AGPL; FOOTPASS root/README Apache versus NC conflict; acato model-card/base-license mismatch; SoccerChat video NDA/data restrictions: unresolved conditions remain explicit. Consult exact pinned cards in the prior audit.
- User footage/annotations: rights and permissions are not established merely by a Kaggle mount path. Keep the experimental Dataset private according to the user's authorized access; no sharing/publication action was performed.

No large weights were fetched and no download/inference was verified. Local checkpoints need a trusted source, expected SHA256, architecture/config match and explicit `license_reviewed` flag before the model adapters load them. Directory fingerprints hash a sorted list of relative paths and file SHA256 values; record the complete folder contents. Source archive hashes identify inspected package versions, not weights.

Reproducibility requires a successful isolated environment solve plus `pip check`, full freeze, Python/CUDA/device metadata, input/annotation/cache hashes, source commits, checkpoint hashes, raw class map, seed, frame/crop order, preprocessing, distance rules, timing scope and all errors. Proposed direct pins are not falsely described as tested lockfiles. No performance improvement or hardware fit is asserted without subsequent comparable execution.
