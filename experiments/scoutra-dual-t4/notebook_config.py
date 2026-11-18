# Run this cell after attaching zhanto-highlights and selecting Kaggle GPU T4 x2.
RUN_MODE = "SMOKE"                   # PREFLIGHT, SMOKE, BENCHMARK, ADVANCED, REPORT
AUTO_ACQUIRE = False                 # First model run: enable Kaggle Internet and set True.
USE_BOTH_GPUS = True
ALLOW_SINGLE_GPU = False             # Explicit consent to one-GPU fallback or a single-GPU run.
DATASET_ROOT = "auto"
MAX_VIDEOS = 12                     # All 12 are always audited.
MODEL_GROUPS = ["detection", "tracking"]
DETECTOR_METHODS = ["rfdetr_small"]  # Add rfdetr_medium, dfine_small, rtdetr_r18vd, rtdetrv2_r18vd.
TRACKER_METHODS = ["bytetrack", "botsort_cmc"]  # strongsort needs OSNET_CHECKPOINT.
TRACK_DETECTOR = "rfdetr_small"      # The sole common detector cache for tracker comparison.
SCALING_MODELS = ["rfdetr_small"]    # BENCHMARK compares this exact workload on 2 GPUs and 1 GPU.
SAVE_ANNOTATED_VIDEOS = True

# Keep the first smoke small. Start at FP32; enable FP16 only with comparison tests.
SMOKE_FRAMES = 8
BATCH_SIZE = 4
PRECISION = "fp32"
DETECTOR_SCORE_FLOOR = 0.05
OPERATING_SCORE = 0.25
TRACKER_SETTINGS = {"high": 0.25, "low": 0.10, "new": 0.25, "match": 0.80, "buffer": 30}
JOB_TIMEOUT_S = 900
ASSET_CACHE = None                  # Optional offline cache path containing the reviewed files.
CACHE_RUN = None                    # Prior run ID or extracted run directory for frozen detections/tracks.
REPORT_RUN = None                   # REPORT uses the latest saved run by default.

# Run advanced methods independently via RUN_MODE="ADVANCED" and this explicit list.
ADVANCED_METHODS = []               # Example ["no_link", "temporal_ball_baseline"] with CACHE_RUN set.
IDENTITY_TRACKER = "bytetrack"
IDENTITY_THRESHOLD = 0.90
IDENTITY_SWEEP = []                  # Optional additional cosine_tracklet_link thresholds, e.g. [0.88, 0.92, 0.95].
CROP_MANIFEST = None                # Or select prepare_crops before OSNet/part extraction in the same run.
CROP_STRIDE = 5
CROPS_PER_TRACK = 12
IDENTITY_FEATURES = None            # Saved, clip/track-fingerprinted global NPZ for identity linking.
LEGACY_REID = {}                    # See repository guide for exact PRTReID/BPBReID preparation.
DEEP_EIOU_CONFIG = {}               # Optional isolated tracker checkout and sports checkpoint.
GTA_CONFIG = {}                     # Optional isolated GTA-Link source/checkpoint configuration, see guide.
PITCH_LANDMARKS = None
OCR_PREDICTIONS = None
SCOUTRA_CHECKPOINT = None           # {'path','sha256','architecture','resolution','class_map','provenance'}
OSNET_CHECKPOINT = None             # {'path','sha256','provenance'}, require verified upstream OSNet filename.

DETECTION_ANNOTATIONS = None
TRACKING_ANNOTATIONS = None
REID_ANNOTATIONS = None
IDENTITY_ANNOTATIONS = None
REID_DISTANCE_THRESHOLD = None     # Set only from a held-out validation split before calculating FMR.
PART_VISIBILITY_MIN = 0.5
DISTANCE_SAMPLE_CROPS = 1200

CONFIG = {
    "mode": RUN_MODE, "auto_acquire": AUTO_ACQUIRE, "use_both": USE_BOTH_GPUS, "allow_single": ALLOW_SINGLE_GPU,
    "dataset_root": DATASET_ROOT, "max_videos": MAX_VIDEOS, "groups": MODEL_GROUPS,
    "detectors": DETECTOR_METHODS, "trackers": TRACKER_METHODS, "track_detector": TRACK_DETECTOR,
    "scaling_models": SCALING_MODELS, "save_videos": SAVE_ANNOTATED_VIDEOS, "smoke_frames": SMOKE_FRAMES,
    "batch": BATCH_SIZE, "precision": PRECISION, "score": DETECTOR_SCORE_FLOOR, "operating_score": OPERATING_SCORE,
    "tracker_settings": TRACKER_SETTINGS, "timeout": JOB_TIMEOUT_S, "asset_cache": ASSET_CACHE,
    "cache_run": CACHE_RUN, "report_run": REPORT_RUN, "advanced_methods": ADVANCED_METHODS,
    "identity_tracker": IDENTITY_TRACKER, "identity_threshold": IDENTITY_THRESHOLD, "identity_sweep": IDENTITY_SWEEP,
    "crop_manifest": CROP_MANIFEST, "crop_stride": CROP_STRIDE, "crops_per_track": CROPS_PER_TRACK,
    "identity_features": IDENTITY_FEATURES, "legacy_reid": LEGACY_REID, "gta_config": GTA_CONFIG, "deep_eiou_config": DEEP_EIOU_CONFIG,
    "landmarks": PITCH_LANDMARKS, "ocr_predictions": OCR_PREDICTIONS, "scoutra_checkpoint": SCOUTRA_CHECKPOINT,
    "osnet_checkpoint": OSNET_CHECKPOINT, "detection_annotations": DETECTION_ANNOTATIONS,
    "tracking_annotations": TRACKING_ANNOTATIONS, "reid_annotations": REID_ANNOTATIONS,
    "identity_annotations": IDENTITY_ANNOTATIONS, "reid_threshold": REID_DISTANCE_THRESHOLD,
    "visibility_min": PART_VISIBILITY_MIN, "distance_sample": DISTANCE_SAMPLE_CROPS,
}
