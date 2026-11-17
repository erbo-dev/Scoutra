# Dataset and annotation requirements

## Available evidence
The user supplied twelve video filenames and one Kaggle mount path, reported in the conversation (the video-filename list is excluded from this source-only delivery). Their bytes are not accessible in this cloud. No Scoutra RF-DETR checkpoint, backend artifacts, reviewed detection/identity/event labels or experiment logs were found. The previous static audit is included under `previous-audit/`; it is research evidence, not a baseline measurement.

`discover()` scans all actual video files recursively, not only the twelve names. It records FPS/container frame-count duration/resolution/file hash; those are measured only when notebook 01 runs. It leaves `source_match` blank; camera/challenge unknown; annotation availability unknown; split unassigned. Hashing is whole-file CPU I/O. Empty rows/templates are not an invented dataset inventory.

## Sampling
Begin with one verified 20–60 s continuous shot, then add distinct difficult segments: crossing/occlusion, pan/zoom, tiny players/ball, front/back change, same-kit ambiguity, true disappearance/re-entry. Choose intervals before comparing models. Add easy controls. Record camera, venue/weather, native dimensions, period, selected player, cuts, replays and challenge labels. A highlight edit is a scene boundary; it is not evidence of recovery after a physical disappearance. Do not claim full-match results from highlights.

Match-disjoint splits are required when provenance can be established. Ideally include at least five independent matches for uncertainty estimates and a held-out venue/camera. Five is a minimum for an exploratory bootstrap, not proof of adequate statistical power. If only one source match is available, report within-match exploratory tests and obtain an independent match before progression. Detect near-duplicate/replayed action manually; footage shared across train/test invalidates independence. No training is planned in this phase.

## Detection / tracking ground truth
File `detections.json`:
```json
{
  "schema_version": "scoutra-ground-truth/1.0",
  "coordinate_system": "native_image_xyxy_pixels",
  "frames": []
}
```
Each real frame record needs `clip_id`, `scene_id`, zero-based **source** `frame_index`, `timestamp_s`, native `width`,`height`, `camera_type`, boolean `reviewed`, `boxes_exhaustive`, `identities_exhaustive`, and `objects`.

Each object has numeric `x1,y1,x2,y2` with half-open native pixel boundaries, `class_id`, `role`, `player_id` (string/null), `team_id` (string/null) and `ignore`. Canonical classes: **1 = on-pitch person** (players, goalkeepers, referees); **2 = ball**. `role` distinguishes player/goalkeeper/referee/unknown, enabling player-only recall. Generic COCO person weights cannot automatically distinguish players from referees. Spectators must have an explicit reviewed exclusion/ignore policy before scoring; current evaluator rejects frames with ignore regions instead of silently mishandling them.

Detection AP/FP requires exhaustively annotated person/ball boxes on each evaluated frame. Tracking official metrics require all persons with consistent physical IDs on **every frame** of the selected continuous sequence. A physical ID describes the same footballer across different tracker IDs; do not copy a tracker output as GT. Identity-uncertain frames remain incomplete and cannot support precision claims. Record visible boxes only, including partial visible bodies under a consistent policy; no invented offscreen boxes. Agree on truncation/occlusion/tiny-ball labeling criteria before annotation. Review a sample independently and report disagreement and unresolved cases.

The generated offline HTML supports manual boxes/IDs and completeness flags. It starts all flags false, exports no automatic labels, and marks edited frames unreviewed. It does not interpolate, track, calibrate or recognize players. Limit pages to ~100 frames using `--limit 100 --every 1 --start-index 0`, then start-index100/200/etc. for subsequent pages; merge their exported frames with `scoutra_lab.annotations.merge`, which rejects duplicates. Dense MOT labeling is labor-intensive, so begin with a smaller contiguous sequence. Keep unknown ID as null. For events, visible target intervals, number transcription, cropping tags and reviewer notes use the separate JSON/manifest files; the HTML does not pretend to label all tasks.

## Selected players and recovery
`selected_players.json` contains `schema_version: scoutra-selected-players/1.0`, `anchors` and `recovery_episodes`. Each anchor: `clip_id`, `player_id`, `frame_index`, an unambiguous human-selected target. The scorer maps the target only at its anchor; it never chooses the best ID over the whole video. Anchor failure is recorded. For deployment, a user-selection anchor is legitimate input; GT elsewhere is evaluation-only.

Each recovery episode should record target `player_id`, last visible frame, first visible return frame, reviewed intervening absence, evaluation window, scene continuity, occlusion/exit type and ambiguity. Report recovery success, delay, false selected-ID assignment, and unknown coverage. Current automatic evaluator supports anchor precision/coverage and false merges; **episode-specific recovery delay, false-confident assignment curves and split correctness still require reviewed episode reports or a validated evaluation extension**. Never describe them as already computed.

## Crop retrieval
One immutable crop JSON list: `crop_id`, `detection_id` if from a detector, `clip_id`, `frame_index`, `timestamp_s`, `path`, `sha256`, verified `source_match`, physical `player_id`, `team_id`, independent `tracklet_id`, `viewpoint`, `occlusion`, `crop_width`,`crop_height`, `reviewed`, `retrieval_role` (`query`/`gallery`). `make_crops` leaves unknown labels blank and retrieval roles unassigned. Use separate corpora for oracle GT crops and deployment detector-track crops. Do not use labels to split tracks in deployment conditions unless explicitly reported as an oracle upper bound.

Query/gallery cannot reuse a crop or source tracklet. Avoid adjacent near-duplicate frames as nominal long-gap positives. Include same-team hard negatives and front/back, partial occlusion, height<32/32–96/>=96 px and gap bins <2/2–10/10–60/>60 s. IDs are namespaced by verified source match; do not infer that jersey 10 is the same person in unrelated matches. Retrieval threshold/false-match operating point is selected on validation matches, locked before test. Report no-positive and missing-label queries separately.

## Events and calibration later
Events: class, start/contact/end time, actor physical ID/null, actor visibility, reviewer confidence, timing source and tolerance. Unknown actor remains unknown. Calibration: surveyed pitch dimensions, visible landmark pixels/world coordinates, camera/scene changes, independent held-out landmarks and feet points. Standard pitch dimensions must not be assumed for youth football. Event timestamps need true decoder PTS when VFR matters; current frame-index/FPS cache only supplies estimates.
