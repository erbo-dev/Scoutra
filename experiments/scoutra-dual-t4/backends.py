"""Pinned model and tracker interfaces. COCO and football-trained labels stay explicit."""
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from common import BOX, Blocked, canonical_box, fetch, valid_score, verified

RF_COMMIT = "f29e2a8bbb7cdcb8920bf74a335062f88a35046d"
HF_COMMIT = "753d61104116eefc8ffc977327b441ee0c8d599f"
ULTRA_COMMIT = "4b884f4a44685f6ea45f4676c1bca50a2d72d443"
RF_TRANSFORMERS_COMMIT = "ddb849abe009d1089e6c691bfc897f27211c663c"
OSNET_SHA256 = "6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99"
BOXMOT_COMMIT = "c877349157a35df6095e703e002c60418d2b34ab"
RF = {
    "rfdetr_small": ("small", "d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33"),
    "rfdetr_medium": ("medium", "749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8"),
}
HF = {
    "dfine_small": (
        "ustc-community/dfine-small-coco", "f79e65b5fbb33ceb9d3ebba042955d7410c608f8", "d_fine",
        "9f206c4c935c67c957af2f15c3333b4ff224e72b1881bc7bbde9f873edf4448e",
        "7e9e2245806aa3ef068a5321f5eebee3116fa4cec560173231954ffaca930eca",
        "a144baff3a27bfd538fe37b11fa8d4fca5744b8c054b5301e7c699a1379cd18f",
    ),
    "rtdetr_r18vd": (
        "PekingU/rtdetr_r18vd", "ac77a11ff0170a41b771c03264987f8ce2b0d753", "rt_detr",
        "8493be71f51a1c0a741f8f71ec151039227579379de7bcba047c0470d9320c3c",
        "ffb4b9461a1dad746be8f0f9c8330ed7743a1ba5fba4f75c232cd281b3d4c64a",
        "fe87a5a30f5daf298d10794c7682a63b6107986f97d6a770ba948d89e4340093",
    ),
    "rtdetrv2_r18vd": (
        "PekingU/rtdetr_v2_r18vd", "5650961749fa93567c0d46fc7f43ea4f9e914107", "rt_detr_v2",
        "ed051ec77cb41c5d9d5e3af21a979b1e890599dfd68434d7636ff82ded4c1527",
        "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1",
        "d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115",
    ),
}
DETECTORS = {**dict.fromkeys(RF, "rf"), **dict.fromkeys(HF, "hf"), "scoutra_rfdetr": "rf"}
TRACKERS = {"bytetrack": "tracker", "botsort_cmc": "tracker", "strongsort": "boxmot"}
CLASS_MAP = {"person": "person", "sports ball": "ball"}
RF_COCO_LABELS = {1: "person", 37: "sports ball"}
HUMANS = {"person", "player", "goalkeeper", "referee"}


def detector_asset(name, config, cache, probe=None):
    offline = Path(config["asset_cache"]) if config.get("asset_cache") else None
    online = config["auto_acquire"]
    if name in RF:
        size, checksum = RF[name]
        release = f"{size}_coco/checkpoint_best_regular.pth"
        url = f"https://repo.roboflow.com/rfdetr/{release}"
        mirror = f"https://storage.googleapis.com/rfdetr/{release}"
        filename = f"rf-detr-{size}.pth"
        local = offline / filename if offline else None
        if local and local.is_file():
            path = verified(local, checksum)
        else:
            path = fetch(url, cache / filename, checksum, online, mirrors=(mirror,), probe=probe)
        return dict(path=str(path), weights_sha256=checksum, weights_source=url, weights_mirrors=[mirror], source_commit=RF_COMMIT,
                    checkpoint_kind="COCO", checkpoint_license="Apache-2.0",
                    transformers_commit=RF_TRANSFORMERS_COMMIT)
    if name in HF:
        repo, revision, model_type, *hashes = HF[name]
        url = f"https://huggingface.co/{repo}/resolve/{revision}"
        files = dict(zip(("config.json", "preprocessor_config.json", "model.safetensors"), hashes))
        local = offline / name if offline else None
        if local and all((local / filename).is_file() for filename in files):
            for filename, checksum in files.items():
                verified(local / filename, checksum)
            directory = local.resolve()
        else:
            directory = cache / name
            for filename, checksum in files.items():
                fetch(f"{url}/{filename}", directory / filename, checksum, online, probe=probe)
        return dict(path=str(directory.resolve()), files=files, weights_source=repo, weights_revision=revision,
                    weights_sha256=hashes[-1], source_commit=HF_COMMIT, model_type=model_type,
                    checkpoint_kind="COCO", checkpoint_license="Apache-2.0")
    spec = config["scoutra_checkpoint"]
    if not spec:
        raise Blocked("Set SCOUTRA_CHECKPOINT with trusted path, SHA-256, class map, base architecture and resolution.")
    required = {"path", "sha256", "architecture", "resolution", "class_map", "provenance"}
    if not required <= set(spec) or spec["architecture"] not in {"RFDETRSmall", "RFDETRMedium", "RFDETRLarge", "RFDETRBase", "RFDETRNano"}:
        raise Blocked(f"SCOUTRA_CHECKPOINT requires {sorted(required)} and a supported RF-DETR architecture.")
    if not spec["class_map"] or not set(spec["class_map"].values()) <= HUMAN_AND_BALL:
        raise Blocked("Custom class_map must match training IDs and use canonical labels person/player/goalkeeper/referee/ball.")
    path = verified(spec["path"], spec["sha256"])
    return dict(path=str(path), weights_sha256=spec["sha256"], weights_source=spec["provenance"],
                source_commit=RF_COMMIT, checkpoint_kind="Scoutra fine-tuned (user-supplied)",
                architecture=spec["architecture"], resolution=spec["resolution"], class_map=spec["class_map"],
                trust_checkpoint=bool(spec.get("trust_checkpoint", False)))


HUMAN_AND_BALL = HUMANS | {"ball"}


def osnet_asset(config):
    spec = config["osnet_checkpoint"]
    if not spec or not spec.get("provenance"):
        raise Blocked("Provide a rights-reviewed OSNET_CHECKPOINT with path, SHA-256 and provenance.")
    if spec.get("sha256") != OSNET_SHA256:
        raise Blocked(f"The isolated BoxMOT condition accepts the verified upstream OSNet checkpoint SHA-256 {OSNET_SHA256}.")
    path = verified(spec["path"], OSNET_SHA256)
    if path.name != "osnet_x0_25_msmt17.pt":
        raise Blocked("BoxMOT infers architecture from filename. Keep the audited osnet_x0_25_msmt17.pt filename.")
    return dict(path=str(path), weights_sha256=spec["sha256"], weights_source=spec["provenance"],
                source_commit=BOXMOT_COMMIT, checkpoint_kind="OSNet x0.25 MSMT17 (user supplied)")


def module_devices(module):
    return {
        "parameters": dict(Counter(str(t.device) for t in module.parameters())),
        "buffers": dict(Counter(str(t.device) for t in module.buffers())),
    }


def place_rfdetr(detector, device):
    """Bind pinned RF-DETR's lazily loaded eager model to this worker's device."""
    import torch

    context = detector.model
    device = torch.device(device)
    if torch.device(context.device) != device:
        raise RuntimeError(f"RF-DETR inference context requested {context.device}, worker assigned {device}")
    before = module_devices(context.model)
    before_postprocess = module_devices(context.postprocess)
    print(f"RF-DETR device before placement: target={device} model={before} postprocess={before_postprocess}", flush=True)
    # rfdetr 1.12.1 constructs the nn.Module on CPU and defers .to(device) to predict().
    # Move its existing eager module here so pre-inference assertions check placed weights.
    context.model = context.model.to(device=device).eval()
    context.postprocess = context.postprocess.to(device=device).eval()
    context.device = device
    after = module_devices(context.model)
    postprocess = module_devices(context.postprocess)
    info = {"requested_device": str(device), "context_device": str(context.device),
            "before": before, "before_postprocess": before_postprocess, "after": after, "postprocess": postprocess}
    print(f"RF-DETR device after placement: {json.dumps(info, sort_keys=True)}", flush=True)
    if (torch.device(context.device) != device or not after["parameters"]
            or any(set(devices) - {str(device)} for states in (after, postprocess) for devices in states.values())):
        raise RuntimeError(f"RF-DETR parameter/buffer placement failed: {info}")
    return info


class Detector:
    def __init__(self, name, asset, device, precision, score):
        import torch

        self.name, self.device = name, torch.device(device)
        self.precision, self.score = precision, score
        self.load_info = {}
        self.forward_evidence = None
        self.is_rf = name not in HF
        if self.is_rf:
            import rfdetr

            if name == "scoutra_rfdetr":
                self.model = rfdetr.RFDETR.from_checkpoint(
                    asset["path"], device=device, trust_checkpoint=asset["trust_checkpoint"]
                )
                if type(self.model).__name__ != asset["architecture"] or self.model.model_config.resolution != asset["resolution"]:
                    raise ValueError("Scoutra checkpoint architecture or input resolution disagrees with the declared settings")
                self.custom = {int(k): value for k, value in asset["class_map"].items()}
                labels = self.model.class_names
                if len(labels) != len(self.custom) or set(self.custom) != set(range(len(labels))):
                    raise ValueError("Scoutra class map must cover all contiguous checkpoint classes")
                self.load_info["checkpoint_class_names"] = labels
            else:
                cls = {"rfdetr_small": rfdetr.RFDETRSmall, "rfdetr_medium": rfdetr.RFDETRMedium}[name]
                self.model = cls(pretrain_weights=asset["path"], device=device)
                self.custom = None
                from rfdetr.assets.coco_classes import COCO_CLASSES
                if {i: COCO_CLASSES.get(i) for i in RF_COCO_LABELS} != RF_COCO_LABELS:
                    raise ValueError("RF-DETR sparse COCO map changed")
                self.load_info["raw_id_map"] = {i: CLASS_MAP[label] for i, label in RF_COCO_LABELS.items()}
            self.load_info["rfdetr_version"] = version("rfdetr")
            self.load_info["placement"] = place_rfdetr(self.model, self.device)
            self.load_info["input_resolution"] = self.model.model_config.resolution
            self.model.model.model.register_forward_hook(self._rf_forward)
        else:
            from transformers import (DFineForObjectDetection, RTDetrForObjectDetection,
                                      RTDetrImageProcessor, RTDetrV2ForObjectDetection)

            cls = {"d_fine": DFineForObjectDetection, "rt_detr": RTDetrForObjectDetection,
                   "rt_detr_v2": RTDetrV2ForObjectDetection}[asset["model_type"]]
            self.processor = RTDetrImageProcessor.from_pretrained(asset["path"], local_files_only=True)
            self.model, info = cls.from_pretrained(
                asset["path"], local_files_only=True, use_safetensors=True, output_loading_info=True
            )
            for key in ("missing_keys", "mismatched_keys", "error_msgs"):
                if info.get(key):
                    raise ValueError(f"{name}: incompatible checkpoint: {key}={info[key]}")
            self.labels = {int(k): value for k, value in self.model.config.id2label.items()}
            if self.model.config.model_type != asset["model_type"] or {k: self.labels.get(k) for k in (0, 32)} != {0: "person", 32: "sports ball"}:
                raise ValueError(f"{name}: incorrect model architecture or COCO label map")
            self.model = self.model.to(self.device).eval()
            self.load_info = {"unexpected_keys": info.get("unexpected_keys", []), "raw_id_map": {0: "person", 32: "ball"},
                              "input_resolution": self.processor.size}

    def _rf_forward(self, _module, inputs, output):
        self._record_forward(inputs[0], output["pred_logits"], output["pred_boxes"])

    def _record_forward(self, inputs, logits, boxes):
        devices = (inputs.device, logits.device, boxes.device)
        if any(device != self.device for device in devices):
            raise RuntimeError(f"{self.name}: expected forward tensors on {self.device}, got {devices}")
        size = inputs.shape[0]
        if size < 1 or logits.ndim != 3 or boxes.ndim != 3 or boxes.shape[-1] != 4 or logits.shape[:2] != boxes.shape[:2] or logits.shape[0] != size:
            raise RuntimeError(f"{self.name}: unexpected forward shapes {inputs.shape}, {logits.shape}, {boxes.shape}")
        if self.forward_evidence is not None:
            raise RuntimeError(f"{self.name}: more than one forward observed for the current batch")
        self.forward_evidence = {"frames": size, "input_device": str(devices[0]), "logits_device": str(devices[1]),
                                 "boxes_device": str(devices[2]), "logits_shape": list(logits.shape),
                                 "boxes_shape": list(boxes.shape)}

    def predict(self, rgb):
        import torch

        self.forward_evidence = None
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.float16, enabled=self.precision == "fp16"):
            if self.is_rf:
                outputs = self.model.predict(rgb, threshold=self.score, include_source_image=False)
                raw = []
                for out in outputs:
                    labels = out.data.get("class_name")
                    if len(out) and labels is None:
                        raise ValueError("RF-DETR did not export class names")
                    raw.append(zip(out.xyxy, out.confidence, out.class_id, labels if labels is not None else []))
            else:
                inputs = self.processor(images=rgb, return_tensors="pt")
                if self.device.type == "cuda":
                    inputs = {key: tensor.pin_memory().to(self.device, non_blocking=True) for key, tensor in inputs.items()}
                outputs = self.model(**inputs)
                self._record_forward(inputs["pixel_values"], outputs.logits, outputs.pred_boxes)
                sizes = torch.as_tensor([im.shape[:2] for im in rgb], device=self.device)
                predictions = self.processor.post_process_object_detection(outputs, threshold=self.score, target_sizes=sizes)
                raw = []
                for out in predictions:
                    ids = out["labels"].detach().cpu().tolist()
                    raw.append(zip(out["boxes"].detach().cpu().numpy(), out["scores"].detach().cpu().numpy(),
                                   ids, (self.labels[i] for i in ids)))
        if not self.forward_evidence or self.forward_evidence["frames"] != len(rgb):
            raise RuntimeError(f"{self.name}: no matching forward batch was observed at the model boundary")
        if len(raw) != len(rgb):
            raise ValueError("Model did not return one prediction object per image")
        return [self._normalize(rows, (image.shape[1], image.shape[0])) for image, rows in zip(rgb, raw)]

    def _normalize(self, rows, shape):
        result = []
        for values, score, raw_id, label in rows:
            raw_id = int(raw_id)
            label = str(label)
            canon = self.custom.get(raw_id) if self.name == "scoutra_rfdetr" else CLASS_MAP.get(label)
            if self.name in RF and (raw_id in RF_COCO_LABELS or label in CLASS_MAP) and RF_COCO_LABELS.get(raw_id) != label:
                raise ValueError(f"Unexpected RF-DETR class mapping {raw_id} -> {label}")
            if canon is None:
                continue
            box = canonical_box(values, shape)
            if box:
                result.append(dict(box, score=valid_score(score), raw_id=raw_id, raw_label=str(label), **{"class": canon}))
        return result


def format_tracker_parameters(name, thresholds, fps):
    return dict(track_high_thresh=thresholds["high"], track_low_thresh=thresholds["low"],
                new_track_thresh=thresholds["new"], match_thresh=thresholds["match"],
                track_buffer=thresholds["buffer"], fuse_score=True, with_reid=False,
                gmc_method="sparseOptFlow", proximity_thresh=0.5, appearance_thresh=0.8,
                model="auto", frame_rate=max(1, round(fps)))


class Tracker:
    def __init__(self, name, fps, settings, asset=None, device=None):
        self.name = name
        self.thresholds = settings
        parameters = format_tracker_parameters(name, settings, fps)
        if name == "strongsort":
            import torch
            from boxmot import StrongSort
            from advanced import verify_osnet_backbone

            self.model = StrongSort(reid_weights=Path(asset["path"]), device=torch.device(device), half=False)
            verify_osnet_backbone(self.model.model, Path(asset["path"]), device)
            self.parameters = {"implementation": "BoxMOT StrongSort upstream defaults", "source_commit": BOXMOT_COMMIT}
        else:
            from ultralytics.trackers.bot_sort import BOTSORT
            from ultralytics.trackers.byte_tracker import BYTETracker

            cls = {"bytetrack": BYTETracker, "botsort_cmc": BOTSORT}[name]
            fps = parameters.pop("frame_rate")
            self.model = cls(SimpleNamespace(**parameters), frame_rate=fps)
            self.parameters = {**parameters, "frame_rate": fps, "source_commit": ULTRA_COMMIT}

    def update(self, detections, frame):
        import numpy as np

        people = [d for d in detections if d["class"] in HUMANS]
        xyxy = np.array([[d[key] for key in BOX] for d in people], dtype=np.float32).reshape(-1, 4)
        confidence = np.array([d["score"] for d in people], dtype=np.float32)
        labels = np.zeros(len(people), dtype=np.float32)
        if self.name == "strongsort":
            out = self.model.update(np.column_stack((xyxy, confidence, labels)), frame)
        else:
            from ultralytics.engine.results import Boxes

            boxes = Boxes(np.column_stack((xyxy, confidence, labels)), orig_shape=frame.shape[:2])
            high = confidence >= self.thresholds["high"]
            high_indices = np.flatnonzero(high)
            low_indices = np.flatnonzero((confidence > self.thresholds["low"]) & ~high)
            out = self.model.update(boxes, frame)
        result = []
        for row in out:
            if len(row) != 8 or not np.isfinite(row).all():
                raise ValueError("Tracker must return Nx8 finite rows including the detection index")
            index = int(row[7])
            if self.name == "strongsort":
                ref = index
            else:
                # The pinned port assigns indices separately to its high- and low-score arrays.
                pool = high_indices if row[5] >= np.float32(self.thresholds["high"]) else low_indices
                if not 0 <= index < len(pool):
                    raise ValueError(f"Tracker index {index} is out of range in its score partition")
                ref = int(pool[index])
            if not 0 <= ref < len(people) or not np.isclose(row[5], confidence[ref], atol=1e-7, rtol=0):
                raise ValueError(f"Tracker provenance disagrees with the frozen input detection at index {ref}")
            box = canonical_box(row[:4], (frame.shape[1], frame.shape[0]))
            if box is not None:
                result.append(dict(box, track_id=int(row[4]), score=valid_score(row[5]),
                                   detection_id=people[ref]["detection_id"], **{"class": people[ref]["class"]}))
        if len({r["track_id"] for r in result}) != len(result):
            raise ValueError("Tracker emitted duplicate IDs within one frame")
        return result
