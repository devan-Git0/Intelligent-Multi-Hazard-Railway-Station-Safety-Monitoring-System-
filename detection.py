"""
STEP 3 -- Detection
  3.1 Run RT-DETR on the preprocessed frame -> person / animal / luggage / platform_edge
  3.2 Run the water classifier on the platform-surface ROI -> water coverage %

Two separate models, two separate outputs -- water never goes through
RT-DETR's detection head, since standing water is a region property,
not a discrete object (this was a deliberate architecture decision, not
an oversight -- see the earlier "Segmentation branch" discussion).

Run this file directly to test both models on a synthetic frame.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import cv2
from ultralytics import RTDETR, YOLO

from config import (
    CLASS_NAMES,
    COCO_CLASS_MAP,
    PERSON_DETECTOR_WEIGHTS,
    RTDETR_WEIGHTS,
    WATER_CLASSIFIER_WEIGHTS,
)
from datatypes import Detection
from models import load_water_classifier


class DetectionEngine:
    def __init__(
        self,
        rtdetr_weights: str = RTDETR_WEIGHTS,
        water_weights: str = WATER_CLASSIFIER_WEIGHTS,
        device: str = "cpu",
        load_water_model: bool = True,
        coco_class_map: Optional[Dict[int, str]] = None,
        person_weights: Optional[str] = None,
    ):
        # 1.1 -- Load RT-DETR. Ultralytics auto-detects a real fine-tuned
        # .pt file vs. a COCO-pretrained checkpoint name (e.g. "rtdetr-l.pt")
        # -- use the latter here if you haven't fine-tuned yet, just to get
        # the pipeline running end-to-end before your own weights exist.
        self.detector = RTDETR(rtdetr_weights)
        self.device = device

        # Pretrained COCO checkpoints have 80 classes with "person" at 0;
        # a model fine-tuned on this project has the CLASS_NAMES instead.
        # In COCO mode only COCO_CLASS_MAP's classes are kept and renamed.
        names = getattr(self.detector, "names", None) or {}
        self.is_coco = len(names) == 80 and names.get(0) == "person"
        self.coco_class_map = coco_class_map or COCO_CLASS_MAP
        self.person_detector = None
        person_weights = person_weights or PERSON_DETECTOR_WEIGHTS
        if not self.is_coco and os.path.isfile(person_weights):
            # The small general-purpose detector supplies person boxes when
            # a fine-tuned railway model is weak on people in CCTV footage.
            self.person_detector = YOLO(person_weights)
            print(f"  Person detection fallback: {person_weights}")
        mode = (f"pretrained COCO -> keeping {sorted(set(self.coco_class_map.values()))}"
                if self.is_coco else f"fine-tuned -> {CLASS_NAMES}")
        print(f"  RT-DETR mode: {mode}")

        # 1.2 -- Load the water classifier. Optional flag so tracking.py's
        # and hazards.py's standalone tests don't need real water weights
        # just to exercise their own logic.
        self.water_model = None
        if load_water_model:
            self.water_model = load_water_classifier(water_weights, device=device)

    def detect_objects(self, frame_resized: np.ndarray, conf: float = 0.15) -> List[Detection]:
        """3.1 -- RT-DETR detections. With the pretrained COCO model these are
        person / animal (per COCO_CLASS_MAP); with a fine-tuned model they are
        person / animal / luggage / platform_edge."""
        kwargs = {"classes": list(self.coco_class_map)} if self.is_coco else {}
        results = self.detector.predict(frame_resized, conf=conf, verbose=False, **kwargs)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            if self.is_coco:
                class_name = self.coco_class_map.get(cls_id)
                if class_name is None:
                    continue
            else:
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if self.person_detector is not None and class_name == "person":
                continue
            detections.append(Detection(
                class_name=class_name,
                confidence=float(box.conf.item()),
                bbox=(x1, y1, x2, y2),
            ))

        if self.person_detector is not None:
            person_results = self.person_detector.predict(
                frame_resized,
                classes=[0],
                conf=0.12,
                iou=0.4,
                max_det=100,
                verbose=False,
            )[0]
            for box in person_results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    class_name="person",
                    confidence=float(box.conf.item()),
                    bbox=(x1, y1, x2, y2),
                ))
        person_detections = [d for d in detections if d.class_name == "person"]
        other_detections = [d for d in detections if d.class_name != "person"]
        if len(person_detections) > 1:
            boxes = []
            scores = []
            for detection in person_detections:
                x1, y1, x2, y2 = detection.bbox
                boxes.append([
                    int(x1), int(y1), max(1, int(x2 - x1)),
                    max(1, int(y2 - y1)),
                ])
                scores.append(float(detection.confidence))
            kept = cv2.dnn.NMSBoxes(
                boxes, scores, score_threshold=0.12, nms_threshold=0.3
            )
            indices = np.asarray(kept).reshape(-1).tolist() if len(kept) else []
            person_detections = [person_detections[i] for i in indices]
        return other_detections + person_detections

    def classify_water(self, roi_patch: np.ndarray) -> float:
        """3.2 -- returns a water probability in [0, 1] for one cropped
        patch of the platform-surface ROI. Coverage % across the whole ROI
        is computed by hazards.py by tiling this over several patches."""
        if self.water_model is None:
            raise RuntimeError("Water classifier not loaded -- init with load_water_model=True")

        # MobileNetV2 expects 224x224, ImageNet-normalized -- independent of
        # the 640x640 detection input size
        patch = np.transpose(roi_patch, (2, 0, 1))  # HWC -> CHW
        tensor = torch.from_numpy(patch).unsqueeze(0).float()
        with torch.no_grad():
            logits = self.water_model(tensor)
            probs = torch.softmax(logits, dim=1)
        return float(probs[0, 1])  # index 1 = "water" class


if __name__ == "__main__":
    import cv2

    # Default weights are the COCO-pretrained RT-DETR-L (auto-downloaded the
    # first time). Only person/animal classes are kept -- see COCO_CLASS_MAP.
    print("Loading the pretrained COCO RT-DETR...")
    engine = DetectionEngine(load_water_model=False)

    fake_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    detections = engine.detect_objects(fake_frame, conf=0.1)
    print(f"Ran inference on a synthetic frame -- {len(detections)} raw detections "
          f"(expected: near-zero on random noise, this just confirms the model runs).")

    print("\nWaterClassifier forward pass (architecture only, untrained weights):")
    from models import WaterClassifier
    water_model = WaterClassifier()
    water_model.eval()
    fake_patch = np.random.rand(224, 224, 3).astype(np.float32)
    tensor = torch.from_numpy(np.transpose(fake_patch, (2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(water_model(tensor), dim=1)
    print("  water probability:", float(probs[0, 1]))
    print("\nDetection pipeline structurally OK.")
