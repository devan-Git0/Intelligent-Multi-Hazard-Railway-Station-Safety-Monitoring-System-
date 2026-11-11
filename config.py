"""
STEP 1.4 -- Hazard parameters, plus every path the detection/hazard demo
needs. Change values here, not inside the logic files.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import os

# ---------------------------------------------------------------------------
# Model + camera paths
# ---------------------------------------------------------------------------
# Pretrained COCO RT-DETR-L -- Ultralytics downloads it automatically on first
# run. Swap in "weights/rtdetr_finetuned.pt" once you have fine-tuned on your
# own dataset (detection.py auto-detects which kind of model it was given).
RTDETR_WEIGHTS = ("weights/rtdetr_finetuned.pt" if os.path.exists("weights/rtdetr_finetuned.pt") else "rtdetr-l.pt")
PERSON_DETECTOR_WEIGHTS = "weights/yolo26n.pt"
WATER_CLASSIFIER_WEIGHTS = "weights/water_classifier.pt"  # produced by fine-tuning
CAMERA_SOURCE = 0                                        # 0 = default webcam; or an RTSP URL string
IMG_SIZE = 640

# Normalized camera calibration for supplied still-image examples. These are
# intentionally source-specific; never apply one camera's edge to another view.
IMAGE_CAMERA_PROFILES = {
    "station.jpg.png": {
        # Track bed lies below the far platform and above the near platform edge.
        "track_crossing_zone": [(0.0, 0.50), (1.0, 0.50), (1.0, 0.94), (0.0, 0.70)],
        "platform_warning_line": [(0.0, 0.77), (0.80, 0.995)],
        "platform_edge_line": [(0.0, 0.74), (0.77, 0.995)],
    },
}

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DB_PATH = "incidents.db"          # SQLite file shared by main.py and dashboard.py
SNAPSHOT_DIR = "snapshots"        # annotated frame saved for every logged incident

# RT-DETR class names, in the order your training config uses.
# NOTE: "platform_edge" is included here even though it wasn't in the
# original step-1 pseudocode ("detects people, animals, luggage") --
# the real Roboflow dataset we processed labels the edge boundary itself
# as a detectable object, which is more robust than a fixed calibrated
# line, so it's added as a 4th class. If you'd rather keep the original
# 3-class scope, delete "platform_edge" here and switch the platform-edge
# hazard check in hazards.py back to a fixed-line distance calculation.
CLASS_NAMES = ["person", "animal", "luggage", "platform_edge"]

# When the pretrained COCO model is used, only these COCO class IDs are kept,
# and each is renamed to one of the project classes above. Everything else
# COCO can detect (cars, chairs, ...) is ignored.
#   0 person | 14 bird 15 cat 16 dog 17 horse 18 sheep 19 cow
#   20 elephant 21 bear 22 zebra 23 giraffe
# COCO luggage-like classes: backpack (24), handbag (26), suitcase (28).
COCO_CLASS_MAP = {
    0: "person",
    14: "animal", 15: "animal", 16: "animal", 17: "animal", 18: "animal",
    19: "animal", 20: "animal", 21: "animal", 22: "animal", 23: "animal",
    24: "luggage", 26: "luggage", 28: "luggage",
}


@dataclass
class HazardConfig:
    # Crossing zone starts at the yellow safety stripe and covers the edge
    # strip and tracks. Coordinates are in the 640x640 resized frame.
    track_crossing_zone: List[Tuple[float, float]] = field(default_factory=lambda: [
        (208, 0), (640, 0), (640, 640), (447, 640)
    ])

    # Animal intrusion -- separate polygon in case animals should trigger
    # from a wider area than people do
    animal_zone: List[Tuple[float, float]] = field(default_factory=lambda: [
        (150, 380), (650, 380), (650, 480), (150, 480)
    ])

    # Platform edge proximity -- pixel distance from a person's foot-point
    # to the calibrated platform boundary. Coordinates are normalized (0..1)
    # against the resized camera frame. Adjust these two points for each
    # fixed camera; set to None to fall back to detector-box estimates.
    platform_edge_line_normalized: List[Tuple[float, float]] = field(
        # Calibrated to railway_safety_system/vid.mp4. Recalibrate if the
        # camera view changes. This is the outer platform/track border.
        default_factory=lambda: [(0.51, 0.27), (0.84, 0.995)]
    )
    platform_warning_line_normalized: List[Tuple[float, float]] = field(
        # Inner edge of the yellow safety stripe; danger zone lies to its right.
        default_factory=lambda: [(0.43, 0.28), (0.70, 0.995)]
    )
    # This is a raw pixel threshold, not a real-world distance, until
    # homography calibration is added.
    edge_danger_distance_px: float = 60.0
    edge_dwell_seconds: float = 2.0

    # Abandoned luggage
    luggage_separation_distance_px: float = 150.0

    # Water accumulation -- fraction of the platform ROI classified as water
    water_coverage_threshold: float = 0.15
    water_confirm_frames: int = 5   # require N consecutive frames above
                                     # threshold before triggering, so one
                                     # noisy frame doesn't fire an alert


# ---------------------------------------------------------------------------
# STEP 4 -- ByteTrack settings (tracking.py)
# ---------------------------------------------------------------------------
@dataclass
class TrackerConfig:
    track_activation_threshold: float = 0.1   # min confidence to start a track
    lost_track_buffer: int = 75               # keep tracks through short detector gaps
    minimum_matching_threshold: float = 0.45  # IoU matching threshold
    frame_rate: int = 30                      # camera FPS (affects the buffer)
    minimum_consecutive_frames: int = 1       # frames before a track gets an ID
    max_history_points: int = 600             # cap per-track history (first point is always kept)
    stale_track_seconds: float = 10.0         # drop a track's history this long after last seen
    track_recovery_seconds: float = 5.0       # retain identity across short detector gaps


# ---------------------------------------------------------------------------
# STEP 6 -- Risk assessment & severity scoring (risk_fusion.py)
# ---------------------------------------------------------------------------
@dataclass
class RiskConfig:
    # Base severity (0-100) per hazard type, before modifiers.
    base_severity: dict = field(default_factory=lambda: {
        "track_crossing": 65.0,
        "edge_line_crossing": 65.0,
        "abandoned_luggage": 70.0,
        "platform_edge": 60.0,
        "animal_intrusion": 55.0,
        "water_accumulation": 40.0,
    })
    # Added once per OTHER hazard active in the same frame (compounding risk).
    concurrent_hazard_bonus: float = 5.0
    max_concurrent_bonus: float = 15.0
    # Score thresholds -> severity level.
    critical_threshold: float = 80.0
    high_threshold: float = 60.0
    medium_threshold: float = 40.0
    # The same (hazard, track) will not be re-reported for this many seconds.
    cooldown_seconds: float = 30.0


# ---------------------------------------------------------------------------
# STEP 7 -- Alert dispatcher (alerts.py)
# ---------------------------------------------------------------------------
MIN_ALERT_LEVEL = "MEDIUM"        # LOW incidents are logged but not pushed anywhere
ENABLE_CONSOLE_ALERTS = True
ENABLE_VOICE_ALERTS = False       # pyttsx3 spoken announcement (HIGH and CRITICAL only)

SLACK_WEBHOOK_URL = ""            # leave empty to disable Slack
EMAIL_ENABLED = False
EMAIL_SMTP_HOST = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587
EMAIL_USERNAME = ""
EMAIL_PASSWORD = ""               # use an app password, never your real one
EMAIL_FROM = ""
EMAIL_TO: List[str] = []
EMAIL_MIN_LEVEL = "HIGH"          # email only for HIGH and CRITICAL
