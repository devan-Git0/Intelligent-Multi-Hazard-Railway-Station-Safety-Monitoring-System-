"""
End-to-end test of the whole pipeline with NO camera, NO weights, NO GPU.
A scripted fake detector replaces RT-DETR/the water model, a synthetic
video replaces the camera, and a fake clock replaces real time, so the
hazard checks that need history (track crossing, platform edge, abandoned
luggage) can be exercised in a second or two.

    python test_pipeline.py
"""

import os
import sys
import tempfile
import types

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(tempfile.mkdtemp())          # keep the test DB/snapshots out of the project

import config  # noqa: E402

config.DB_PATH = os.path.join(os.getcwd(), "test_incidents.db")
config.SNAPSHOT_DIR = os.path.join(os.getcwd(), "snaps")
config.CAMERA_SOURCE = os.path.join(os.getcwd(), "fake.avi")

# ---- fake video: 80 black frames --------------------------------------------
N_FRAMES = 80
vw = cv2.VideoWriter(config.CAMERA_SOURCE, cv2.VideoWriter_fourcc(*"MJPG"), 1, (640, 480))
for _ in range(N_FRAMES):
    vw.write(np.zeros((480, 640, 3), dtype=np.uint8))
vw.release()

# ---- fake detection module (replaces torch / ultralytics) ------------------
from datatypes import Detection  # noqa: E402

frame_counter = {"n": 0}


class FakeEngine:
    def __init__(self, *a, **kw):
        self.water_model = object()  # non-None -> water branch active

    def detect_objects(self, frame, conf=0.4):
        f = frame_counter["n"]
        dets = [
            Detection("platform_edge", 0.9, (0, 450, 640, 470)),
            Detection("luggage", 0.9, (100, 100, 140, 140)),          # never moves, nobody near
            Detection("person", 0.9, (300, 400, 320, 440)),           # stands at the edge
        ]
        if 10 <= f <= 30:                                              # walks into the track zone
            x = 380 + (f - 10) * 2
            dets.append(Detection("person", 0.9, (x, 430, x + 20, 470)))
        if 20 <= f <= 25:
            dets.append(Detection("animal", 0.9, (390, 420, 430, 460)))
        frame_counter["n"] += 1
        return dets

    def classify_water(self, patch):
        return 0.6 if frame_counter["n"] >= 50 else 0.0


fake = types.ModuleType("detection")
fake.DetectionEngine = FakeEngine
sys.modules["detection"] = fake

# ---- fake clock: each frame = 1 simulated second ----------------------------
import time as _time  # noqa: E402

_clock = {"t": 1_700_000_000.0}
_time.time = lambda: _clock["t"]
_time.sleep = lambda s: None

import main  # noqa: E402

_orig = main.RailwaySafetySystem._process_one_frame


def _tick(self):
    _clock["t"] += 1.0
    return _orig(self)


main.RailwaySafetySystem._process_one_frame = _tick

system = main.RailwaySafetySystem()
system.run_forever(max_frames=N_FRAMES)

# ---- check what landed in the database --------------------------------------
from database import IncidentDatabase  # noqa: E402

db = IncidentDatabase(config.DB_PATH)
counts = db.count_by_type()
rows = db.recent_incidents()
db.close()

print("\nIncidents by type:", counts)
for expected in ("track_crossing", "platform_edge", "animal_intrusion",
                 "abandoned_luggage", "water_accumulation"):
    assert counts.get(expected, 0) >= 1, f"{expected} never fired"
assert all(os.path.exists(r["snapshot_path"]) for r in rows), "missing snapshot"
print("Snapshots saved:", len(os.listdir(config.SNAPSHOT_DIR)))
print("\nEnd-to-end pipeline OK: all five hazards fired, scored, alerted and logged.")
