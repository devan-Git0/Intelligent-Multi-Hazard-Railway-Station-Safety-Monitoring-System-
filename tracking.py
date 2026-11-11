"""
STEP 4 -- Multi-Object Tracking (ByteTrack)
Turns per-frame RT-DETR detections into TrackedObjects with persistent IDs
and a position history, which is what makes heading (track crossing) and
dwell time (platform edge, abandoned luggage) computable in hazards.py.

platform_edge detections are static scenery, not moving objects, so they
bypass ByteTrack and are passed straight through with a negative ID --
hazards.py only needs their boxes.

Run this file directly to test continuous IDs, dwell and heading on a
synthetic moving box -- needs only numpy and supervision.
"""

from typing import Dict, List, Optional

import numpy as np
import supervision as sv

from config import CLASS_NAMES, TrackerConfig
from datatypes import Detection, TrackedObject

STATIC_CLASSES = {"platform_edge"}


class ObjectTracker:
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        c = self.config
        self.byte_track = sv.ByteTrack(
            track_activation_threshold=c.track_activation_threshold,
            lost_track_buffer=c.lost_track_buffer,
            minimum_matching_threshold=c.minimum_matching_threshold,
            frame_rate=c.frame_rate,
            minimum_consecutive_frames=c.minimum_consecutive_frames,
        )
        self._tracks: Dict[int, TrackedObject] = {}
        self._last_seen: Dict[int, float] = {}
        self._raw_to_stable: Dict[int, int] = {}
        self._next_stable_id = 1

    def update(self, detections: List[Detection], timestamp: float) -> List[TrackedObject]:
        """Feed one frame's detections; returns every currently visible
        TrackedObject (moving objects with history + static edge boxes)."""
        moving = [d for d in detections if d.class_name not in STATIC_CLASSES
                  and d.class_name in CLASS_NAMES]
        static = [d for d in detections if d.class_name in STATIC_CLASSES]

        if moving:
            sv_dets = sv.Detections(
                xyxy=np.array([d.bbox for d in moving], dtype=np.float32),
                confidence=np.array([d.confidence for d in moving], dtype=np.float32),
                class_id=np.array([CLASS_NAMES.index(d.class_name) for d in moving], dtype=int),
            )
        else:
            sv_dets = sv.Detections.empty()

        tracked = self.byte_track.update_with_detections(sv_dets)

        result: List[TrackedObject] = []
        claimed_stable_ids = set()
        for i in range(len(tracked)):
            raw_id = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            cls_name = CLASS_NAMES[int(tracked.class_id[i])]
            bbox = (x1, y1, x2, y2)
            tid = self._raw_to_stable.get(raw_id)
            if tid in claimed_stable_ids or tid not in self._tracks:
                tid = self._recover_identity(
                    cls_name, bbox, timestamp, claimed_stable_ids
                )
                if tid is None:
                    tid = self._next_stable_id
                    self._next_stable_id += 1
                self._raw_to_stable[raw_id] = tid

            obj = self._tracks.get(tid)
            if obj is None:
                obj = TrackedObject(track_id=tid, class_name=cls_name,
                                    bbox=bbox, history=[])
                self._tracks[tid] = obj
            obj.bbox = bbox
            obj.history.append((timestamp, (x1 + x2) / 2, (y1 + y2) / 2))
            # Cap memory but ALWAYS keep the first point -- dwell_seconds
            # is measured from history[0].
            if len(obj.history) > self.config.max_history_points:
                del obj.history[1]
            self._last_seen[tid] = timestamp
            claimed_stable_ids.add(tid)
            result.append(obj)

        for k, d in enumerate(static):
            result.append(TrackedObject(track_id=-(k + 2), class_name=d.class_name,
                                        bbox=d.bbox, history=[]))

        self._drop_stale(timestamp)
        return result

    def _recover_identity(self, class_name, bbox, timestamp, claimed_ids):
        """Reuse a recent public ID when ByteTrack starts a replacement track."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        new_height = max(1.0, y2 - y1)
        best_id, best_distance = None, float("inf")

        for tid, obj in self._tracks.items():
            if tid in claimed_ids or obj.class_name != class_name:
                continue
            elapsed = timestamp - self._last_seen.get(tid, timestamp)
            if elapsed < 0 or elapsed > self.config.track_recovery_seconds:
                continue

            old_x, old_y = obj.centroid
            if len(obj.history) >= 2:
                (t0, x0, y0), (t1, x1_old, y1_old) = obj.history[-2:]
                history_dt = t1 - t0
                if history_dt > 0:
                    predict_dt = min(elapsed, 0.75)
                    old_x += ((x1_old - x0) / history_dt) * predict_dt
                    old_y += ((y1_old - y0) / history_dt) * predict_dt

            distance = ((cx - old_x) ** 2 + (cy - old_y) ** 2) ** 0.5
            old_height = max(1.0, obj.bbox[3] - obj.bbox[1])
            max_distance = min(
                160.0,
                max(45.0, 0.7 * max(old_height, new_height) + 50.0 * elapsed),
            )
            if distance <= max_distance and distance < best_distance:
                best_id, best_distance = tid, distance

        return best_id

    def _drop_stale(self, now: float) -> None:
        cutoff = now - self.config.stale_track_seconds
        stale_ids = [t for t, seen in self._last_seen.items() if seen < cutoff]
        for tid in stale_ids:
            self._tracks.pop(tid, None)
            self._last_seen.pop(tid, None)
        if stale_ids:
            stale_ids = set(stale_ids)
            self._raw_to_stable = {
                raw: stable for raw, stable in self._raw_to_stable.items()
                if stable not in stale_ids
            }


if __name__ == "__main__":
    tracker = ObjectTracker(TrackerConfig(minimum_consecutive_frames=1))
    ids = set()
    obj = None
    for f in range(40):
        t = f * 0.1
        x = 100 + f * 5  # person walking right at 50 px/s
        dets = [Detection("person", 0.9, (x, 200, x + 40, 300)),
                Detection("platform_edge", 0.8, (0, 450, 640, 470))]
        out = tracker.update(dets, timestamp=t)
        person = [o for o in out if o.class_name == "person"]
        if person:
            ids.add(person[0].track_id)
            obj = person[0]
    print("distinct person IDs over 40 frames:", ids)
    print("dwell_seconds:", round(obj.dwell_seconds, 2), "| heading:", obj.heading)
    print("edge passthrough id:", [o.track_id for o in out if o.class_name == "platform_edge"])
    assert len(ids) == 1, "ID switched -- tracker not continuous"
    assert obj.dwell_seconds > 3.5 and obj.heading[0] > 0
    print("Tracking OK.")
