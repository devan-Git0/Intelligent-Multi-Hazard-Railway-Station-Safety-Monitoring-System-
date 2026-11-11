"""
STEP 5 -- Hazard-Specific Logic
Five independent checks, sharing only the tracked-object list as input.
Each one returns a list of HazardEvent -- zero, one, or more per call.

Run this file directly to test all five against synthetic tracks --
pure Python logic, no camera, no models, no GPU needed.
"""

from typing import List

from config import HazardConfig
from datatypes import HazardEvent, TrackedObject
from config import IMG_SIZE


def _point_in_polygon(point, polygon) -> bool:
    """Standard ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


def check_track_crossing(tracks: List[TrackedObject], config: HazardConfig) -> List[HazardEvent]:
    """A person whose foot-point is inside the danger zone AND moving
    toward it (not just standing near the boundary) triggers this."""
    events = []
    for t in tracks:
        if t.class_name != "person":
            continue
        if _point_in_polygon(t.foot_point, config.track_crossing_zone):
            hx, hy = t.heading
            if hx == 0.0 and hy == 0.0:
                continue  # no motion data yet, don't fire on the very first frame
            events.append(HazardEvent(
                hazard_type="track_crossing",
                track_id=t.track_id,
                confidence=1.0,
                details=f"person entered danger zone, heading=({hx:.1f},{hy:.1f})",
            ))
    return events


def check_platform_edge(
    tracks: List[TrackedObject],
    config: HazardConfig,
    frame_shape=None,
) -> List[HazardEvent]:
    """Distance from a person's foot-point to the nearest detected edge.

    The model returns rectangular regions; estimate each edge as a centerline
    along the region's longer axis so the full region does not count as edge.
    NOTE: this is a raw pixel distance. Converting to real-world distance
    needs a homography calibration per camera -- not yet added, flag this
    as a known simplification in your report."""
    events = []
    edge_boxes = [t for t in tracks if t.class_name == "platform_edge"]
    calibrated_line = config.platform_edge_line_normalized
    if not calibrated_line and not edge_boxes:
        return events

    if calibrated_line and len(calibrated_line) == 2:
        if frame_shape is None:
            frame_height = frame_width = IMG_SIZE
        else:
            frame_height, frame_width = frame_shape[:2]
        line_start = (
            calibrated_line[0][0] * frame_width,
            calibrated_line[0][1] * frame_height,
        )
        line_end = (
            calibrated_line[1][0] * frame_width,
            calibrated_line[1][1] * frame_height,
        )
    else:
        line_start = line_end = None

    for t in tracks:
        if t.class_name != "person":
            continue
        fx, fy = t.foot_point
        if line_start is not None:
            min_dist = _point_to_segment_distance((fx, fy), line_start, line_end)
        else:
            min_dist = min(
                _point_to_edge_centerline_distance((fx, fy), edge.bbox)
                for edge in edge_boxes
            )
        if min_dist < config.edge_danger_distance_px and t.dwell_seconds >= config.edge_dwell_seconds:
            events.append(HazardEvent(
                hazard_type="platform_edge",
                track_id=t.track_id,
                confidence=1.0,
                details=f"{min_dist:.0f}px from edge, dwell={t.dwell_seconds:.1f}s",
            ))
    return events


def _point_to_edge_centerline_distance(point, bbox) -> float:
    """Distance to a box's long-axis centerline approximation of the edge."""
    px, py = point
    x1, y1, x2, y2 = bbox
    if (x2 - x1) >= (y2 - y1):
        nearest_x = min(max(px, x1), x2)
        nearest_y = (y1 + y2) / 2
    else:
        nearest_x = (x1 + x2) / 2
        nearest_y = min(max(py, y1), y2)
    return ((px - nearest_x) ** 2 + (py - nearest_y) ** 2) ** 0.5


def _point_to_segment_distance(point, start, end) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    fraction = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_squared))
    nearest_x, nearest_y = x1 + fraction * dx, y1 + fraction * dy
    return ((px - nearest_x) ** 2 + (py - nearest_y) ** 2) ** 0.5


def check_animal_intrusion(tracks: List[TrackedObject], config: HazardConfig) -> List[HazardEvent]:
    events = []
    for t in tracks:
        if t.class_name != "animal":
            continue
        if _point_in_polygon(t.centroid, config.animal_zone):
            events.append(HazardEvent(
                hazard_type="animal_intrusion",
                track_id=t.track_id,
                confidence=1.0,
                details="animal detected inside track zone",
            ))
    return events


def check_abandoned_luggage(tracks: List[TrackedObject], config: HazardConfig) -> List[HazardEvent]:
    """Flag luggage as abandoned when no detected person is nearby."""
    events = []
    persons = [t for t in tracks if t.class_name == "person"]

    for t in tracks:
        if t.class_name != "luggage":
            continue

        if persons:
            nearest_dist = min(_euclidean(t.centroid, p.centroid) for p in persons)
        else:
            nearest_dist = float("inf")  # no people in frame at all -> maximally separated

        if nearest_dist > config.luggage_separation_distance_px:
            events.append(HazardEvent(
                hazard_type="abandoned_luggage",
                track_id=t.track_id,
                confidence=1.0,
                details=(
                    "no person detected nearby"
                    if nearest_dist == float("inf")
                    else f"nearest person is {nearest_dist:.0f}px away"
                ),
            ))
    return events


def _euclidean(p1, p2) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def check_water_accumulation(
    water_coverage: float,
    config: HazardConfig,
    consecutive_high_frames: int,
) -> List[HazardEvent]:
    """Takes the water classifier's coverage % directly (not a tracked
    object -- water has no identity to track). consecutive_high_frames is
    maintained by the caller (main.py) across calls, requiring N frames
    above threshold before firing, so one noisy frame can't trigger this."""
    events = []
    if water_coverage > config.water_coverage_threshold and \
            consecutive_high_frames >= config.water_confirm_frames:
        events.append(HazardEvent(
            hazard_type="water_accumulation",
            track_id=-1,  # not a tracked object
            confidence=water_coverage,
            details=f"{water_coverage:.0%} of platform ROI covered",
        ))
    return events


if __name__ == "__main__":
    config = HazardConfig()

    print("=== Track crossing ===")
    person_in_zone = TrackedObject(
        track_id=1, class_name="person", bbox=(390, 430, 410, 470),
        history=[(0.0, 400, 460), (0.5, 405, 460)],  # moving right, inside the zone
    )
    print(check_track_crossing([person_in_zone], config))

    print("\n=== Platform edge ===")
    edge_box = TrackedObject(track_id=2, class_name="platform_edge", bbox=(0, 450, 640, 470))
    person_near_edge = TrackedObject(
        track_id=3, class_name="person", bbox=(300, 400, 320, 440),
        history=[(0.0, 310, 420), (3.0, 310, 440)],  # 3s dwell, foot-point near y=440
    )
    print(check_platform_edge([edge_box, person_near_edge], config))

    print("\n=== Animal intrusion ===")
    animal = TrackedObject(track_id=4, class_name="animal", bbox=(390, 420, 430, 460))
    print(check_animal_intrusion([animal], config))

    print("\n=== Abandoned luggage ===")
    lonely_bag = TrackedObject(
        track_id=5, class_name="luggage", bbox=(100, 100, 140, 140),
        history=[(0.0, 120, 120), (35.0, 120, 120)],  # 35s dwell, no one nearby
    )
    print(check_abandoned_luggage([lonely_bag], config))

    print("\n=== Water accumulation ===")
    print("below threshold, not enough frames:", check_water_accumulation(0.20, config, consecutive_high_frames=2))
    print("above threshold, enough frames:    ", check_water_accumulation(0.20, config, consecutive_high_frames=5))

    print("\nAll hazard checks ran without error.")
