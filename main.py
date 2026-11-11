
import argparse
import os
import time

import cv2
import numpy as np

import config
from alerts import AlertDispatcher
from database import IncidentDatabase
from detection import DetectionEngine
from datatypes import HazardEvent, Incident, TrackedObject
from hazards import (
    check_abandoned_luggage,
    check_animal_intrusion,
    check_platform_edge,
    check_water_accumulation,
)
from preprocessing import capture_and_preprocess, open_camera
from risk_fusion import RiskFusion
from tracking import ObjectTracker


RED = (0, 0, 255)
DARK_RED = (0, 0, 150)
ORANGE = (0, 140, 255)
GREEN = (0, 200, 0)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)


class RailwaySafetySystem:
    def __init__(
        self,
        source=None,
        rtdetr_weights=None,
        water_weights=None,
        show=False,
    ):
        self.show = show
        self._window_name = "Railway Safety - RT-DETR"
        self._window_created = False
        self.hazard_config = config.HazardConfig()

        print("[1.1] Loading RT-DETR and optional water model...")

        water_weights = (
            water_weights or config.WATER_CLASSIFIER_WEIGHTS
        )
        have_water = os.path.exists(water_weights)

        if not have_water:
            print("Water classifier unavailable; water detection disabled.")

        self.detection = DetectionEngine(
            rtdetr_weights=(
                rtdetr_weights or config.RTDETR_WEIGHTS
            ),
            water_weights=water_weights,
            load_water_model=have_water,
        )

        self.tracker = ObjectTracker()
        self.risk = RiskFusion()
        self.alerts = AlertDispatcher()
        self.db = IncidentDatabase(config.DB_PATH)

        os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)

        self._water_high_streak = 0
        self.cap = None

        # A source image must NOT be opened with VideoCapture.
        if source is not None and isinstance(source, str):
            if os.path.isfile(source) and self._is_image_path(source):
                self.image_mode = True
                self.image_path = source
                print(f"[1.2] Single-image input: {source}")
                return

        self.image_mode = False
        self.image_path = None

        camera_source = (
            config.CAMERA_SOURCE if source is None else source
        )
        print("[1.2] Opening video/camera source...")
        self.cap = open_camera(camera_source)

        print("Initialization complete.")

    @staticmethod
    def _is_image_path(path):
        return os.path.splitext(path)[1].lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
        }

    def run_forever(self, max_frames=None):
        try:
            if self.image_mode:
                self.process_single_image(self.image_path)
                return

            frame_count = 0
            while max_frames is None or frame_count < max_frames:
                success = self._process_one_frame()

                if not success:
                    if self.cap is not None and (
                        self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
                    ):
                        break
                    time.sleep(0.1)
                    continue

                frame_count += 1

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            self.release()

    @staticmethod
    def _filter_duplicate_edge_detections(detections, iou_threshold=0.45):
        """Suppress duplicate platform_edge boxes while preserving other classes."""
        is_edge = lambda d: str(d.class_name).strip().lower().replace(" ", "_") == "platform_edge"
        edges = [d for d in detections if is_edge(d)]
        others = [d for d in detections if not is_edge(d)]
        if len(edges) <= 1:
            return others + edges

        boxes, scores = [], []
        for d in edges:
            x1, y1, x2, y2 = d.bbox
            boxes.append([int(x1), int(y1), max(1, int(x2-x1)), max(1, int(y2-y1))])
            scores.append(float(d.confidence))
        kept = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.01,
                                nms_threshold=iou_threshold)
        if len(kept) == 0:
            return others
        indices = np.asarray(kept).reshape(-1).tolist()
        return others + [edges[i] for i in indices]

    def _person_inside_track_zone(self, obj):
        """Test the person's bottom-centre against the configured track polygon."""
        if self._normalise_class(getattr(obj, "class_name", "")) not in {"person", "people", "human"}:
            return False
        zone = getattr(self.hazard_config, "track_crossing_zone", None)
        if zone is None or len(zone) < 3:
            return False
        x1, y1, x2, y2 = obj.bbox
        foot = ((float(x1) + float(x2)) / 2.0, float(y2))
        return cv2.pointPolygonTest(np.asarray(zone, dtype=np.float32), foot, False) >= 0

    def _has_person_on_track(self, objects):
        return any(self._person_inside_track_zone(obj) for obj in objects)

    def _process_one_frame(self):
        frame_norm, frame_display = capture_and_preprocess(
            self.cap,
            img_size=config.IMG_SIZE,
        )

        if frame_display is None:
            return False

        now = time.time()
        detections = self.detection.detect_objects(frame_display)
        detections = self._filter_duplicate_edge_detections(detections)

        tracked_objects = self.tracker.update(
            detections,
            timestamp=now,
        )

        water_coverage = self._estimate_water_coverage(frame_display)

        if (
            water_coverage
            > self.hazard_config.water_coverage_threshold
        ):
            self._water_high_streak += 1
        else:
            self._water_high_streak = 0

        hazard_events = self._check_hazards(
            tracked_objects,
            water_coverage,
            frame_shape=frame_display.shape[:2],
        )

        self._debug_objects(tracked_objects)
        self._debug_hazards(hazard_events)

        incidents = self.risk.assess(
            hazard_events,
            timestamp=now,
        )

        for incident in incidents:
            incident.snapshot_path = self._save_snapshot(
                frame_display,
                tracked_objects,
                incident,
            )
            self.db.insert_incident(incident)
            self.alerts.dispatch(incident)

        if self.show:
            self._show(frame_display, tracked_objects, hazard_events, detections)

        return True

    def process_single_image(self, image_path):
        """Run detection and display results for a still image."""
        frame = cv2.imread(image_path)

        if frame is None:
            raise FileNotFoundError(
                f"Cannot read image: {image_path}"
            )

        self._apply_image_camera_profile(image_path, frame.shape[:2])

        print(f"\nProcessing image: {image_path}")

        # Detect directly on the image; no VideoCapture or tracking
        # is needed for a single still image.
        detections = self.detection.detect_objects(frame)
        detections = self._filter_duplicate_edge_detections(detections)

        print("\n--- FILTERED IMAGE DETECTIONS ---")
        for detection in detections:
            print(detection)

        # Some hazard functions expect tracked objects with track IDs.
        # Create a one-frame tracker update for compatibility.
        tracked_objects = self.tracker.update(
            detections,
            timestamp=time.time(),
        )

        water_coverage = self._estimate_water_coverage(frame)

        hazard_events = self._check_hazards(
            tracked_objects,
            water_coverage,
            frame_shape=frame.shape[:2],
        )

        self._debug_objects(tracked_objects)
        self._debug_hazards(hazard_events)

        output = self._draw(frame, tracked_objects, detections, hazard_events)

        if self._has_person_on_track(tracked_objects):
            self._draw_banner(output, "HIGH RISK DETECTED")
        elif hazard_events:
            names = sorted({
                str(getattr(event, "hazard_type", "HAZARD")).upper()
                for event in hazard_events
            })
            self._draw_banner(output, "DANGER: " + ", ".join(names))
        else:
            cv2.rectangle(output, (0, 0), (output.shape[1], 42), (50, 50, 50), -1)
            cv2.putText(
                output, "NO HAZARD EVENT TRIGGERED", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA,
            )

        root, _ = os.path.splitext(image_path)
        output_path = root + "_annotated.jpg"

        if not cv2.imwrite(output_path, output):
            raise RuntimeError(
                f"Could not save output to {output_path}"
            )

        print(f"\nAnnotated image saved to: {output_path}")

        if self.show:
            self._display_fullscreen(output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _apply_image_camera_profile(self, image_path, frame_shape):
        basename = os.path.basename(image_path).lower()
        profile = config.IMAGE_CAMERA_PROFILES.get(basename)
        if profile is None and basename.startswith("station."):
            profile = config.IMAGE_CAMERA_PROFILES.get("station.jpg.png")
        if profile is None:
            return

        height, width = frame_shape[:2]
        self.hazard_config.track_crossing_zone = [
            (x * width, y * height)
            for x, y in profile["track_crossing_zone"]
        ]
        self.hazard_config.platform_warning_line_normalized = profile[
            "platform_warning_line"
        ]
        self.hazard_config.platform_edge_line_normalized = profile[
            "platform_edge_line"
        ]
        print(f"Using calibrated camera profile: {basename}")

    def _check_hazards(self, tracked_objects, water_coverage, frame_shape=None):
        person_events = [
            HazardEvent(
                hazard_type="edge_line_crossing",
                track_id=obj.track_id,
                confidence=1.0,
                details="person crossed the platform yellow safety line",
            )
            for obj in tracked_objects
            if self._normalise_class(getattr(obj, "class_name", ""))
            in {"person", "people", "human"}
            and self._person_inside_track_zone(obj)
        ]
        events = person_events + (
            check_platform_edge(
                tracked_objects, self.hazard_config, frame_shape=frame_shape
            )
            + check_animal_intrusion(
                tracked_objects, self.hazard_config
            )
            + check_abandoned_luggage(
                tracked_objects, self.hazard_config
            )
            + check_water_accumulation(
                water_coverage,
                self.hazard_config,
                self._water_high_streak,
            )
        )
        return events

    @staticmethod
    def _debug_objects(objects):
        print("\n--- DETECTIONS / TRACKED OBJECTS ---")

        if not objects:
            print("No objects detected.")

        for obj in objects:
            print(
                "Class:", getattr(obj, "class_name", "unknown"),
                "| ID:", getattr(obj, "track_id", -1),
                "| Confidence:", getattr(
                    obj, "confidence",
                    getattr(obj, "conf", "not available")
                ),
                "| Box:", getattr(obj, "bbox", None),
            )

    @staticmethod
    def _debug_hazards(events):
        print("--- HAZARD EVENTS ---")

        if not events:
            print("No hazard event triggered.")
            return

        for event in events:
            print(
                "ALERT:",
                getattr(event, "hazard_type", str(event)),
            )

    def _estimate_water_coverage(self, frame):
        if self.detection.water_model is None:
            return 0.0

        height, width = frame.shape[:2]
        roi = frame[int(0.6 * height):, :]
        patch_size = 224
        scores = []

        for y in range(0, roi.shape[0] - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                patch = roi[y:y + patch_size, x:x + patch_size]
                patch_rgb = cv2.cvtColor(
                    patch, cv2.COLOR_BGR2RGB
                ).astype("float32") / 255.0
                scores.append(
                    self.detection.classify_water(patch_rgb)
                )

        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _normalise_class(name):
        return (
            str(name).strip().lower()
            .replace(" ", "_").replace("-", "_")
        )

    @staticmethod
    def _box_iou(first, second):
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = iw * ih
        union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(
            0.0, (bx2 - bx1) * (by2 - by1)
        ) - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _box_overlap_ratio(first, second):
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )
        smaller_area = min(
            max(0.0, (ax2 - ax1) * (ay2 - ay1)),
            max(0.0, (bx2 - bx1) * (by2 - by1)),
        )
        return intersection / smaller_area if smaller_area > 0 else 0.0

    def _draw(self, frame, tracked_objects, detections=(), hazard_events=()):
        output = frame.copy()
        height, width = output.shape[:2]
        abandoned_luggage_ids = {
            getattr(event, "track_id", None)
            for event in hazard_events
            if getattr(event, "hazard_type", "") == "abandoned_luggage"
        }
        draw_objects = list(tracked_objects)
        tracked_persons = [
            obj for obj in tracked_objects
            if self._normalise_class(getattr(obj, "class_name", ""))
            in {"person", "people", "human"}
        ]
        # Keep the person box visible on frames where the tracker rejects a
        # detector result. It is visual-only and cannot trigger hazards.
        for detection in detections:
            if self._normalise_class(getattr(detection, "class_name", "")) not in {
                "person", "people", "human"
            }:
                continue
            if any(
                self._box_iou(detection.bbox, obj.bbox) >= 0.15
                or self._box_overlap_ratio(detection.bbox, obj.bbox) >= 0.5
                   for obj in tracked_persons):
                continue
            draw_objects.append(TrackedObject(
                track_id=-1,
                class_name=detection.class_name,
                bbox=detection.bbox,
                history=[],
            ))
        calibrated_edge = self.hazard_config.platform_edge_line_normalized
        warning_line = self.hazard_config.platform_warning_line_normalized
        if calibrated_edge and len(calibrated_edge) == 2:
            edge_start = (
                max(0, min(round(calibrated_edge[0][0] * (width - 1)), width - 1)),
                max(0, min(round(calibrated_edge[0][1] * (height - 1)), height - 1)),
            )
            edge_end = (
                max(0, min(round(calibrated_edge[1][0] * (width - 1)), width - 1)),
                max(0, min(round(calibrated_edge[1][1] * (height - 1)), height - 1)),
            )
            if warning_line and len(warning_line) == 2:
                warning_start = (
                    round(warning_line[0][0] * (width - 1)),
                    round(warning_line[0][1] * (height - 1)),
                )
                warning_end = (
                    round(warning_line[1][0] * (width - 1)),
                    round(warning_line[1][1] * (height - 1)),
                )
                danger_band = np.asarray(
                    [warning_start, edge_start, edge_end, warning_end],
                    dtype=np.int32,
                )
                overlay = output.copy()
                cv2.fillPoly(overlay, [danger_band], ORANGE)
                cv2.addWeighted(overlay, 0.16, output, 0.84, 0, output)
                dark_yellow = (0, 170, 205)
                cv2.line(output, warning_start, warning_end, dark_yellow, 1, cv2.LINE_AA)
                cv2.circle(output, warning_start, 2, dark_yellow, -1, cv2.LINE_AA)
                cv2.circle(output, warning_end, 2, dark_yellow, -1, cv2.LINE_AA)
            cv2.line(output, edge_start, edge_end, ORANGE, 1, cv2.LINE_AA)
            cv2.circle(output, edge_start, 2, ORANGE, -1, cv2.LINE_AA)
            cv2.circle(output, edge_end, 2, ORANGE, -1, cv2.LINE_AA)

        for obj in draw_objects:
            try:
                x1, y1, x2, y2 = (
                    int(v) for v in obj.bbox
                )
            except (TypeError, ValueError, AttributeError):
                continue

            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            class_name = str(
                getattr(obj, "class_name", "unknown")
            )
            cls = self._normalise_class(class_name)

            if calibrated_edge and len(calibrated_edge) == 2 and cls in {
                "platform_edge", "platformedge", "edge"
            }:
                # The fixed camera calibration is more precise than the
                # detector's broad box for this static boundary.
                continue

            track_id = getattr(obj, "track_id", -1)
            if cls in {"person", "people", "human"}:
                on_track = self._person_inside_track_zone(obj)
                color, thickness = (RED, 1) if on_track else (GREEN, 1)
                label = "HIGH RISK" if on_track else "PERSON"
                
            elif cls in {"luggage", "suitcase", "bag", "baggage"}:
                abandoned = track_id in abandoned_luggage_ids
                color, thickness = (RED, 1) if abandoned else (GREEN, 1)
                label = "ABANDONED" if abandoned else "LUGGAGE"
            elif cls in {"platform_edge", "platformedge", "edge"}:
                color, thickness = ORANGE, 3
                label = "PLATFORM EDGE"
            elif cls in {"animal", "dog", "cow"}:
                color, thickness = MAGENTA, 2
                label = class_name.upper()
            else:
                color, thickness = GREEN, 2
                label = class_name.upper()

            if track_id is not None and track_id >= 0:
                label += f" #{track_id}"

            if cls in {"platform_edge", "platformedge", "edge"}:
                # The model predicts a broad region around the platform edge.
                # Show a thin centerline along that region's long axis instead
                # of outlining the entire (often oversized) detection box.
                if (x2 - x1) >= (y2 - y1):
                    edge_start = (x1, (y1 + y2) // 2)
                    edge_end = (x2, (y1 + y2) // 2)
                else:
                    edge_start = ((x1 + x2) // 2, y1)
                    edge_end = ((x1 + x2) // 2, y2)
                cv2.line(output, edge_start, edge_end, color, 3, cv2.LINE_AA)
                cv2.circle(output, edge_start, 4, color, -1, cv2.LINE_AA)
                cv2.circle(output, edge_end, 4, color, -1, cv2.LINE_AA)
                label_x, label_y = edge_start
            else:
                cv2.rectangle(
                    output, (x1, y1), (x2, y2), color, thickness
                )
                label_x, label_y = x1, y1

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5 if cls in {"person", "people", "human"} else 0.65
            (tw, th), baseline = cv2.getTextSize(
                label, font, scale, 2
            )

            top = max(0, label_y - th - baseline - 8)
            right = min(width - 1, label_x + tw + 10)
            bottom = min(height - 1, top + th + baseline + 8)

            cv2.rectangle(
                output, (label_x, top), (right, bottom), color, -1
            )
            cv2.putText(
                output, label,
                (label_x + 5, bottom - baseline - 4),
                font, scale, WHITE, 2, cv2.LINE_AA
            )

        return output

    @staticmethod
    def _draw_banner(frame, message):
        cv2.rectangle(
            frame, (0, 0), (frame.shape[1], 55), DARK_RED, -1
        )
        cv2.putText(
            frame, message[:100], (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            WHITE, 2, cv2.LINE_AA
        )

    def _show(self, frame, tracked_objects, hazard_events, detections=()):
        output = self._draw(
            frame, tracked_objects, detections, hazard_events=hazard_events
        )
        raw_person_on_track = any(
            self._person_inside_track_zone(TrackedObject(
                track_id=-1,
                class_name=detection.class_name,
                bbox=detection.bbox,
                history=[],
            ))
            for detection in detections
            if self._normalise_class(getattr(detection, "class_name", ""))
            in {"person", "people", "human"}
        )
        if self._has_person_on_track(tracked_objects) or raw_person_on_track:
            self._draw_banner(output, "HIGH RISK DETECTED")
        elif hazard_events:
            names = sorted({
                str(getattr(event, "hazard_type", "HAZARD")).upper()
                for event in hazard_events
            })
            self._draw_banner(
                output, "DANGER: " + ", ".join(names)
            )

        self._display_fullscreen(output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def _display_fullscreen(self, output):
        if not self._window_created:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                self._window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
            self._window_created = True

        # Show the complete frame at the largest size that fits the display.
        # Letterboxing preserves the camera geometry used by the overlays.
        cv2.imshow(self._window_name, output)
        try:
            _, _, view_width, view_height = cv2.getWindowImageRect(
                self._window_name
            )
        except cv2.error:
            view_width = view_height = 0

        if view_width > 0 and view_height > 0:
            frame_height, frame_width = output.shape[:2]
            scale = min(view_width / frame_width, view_height / frame_height)
            display_width = max(1, round(frame_width * scale))
            display_height = max(1, round(frame_height * scale))
            resized = cv2.resize(
                output,
                (display_width, display_height),
                interpolation=cv2.INTER_LINEAR,
            )
            display = np.full(
                (view_height, view_width, 3), (24, 31, 39), dtype=np.uint8
            )
            left = (view_width - display_width) // 2
            top = (view_height - display_height) // 2
            display[top:top + display_height, left:left + display_width] = resized
            cv2.imshow(self._window_name, display)

    def _save_snapshot(self, frame, tracked_objects, incident):
        snapshot_events = [
            HazardEvent(
                hazard_type=getattr(incident, "hazard_type", ""),
                track_id=getattr(incident, "track_id", -1),
                confidence=1.0,
            )
        ]
        output = self._draw(
            frame, tracked_objects, hazard_events=snapshot_events
        )
        severity = str(
            getattr(incident, "severity_level", "HIGH")
        ).upper()
        hazard = str(
            getattr(incident, "hazard_type", "HAZARD")
        ).upper()

        self._draw_banner(output, f"{severity}: {hazard}")

        filename = (
            f"{int(time.time() * 1000)}_"
            f"{hazard.replace(' ', '_')}.jpg"
        )
        path = os.path.join(config.SNAPSHOT_DIR, filename)
        cv2.imwrite(path, output)
        return path

    def release(self):
        if self.cap is not None:
            self.cap.release()

        if getattr(self, "alerts", None) is not None:
            self.alerts.close()

        if getattr(self, "db", None) is not None:
            self.db.close()

        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Railway platform safety monitor"
    )
    parser.add_argument(
        "--source", default=None,
        help="Image path, video path, camera index or RTSP URL"
    )
    parser.add_argument(
        "--rtdetr-weights", default=None,
        help="Path to trained RT-DETR weights"
    )
    parser.add_argument(
        "--water-weights", default=None,
        help="Path to water classifier weights"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display annotated image/video"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum video frames to process"
    )

    args = parser.parse_args()

    if args.source is not None and args.source.isdigit():
        args.source = int(args.source)

    return args


if __name__ == "__main__":
    args = parse_args()

    system = RailwaySafetySystem(
        source=args.source,
        rtdetr_weights=args.rtdetr_weights,
        water_weights=args.water_weights,
        show=args.show,
    )

    system.run_forever(max_frames=args.max_frames)
