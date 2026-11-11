"""
Shared data structures used across the whole pipeline
(detection -> tracking -> hazards -> risk fusion -> alerts / database).
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Detection:
    """One raw output from RT-DETR or the water classifier, before tracking."""
    class_name: str          # "person" | "animal" | "luggage" | "platform_edge" | "water"
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coordinates


@dataclass
class TrackedObject:
    """One object with a persistent ID, maintained across frames by tracking.py."""
    track_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    history: List[Tuple[float, float, float]] = field(default_factory=list)  # (t, cx, cy)

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def foot_point(self) -> Tuple[float, float]:
        """Bottom-center of the box -- where the object actually touches the
        ground, which is what matters for distance-to-edge and zone checks,
        not the box center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def dwell_seconds(self) -> float:
        """How long this track has existed, from its own history -- used by
        the platform-edge, luggage, and animal hazard checks."""
        if len(self.history) < 2:
            return 0.0
        return self.history[-1][0] - self.history[0][0]

    @property
    def heading(self) -> Tuple[float, float]:
        """Rough direction of travel as a unit-ish vector, from the last two
        recorded positions. Returns (0, 0) if there isn't enough history yet."""
        if len(self.history) < 2:
            return (0.0, 0.0)
        (_, x0, y0), (_, x1, y1) = self.history[-2], self.history[-1]
        return (x1 - x0, y1 - y0)


@dataclass
class HazardEvent:
    """One triggered hazard, before risk scoring."""
    hazard_type: str
    track_id: int
    confidence: float
    details: str = ""


@dataclass
class Incident:
    """A HazardEvent after risk scoring -- this is what gets dispatched to
    the alert channels and written to the incident database."""
    hazard_type: str
    track_id: int
    severity_score: float      # 0-100
    severity_level: str        # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    details: str
    timestamp: float           # unix seconds
    snapshot_path: str = ""    # filled in by main.py once the frame is saved
