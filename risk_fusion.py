"""
STEP 6 -- Risk Assessment & Severity Scoring
Takes the HazardEvents from all five hazard checks in one frame and turns
them into scored Incidents (0-100 + LOW/MEDIUM/HIGH/CRITICAL).

  * base severity per hazard type (config.RiskConfig.base_severity)
  * scaled by the event's confidence
  * small bonus when several hazards are active in the same frame
  * per-(hazard, track) cooldown so one lingering person/bag doesn't
    create a new incident every frame

Pure Python -- run this file directly to test scoring and cooldown.
"""

from typing import Dict, List, Optional, Tuple

from config import RiskConfig
from datatypes import HazardEvent, Incident

LEVEL_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


class RiskFusion:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self._last_reported: Dict[Tuple[str, int], float] = {}

    def level_for(self, score: float) -> str:
        c = self.config
        if score >= c.critical_threshold:
            return "CRITICAL"
        if score >= c.high_threshold:
            return "HIGH"
        if score >= c.medium_threshold:
            return "MEDIUM"
        return "LOW"

    def score_event(self, event: HazardEvent, n_other_hazards: int) -> float:
        c = self.config
        base = c.base_severity.get(event.hazard_type, 50.0)
        if event.hazard_type == "water_accumulation":
            # confidence here is coverage fraction -- more water, more severe
            score = base + 60.0 * event.confidence
        else:
            score = base * (0.7 + 0.3 * max(0.0, min(1.0, event.confidence)))
        score += min(c.max_concurrent_bonus, c.concurrent_hazard_bonus * n_other_hazards)
        return max(0.0, min(100.0, score))

    def assess(self, events: List[HazardEvent], timestamp: float) -> List[Incident]:
        """Returns only NEW incidents -- events still inside their cooldown
        window are suppressed. Sorted most severe first."""
        incidents: List[Incident] = []
        for ev in events:
            key = (ev.hazard_type, ev.track_id)
            last = self._last_reported.get(key)
            if last is not None and timestamp - last < self.config.cooldown_seconds:
                continue
            score = self.score_event(ev, n_other_hazards=len(events) - 1)
            incidents.append(Incident(
                hazard_type=ev.hazard_type,
                track_id=ev.track_id,
                severity_score=round(score, 1),
                severity_level=self.level_for(score),
                details=ev.details,
                timestamp=timestamp,
            ))
            self._last_reported[key] = timestamp
        incidents.sort(key=lambda i: i.severity_score, reverse=True)
        return incidents


if __name__ == "__main__":
    fusion = RiskFusion()
    ev_cross = HazardEvent("track_crossing", 1, 1.0, "person in zone")
    ev_water = HazardEvent("water_accumulation", -1, 0.30, "30% covered")

    print("=== Single hazard ===")
    out = fusion.assess([ev_cross], timestamp=0.0)
    print(out)
    assert out[0].severity_level == "CRITICAL"

    print("\n=== Same hazard 5s later (inside cooldown) -> suppressed ===")
    assert fusion.assess([ev_cross], timestamp=5.0) == []
    print("suppressed OK")

    print("\n=== After cooldown -> reported again ===")
    assert len(fusion.assess([ev_cross], timestamp=40.0)) == 1
    print("re-reported OK")

    print("\n=== Two simultaneous hazards (bonus applied, sorted) ===")
    fusion2 = RiskFusion()
    out = fusion2.assess([ev_water, ev_cross], timestamp=0.0)
    for i in out:
        print(f"  {i.hazard_type}: {i.severity_score} {i.severity_level}")
    assert out[0].hazard_type == "track_crossing"
    print("\nRisk fusion OK.")
