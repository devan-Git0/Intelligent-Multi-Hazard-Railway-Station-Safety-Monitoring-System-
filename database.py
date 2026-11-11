"""
STEP 8 -- Incident Database (SQLite)
Stores every Incident that passes risk scoring. main.py writes; dashboard.py
reads. WAL mode lets both run at the same time without locking each other.

Run this file directly to test schema + insert/read on a temp database.
"""

import os
import sqlite3
import threading
from typing import Dict, List

from datatypes import Incident

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      REAL    NOT NULL,
    hazard_type    TEXT    NOT NULL,
    track_id       INTEGER NOT NULL,
    severity_score REAL    NOT NULL,
    severity_level TEXT    NOT NULL,
    details        TEXT,
    snapshot_path  TEXT
);
CREATE INDEX IF NOT EXISTS idx_incidents_time ON incidents(timestamp DESC);
"""


class IncidentDatabase:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def insert_incident(self, incident: Incident) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO incidents (timestamp, hazard_type, track_id, severity_score,"
                " severity_level, details, snapshot_path) VALUES (?,?,?,?,?,?,?)",
                (incident.timestamp, incident.hazard_type, incident.track_id,
                 incident.severity_score, incident.severity_level,
                 incident.details, incident.snapshot_path),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def recent_incidents(self, limit: int = 100) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM incidents ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count_by_type(self) -> Dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT hazard_type, COUNT(*) AS n FROM incidents GROUP BY hazard_type"
            ).fetchall()
        return {r["hazard_type"]: r["n"] for r in rows}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


if __name__ == "__main__":
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "test_incidents.db")
    db = IncidentDatabase(path)
    db.insert_incident(Incident("track_crossing", 1, 95.0, "CRITICAL", "test", 1000.0, "a.jpg"))
    db.insert_incident(Incident("animal_intrusion", 2, 55.0, "MEDIUM", "dog", 1001.0))
    rows = db.recent_incidents()
    print(rows)
    print(db.count_by_type())
    assert len(rows) == 2 and rows[0]["hazard_type"] == "animal_intrusion"
    db.close()
    print("Database OK.")
