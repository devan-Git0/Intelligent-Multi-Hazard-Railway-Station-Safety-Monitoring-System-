# Railway Platform Safety Monitoring System

Real-time, single-camera safety monitoring for railway platforms. An RT-DETR detector finds people, animals, luggage and the platform edge; ByteTrack gives each object a persistent ID; rule-based hazard checks turn tracks into events; a risk engine scores them; and incidents are alerted, saved with a snapshot, and shown on a live dashboard.

## Pipeline

```
Camera / video / image
   -> preprocessing.py   resize to 640x640, normalise, optional ROI mask
   -> detection.py       RT-DETR (person / animal / luggage / platform_edge)
   |                     + MobileNetV2 water patch classifier (optional)
   -> tracking.py        ByteTrack -> stable IDs, position history, dwell, heading
   -> hazards.py + main.py   hazard checks -> HazardEvent
   -> risk_fusion.py     severity score 0-100, LOW..CRITICAL, per-hazard cooldown
   -> alerts.py          console / voice / Slack / email (background threads)
   -> database.py        SQLite incident log (+ annotated snapshot per incident)
   -> dashboard.py       Streamlit live dashboard
```

## Files

| File | Role |
|---|---|
| `main.py` | Entry point. `RailwaySafetySystem` wires every stage; handles video/camera loop, single-image mode, overlay drawing, snapshots, CLI. |
| `config.py` | All paths, thresholds, class names, zone polygons, risk weights, alert settings. Change values here, not in logic files. |
| `datatypes.py` | Shared dataclasses: `Detection`, `TrackedObject`, `HazardEvent`, `Incident`. |
| `preprocessing.py` | Camera open, frame capture, resize, 0-1 normalisation, ROI mask. |
| `detection.py` | `DetectionEngine`: RT-DETR inference (COCO-pretrained or fine-tuned), optional YOLO person-detector fallback, person NMS, water patch scoring. |
| `models.py` | `WaterClassifier` (MobileNetV2, 2-class) and weight loader. |
| `tracking.py` | `ObjectTracker`: `supervision` ByteTrack plus stable-ID recovery after tracker re-assignments. |
| `hazards.py` | Hazard checks (see below). |
| `risk_fusion.py` | `RiskFusion`: severity scoring, concurrent-hazard bonus, cooldown. |
| `alerts.py` | `AlertDispatcher`: console, pyttsx3 voice, Slack webhook, SMTP email. |
| `database.py` | `IncidentDatabase`: SQLite in WAL mode, so the monitor and dashboard can run together. |
| `dashboard.py` | Streamlit dashboard: latest alert, KPIs, incident table, per-hazard chart, snapshot viewer, auto-refresh. |
| `train_detector.py` | Converts Roboflow CSV / TFRecord / Pascal VOC datasets to YOLO format and fine-tunes RT-DETR. |
| `test_pipeline.py` | End-to-end test with a fake detector, synthetic video and simulated clock. |

## Hazards

| Hazard | Logic | Base severity |
|---|---|---|
| `edge_line_crossing` | Person's foot-point (bottom-centre of box) is inside `track_crossing_zone`. Implemented inline in `main.py`. | 65 |
| `track_crossing` | Foot-point in the zone *and* moving (non-zero heading). Defined in `hazards.py`, **not currently called by `main.py`**. | 65 |
| `platform_edge` | Person's foot-point is within `edge_danger_distance_px` (60 px) of the edge line and the track is at least `edge_dwell_seconds` (2 s) old. | 60 |
| `animal_intrusion` | Animal centroid is inside `animal_zone`. | 55 |
| `abandoned_luggage` | Luggage with no person within `luggage_separation_distance_px` (150 px). | 70 |
| `water_accumulation` | Mean water probability over 224x224 patches of the bottom 40% of the frame exceeds 15% for 5 consecutive frames. | 40 + 60 x coverage |

**Scoring** (`risk_fusion.py`): `base x (0.7 + 0.3 x confidence)`, plus 5 per other concurrent hazard (capped at +15), clipped to 0-100. Levels: LOW < 40 <= MEDIUM < 60 <= HIGH < 80 <= CRITICAL. The same `(hazard, track)` pair is not re-reported for 30 s.

**Alert routing** (`alerts.py`): incidents below `MIN_ALERT_LEVEL` (MEDIUM) are logged but not pushed. Voice fires for HIGH and above, email for `EMAIL_MIN_LEVEL` (HIGH) and above. Slack and email run on a background thread pool so they cannot stall the video loop. Channels with no credentials are skipped.

## Setup

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch torchvision ultralytics supervision opencv-python numpy requests streamlit pandas
pip install pyttsx3        # optional, voice alerts
pip install tensorflow     # optional, only for TFRecord training data
```

`supervision` deprecated ByteTrack in 0.28 and removes it in 0.31, so pin `supervision<0.31`.
There is no `requirements.txt` in this archive; the list above is derived from the imports.

## Run

```bash
python main.py                                   # default webcam
python main.py --source clip.mp4 --show          # video file, live window (q quits)
python main.py --source station.png --show       # single still image -> station_annotated.png
python main.py --source rtsp://...               # RTSP stream
python main.py --rtdetr-weights weights/rtdetr_finetuned.pt --water-weights weights/water_classifier.pt

streamlit run dashboard.py                       # second terminal
```

Options: `--source`, `--rtdetr-weights`, `--water-weights`, `--show`, `--max-frames`.

Outputs: incidents in `incidents.db`, annotated frames in `snapshots/`, and `<image>_annotated.jpg` in image mode.

### Detector weights

- **Default:** `rtdetr-l.pt` (COCO, auto-downloaded). Only `person`, animal classes and COCO luggage classes (backpack, handbag, suitcase) are kept and renamed to the project classes. There is no `platform_edge` class, so the platform-edge check runs only against the calibrated edge line.
- **Fine-tuned:** if `weights/rtdetr_finetuned.pt` exists, `config.py` picks it up automatically. Classes: `person`, `animal`, `luggage`, `platform_edge`.
- **Person fallback:** if a fine-tuned model is used and `weights/yolo26n.pt` exists, it supplies person boxes instead.
- **Water:** needs `weights/water_classifier.pt`. Without it the water check is disabled and everything else still runs.

## Training the detector

```bash
python train_detector.py \
  --dataset1 "Platform Edge.v1i.tensorflow" \
  --dataset2 "Railway track obstacle detection.v1i.tensorflow" \
  --dataset3 "abandoned_luggage" \
  --epochs 10 --imgsz 416 --batch 1 --device 0
```

- Accepts Roboflow CSV (`_annotations.csv` per split), TFRecord, and Pascal VOC XML. Existing test splits stay held out; if no validation split exists, 15% is reserved.
- Source labels are mapped to the four project classes (e.g. `pedestrian` -> `person`, `suitcase` -> `luggage`, `platform edge` -> `platform_edge`). Unmapped labels are skipped and counted in the summary. Override with `--class-map my_map.json`.
- Other flags: `--base-weights`, `--output`, `--work-dir`, `--workers`, `--patience`, `--seed`, `--resume`.
- Result is copied to `weights/rtdetr_finetuned.pt`; best checkpoint is `training_runs/railway_rtdetr/weights/best.pt`.
- This does not train the water classifier.

## Calibration

Everything geometric in `config.py` is a placeholder for one camera view and must be recalibrated per camera:

- `HazardConfig.track_crossing_zone` (640x640 pixel polygon)
- `HazardConfig.animal_zone`
- `platform_edge_line_normalized` and `platform_warning_line_normalized` (0-1 coordinates)
- `IMAGE_CAMERA_PROFILES`: per-filename normalised zones for still images (currently only `station.jpg.png`).

## Testing

Each module has a standalone self-test:

```bash
python preprocessing.py   python models.py      python tracking.py
python hazards.py         python risk_fusion.py python database.py
python alerts.py          python detection.py   # needs torch + ultralytics
python test_pipeline.py   # full pipeline, no camera / weights / GPU
```

`test_pipeline.py` replaces the detector, so it does not exercise real weights.
**Known issue:** it asserts a `track_crossing` incident, but `main.py` emits `edge_line_crossing` for that situation, so the assertion needs updating (or `main.py` switched to `check_track_crossing`). I did not run the test because the dependencies are not installed in my environment; this is from reading the code.

## Known limitations

- Platform-edge distance is in raw pixels; no homography to real-world units.
- Water uses patch classification, not segmentation. Patches are scaled to 0-1 but not ImageNet-normalised, so training must match.
- Abandoned luggage is person-proximity only; there is no time-based dwell requirement in the current check.
- "Dwell" is track lifetime, not continuous time near the edge.
- `platform_edge` boxes bypass ByteTrack and go straight to the hazard checks.
- No trained weights ship with the code.
- Slack/email credentials in `config.py` are empty placeholders; use an app password for Gmail SMTP and keep secrets out of version control.
- Referenced but missing from this archive: `requirements.txt`, `class_map.example.json`, `weights/`.
