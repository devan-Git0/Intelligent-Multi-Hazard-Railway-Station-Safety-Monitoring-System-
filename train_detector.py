"""Fine-tune the project's Ultralytics RT-DETR detector from CSV, TFRecord, and Pascal VOC XML datasets.

Expected Roboflow TensorFlow Object Detection export: *.tfrecord / *.record plus
label_map.pbtxt. The script converts both datasets into YOLO detection format,
then fine-tunes RT-DETR. TensorFlow is used only to read TFRecords; PyTorch/
Ultralytics performs training so the resulting .pt file works with detection.py.
"""

import argparse
import hashlib
import random
import re
import shutil
import sys
from pathlib import Path

PROJECT_CLASSES = ["person", "animal", "luggage", "platform_edge"]
ALIASES = {
    "person": "person", "people": "person", "human": "person", "pedestrian": "person",
    "animal": "animal", "bird": "animal", "cat": "animal", "dog": "animal",
    "horse": "animal", "cow": "animal", "sheep": "animal", "goat": "animal",
    "elephant": "animal", "bear": "animal", "zebra": "animal", "giraffe": "animal",
    "luggage": "luggage", "bag": "luggage", "baggage": "luggage", "suitcase": "luggage",
    "backpack": "luggage", "handbag": "luggage", "suitcases": "luggage",
    "platform_edge": "platform_edge", "platform edge": "platform_edge",
    "platform-edge": "platform_edge", "edge": "platform_edge", "platform boundary": "platform_edge",
    "platform_boundary": "platform_edge", "platform": "platform_edge",
}


def parse_args():
    p = argparse.ArgumentParser(description="Train railway RT-DETR using one or two Roboflow TensorFlow datasets.")
    p.add_argument("--dataset1", required=True, help="First Roboflow TensorFlow export folder or TFRecord file")
    p.add_argument("--dataset2", default=None, help="Optional second dataset folder or TFRecord file")
    p.add_argument("--dataset3", default=None, help="Optional third dataset folder; supports Pascal VOC XML annotations")
    p.add_argument("--base-weights", default="rtdetr-l.pt", help="Starting Ultralytics RT-DETR weights (default: rtdetr-l.pt)")
    p.add_argument("--output", default="weights/rtdetr_finetuned.pt", help="Final trained .pt path")
    p.add_argument("--work-dir", default="training_work/combined_roboflow", help="Generated YOLO dataset directory")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=416, help="512 is a safer starting size for a 6 GB laptop GPU")
    p.add_argument("--batch", type=int, default=1, help="Start with 2 for a 6 GB GPU; lower to 1 if CUDA runs out of memory")
    p.add_argument("--device", default="0", help="0 for NVIDIA GPU, cpu for CPU training")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class-map", default=None, help="Optional JSON mapping source label names to person/animal/luggage/platform_edge")
    p.add_argument("--resume", action="store_true", help="Resume from Ultralytics' last checkpoint if available")
    return p.parse_args()


def read_label_map(root: Path):
    mapping = {}
    candidates = list(root.rglob("label_map.pbtxt")) + list(root.rglob("labelmap.pbtxt"))
    if not candidates:
        return mapping
    text = candidates[0].read_text(encoding="utf-8", errors="ignore")
    for block in re.findall(r"item\s*\{(.*?)\}", text, flags=re.S):
        id_match = re.search(r"\bid\s*:\s*(\d+)", block)
        name_match = re.search(r"\b(?:name|display_name)\s*:\s*['\"]([^'\"]+)['\"]", block)
        if id_match and name_match:
            mapping[int(id_match.group(1))] = name_match.group(1).strip()
    return mapping


def feature_value(features, key, kind, default=None):
    item = features.feature.get(key)
    if item is None:
        return default
    values = getattr(item, kind).value
    return values[0] if values else default


def feature_values(features, key, kind):
    item = features.feature.get(key)
    if item is None:
        return []
    return list(getattr(item, kind).value)


def source_split(path: Path):
    text = (str(path.parent) + " " + path.stem).lower()
    if re.search(r"(^|[/_ .-])(valid|validation|val)([/_ .-]|$)", text):
        return "val"
    if re.search(r"(^|[/_ .-])test([/_ .-]|$)", text):
        return "test"
    return "train"


def discover_records(dataset_path: Path):
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if dataset_path.is_file():
        if dataset_path.suffix.lower() not in (".record", ".tfrecord", ".tfrecords"):
            raise ValueError(f"Expected a TFRecord file, got: {dataset_path}")
        return [dataset_path]
    records = sorted(p for p in dataset_path.rglob("*") if p.is_file() and p.suffix.lower() in (".record", ".tfrecord", ".tfrecords"))
    if not records:
        raise FileNotFoundError(f"No .record/.tfrecord files found under {dataset_path}. Use Roboflow's TensorFlow Object Detection export.")
    return records


def normalize_class(label, custom_map):
    key = label.decode("utf-8", errors="ignore") if isinstance(label, bytes) else str(label)
    key = key.strip().lower().replace("-", "_")
    if custom_map:
        for source_name, target in custom_map.items():
            if source_name.strip().lower().replace("-", "_") == key:
                target = target.strip().lower().replace("-", "_")
                return target if target in PROJECT_CLASSES else None
    return ALIASES.get(key) or ALIASES.get(key.replace("_", " "))


def convert_csv_datasets(dataset_paths, output_dir: Path, custom_map, seed):
    """Convert Roboflow CSV exports with image and _annotations.csv together in each split."""
    import csv
    from collections import defaultdict
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    counters = {"train": 0, "val": 0, "test": 0, "unmapped": 0, "bad": 0}
    for dataset_idx, root in enumerate(dataset_paths, start=1):
        root = root.resolve()
        if not root.is_dir():
            raise ValueError(f"CSV dataset must be a folder containing train/valid/test splits: {root}")
        csv_files = sorted(root.rglob("*.csv"))
        csv_files = [p for p in csv_files if "annotation" in p.name.lower()]
        if not csv_files:
            raise FileNotFoundError(f"No annotation CSV found under {root}")
        print(f"Dataset {dataset_idx}: {root}; annotation CSV files: {len(csv_files)}")
        for csv_path in csv_files:
            parts = [part.lower() for part in csv_path.parts]
            if "train" in parts:
                split = "train"
            elif any(x in parts for x in ("valid", "validation", "val")):
                split = "val"
            elif "test" in parts:
                split = "test"
            else:
                print(f"Warning: cannot determine split from {csv_path}; skipping")
                continue
            grouped = defaultdict(list)
            try:
                with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    required = {"filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"}
                    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                        raise ValueError(f"CSV needs columns {sorted(required)}; found {reader.fieldnames}")
                    for row in reader:
                        grouped[row["filename"]].append(row)
            except Exception as exc:
                raise RuntimeError(f"Could not read annotation CSV {csv_path}: {exc}") from exc
            image_dir = csv_path.parent
            for filename, rows in grouped.items():
                image_path = image_dir / filename
                if not image_path.is_file():
                    # Some exports use a relative path in the CSV.
                    matches = list(image_dir.rglob(Path(filename).name))
                    if not matches:
                        counters["bad"] += 1
                        continue
                    image_path = matches[0]
                labels = []
                for row in rows:
                    label = normalize_class(row.get("class", ""), custom_map)
                    if label is None:
                        counters["unmapped"] += 1
                        continue
                    try:
                        w, h = float(row["width"]), float(row["height"])
                        x1, y1 = float(row["xmin"]), float(row["ymin"])
                        x2, y2 = float(row["xmax"]), float(row["ymax"])
                        if w <= 0 or h <= 0:
                            raise ValueError("invalid image dimensions")
                        x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2))
                        y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        xc, yc = ((x1+x2)/2)/w, ((y1+y2)/2)/h
                        bw, bh = (x2-x1)/w, (y2-y1)/h
                        class_id = PROJECT_CLASSES.index(label)
                        labels.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    except (ValueError, TypeError, ZeroDivisionError):
                        counters["bad"] += 1
                if not labels:
                    continue
                digest = hashlib.sha1(f"{dataset_idx}:{split}:{filename}".encode()).hexdigest()[:16]
                suffix = image_path.suffix.lower() if image_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp") else ".jpg"
                target_image = output_dir / "images" / split / f"{digest}{suffix}"
                target_label = output_dir / "labels" / split / f"{digest}.txt"
                shutil.copy2(image_path, target_image)
                target_label.write_text("\n".join(labels) + "\n", encoding="utf-8")
                counters[split] += 1
    # Use existing validation/test split when present; otherwise reserve validation from train.
    train_images = sorted((output_dir / "images" / "train").glob("*.*"))
    val_images = sorted((output_dir / "images" / "val").glob("*.*"))
    if not val_images and len(train_images) >= 3:
        rng = random.Random(seed)
        rng.shuffle(train_images)
        count = max(1, round(len(train_images) * 0.15))
        for image in train_images[:count]:
            label = output_dir / "labels" / "train" / f"{image.stem}.txt"
            shutil.move(str(image), str(output_dir / "images" / "val" / image.name))
            if label.exists():
                shutil.move(str(label), str(output_dir / "labels" / "val" / label.name))
        counters["val"] += count
        counters["train"] -= count
    print("CSV conversion summary:", counters)
    train_total = len(list((output_dir / "images" / "train").glob("*.*")))
    val_total = len(list((output_dir / "images" / "val").glob("*.*")))
    if train_total < 2 or val_total < 1:
        raise RuntimeError("Not enough usable labeled images. Need at least 2 training images and 1 validation image; check class names and split folders.")
    return counters


def convert_voc_datasets(dataset_paths, output_dir: Path, custom_map, seed):
    """Convert Pascal VOC XML annotations and matching images to YOLO labels."""
    import xml.etree.ElementTree as ET
    from collections import defaultdict

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counters = {"train": 0, "val": 0, "test": 0, "unmapped": 0, "bad": 0}
    image_suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for dataset_idx, root in enumerate(dataset_paths, start=1):
        root = root.resolve()
        xml_files = sorted(root.rglob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No Pascal VOC XML annotations found under {root}")

        image_index = defaultdict(list)
        for image_path in root.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in image_suffixes:
                image_index[image_path.name.lower()].append(image_path)

        records = []
        for xml_path in xml_files:
            try:
                annotation = ET.parse(xml_path).getroot()
                filename = Path((annotation.findtext("filename") or "").strip()).name
                matches = image_index.get(filename.lower(), [])
                if not matches:
                    counters["bad"] += 1
                    continue
                image_path = matches[0]
                size = annotation.find("size")
                width = float(size.findtext("width")) if size is not None else 0.0
                height = float(size.findtext("height")) if size is not None else 0.0
                if width <= 0 or height <= 0:
                    counters["bad"] += 1
                    continue

                labels = []
                for obj in annotation.findall("object"):
                    class_name = normalize_class(obj.findtext("name", ""), custom_map)
                    box = obj.find("bndbox")
                    if class_name is None or box is None:
                        counters["unmapped"] += 1
                        continue
                    try:
                        x1 = max(0.0, min(width, float(box.findtext("xmin"))))
                        y1 = max(0.0, min(height, float(box.findtext("ymin"))))
                        x2 = max(0.0, min(width, float(box.findtext("xmax"))))
                        y2 = max(0.0, min(height, float(box.findtext("ymax"))))
                        if x2 <= x1 or y2 <= y1:
                            counters["bad"] += 1
                            continue
                        xc, yc = ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height
                        bw, bh = (x2 - x1) / width, (y2 - y1) / height
                        class_id = PROJECT_CLASSES.index(class_name)
                        labels.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    except (ValueError, TypeError, ZeroDivisionError):
                        counters["bad"] += 1
                if labels:
                    records.append((xml_path, image_path, labels))
            except (ET.ParseError, OSError, ValueError, AttributeError) as exc:
                counters["bad"] += 1
                print(f"Warning: skipping invalid VOC annotation {xml_path}: {exc}")

        if not records:
            raise RuntimeError(f"No usable Pascal VOC image/annotation pairs found under {root}")

        explicit_split = any(
            any(part.lower() in ("train", "val", "valid", "validation", "test") for part in xml_path.parts)
            for xml_path, _, _ in records
        )
        if explicit_split:
            split_records = defaultdict(list)
            for record in records:
                parts = [part.lower() for part in record[0].parts]
                if "test" in parts:
                    split = "test"
                elif any(part in ("val", "valid", "validation") for part in parts):
                    split = "val"
                else:
                    split = "train"
                split_records[split].append(record)
        else:
            random.Random(seed + dataset_idx).shuffle(records)
            val_count = max(1, round(len(records) * 0.15)) if len(records) >= 3 else 1
            split_records = {"val": records[:val_count], "train": records[val_count:]}

        for split, items in split_records.items():
            for xml_path, image_path, labels in items:
                rel_path = xml_path.relative_to(root).as_posix()
                digest = hashlib.sha1(f"voc:{dataset_idx}:{split}:{rel_path}".encode()).hexdigest()[:16]
                target_image = output_dir / "images" / split / f"{digest}{image_path.suffix.lower()}"
                target_label = output_dir / "labels" / split / f"{digest}.txt"
                shutil.copy2(image_path, target_image)
                target_label.write_text("\n".join(labels) + "\n", encoding="utf-8")
                counters[split] += 1

        print(f"Pascal VOC dataset {dataset_idx}: {root}; usable images: {len(records)}")

    print("Pascal VOC conversion summary:", counters)
    return counters


def convert_records(dataset_paths, output_dir: Path, custom_map, seed):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required only for reading Roboflow TFRecords. Install it in this Python environment: python -m pip install tensorflow") from exc
    # Prevent TensorFlow's TFRecord reader from reserving GPU memory needed by PyTorch training.
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counters = {"train": 0, "val": 0, "test": 0, "unmapped": 0, "bad": 0}
    all_sources = []
    for dataset_idx, root in enumerate(dataset_paths, start=1):
        root = root.resolve()
        label_map = read_label_map(root if root.is_dir() else root.parent)
        records = discover_records(root)
        print(f"Dataset {dataset_idx}: {root} ({len(records)} TFRecord file(s)); label map IDs: {label_map or 'not found'}")
        for record in records:
            split = source_split(record)
            # Test samples are preserved separately and never included in training.
            for row_idx, serialized in enumerate(tf.data.TFRecordDataset([str(record)], num_parallel_reads=1)):
                try:
                    example = tf.train.Example.FromString(bytes(serialized.numpy()))
                    f = example.features.feature
                    encoded = feature_value(f, "image/encoded", "bytes_list")
                    if not encoded:
                        counters["bad"] += 1
                        continue
                    xmin = feature_values(f, "image/object/bbox/xmin", "float_list")
                    xmax = feature_values(f, "image/object/bbox/xmax", "float_list")
                    ymin = feature_values(f, "image/object/bbox/ymin", "float_list")
                    ymax = feature_values(f, "image/object/bbox/ymax", "float_list")
                    ids = feature_values(f, "image/object/class/label", "int64_list")
                    texts = feature_values(f, "image/object/class/text", "bytes_list")
                    n = min(len(xmin), len(xmax), len(ymin), len(ymax), max(len(ids), len(texts)))
                    labels = []
                    for j in range(n):
                        raw_name = texts[j] if j < len(texts) else label_map.get(int(ids[j]), "") if j < len(ids) else ""
                        mapped = normalize_class(raw_name, custom_map)
                        if mapped is None and j < len(ids):
                            mapped = normalize_class(label_map.get(int(ids[j]), ""), custom_map)
                        if mapped is None:
                            counters["unmapped"] += 1
                            continue
                        x1, x2 = max(0.0, float(xmin[j])), min(1.0, float(xmax[j]))
                        y1, y2 = max(0.0, float(ymin[j])), min(1.0, float(ymax[j]))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        class_id = PROJECT_CLASSES.index(mapped)
                        labels.append(f"{class_id} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}")
                    if not labels:
                        continue
                    identity = f"{dataset_idx}:{record}:{row_idx}:{hashlib.sha1(encoded).hexdigest()[:10]}"
                    name = hashlib.sha1(identity.encode()).hexdigest()[:16]
                    ext = feature_value(f, "image/format", "bytes_list", b"jpg")
                    ext = ext.decode("ascii", errors="ignore").lower() if isinstance(ext, bytes) else "jpg"
                    ext = "png" if "png" in ext else "jpg"
                    image_path = output_dir / "images" / split / f"{name}.{ext}"
                    label_path = output_dir / "labels" / split / f"{name}.txt"
                    image_path.write_bytes(encoded)
                    label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
                    counters[split] += 1
                except Exception as exc:
                    counters["bad"] += 1
                    if counters["bad"] <= 3:
                        print(f"Warning: skipped unreadable TFRecord example in {record}: {exc}")

    # If export contains no validation split, reserve 15% of training examples.
    train_images = sorted((output_dir / "images" / "train").glob("*.*"))
    val_images = sorted((output_dir / "images" / "val").glob("*.*"))
    if not val_images and len(train_images) >= 2:
        rng = random.Random(seed)
        rng.shuffle(train_images)
        count = max(1, round(len(train_images) * 0.15))
        for image in train_images[:count]:
            label = output_dir / "labels" / "train" / f"{image.stem}.txt"
            shutil.move(str(image), str(output_dir / "images" / "val" / image.name))
            if label.exists():
                shutil.move(str(label), str(output_dir / "labels" / "val" / label.name))
        counters["val"] += count
        counters["train"] -= count

    print("Conversion summary:", counters)
    if counters["train"] < 2 or counters["val"] < 1:
        raise RuntimeError("Not enough usable labeled images after conversion. Check TFRecord annotations, label_map.pbtxt, and class names. Need at least 2 train images and 1 validation image.")
    return counters


def main():
    args = parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("epochs and batch must be >= 1; imgsz must be >= 32")

    dataset_paths = [Path(args.dataset1)]
    if args.dataset2:
        dataset_paths.append(Path(args.dataset2))
    if args.dataset3:
        dataset_paths.append(Path(args.dataset3))
    custom_map = None
    if args.class_map:
        import json
        custom_map = json.loads(Path(args.class_map).read_text(encoding="utf-8"))
        if not isinstance(custom_map, dict):
            raise ValueError("--class-map JSON must be an object, e.g. {\"human\": \"person\"}")

    work_dir = Path(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = [path for path in dataset_paths if path.is_dir() and any(
        "annotation" in p.name.lower() and p.suffix.lower() == ".csv" for p in path.rglob("*.csv")
    )]
    voc_paths = [path for path in dataset_paths if path.is_dir() and any(path.rglob("*.xml"))]
    record_paths = [path for path in dataset_paths if path not in csv_paths and path not in voc_paths]

    # Convert XML first so its validation split is available to CSV or
    # TFRecord datasets that need to reserve validation examples.
    if voc_paths:
        convert_voc_datasets(voc_paths, work_dir, custom_map, args.seed)
    if csv_paths:
        convert_csv_datasets(csv_paths, work_dir, custom_map, args.seed)
    if record_paths:
        convert_records(record_paths, work_dir, custom_map, args.seed)

    train_total = len(list((work_dir / "images" / "train").glob("*.*")))
    val_total = len(list((work_dir / "images" / "val").glob("*.*")))
    if train_total < 2 or val_total < 1:
        raise RuntimeError(
            f"Not enough usable labeled images after conversion (train={train_total}, val={val_total}). "
            "Need at least 2 training images and 1 validation image."
        )

    yaml_path = work_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {work_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names: {PROJECT_CLASSES!r}\n",
        encoding="utf-8",
    )

    try:
        import torch
        from ultralytics import RTDETR
    except ImportError as exc:
        raise RuntimeError("Install project requirements first: python -m pip install -r requirements.txt") from exc
    if args.device != "cpu" and not torch.cuda.is_available():
        print("WARNING: CUDA is not available to PyTorch; switching to CPU (training will be much slower).")
        args.device = "cpu"
    elif args.device != "cpu":
        print("PyTorch GPU:", torch.cuda.get_device_name(0))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting RT-DETR training: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device={args.device}")
    resume_checkpoint = Path("training_runs/railway_rtdetr/weights/last.pt")
    if args.resume and resume_checkpoint.exists():
        print(f"Resuming from {resume_checkpoint}")
        model = RTDETR(str(resume_checkpoint))
    else:
        model = RTDETR(args.base_weights)
    results = model.train(
        data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, patience=args.patience, seed=args.seed,
        project="training_runs", name="railway_rtdetr", exist_ok=True,
        cache=False, amp=True, plots=False, save=True, verbose=True,
        resume=args.resume,
    )
    best = Path(model.trainer.best) if getattr(model, "trainer", None) and getattr(model.trainer, "best", None) else None
    last = Path(model.trainer.last) if getattr(model, "trainer", None) and getattr(model.trainer, "last", None) else None
    selected = best if best and best.exists() else last
    if selected is None or not selected.exists():
        # Ultralytics normally writes to training_runs/railway_rtdetr/weights.
        candidates = [Path("training_runs/railway_rtdetr/weights/best.pt"), Path("training_runs/railway_rtdetr/weights/last.pt")]
        selected = next((p for p in candidates if p.exists()), None)
    if selected is None:
        raise RuntimeError("Training ended but no best.pt/last.pt checkpoint was found; inspect the Ultralytics training log.")
    shutil.copy2(selected, output)
    print(f"\nTraining complete. Selected checkpoint: {selected}\nCopied trained detector to: {output.resolve()}")
    print("Run inference with: python main.py --source clip.mp4 --rtdetr-weights \"" + str(output) + "\" --show")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Ultralytics usually keeps training_runs/railway_rtdetr/weights/last.pt; rerun with --resume if present.")
        sys.exit(130)
