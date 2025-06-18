#!/usr/bin/env python3
"""
YOLOv8 — add four custom objects to the 80-class COCO model (total 84 classes).

Dataset folder expected:
    /home/lab/Desktop/TFG/save_images/
        ├── train/
        │     ├── images/   *.jpg | *.png
        │     └── labels/   *.txt  (class_id  cx  cy  w  h)
        └── val/  OR  valid/
              ├── images/
              └── labels/

Each label file currently uses local IDs 0-3 → ['screwdriver','highlighter','glue','box'].
The script rewrites them in-place to global IDs 80-83 and fine-tunes the model.
"""

from pathlib import Path
import yaml
from ultralytics import YOLO

# ────────────────────────────── USER - EDITABLE PATHS ─────────────────────────────
DATA_ROOT = Path("/home/lab/Desktop/TFG/save_images")    # dataset root (contains train/, val/)
RUNS_DIR  = Path("/home/lab/Desktop/TFG/runs")           # where YOLO will write logs/weights
# ───────────────────────────────────────────────────────────────────────────────────

# 0. safety check ------------------------------------------------------------------
assert (DATA_ROOT / "train/images").exists(), "train/images not found"
assert (DATA_ROOT / "train/labels").exists(), "train/labels not found"

# 1. remap local class IDs 0-3 → 80-83 --------------------------------------------
shift = {0: 80, 1: 81, 2: 82, 3: 83, 4: 84, 5: 85}

def remap_file(txt_path: Path) -> None:
    with open(txt_path) as f:
        lines = f.readlines()

    changed, fixed = False, []
    for ln in lines:
        parts = ln.strip().split()
        cid_old = int(parts[0])
        cid_new = shift.get(cid_old, cid_old)
        if cid_new != cid_old:
            parts[0] = str(cid_new)
            changed = True
        fixed.append(" ".join(parts) + "\n")

    if changed:
        with open(txt_path, "w") as f:
            f.writelines(fixed)

for split in ("train", "val", "valid"):
    lbl_dir = DATA_ROOT / split / "labels"
    if lbl_dir.exists():
        for lbl in lbl_dir.rglob("*.txt"):
            remap_file(lbl)

print("✅ label IDs rewritten to 80-85")

# 2. build combined.yaml -----------------------------------------------------------
coco_names = [YOLO("yolov8n.pt").names[i] for i in range(80)]
custom_names = ["screwdriver", "highlighter", "glue", "box", "ball", "rubik's cube"]

yaml_cfg = {
    "train": str(DATA_ROOT / "train/images"),
    "val":   str((DATA_ROOT / "val/images") if (DATA_ROOT / "val").exists()
                 else (DATA_ROOT / "valid/images")),
    "nc":    86,
    "names": coco_names + custom_names,
}

yaml_path = DATA_ROOT / "combined.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(yaml_cfg, f, sort_keys=False)

print(f"✅ {yaml_path} written (85 classes)")

# 3. fine-tune ---------------------------------------------------------------------
model = YOLO("yolov8n.pt")          # replace with yolov8s/m/l/x.pt if desired
model.train(
    data=str(yaml_path),
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=3e-3,                       # small LR so COCO weights stay stable
    freeze=10,                      # freeze first 10 layers (optional but helps)
    project=str(RUNS_DIR),
    name="yolov8_6class",
    exist_ok=True,
)

print("\n🎉 Training complete – check the runs/ folder for results.")
