from pathlib import Path
import yaml
from ultralytics import YOLO

DATA_ROOT = Path("/home/lab/Desktop/TFG/save_images")    # dataset (contains train/, val/)
RUNS_DIR  = Path("/home/lab/Desktop/TFG/runs")           # save folder
# ───────────────────────────────────────────────────────────────────────────────────

assert (DATA_ROOT / "train/images").exists(), "train/images not found"
assert (DATA_ROOT / "train/labels").exists(), "train/labels not found"

# Remap local class IDs 0-5 → 80-85
shift = {0: 80, 1: 81, 2: 82, 3: 83, 4: 84, 5: 85}

def remap_file(txt_path):
    """
    Reads YOLO label file and remaps class IDs according to the `shift` dictionary.
    """
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

print("Label IDs rewritten to 80-85")

# Build combined.yaml
coco_names = [YOLO("yolov8n.pt").names[i] for i in range(80)]
custom_names = ["screwdriver", "highlighter", "glue", "box", "ball", "rubik's cube"]

yaml_cfg = {
    "train": str(DATA_ROOT / "train/images"),
    "val": str((DATA_ROOT / "val/images") if (DATA_ROOT / "val").exists()
                 else (DATA_ROOT / "valid/images")),
    "nc": 86,
    "names": coco_names + custom_names,
}

yaml_path = DATA_ROOT / "combined.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(yaml_cfg, f, sort_keys=False)

print(f"✅ {yaml_path} written (85 classes)")

# Fine-tune
model = YOLO("yolov8n.pt")
model.train(
    data=str(yaml_path),
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=3e-3,                       # small LR so COCO weights stay stable
    freeze=10,                      # freeze first 10 layers
    project=str(RUNS_DIR),
    name="yolov8_6class",
    exist_ok=True,
)

print("\nTraining complete.")
