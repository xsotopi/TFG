import os, shutil, yaml
from pathlib import Path
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# ───────── 1. Rutas ─────────
BASE        = Path("/home/lab/Desktop/TFG/yolo_finetune")
DS_TOOLS    = BASE / "data_split"
DS_MARKERS  = BASE / "markers"
DS_GLUE     = BASE / "glue"

COMB        = BASE / "combined"
TR_IMG, TR_LBL = COMB / "train/images", COMB / "train/labels"
VA_IMG, VA_LBL = COMB / "val/images",   COMB / "val/labels"
for p in [TR_IMG, TR_LBL, VA_IMG, VA_LBL]:
    p.mkdir(parents=True, exist_ok=True)

# ───────── 2. Función genérica de copia ─────────
def cp_dataset(src_img, src_lbl, dst_img, dst_lbl, prefix=""):
    """
    Copia todas las imágenes + labels añadiendo un prefijo al nombre.
    Mantiene la extensión (.jpg, .png, .txt).
    """
    for img in src_img.glob("*.*"):
        new_img_name = prefix + img.name
        shutil.copy2(img, dst_img / new_img_name)
        lbl = src_lbl / f"{img.stem}.txt"
        if lbl.exists():
            new_lbl_name = prefix + lbl.name
            shutil.copy2(lbl, dst_lbl / new_lbl_name)

# ───────── 3. Copiar datasets ─────────
# 3-A  tools  (ya dividido, sin prefijo para no romper índices 80-84)
cp_dataset(DS_TOOLS / "train/images", DS_TOOLS / "train/labels", TR_IMG, TR_LBL)
cp_dataset(DS_TOOLS / "val/images",   DS_TOOLS / "val/labels",   VA_IMG, VA_LBL)

# 3-B  glue   (prefijo 'glue_')
cp_dataset(DS_GLUE / "train/images", DS_GLUE / "train/labels", TR_IMG, TR_LBL, prefix="glue_")
cp_dataset(DS_GLUE / "val/images",   DS_GLUE / "val/labels",   VA_IMG, VA_LBL, prefix="glue_")

# 3-C  markers 80/20 + prefijo 'high_'
all_m_imgs = sorted((DS_MARKERS / "train/images").glob("*.*"))
train_m, val_m = train_test_split(all_m_imgs, test_size=0.2, random_state=42)
def cp_markers(img_list, dst_img, dst_lbl):
    for img in img_list:
        new_img = "high_" + img.name
        shutil.copy2(img, dst_img / new_img)
        lbl = DS_MARKERS / "train/labels" / f"{img.stem}.txt"
        shutil.copy2(lbl, dst_lbl / ("high_" + lbl.name))
cp_markers(train_m, TR_IMG, TR_LBL)
cp_markers(val_m,   VA_IMG, VA_LBL)

# ───────── 4. Re-indexar labels ─────────
OFFSET_HIGHLIGHTER, OFFSET_GLUE = 85, 86
def rewrite(txt_path: Path, new_cls: int):
    with open(txt_path) as f:
        lines = [" ".join([str(new_cls), *ln.split()[1:]]) + "\n"
                 for ln in f if ln.strip()]
    with open(txt_path, "w") as f:
        f.writelines(lines)

# a) ficheros que empiezan por 'high_' → 85
for txt in list(TR_LBL.glob("high_*.txt")) + list(VA_LBL.glob("high_*.txt")):
    rewrite(txt, OFFSET_HIGHLIGHTER)

# b) ficheros que empiezan por 'glue_' → 86
for txt in list(TR_LBL.glob("glue_*.txt")) + list(VA_LBL.glob("glue_*.txt")):
    rewrite(txt, OFFSET_GLUE)

# ───────── 5. YAML combinado ─────────
coco_names   = [YOLO("yolov8n.pt").names[i] for i in range(80)]
custom_names = ["pliers", "screwdriver", "wrench", "ratchet",
                "ratchet_parts", "highlighter", "glue"]
yaml_cfg = {
    "train": str(TR_IMG),
    "val":   str(VA_IMG),
    "nc":    87,
    "names": coco_names + custom_names
}
with open(BASE / "combined.yaml", "w") as f:
    yaml.dump(yaml_cfg, f, sort_keys=False)
print("✅ combined.yaml listo — imágenes val:", len(list(VA_IMG.glob('*.*'))))

# ───────── 6. Entrenamiento ─────────
model = YOLO("yolov8n.pt")
model.train(
    data=str(BASE / "combined.yaml"),
    epochs=50,
    batch=16,
    imgsz=640,
    rect=True,
    lr0=1e-2,
    project=str(BASE / "runs"), name="yolov8_all", exist_ok=True,
)
