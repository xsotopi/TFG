import os, shutil, yaml
from pathlib import Path
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# ---------- 1. Rutas ----------
BASE        = Path("/home/lab/Desktop/TFG/yolo_finetune")
DS_TOOLS    = BASE / "data_split"     # 5 herramientas (ya estaba dividido)
DS_MARKERS  = BASE / "markers"        # 1 highlighter

COMB        = BASE / "combined"       # dataset resultante
TR_IMG      = COMB / "train/images"
TR_LBL      = COMB / "train/labels"
VA_IMG      = COMB / "val/images"
VA_LBL      = COMB / "val/labels"

for p in [TR_IMG, TR_LBL, VA_IMG, VA_LBL]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- 2. Copiar data_split tal cual ----------
def cp_tree(src_dir, dst_dir):
    for f in src_dir.rglob("*.*"):
        rel = f.relative_to(src_dir)
        shutil.copy(f, dst_dir / rel)

cp_tree(DS_TOOLS / "train/images",  TR_IMG)
cp_tree(DS_TOOLS / "train/labels",  TR_LBL)
cp_tree(DS_TOOLS / "val/images",    VA_IMG)
cp_tree(DS_TOOLS / "val/labels",    VA_LBL)

# ---------- 3. Acomodar markers (no está splitteado) ----------
# haremos un split 80/20 rápido
M_IMG = DS_MARKERS / "train/images"
M_LBL = DS_MARKERS / "train/labels"
imgs  = sorted([f for f in M_IMG.iterdir() if f.suffix.lower() in {'.jpg','.png','.jpeg'}])
tr, va = train_test_split(imgs, test_size=0.2, random_state=42)

def move_list(lista, dst_img, dst_lbl):
    for img_path in lista:
        lbl_path = M_LBL / (img_path.stem + ".txt")
        shutil.copy(img_path, dst_img / img_path.name)
        shutil.copy(lbl_path, dst_lbl / lbl_path.name)

move_list(tr, TR_IMG, TR_LBL)
move_list(va, VA_IMG, VA_LBL)

# ---------- 4. Re-indexar clases ----------
#   COCO: 0-79
#   tools: 80-84                (mantenemos el offset que ya usabas)
#   highlighter: 85
offset_tools = 80
offset_high  = 85               # solo 1 clase -> +85

# (a) Archivos de herramientas ya estaban corregidos en tu script original.
#     No volvemos a tocarlos.

# (b) Corregir ONLY los labels de highlighter (clase 0 -> 85)
for txt in list(TR_LBL.glob("*.txt")) + list(VA_LBL.glob("*.txt")):
    with open(txt) as f:
        lines = f.readlines()
    if all(l.startswith("0 ") for l in lines):  # solo highlighter
        new = [' '.join([str(offset_high)] + l.split()[1:]) + '\n' for l in lines]
        with open(txt, "w") as f:
            f.writelines(new)

# ---------- 5. YAML combinado ----------
coco_names  = [YOLO("yolov8n.pt").names[i] for i in range(80)]
custom_names = ["pliers", "screwdriver", "wrench", "ratchet", "ratchet_parts", "highlighter"]
all_names   = coco_names + custom_names      # 86 nombres

yaml_cfg = {
    "train": str(TR_IMG),
    "val":   str(VA_IMG),
    "nc":    len(all_names),
    "names": all_names
}
with open(BASE / "combined.yaml", "w") as f:
    yaml.dump(yaml_cfg, f, sort_keys=False)
print("✅ combined.yaml creado")

# ---------- 6. Fine-tune ----------
model = YOLO("yolov8n.pt")
model.train(
    data=str(BASE / "combined.yaml"),
    epochs=50,
    batch=16,
    imgsz=640,
    lr0=1e-2,
    project=str(BASE/"runs"),
    name="yolov8_tools_highlighters",
    exist_ok=True,
)
