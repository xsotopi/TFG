import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO

def main():
    # 1. RUTAS Y PARÁMETROS
    BASE       = Path("/home/lab/Desktop/TFG/yolo_finetune")
    ORIG_IMG   = BASE / "train" / "images"
    ORIG_LBL   = BASE / "train" / "labels"

    # Split 80% train / 20% val
    SPLIT_DIR  = BASE / "data_split"
    TRAIN_IMG  = SPLIT_DIR / "train" / "images"
    TRAIN_LBL  = SPLIT_DIR / "train" / "labels"
    VAL_IMG    = SPLIT_DIR / "val"   / "images"
    VAL_LBL    = SPLIT_DIR / "val"   / "labels"
    TEST_SIZE  = 0.2
    SEED       = 42

    # Entrenamiento
    PRETRAINED = "yolov8n.pt"   # modelo base (COCO)
    EPOCHS     = 50
    BATCH      = 16
    IMGSZ      = 640
    LR0        = 1e-2

    # 2. Crear carpetas de split
    for p in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
        p.mkdir(parents=True, exist_ok=True)

    # 3. Listado de imágenes y split
    all_imgs = [f.name for f in ORIG_IMG.iterdir() if f.suffix.lower() in {".jpg",".png",".jpeg"}]
    train_imgs, val_imgs = train_test_split(all_imgs, test_size=TEST_SIZE, random_state=SEED)

    def copiar(lista, dst_img, dst_lbl):
        for img in lista:
            lbl_name = img.rsplit('.', 1)[0] + '.txt'
            # copiar imagen
            shutil.copy(ORIG_IMG / img, dst_img / img)
            # copiar label (si existe)
            src_lbl = ORIG_LBL / lbl_name
            if src_lbl.exists():
                shutil.copy(src_lbl, dst_lbl / lbl_name)
            else:
                print(f"⚠️ Falta label para {img}")

    copiar(train_imgs, TRAIN_IMG, TRAIN_LBL)
    copiar(val_imgs,   VAL_IMG,   VAL_LBL)

    # 4. Clases combinadas COCO + custom
    # 4.1 Extraer nombres COCO del modelo pretrained
    coco_model = YOLO(PRETRAINED)
    coco_dict  = coco_model.names               # dict {0:'person',1:'bicycle',...}
    coco_names = [coco_dict[i] for i in sorted(coco_dict.keys())]

    # 4.2 Tus 5 herramientas en inglés
    custom_names = [
        "pliers",
        "screwdriver",
        "wrench",
        "ratchet",
        "ratchet_parts"
    ]

    all_names = coco_names + custom_names
    NC_total  = len(all_names)  # 85

    # 4.3 Ajustar índices en etiquetas custom (0-4 → 80-84)
    offset = len(coco_names)
    for lbl_dir in [TRAIN_LBL, VAL_LBL]:
        for txt_file in lbl_dir.glob('*.txt'):
            lines = []
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0]) + offset
                    coords = parts[1:]
                    lines.append(' '.join([str(cls_id)] + coords))
            with open(txt_file, 'w') as f:
                f.write('\n'.join(lines) + '\n')

    # 5. Generar data_split.yaml
    cfg = {
        'train': str(TRAIN_IMG),
        'val':   str(VAL_IMG),
        'test':  "",
        'nc':    NC_total,
        'names': all_names
    }
    with open(BASE / "data_split.yaml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    print(f"✅ data_split.yaml creado con nc={NC_total}")

    # 6. Fine-tuning YOLOv8 sobre dataset combinado
    model = YOLO(PRETRAINED)
    model.train(
        data=str(BASE / "data_split.yaml"),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        lr0=LR0,
        project=str(BASE / "runs"),
        name="yolov8_coco_plus_tools_en",
        exist_ok=True,
    )
    print("🚀 Entrenamiento COCO (80) + Tools (5) completado.")

if __name__ == '__main__':
    main()
