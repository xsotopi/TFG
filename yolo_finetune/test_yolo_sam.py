# yolo_sam_stream.py – live YOLO‑v8 detection + SAM segmentation
# -----------------------------------------------------------------------------
# • Change OBJECT_NAME below to the class you want to segment.
# • Make sure the "Camera" window has keyboard focus; then:
#     d  → run detection + segmentation on the current frame
#     q  → quit the program
# -----------------------------------------------------------------------------

import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import torch

# ─────────────────────── USER SETTINGS ────────────────────────────
WEIGHTS_YOLO   = "/home/lab/Desktop/TFG/runs/yolov8_6class/weights/best.pt"
SAM_CHECKPOINT = "/home/lab/Desktop/TFG/sam_vit_b_01ec64.pth"
OBJECT_NAME    = "highlighter"           # ← change this manually
CAM_INDEX      = "/dev/video6"    # camera device or index (int or str)
CONF_THRESHOLD = 0.25              # YOLO confidence threshold
IMG_SIZE       = 640               # YOLO inference size
MASK_COLOR     = (0, 0, 255)       # BGR overlay color (red)
MASK_ALPHA     = 0.35              # transparency of mask
# ------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load models --------------------------------------------------------------
print("⏳ Loading YOLO weights…")
model = YOLO(WEIGHTS_YOLO)
model.to(device)
CLASS_NAMES = model.names

print("⏳ Loading SAM checkpoint…")
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to(device)
predictor = SamPredictor(sam)
print("✅ Models ready on", device)

# 2. Open camera --------------------------------------------------------------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print(f"❌ Cannot open camera at {CAM_INDEX}")
    sys.exit(1)
print("🎥 Stream abierto. Haz clic en la ventana y presiona 'd' para segmentar, 'q' para salir.")

# 3. Main loop ----------------------------------------------------------------
while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️  Error al leer frame de la cámara")
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKeyEx(1)
    if key == -1:
        continue
    key &= 0xFF  # ASCII

    # ----------------------------- DETECT & SEGMENT -------------------------
    if key in (ord('d'), ord('D')):
        results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        annotated = frame.copy()
        mask_union = np.zeros(frame.shape[:2], dtype=bool)
        found = False

        for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
            label = CLASS_NAMES[int(cls_id)]
            if OBJECT_NAME.lower() not in label.lower():
                continue
            found = True
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)

            # draw YOLO bbox + label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # run SAM on that bbox
            masks, _, _ = predictor.predict(box=np.array([x1, y1, x2, y2], dtype=np.float32),
                                            multimask_output=False)
            mask_union |= masks[0]

        if found and mask_union.any():
            overlay = np.full_like(annotated, MASK_COLOR, dtype=np.uint8)
            annotated = np.where(mask_union[:, :, None],
                                 cv2.addWeighted(overlay, MASK_ALPHA, annotated, 1 - MASK_ALPHA, 0),
                                 annotated)
        else:
            print(f"ℹ️  No '{OBJECT_NAME}' detectado en este frame.")

        cv2.imshow("Detections", annotated)
        cv2.waitKey(0)
        cv2.destroyWindow("Detections")

    # ---------------------------------------------------------------- QUIT
    elif key in (ord('q'), ord('Q')):
        break

# 4. Clean‑up -----------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
