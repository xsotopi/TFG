import sys
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import torch

WEIGHTS_YOLO   = "/home/lab/Desktop/TFG/runs/yolov8_6class/weights/best.pt"
SAM_CHECKPOINT = "/home/lab/Desktop/TFG/sam_vit_b_01ec64.pth"
OBJECT_NAME    = "highlighter" # Class to detect
CAM_INDEX      = "/dev/video6"    
CONF_THRESHOLD = 0.25
IMG_SIZE       = 640
MASK_COLOR     = (0, 0, 255)
MASK_ALPHA     = 0.35

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
print("⏳ Loading YOLO weights…")
model = YOLO(WEIGHTS_YOLO)
model.to(device)
CLASS_NAMES = model.names

print("⏳ Loading SAM checkpoint…")
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to(device)
predictor = SamPredictor(sam)
print("✅ Models ready on", device)

# Open camera
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print(f"❌ Cannot open camera at {CAM_INDEX}")
    sys.exit(1)
print("🎥 Stream abierto. Haz clic en la ventana y presiona 'd' para segmentar, 'q' para salir.")

# Main loop
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

    # DETECT & SEGMENT
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

            # Draw YOLO bbox + label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # Run SAM on that bbox
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

    elif key in (ord('q'), ord('Q')):
        break

# Clean‑up
cap.release()
cv2.destroyAllWindows()
