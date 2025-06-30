import cv2
from ultralytics import YOLO

WEIGHTS = "/home/lab/Desktop/TFG/runs/yolov8_6class/weights/best.pt"
model = YOLO(WEIGHTS)

cap = cv2.VideoCapture("/dev/video6")
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara en /dev/6")
    exit(1)

print("🎥 Stream abierto. Presiona 'd' para detección, 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Error al leer frame de la cámara")
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('d'):
        results = model(frame, imgsz=640, conf=0.25)[0]
        annotated_frame = results.plot()
        cv2.imshow("Detections", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Detections")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
