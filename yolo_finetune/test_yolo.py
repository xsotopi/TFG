import cv2
from ultralytics import YOLO

# 1. Carga tu modelo fine-tuneado
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

    # Muestra el vídeo en directo
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('d'):
        # Ejecuta detección en el frame actual
        results = model(frame, imgsz=640, conf=0.25)[0]
        # Dibuja las cajas y etiquetas sobre el frame
        annotated_frame = results.plot()
        cv2.imshow("Detections", annotated_frame)
        # Espera a que el usuario presione cualquier tecla para volver al stream
        cv2.waitKey(0)
        cv2.destroyWindow("Detections")

    elif key == ord('q'):
        # Salir del bucle
        break

# Limpieza
cap.release()
cv2.destroyAllWindows()
