import cv2
import time
from pathlib import Path

CAMERA_INDEX = 6
IMAGE_DIR    = Path("images")
IMAGE_DIR.mkdir(exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_ANY)
if not cap.isOpened():
    raise RuntimeError("❌ No se pudo abrir la cámara")

print("[INFO] SPACE to capture (ESC to exit).")
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        print("Invalid frame")
        time.sleep(0.1)
        continue

    cv2.imshow("Vista cámara", frame)
    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        print("🔚 Salida solicitada.")
        break

    if key == 32:
        fname = IMAGE_DIR / f"img_{idx:04d}.png"
        cv2.imwrite(str(fname), frame)
        print(f"✅ Imagen guardada en {fname}")
        idx += 1

cap.release()
cv2.destroyAllWindows()
print("✅ Sesión terminada.")
