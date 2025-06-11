#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live-view camera window.
SPACE → save current frame as PNG in ./images/
ESC   → exit.
"""

import cv2
import time
from pathlib import Path

# ---------- CONFIG -----------------------------------------------------------
CAMERA_INDEX = 6          # change if your webcam is on a different index
IMAGE_DIR    = Path("images")
IMAGE_DIR.mkdir(exist_ok=True)

# ---------- CAMERA -----------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_ANY)
if not cap.isOpened():
    raise RuntimeError("❌ No se pudo abrir la cámara")

print("[INFO] Pulsa ESPACIO para capturar (ESC sale).")
cv2.namedWindow("Vista cámara", cv2.WINDOW_NORMAL)

idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️ Frame inválido"); time.sleep(0.1); continue

    cv2.imshow("Vista cámara", frame)
    key = cv2.waitKey(10) & 0xFF

    if key == 27:                   # ESC
        print("🔚 Salida solicitada."); break

    if key == 32:                   # SPACE
        fname = IMAGE_DIR / f"img_{idx:04d}.png"
        cv2.imwrite(str(fname), frame)
        print(f"✅ Imagen guardada en {fname}")
        idx += 1

cap.release()
cv2.destroyAllWindows()
print("✅ Sesión terminada.")
