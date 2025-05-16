import cv2
import numpy as np
import socket, struct
import os
import time
import re

# --- CONFIGURA TU IP DEL ROBOT ---
UR_IP = "192.168.0.102"
UR_PORT = 30003

# --- CONFIGURACIÓN ---
N_CAPTURES = 15
IMAGE_DIR = "images"
POSE_DIR = "poses"
CAMERA_INDEX = 6  # cambia esto si es necesario

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(POSE_DIR, exist_ok=True)

# Mantén el socket abierto durante toda la sesión
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((UR_IP, UR_PORT))

def get_tcp_pose():
    """
    Recibe un paquete RT (< 125 Hz) y devuelve [x,y,z,rx,ry,rz] en metros / rad.
    """
    # Cabecera: 4 bytes longitud (big-endian UInt32)
    hdr = sock.recv(4)
    if len(hdr) < 4:
        raise RuntimeError("Conexión cerrada por el robot")
    (pkt_len,) = struct.unpack(">I", hdr)

    # Resto del paquete
    data = b""
    while len(data) < pkt_len - 4:
        chunk = sock.recv(pkt_len - 4 - len(data))
        if not chunk:
            raise RuntimeError("Conexión cortada")
        data += chunk

    # 6 doubles a partir del byte 444 (0-based)
    start = 444 - 4            # porque ya consumimos los 4 bytes de la longitud
    tcp_pose = struct.unpack(">6d", data[start : start + 48])
    return list(tcp_pose)       # [x,y,z,rx,ry,rz]




def pose_to_matrix(pose):
    x, y, z, rx, ry, rz = pose
    angle = np.linalg.norm([rx, ry, rz])
    if angle < 1e-6:
        R = np.eye(3)
    else:
        axis = np.array([rx, ry, rz]) / angle
        R, _ = cv2.Rodrigues(axis * angle)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

# --- INICIA CÁMARA ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("❌ No se pudo abrir la cámara. Verifica el índice.")

print("[INFO] Pulsa ESPACIO para capturar imagen + pose. ESC para salir.")
cv2.namedWindow("Vista cámara", cv2.WINDOW_NORMAL)

idx = 0
while idx < N_CAPTURES:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Error leyendo la cámara.")
        time.sleep(0.1)
        continue

    cv2.imshow("Vista cámara", frame)
    key = cv2.waitKey(10) & 0xFF

    if key == 27:        # ESC
        print("🔚 Salida solicitada.")
        break

    if key == 32:        # SPACE
        print(f"[INFO] Capturando imagen {idx:02d}...")
        pose = get_tcp_pose()
        if pose is None:
            continue

        # Guardar imagen
        img_path = os.path.join(IMAGE_DIR, f"img_{idx:02d}.png")
        cv2.imwrite(img_path, frame)

        # Guardar pose
        T = pose_to_matrix(pose)
        pose_path = os.path.join(POSE_DIR, f"pose_{idx:02d}.npy")
        np.save(pose_path, T)

        print(f"✅ Imagen y pose {idx:02d} guardadas.")
        idx += 1
        time.sleep(0.3)

cap.release()
cv2.destroyAllWindows()
print("✅ Captura finalizada.")