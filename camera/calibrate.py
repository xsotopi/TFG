import cv2
import numpy as np
import glob, os

# ---------- CONFIGURACIÓN ----------
CHECKERBOARD = (8, 7)          # 8 columnas internas × 7 filas internas
SQUARE_SIZE  = 0.0127           # 15 mm  (lo dejas así)
IMAGE_DIR    = "images"
POSE_DIR     = "poses"
SAVE_PATH    = "T_cam2tcp.npy"

# ---------- PATRÓN 3D ----------
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints, imgpoints = [], []            # para calibrar K, dist
valid_sets = []                          # (corners, T_tcp_base)

# ---------- CARGAR DATOS ----------
images = sorted(glob.glob(os.path.join(IMAGE_DIR, "img_*.png")))
pose_files = sorted(glob.glob(os.path.join(POSE_DIR, "pose_*.npy")))
robot_poses = [np.load(p) for p in pose_files]

if len(images) != len(robot_poses):
    raise ValueError("Nº de imágenes y poses no coincide")

# ---------- DETECTAR EL TABLERO ----------
for img_path, T_tcp_base in zip(images, robot_poses):
    img  = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ok, corners = cv2.findChessboardCornersSB(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)

    if not ok:
        print(f"[WARN] Patrón no detectado: {img_path}")
        continue

    objpoints.append(objp)
    imgpoints.append(corners)
    valid_sets.append((corners, T_tcp_base))

print(f"[INFO] Imágenes válidas: {len(objpoints)}")
if len(objpoints) < 3:
    raise RuntimeError("Necesitas al menos 3 vistas válidas para calibrar.")

# ---------- CALIBRACIÓN INTRÍNSECA ----------
_, K, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
print("[INFO] Intrínseca K:\n", K)

# ---------- LISTAS PARA HAND-EYE ----------
R_t2c, t_t2c = [], []
R_g2b, t_g2b = [], []

for corners, T_tcp_base in valid_sets:
    # Pose tablero→cámara con K & dist reales
    ok, rvec, tvec = cv2.solvePnP(
        objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        continue
    R,_ = cv2.Rodrigues(rvec)
    R_t2c.append(R)
    t_t2c.append(tvec)

    # TCP→base
    Rg = T_tcp_base[:3,:3]
    tg = T_tcp_base[:3,3].reshape(3,1)
    R_g2b.append(Rg)
    t_g2b.append(tg)

# ---------- HAND-EYE ----------
R_c2g, t_c2g = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI)

T_cam2tcp = np.eye(4)
T_cam2tcp[:3,:3] = R_c2g
T_cam2tcp[:3, 3] = t_c2g.ravel()

np.save(SAVE_PATH, T_cam2tcp)
print("\n✅  Matriz cámara → TCP guardada en", SAVE_PATH)
print(T_cam2tcp)

np.savez("intrinsics.npz", K=K, dist=dist)
print("✅  Intrínseca y distorsión guardadas en intrinsics.npz")
