#!/usr/bin/env python3
"""
Hand‑eye calibration **completa** para un UR + cámara en la muñeca
=================================================================

Cambios clave respecto a la versión original
-------------------------------------------
1. **Chequeo y corrección automática de unidades** (m ↔ mm) entre las
   traslaciones de la cámara (`cam_tvecs`) y del robot (`robot_tvecs`).
2. Construcción rigurosa de las matrices de movimiento **A** y **B** en SE(3)
   (no se resta directamente `rvec`/`tvec`).
3. Métrica ‖AX − XB‖ calculada con log(SE3).

Requisitos
----------
```bash
pip install opencv-python numpy scipy
```
"""
from __future__ import annotations
import json, math, cv2, numpy as np
from pathlib import Path
from typing import List
from scipy.linalg import logm

# ───────────── CONFIG ─────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
DATASET_DIR  = SCRIPT_DIR / "dataset_flange_hd"      # carpeta con images/ y poses/
BOARD_SIZE   = (8, 7)     # inner corners (cols, rows)
SQUARE_LEN_M = 0.015      # lado del cuadrado en **metros**
VISUALIZE    = False       # mostrar cada detección
MAX_WIDTH    = 960        # px máx. ventana
METHOD       = cv2.CALIB_HAND_EYE_TSAI        # o .DANIILIDIS, .HORAUD…
RESULT_DIR   = SCRIPT_DIR / "calib_results"
RESULT_DIR.mkdir(exist_ok=True)
# ──────────────────────────────────

IMG_DIR   = DATASET_DIR / "images"
POSES_DIR = DATASET_DIR / "poses"
WIN       = "Detección"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

# ───────── utilidades ─────────

def make_objp():
    grid = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1,2).astype(np.float32)
    obj = np.zeros((len(grid),3), np.float32)
    obj[:,:2] = grid * SQUARE_LEN_M
    return obj


def load_pose(stem: str):
    """Devuelve rvec, tvec TOMADOS DEL ROBOT (TCP→base)."""
    jpath = POSES_DIR / (stem.replace("img_", "pose_") + ".json")
    if not jpath.exists():
        raise FileNotFoundError(jpath)
    tcp = np.array(json.load(open(jpath))['pose_tcp'], np.float64)
    x, y, z, rx, ry, rz = tcp
    rvec = tcp[3:].reshape(3,1)                      # UR ya devuelve formato angle‑axis (rad)
    tvec = np.array([[x], [y], [z]], np.float64)     # el usuario decide la unidad 🙂
    return rvec, tvec


def show(img, lbl, col):
    h, w = img.shape[:2]
    scale = min(1.0, MAX_WIDTH / w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), cv2.INTER_AREA)
    cv2.putText(img, lbl, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2, cv2.LINE_AA)
    cv2.imshow(WIN, img)


def vec_rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convierte (rvec, tvec) en matriz homogénea 4×4."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tvec.ravel()
    return T


def log_se3(R: np.ndarray, t: np.ndarray) -> float:
    """Norma de la se(3) resultante (6‑D) ‖ξ‖."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t.ravel()
    Xi = logm(T)
    return np.linalg.norm([Xi[2,1], Xi[0,2], Xi[1,0], Xi[0,3], Xi[1,3], Xi[2,3]])

# ───────── carga datos ─────────
print("▶︎ Buscando imágenes en", IMG_DIR)
img_paths = sorted(IMG_DIR.glob("img_*.png"))
if not img_paths:
    raise SystemExit("No PNG en " + str(IMG_DIR))

objp_single = make_objp()
objpoints: List[np.ndarray] = []
imgpoints: List[np.ndarray] = []
robot_rvecs: List[np.ndarray] = []
robot_tvecs: List[np.ndarray] = []

for p in img_paths:
    img = cv2.imread(str(p))
    if img is None:
        print("⚠️  No se abre", p.name); continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, BOARD_SIZE,
                    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    lbl, col = ("OK", (0,255,0)) if ok else ("FAIL", (0,0,255))

    if VISUALIZE:
        vis = img.copy()
        if ok:
            cv2.drawChessboardCorners(vis, BOARD_SIZE, corners, True)
        show(vis, lbl, col)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            VISUALIZE = False
            cv2.destroyWindow(WIN)

    if not ok:
        continue

    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    objpoints.append(objp_single)
    imgpoints.append(corners)
    rv, tv = load_pose(p.stem)
    robot_rvecs.append(rv)
    robot_tvecs.append(tv)

n = len(objpoints)
if n < 10:
    raise SystemExit(f"Sólo {n} capturas válidas; se necesitan ≥10.")
print(f"✔ Se usarán {n} imágenes con tablero detectado.")

# ───────── intrínsecos ─────────
print("▶︎ calibrateCamera…")
ret, K, dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("  RMS intrínsecos =", ret)

# Error reproyección cámara
err_cam = 0.0
for i in range(n):
    proj,_ = cv2.projectPoints(objp_single, cam_rvecs[i], cam_tvecs[i], K, dist)
    err_cam += cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
err_cam /= n
print(f"  Error reproyección intrínsecos ≈ {err_cam:.3f} px")

# ───────── unidad check ─────────
print("\n▶︎ chequeando unidades …")
scales = []
for i in range(1, n):
    d_cam   = np.linalg.norm(cam_tvecs[i]   - cam_tvecs[0])
    d_robot = np.linalg.norm(robot_tvecs[i] - robot_tvecs[0])
    if d_cam > 1e-9:
        scales.append(d_robot / d_cam)
mean_scale = float(np.mean(scales))
print(f"  Escala robot / cámara ≈ {mean_scale:.3f}")

if mean_scale > 10:                      # cámara en milímetros
    print("  ⚠️  Detectados cam_tvecs en mm  → dividiendo por 1000")
    cam_tvecs = [tv / 1000.0 for tv in cam_tvecs]
elif mean_scale < 0.1:                   # robot en milímetros
    print("  ⚠️  Detectados robot_tvecs en mm → dividiendo por 1000")
    robot_tvecs = [tv / 1000.0 for tv in robot_tvecs]
else:
    print("  ✔ Unidades coherentes (metros)")

# ───────── hand–eye ───────────
print("\n▶︎ calibrateHandEye …")
R_cam_tcp, t_cam_tcp = cv2.calibrateHandEye(robot_rvecs, robot_tvecs,
                                            cam_rvecs,   cam_tvecs,
                                            method=METHOD)
print("  → hand‑eye OK")

t_cam_tcp = t_cam_tcp.reshape(3,1)  # 3×1 columna

# ───────── chequeo AX = XB ───────
print("\n▶︎ chequeo ‖AX−XB‖ …")

T_tcp = [vec_rt_to_T(r, t) for r, t in zip(robot_rvecs, robot_tvecs)]
T_cam = [vec_rt_to_T(r, t) for r, t in zip(cam_rvecs,  cam_tvecs)]

X = np.eye(4)
X[:3, :3] = R_cam_tcp
X[:3, 3]  = t_cam_tcp.ravel()

ax_xb = []
for i in range(1, n):
    A = T_tcp[i] @ np.linalg.inv(T_tcp[0])
    B = T_cam[i] @ np.linalg.inv(T_cam[0])

    Res = A @ X - X @ B
    ax_xb.append(log_se3(Res[:3,:3], Res[:3,3,None]))

ax_xb = np.array(ax_xb)
print(f"  ‖AX−XB‖   media = {ax_xb.mean():.4e}   desv = {ax_xb.std():.4e}")

# ───────── guardar resultados ─────────
RESULT_FILE = RESULT_DIR / "hand_eye.json"
output = {
    "camera_matrix"   : K.tolist(),
    "dist_coeffs"      : dist.ravel().tolist(),
    "rvec_cam2tcp"     : cv2.Rodrigues(R_cam_tcp)[0].ravel().tolist(),
    "tvec_cam2tcp"     : t_cam_tcp.ravel().tolist(),
    "rms_intrinsics_px": float(ret),
    "reproj_error_px"  : float(err_cam),
    "ax_xb_mean"       : float(ax_xb.mean()),
    "ax_xb_std"        : float(ax_xb.std())
}
with open(RESULT_FILE, "w") as f:
    json.dump(output, f, indent=2)
print("\n✅ Resultados guardados en", RESULT_FILE)

cv2.destroyAllWindows()
