#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprobación rápida de la calibración cámara-robot
— dibuja P0 y calcula la pose objetivo trasladando solo X-Y en marco herramienta —
"""

import numpy as np
import cv2 as cv
from pathlib import Path

# ---------- AJUSTES DEL TABLERO --------------------------------------------
CHESSBOARD_SIZE = (8, 7)           # 8×7 esquinas internas
SQUARE_SIZE     = 0.0127           # 12,7 mm  (ajusta al real)

# ---------- FICHEROS (NO TOCAR) --------------------------------------------
IMG_PATH        = "/home/lab/Desktop/TFG/img_00.png"
INTR_FILE       = "intrinsics.npz"
T_CAM2TCP_FILE  = "T_cam2tcp.npy"               # cámara → TCP
TCP2BASE_FILE   = "/home/lab/Desktop/TFG/pose_00.npy"
ANNOTATED_PATH  = "img_00_with_P0.png"

# ---------------------------------------------------------------------------

def load_calibration():
    with np.load(INTR_FILE) as f:
        return f["K"], f["dist"]

def detect_chessboard(gray):
    ok, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if not ok:
        raise RuntimeError("Tablero NO detectado en la imagen")
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

def build_objp():
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                          0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
    return objp * SQUARE_SIZE

# ---------------------------------------------------------------------------

def main():
    # ---------- carga de parámetros ----------
    K, dist         = load_calibration()

    T_tcp2cam       = np.load(T_CAM2TCP_FILE)          # ← guardaste cam→tcp
    if np.linalg.norm(T_tcp2cam[:3,3]) > 2.0:          # mm → m si hace falta
        T_tcp2cam[:3,3] *= 1e-3
    T_cam2tcp       = np.linalg.inv(T_tcp2cam)         # cámara → TCP

    T_tcp2base      = np.load(TCP2BASE_FILE)           # pose actual TCP

    # ---------- lee imagen ----------
    if not Path(IMG_PATH).is_file():
        raise FileNotFoundError(IMG_PATH)
    img  = cv.imread(IMG_PATH)
    if img is None:
        raise IOError(f"No se pudo cargar {IMG_PATH}")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ---------- detecta tablero y marca P0 ----------
    corners = detect_chessboard(gray)
    corner0 = tuple(corners[0].ravel().astype(int))
    cv.circle(img, corner0, 12, (0,0,255), 3)
    cv.putText(img, "P0", (corner0[0]+15, corner0[1]-10),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    cv.imwrite(ANNOTATED_PATH, img)
    print(f"✅  Imagen anotada guardada en '{ANNOTATED_PATH}'")

    # ---------- solvePnP ----------
    objp             = build_objp()
    ok, rvec, tvec   = cv.solvePnP(objp, corners, K, dist)
    if not ok:
        raise RuntimeError("solvePnP ha fallado")

    R_cam_board, _   = cv.Rodrigues(rvec)
    pt_cam           = R_cam_board @ objp[0] + tvec.ravel()

    print("\nP0 en marco CÁMARA [m]:",
        f"x={pt_cam[0]:.4f}  y={pt_cam[1]:.4f}  z={pt_cam[2]:.4f}")

    # ---------- vector (x,y) en cámara  → herramienta → base -------------------
    R_cam2tcp   = T_cam2tcp[:3, :3]            # solo rotación
    vec_cam_xy  = np.array([pt_cam[0], pt_cam[1], 0.0])    # z=0

    delta_tool  = R_cam2tcp @ vec_cam_xy       # (Δx,Δy,0) en marco TCP
    delta_base  = T_tcp2base[:3, :3] @ delta_tool

    delta_base[1] *= -1

    # ---------- nueva pose -----------------------------------------------------
    target_pose           = T_tcp2base.copy()  # misma R y Z
    target_pose[:3, 3]   += delta_base         # solo X-Y deseados

    # ---------- imprime --------------------------------------------------------
    print("\n--- POSICIONES EN BASE DEL ROBOT (m) ---")
    print(f"Pose actual  TCP: x={T_tcp2base[0,3]:.4f}  y={T_tcp2base[1,3]:.4f}  z={T_tcp2base[2,3]:.4f}")
    print(f"Pose objetivo:   x={target_pose[0,3]:.4f}  y={target_pose[1,3]:.4f}  z={target_pose[2,3]:.4f}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
