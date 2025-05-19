import numpy as np
import glob
import os
import cv2

# Carpeta donde guardaste los .npy de las poses
POSE_DIR = "poses"

def matrix_to_pose6(T):
    """
    Convierte una matriz 4×4 [R|t] en un vector [x,y,z, rx,ry,rz]
    donde [rx,ry,rz] es el vector de ángulo de Rodrigues, con signo
    invertido para coincidir con lo que muestra Polyscope.
    """
    t = T[:3, 3]
    R = T[:3, :3]

    # vector de Rodrigues
    rvec, _ = cv2.Rodrigues(R)

    # ── AQUÍ INVIRTIENDO EL SIGNO ─────────────────────────────────────
    rvec = -rvec

    rx, ry, rz = rvec.ravel()
    return [float(t[0]), float(t[1]), float(t[2]),
            float(rx), float(ry), float(rz)]

def main():
    files = sorted(glob.glob(os.path.join(POSE_DIR, "pose_01.npy")))
    if not files:
        print("❌ No hay archivos .npy en", POSE_DIR)
        return

    print(f"✅ Encontrados {len(files)} ficheros de pose.\n")
    for fn in files:
        T = np.load(fn)
        pose6 = matrix_to_pose6(T)
        # Además, si quieres ver los ángulos en grados:
        degs = np.degrees(pose6[3:6])
        print(f"{os.path.basename(fn)}:")
        print("  Matriz T =\n", T)
        print("  Pose 6d [x,y,z,rx,ry,rz] (rad) =", 
              [f"{x:.4f}" for x in pose6])
        print("  Rotación en grados    =", 
              [f"{d:.2f}°" for d in degs])
        print()

if __name__ == "__main__":
    main()
