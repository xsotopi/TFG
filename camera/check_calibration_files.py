import numpy as np

# ---------- carga intrínsecos ----------
data = np.load("intrinsics.npz")
K    = data["K"]       # 3×3
dist = data["dist"]    # (5,) ó (8,)

print("\n=======  INTRINSICS  =======")
print("K (matriz 3×3):\n", K)
print("dist (coef. distorsión):", dist)

# ---------- carga mano-cámara ----------
T_cam2tcp = np.load("T_cam2tcp.npy").astype(np.float64)   # 4×4

print("\n=======  T_cam2tcp  =======")
print(T_cam2tcp)

# --- diagnóstico rápido ---
detR   = np.linalg.det(T_cam2tcp[:3,:3])
t_norm = np.linalg.norm(T_cam2tcp[:3,3])

print("\n=======  DIAGNÓSTICO  =======")
print(f"Determinante de R  : {detR:+.6f}  (≈ +1 implica rotación válida)")
print(f"Traslación módulo  : {t_norm:.1f}  (unidades *tal cual* están en el archivo)")
print("Traslación vector  :", T_cam2tcp[:3,3])
