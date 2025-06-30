from pathlib import Path
import json, cv2, numpy as np, tqdm

DATASET = Path("dataset_flange_hd")
BOARD   = (8, 7)
SQUARE = 0.015

# load calibration
calib_dir = DATASET / "calib_results"
cfg       = json.loads((calib_dir / "hand_eye.json").read_text())
keep      = set(cfg["frames_kept"])

K         = np.load(calib_dir / "K.npy")
dist      = np.load(calib_dir / "distCoeffs.npy")
T_cam_tcp = np.load(calib_dir / "cam_to_tcp.npy")

def rt_to_T(r, t):
    """Convert rotation vector r and translation vector t to a 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(r)
    T    = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t.ravel()
    return T

# Generate 3D model points for chessboard
objp = np.zeros((BOARD[0]*BOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * SQUARE

Ts_board_base = []

for img_p in tqdm.tqdm(sorted((DATASET / "images").glob("img_*.png")), desc="eval"):
    if img_p.stem not in keep:
        continue

    img  = cv2.imread(str(img_p))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ok, corners = cv2.findChessboardCorners(gray, BOARD)
    if not ok:
        continue
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

    ok, rvec_bc, tvec_bc = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        continue

    T_bc = rt_to_T(rvec_bc, tvec_bc)          # board → camera

    pose_json = DATASET / "poses" / (img_p.stem.replace("img", "pose") + ".json")
    tcp = np.array(json.loads(pose_json.read_text())["pose_tcp"], np.float64)
    T_tcp_base = rt_to_T(tcp[3:], tcp[:3])    # tcp → base

    T_board_base = T_tcp_base @ T_cam_tcp @ T_bc
    Ts_board_base.append(T_board_base)

Ts_board_base = np.stack(Ts_board_base)

# --- translation (mm)
origins = Ts_board_base[:, :3, 3]
o_t_mm  = np.linalg.norm(origins - origins.mean(0), axis=1).std() * 1000

# --- rotation (deg)
rvecs = np.array([cv2.Rodrigues(T[:3, :3])[0].ravel()
                  for T in Ts_board_base])
o_r_deg = np.linalg.norm(rvecs - rvecs.mean(0), axis=1).std() * 180 / np.pi

print("\n--- Validation Results ---")
print(f"Frames used: {len(Ts_board_base)}")
print(f"Translation: {o_t_mm:.2f}  mm")
print(f"Rotation: {o_r_deg:.2f} °")
