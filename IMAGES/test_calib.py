import cv2, json, glob, numpy as np, tqdm
from pathlib import Path

DS      = Path("/home/lab/Desktop/TFG/new_calibration/dataset_flange")
K       = np.load("K.npy")
dist    = np.load("distCoeffs.npy")
T_cf    = np.load("cam_to_flange.npy")         # cam→flange

# -- helper -----------------------------------------------------------
def pose_to_Rt(p):       #  *inverted* base→flange  →  flange→base
    x,y,z, rx,ry,rz = p
    Rb2f,_ = cv2.Rodrigues(np.array([rx,ry,rz],np.float64))
    tb2f   = np.array([x,y,z],np.float64)
    Rf2b   = Rb2f.T
    tf2b   = -Rf2b @ tb2f
    return Rf2b, tf2b

R_cf, t_cf = T_cf[:3,:3], T_cf[:3,3]

# chessboard model (8×7, 15 mm)
objp = np.zeros((8*7,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:7].T.reshape(-1,2) * 0.015

errs = []
for img_path in tqdm.tqdm(sorted((DS/"images").glob("*.png"))):
    img   = cv2.imread(str(img_path))
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, rvec_bc, tvec_bc = cv2.solvePnP(objp,           # board→cam
                                        cv2.findChessboardCorners(
                                            gray,(8,7))[1],
                                        K, dist, flags=cv2.SOLVEPNP_IPPE)

    if not ok: continue
    R_bc,_ = cv2.Rodrigues(rvec_bc); t_bc = tvec_bc.ravel()

    # load flange→base for this frame
    pose_json = next((DS/"poses").glob(img_path.stem.replace("img","pose")+".json"))
    R_fb, t_fb = pose_to_Rt(json.loads(pose_json.read_text())["pose_tcp"])

    # full chain: board-pt → … → img pixels
    proj, _ = cv2.projectPoints(objp, rvec_bc, tvec_bc, K, dist)
    proj = proj.squeeze()

    err = np.linalg.norm(proj -
                         cv2.findChessboardCorners(gray,(8,7))[1].squeeze(),
                         axis=1).mean()
    errs.append(err)

print("mean reprojection error  =  {:.2f} px".format(np.mean(errs)))
print("max  reprojection error  =  {:.2f} px".format(np.max(errs)))
