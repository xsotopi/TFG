from __future__ import annotations
import json, cv2, numpy as np
from pathlib import Path
import tqdm, itertools, math, sys

# CONFIG ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR= SCRIPT_DIR / "dataset_flange_hd"
BOARD_SIZE = (8,7)
SQUARE_LEN_M = 0.015
SIGMA_THRS = 0.5
MIN_FRAMES = 10
METHOD = cv2.CALIB_HAND_EYE_TSAI

IMG_DIR = DATASET_DIR/"images"
POSE_DIR=DATASET_DIR/"poses"
RES_DIR = DATASET_DIR/"calib_results"
RES_DIR.mkdir(exist_ok=True)

def skew(w):
    """ Creates a skew-symmetric matrix from a 3D vector w. """
    x,y,z=w
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],dtype=w.dtype)

def se3_log_norm(T):
    """
    Calculate norm of the logratihm of a 4x4 SE(3) transformation matrix.
    Returns a scalar value representing the distance of the transformation
    from the identity. AX - BX transformation. 
    """
    R=T[:3,:3]
    t=T[:3,3]
    th=math.acos(max(-1,min(1,(np.trace(R)-1)/2)))
    if th<1e-9:
        w=0.5*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
        Vinv=np.eye(3)-0.5*skew(w)
    else:
        w=(th/(2*math.sin(th)))*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
        A=math.sin(th)/th
        B=(1-math.cos(th))/th**2
        Vinv=np.eye(3)-0.5*skew(w)+(1/th**2)*(1-A/(2*B))*skew(w)@skew(w)
    v=Vinv@t
    return float(np.linalg.norm(np.hstack((w,v))))

def obj_corners():
    """Generate ideal 3D coordinates of the chessboard corners in the object space."""
    g=np.mgrid[0:BOARD_SIZE[0],0:BOARD_SIZE[1]].T.reshape(-1,2).astype(np.float32)
    pts=np.zeros((len(g),3),np.float32)
    pts[:,:2]=g*SQUARE_LEN_M
    return pts


def rt_to_T(r,t):
    """Convert rotation vector r and translation vector t to a 4x4 transformation matrix."""
    R,_=cv2.Rodrigues(r)
    T=np.eye(4)
    T[:3,:3]=R
    T[:3,3]=t.ravel()
    return T

# Load data and detect corners.
print("Scanning", IMG_DIR)
paths=sorted(IMG_DIR.glob("img_*.png"))

if not paths: 
    sys.exit("✗ no images found")

objp=obj_corners()
objpts=[] 
imgpts=[] 
rv_r=[] 
tv_r=[] 
keep_names=[]

for p in tqdm.tqdm(paths,desc="detect"):
    im=cv2.imread(str(p))
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ok,c=cv2.findChessboardCorners(gray,BOARD_SIZE,cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok: 
        continue
    c=cv2.cornerSubPix(gray,c,(11,11),(-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.1))
    objpts.append(objp)
    imgpts.append(c)
    keep_names.append(p.stem)
    tcp=np.array(json.loads((POSE_DIR/(p.stem.replace('img','pose')+'.json')).read_text())['pose_tcp'],np.float64)
    rv_r.append(tcp[3:].reshape(3,1))
    tv_r.append(tcp[:3,None])

if len(objpts)<MIN_FRAMES: 
    sys.exit("✗ not enough detections")

img_size=cv2.imread(str(paths[0])).shape[1::-1]

# First calibration with all frames.
_,K,dist,rv_c,tv_c=cv2.calibrateCamera(objpts,imgpts,img_size,None,None)
reproj=np.array([cv2.norm(imgpts[i],cv2.projectPoints(objpts[i],rv_c[i],tv_c[i],K,dist)[0],cv2.NORM_L2)/len(objpts[i]) for i in range(len(objpts))])
R,t=cv2.calibrateHandEye(rv_r,tv_r,rv_c,tv_c,method=METHOD)
X=np.eye(4)
X[:3,:3]=R
X[:3,3]=t.ravel()

# Calculate AX=BX error.
ax_frame=[0.0]+[se3_log_norm(rt_to_T(rv_r[i],tv_r[i]) @ np.linalg.inv(rt_to_T(rv_r[0],tv_r[0])) @ X - X @ rt_to_T(rv_c[i],tv_c[i]) @ np.linalg.inv(rt_to_T(rv_c[0],tv_c[0]))) for i in range(1,len(rv_r))]
ax_frame=np.array(ax_frame)

# Prune outliers
mask=(reproj<reproj.mean()+SIGMA_THRS*reproj.std()) & (ax_frame<ax_frame.mean()+SIGMA_THRS*ax_frame.std())
print(f"Discarding {len(mask)-mask.sum()} frames beyond {SIGMA_THRS} σ")
if mask.sum()<MIN_FRAMES: 
    sys.exit("would fall below MIN_FRAMES – abort")

objpts=[o for o,m in zip(objpts,mask) if m]
imgpts=[i for i,m in zip(imgpts,mask) if m]
rv_r  =[r for r,m in zip(rv_r,mask) if m]
tv_r  =[t for t,m in zip(tv_r,mask) if m]
keep_names=[n for n,m in zip(keep_names,mask) if m]

# Recalibrate with pruned data
_,K,dist,rv_c,tv_c=cv2.calibrateCamera(objpts,imgpts,img_size,None,None)
R,t=cv2.calibrateHandEye(rv_r,tv_r,rv_c,tv_c,method=METHOD)
X[:3,:3]=R
X[:3,3]=t.ravel()

# Final AX=BX error calculation
A_list,B_list=[],[]
for i,j in itertools.combinations(range(len(rv_r)),2):
    A_list.append(rt_to_T(rv_r[j],tv_r[j]) @ np.linalg.inv(rt_to_T(rv_r[i],tv_r[i])))
    B_list.append(rt_to_T(rv_c[j],tv_c[j]) @ np.linalg.inv(rt_to_T(rv_c[i],tv_c[i])))
ax_pair=np.array([se3_log_norm(A@X - X@B) for A,B in zip(A_list,B_list)])
print(f"Final pair-wise mean ‖AX-XB‖ = {ax_pair.mean():.4f}")

# Save results
np.save(RES_DIR/"K.npy",K)
np.save(RES_DIR/"distCoeffs.npy",dist)
np.save(RES_DIR/"cam_to_tcp.npy",X)

json.dump({
    "camera_matrix":K.tolist(),
    "dist_coeffs":dist.ravel().tolist(),
    "rvec_cam2tcp":cv2.Rodrigues(X[:3,:3])[0].ravel().tolist(),
    "tvec_cam2tcp":X[:3,3].ravel().tolist(),
    "pairwise_mean":float(ax_pair.mean()),
    "frames_kept":keep_names
}, open(RES_DIR/"hand_eye.json","w"), indent=2)
print("✅ saved to",RES_DIR)
