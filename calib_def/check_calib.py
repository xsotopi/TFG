#!/usr/bin/env python3
"""
Print the base-frame pose that would place the TCP origin (flange / tool frame)
exactly on the FIRST inner chessboard corner (blue dot).  No motion is sent.

• SPACE → compute & print pose
• ESC   → quit
"""
import cv2, numpy as np, json, socket, struct, math
from pathlib import Path

# ───── adjustable parameters ─────
UR_IP       = "192.168.0.102"
CAM_ID      = 6
BOARD_SIZE  = (8, 7)          # cols, rows – inner corners
SQUARE_M    = 0.015           # 15 mm
SAFETY_Z    = 0          # lift TCP +Z by 3 cm after aligning
CAL_FILE    = Path("calib_results/hand_eye.json")
TARGET_ID   = 0               # 0 = top-left inner corner
# ──────────────────────────────────

# ───── load calibration data ─────
cal = json.load(open(CAL_FILE))
K    = np.array(cal["camera_matrix"])
dist = np.array(cal["dist_coeffs"])
R_ct = cv2.Rodrigues(np.array(cal["rvec_cam2tcp"]))[0]
t_ct = np.array(cal["tvec_cam2tcp"]).reshape(3,1)

# helpers -----------------------------------------------------------
def tcp_pose_rt():
    with socket.create_connection((UR_IP, 30003), timeout=0.4) as s:
        hdr = s.recv(4); (ln,) = struct.unpack(">I", hdr)
        pkt = s.recv(ln-4, socket.MSG_WAITALL)
    return struct.unpack(">6d", pkt[440:488])

def posevec_to_T(p):
    x,y,z,rx,ry,rz = p
    ang = math.sqrt(rx*rx+ry*ry+rz*rz)
    rvec = np.array([rx,ry,rz], dtype=np.float64)
    R = np.eye(3) if ang<1e-9 else cv2.Rodrigues(rvec)[0]
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=(x,y,z)
    return T

def T_to_posevec(T):
    rvec,_ = cv2.Rodrigues(T[:3,:3])
    return np.r_[T[:3,3], rvec.flat]

# object-space corners ----------
objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:BOARD_SIZE[0],0:BOARD_SIZE[1]].T.reshape(-1,2)*SQUARE_M

cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened(): raise RuntimeError("Camera error")

print("SPACE = compute pose | ESC = quit\n")

while True:
    ok, frame = cap.read();  show = frame.copy()
    if not ok: continue
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        g, BOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

    if found:
        # draw chessboard and highlight the chosen corner
        cv2.drawChessboardCorners(show, BOARD_SIZE, corners, True)
        x,y = corners[TARGET_ID].ravel().astype(int)
        cv2.circle(show, (x,y), 8, (255,0,0), -1)

    cv2.putText(show, "Detected" if found else "No board",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0) if found else (0,0,255), 2)
    cv2.imshow("Checker", show)

    k = cv2.waitKey(5) & 0xFF
    if k == 27: break               # ESC
    if k == 32 and found:           # SPACE
        # camera → board pose
        _, rvec_cb, tvec_cb = cv2.solvePnP(objp, corners, K, dist)
        R_cb,_ = cv2.Rodrigues(rvec_cb)

        # corner expressed in camera frame
        p_cam = R_cb @ objp[TARGET_ID].reshape(3,1) + tvec_cb

        # corner expressed in tcp frame
        p_tcp = R_ct @ p_cam + t_ct        # 3×1

        # live tcp → base
        # --- live TCP → base --------------------------------------------
        tcp_vec = tcp_pose_rt()                 # 6-vector at this instant
        T_tb    = posevec_to_T(tcp_vec)         # 4×4 matrix

        delta_base = T_tb[:3, :3] @ p_tcp          # (Δx,Δy,Δz) en base
        delta_base[1, 0] *= -1        
        # --- compute NEW position (TCP on corner, +safety) --------------
        p_base = T_tb[:3,3,None] - delta_base

        tcp_z_base = T_tb[:3,:3] @ np.array([[0],[0],[1]])
        p_base += SAFETY_Z * tcp_z_base
        

        # keep orientation as is
        T_goal = T_tb.copy();  T_goal[:3,3] = p_base.ravel()

        # convert both poses back to 6-vectors
        pose_goal = T_to_posevec(T_goal)
        pose_curr = np.array(tcp_vec)

        # ─── NEW : info about the corner in the camera frame ────────────
        Xc, Yc, Zc = p_cam.ravel()             # metres
        pix, _     = cv2.projectPoints(p_cam, np.zeros((3,1)), np.zeros((3,1)),
                                       K, dist)
        u, v = pix.ravel()                     # pixels
        # ────────────────────────────────────────────────────────────────

        print("\nCURRENT TCP pose:")
        print(f"p[{pose_curr[0]:.4f}, {pose_curr[1]:.4f}, {pose_curr[2]:.4f}, "
              f"{pose_curr[3]:.4f}, {pose_curr[4]:.4f}, {pose_curr[5]:.4f}]")

        print("SUGGESTED TCP pose (corner + safety):")
        print(f"p[{pose_goal[0]:.4f}, {pose_goal[1]:.4f}, {pose_goal[2]:.4f}, "
              f"{pose_goal[3]:.4f}, {pose_goal[4]:.4f}, {pose_goal[5]:.4f}]")

        # ─── NEW : print corner location in camera frame ────────────────
        print(f"Corner in camera frame  (m):  X={Xc:.4f}  Y={Yc:.4f}  Z={Zc:.4f}")
        print(f"Corner projects to pixel     :  u={u:.1f}  v={v:.1f}\n")
        # ────────────────────────────────────────────────────────────────



cap.release(); cv2.destroyAllWindows()
