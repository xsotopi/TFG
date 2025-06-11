# pipeline_sam.py

import os
import time
import threading
import math

import cv2
import numpy as np
import torch
import whisper
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer
from segment_anything import SamPredictor, sam_model_registry

from send2robot2 import get_tcp_pose_persistent, send_pose_secondary_monitor
from ai.audio2text import transcribe_audio
from ai.detect      import detect_target_objects_realtime
from chats.chat_bot import extract_intent_and_object
from camera.cameras import capture_color_frame, capture_depth_frame, release_cameras

# ── CALIBRATION ───────────────────────────────────────────────────
CALIB_DIR = "/home/lab/Desktop/TFG/new_calibration"
K         = np.load(os.path.join(CALIB_DIR, "K.npy"))
dist      = np.load(os.path.join(CALIB_DIR, "distCoeffs.npy"))
T_cam2fl  = np.load(os.path.join(CALIB_DIR, "cam_to_flange.npy"))
print("[INFO] Loaded camera intrinsics and T_cam2flange.")
R_mat = T_cam2fl[:3, :3]
angles = cv2.Rodrigues(R_mat)[0].ravel() * 180 / np.pi
print("Rodrigues (deg):", angles)

t_fl_tcp = np.array([-1.6, 3.1, 212.8]) * 0.001      # → metres
R_fl_tcp, _ = cv2.Rodrigues(
    np.deg2rad([-0.0, 0.0, -139.93]).astype(np.float64)
)
T_fl_tcp = np.eye(4, dtype=np.float64)
T_fl_tcp[:3, :3] = R_fl_tcp
T_fl_tcp[:3,  3] = t_fl_tcp
T_tcp_fl = np.linalg.inv(T_fl_tcp)   # will be used later

# ── DEVICE SETUP ───────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ── LOAD MODELS ────────────────────────────────────────────────────
# 1) Whisper ASR
whisper_model = whisper.load_model("medium").to(device)
print("[INFO] Whisper loaded.")

# 2) YOLO detector
WEIGHTS = "/home/lab/Desktop/TFG/yolo_finetune/runs/yolov8_tools_highlighters/weights/best.pt"
yolo_model = YOLO(WEIGHTS)
yolo_model.to(device)
print("[INFO] YOLO loaded.")

# 3) SAM segmenter
sam_path  = "/home/lab/Desktop/TFG/sam_vit_b_01ec64.pth"
sam_model = sam_model_registry["vit_b"](checkpoint=sam_path)
sam_model.to(device)
sam_predictor = SamPredictor(sam_model)
print("[INFO] SAM loaded.")

# 4) Instruct‐tuned LLM for intent+object extraction
ckpt = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
slm_tokenizer = AutoTokenizer.from_pretrained(ckpt)
slm_model     = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
print("[INFO] LLM loaded.")

# ── CONSTANTS ──────────────────────────────────────────────────────
UPLOAD_FOLDER = "app/uploads"
AUDIO_EXT     = ".mp3"
pix_per_mm    = 3.2
max_width_mm  = 100
base_open_px  = int(max_width_mm * pix_per_mm)

TCP_TO_FINGERTIP = 0.030          # distance from TCP to jaw tips
APPROACH_GAP     = 0.010          # leave 10 mm air before closing
Z_OFFSET_PICK    = -(TCP_TO_FINGERTIP - APPROACH_GAP)   # ≈ -0.020

# ── GLOBAL STATE ───────────────────────────────────────────────────
pose_sent             = False
robot_thread          = None
current_intent        = None
current_target_object = None
show_detections       = True
show_depth            = False
latest_audio_file     = None


def overlay_mask(img: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.4):
    """Overlay a binary mask onto img in the given BGR color."""
    out = img.copy()
    col = np.full_like(img, color, dtype=np.uint8)
    return np.where(mask[:, :, None], cv2.addWeighted(col, alpha, out, 1 - alpha, 0), out)


def find_best_grasp(mask: np.ndarray,
                    max_width_px: int,
                    step_px: int = 3):
    """
    Return:
        (gx, gy)            – grasp centre in px (x-then-y)
        grasp_theta         – grasp angle in rad  (0 = → , π/2 = ↓)
        centroid            – global centroid (x, y)
        major_vec, minor_vec  – unit PCA axes (x, y)
    """
    # ── Basic checks ───────────────────────────────────────────────
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    ys, xs = np.where(mask > 0)                 # rows, cols
    if xs.size == 0:
        raise ValueError("Mask is empty – cannot compute grasp.")

    # ── PCA in (x, y) coords ───────────────────────────────────────
    pts_xy   = np.column_stack((xs, ys)).astype(np.float32)
    centroid = pts_xy.mean(axis=0)                  # (cx, cy)
    pts_c    = pts_xy - centroid                    # centred
    _, _, vt = np.linalg.svd(pts_c, full_matrices=False)
    major_vec, minor_vec = vt[0], vt[1]             # already unit
    grasp_theta = np.arctan2(minor_vec[1], minor_vec[0])

    # ── Half length of object along major axis ─────────────────────
    proj     = pts_c @ major_vec
    half_len = int(np.percentile(np.abs(proj), 95))

    # ── Slide candidate grip centre along major axis ───────────────
    best_score, best_center = -np.inf, None
    for offset in range(-half_len, half_len + 1, step_px):
        c  = centroid + offset * major_vec          # candidate (x, y)
        p1 = c + (max_width_px / 2) * minor_vec
        p2 = c - (max_width_px / 2) * minor_vec

        # sample mask along the line p1-p2
        line = np.linspace(p1, p2, max_width_px)
        xs_l = np.clip(line[:, 0].astype(int), 0, mask.shape[1] - 1)
        ys_l = np.clip(line[:, 1].astype(int), 0, mask.shape[0] - 1)
        in_mask_ratio = mask[ys_l, xs_l].mean()
        if in_mask_ratio > 0.8:
            torque = np.linalg.norm(c - centroid)
            score  = -torque
            if score > best_score:
                best_score, best_center = score, c

    if best_center is None:
        raise RuntimeError(
            "No valid grasp found – increase max_width_px or check mask."
        )

    gx, gy = best_center
    return (float(gx), float(gy)), float(grasp_theta), centroid, major_vec, minor_vec



def process_audio(path: str):
    """
    Transcribe the latest .mp3 via Whisper, extract intent+object using LLM,
    set global current_intent & current_target_object.
    """
    global current_intent, current_target_object, pose_sent, latest_audio_file

    transcript = transcribe_audio(whisper_model, path, "es")
    actions = extract_intent_and_object(transcript, slm_model, slm_tokenizer, device)
    print(f"[AUDIO] Transcript: {transcript} → Actions: {actions}")

    current_intent        = None
    current_target_object = None
    if actions and "pick" in actions and actions["pick"]:
        current_intent        = "pick"
        current_target_object = actions["pick"][0]

    pose_sent = False
    latest_audio_file = None
    os.remove(path)


def monitor_upload_folder():
    """
    Watches UPLOAD_FOLDER for new .mp3 files; whenever a new one arrives,
    calls process_audio() on it.
    """
    global latest_audio_file
    while True:
        time.sleep(1)
        files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(AUDIO_EXT)]
        if not files:
            continue
        newest = max(files, key=lambda f: os.path.getctime(os.path.join(UPLOAD_FOLDER, f)))
        if newest != latest_audio_file:
            latest_audio_file = newest
            process_audio(os.path.join(UPLOAD_FOLDER, newest))


def project_pixel_to_table(u: float, v: float, T_base_fl: np.ndarray):
    """
    Given a pixel (u, v), undistort → build ray in camera coords → transform to base → intersect with z=0
    Returns [X, Y, Zflange], or None if no valid intersection.
    """
    # 1) Undistort
    xn, yn = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, dist).reshape(-1)
    ray_cam = np.array([xn, yn, 1.0], dtype=np.float64)

    # 2) Build base ← cam transform
    T_base_cam = T_base_fl @ T_cam2fl
    R_bc = T_base_cam[:3, :3]
    O_bc = T_base_cam[:3, 3]

    # 3) Ray in base coords
    ray_base = R_bc @ ray_cam
    if abs(ray_base[2]) < 1e-6:
        return None
    s = -O_bc[2] / ray_base[2]
    if s < 0:
        return None

    table_z = 0.0            # base-frame height of your table surface (metres)
    P = O_bc + s * ray_base
    P[2] = table_z 
    return [float(P[0]), float(P[1]), float(P[2])]


def robot_communication_thread(pose):
    """
    Spawns send_pose_secondary_monitor(...) in its own thread, sending exactly one pose.
    """
    try:
        print(f"[↗] Sending pose to UR: {pose}")
        send_pose_secondary_monitor(pose)
    except Exception as e:
        print(f"[ERROR] robot_communication_thread: {e}")


def main():
    global pose_sent, robot_thread, show_detections, show_depth

    # 1) Create upload folder if needed, and start audio‐monitoring thread:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    threading.Thread(target=monitor_upload_folder, daemon=True).start()

    # 2) Create OpenCV windows
    cv2.namedWindow("Detection Feed", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Depth Camera Feed", cv2.WINDOW_NORMAL)
    print("Loop: q=quit, r=toggle detections")

    while True:
        color_frame = capture_color_frame()
        if color_frame is None:
            print("[ERROR] No color frame; exiting.")
            break

        depth_frame      = None
        depth_vis_color  = None
        if show_depth:
            depth_frame = capture_depth_frame()
            if depth_frame is not None:
                depth_vis = cv2.convertScaleAbs(depth_frame, alpha=0.05)
                depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        display_frame = color_frame.copy()

        # If user said “pick <object>”:
        if current_intent == "pick" and current_target_object:
            # 3) Run YOLO detection
            detections = detect_target_objects_realtime(
                yolo_model, display_frame, current_target_object, False
            )

            if detections:
                # 4) Pick the highest‐confidence detection
                last_det = max(detections, key=lambda d: d["confidence"])
                bbox = last_det["bbox"]
                conf = last_det["confidence"]
                x1, y1, x2, y2 = map(int, bbox)

                # 5) Draw bounding box
                if show_detections:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        f"{current_target_object}:{conf:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                    if show_depth and depth_vis_color is not None:
                        db = (int(x1 * depth_frame.shape[1] / display_frame.shape[1]),
                              int(y1 * depth_frame.shape[0] / display_frame.shape[0]),
                              int(x2 * depth_frame.shape[1] / display_frame.shape[1]),
                              int(y2 * depth_frame.shape[0] / display_frame.shape[0]))
                        cv2.rectangle(depth_vis_color, (db[0], db[1]), (db[2], db[3]), (0, 255, 0), 2)

                # 6) SAM segmentation on that bbox
                sam_predictor.set_image(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
                masks, _, _ = sam_predictor.predict(
                    box=np.array([x1, y1, x2, y2], dtype=np.float32),
                    multimask_output=False,
                )
                mask = masks[0]

                # 7) Compute “safe” gripper opening in px
                pts = np.column_stack(np.where(mask > 0))
                if pts.size > 0:
                    pts_center = pts - pts.mean(axis=0)
                    _, _, vt = np.linalg.svd(pts_center, full_matrices=False)
                    minor_vec = vt[1] / np.linalg.norm(vt[1])
                    proj_minor = pts_center @ minor_vec
                    width_minor_px = proj_minor.max() - proj_minor.min()
                    safe_open_px = max(1, int(width_minor_px - 4))
                    max_width_px = min(base_open_px, safe_open_px)
                else:
                    max_width_px = base_open_px

                # 8) Find best grasp (contour/PCA)
                try:
                    (gx, gy), theta, centroid_rc, major_vec, minor_vec = find_best_grasp(mask, max_width_px)
                except Exception as e:
                    print(f"[WARN] No valid grasp: {e}")
                    (gx, gy), theta, centroid_rc, major_vec, minor_vec = (None, None), None, None, None, None

                # 9) Overlay mask + draw axes
                vis = overlay_mask(display_frame, mask)
                if centroid_rc is not None:
                    c_x, c_y = map(int, centroid_rc)          # rename for clarity
                    L = max_width_px

                    p_major = (int(c_x + L * major_vec[0]),  int(c_y + L * major_vec[1]))
                    p_minor = (int(c_x + L * minor_vec[0]),  int(c_y + L * minor_vec[1]))

                    cv2.arrowedLine(vis, (c_x, c_y), p_major,  (0, 255, 255), 2)   # yellow
                    cv2.arrowedLine(vis, (c_x, c_y), p_minor, (0, 165, 255), 2)   # orange
                if gx is not None and gy is not None:
                    length = max_width_px // 2
                    dx     = math.cos(theta) * length          # Δx
                    dy     = math.sin(theta) * length          # Δy
                    pA     = (int(gx + dx), int(gy + dy))
                    pB     = (int(gx - dx), int(gy - dy))
                    cv2.line  (vis, pA, pB, (0, 255, 0), 3)    # green grasp line
                    cv2.circle(vis, (int(gx), int(gy)), 5, (0, 0, 255), -1)        # centre

                # 10) Show overlays
                cv2.imshow("Detection Feed", vis)
                if show_depth and depth_vis_color is not None:
                    cv2.imshow("Depth Camera Feed", depth_vis_color)

                # 11) Compute + send pose exactly once
                if not pose_sent and centroid_rc is not None:
                    # a) Read base←flange transform ONE time
                    try:
                        T_base_fl = get_tcp_pose_persistent()
                    except Exception as e:
                        print(f"[ERROR] Could not read RT pose: {e}")
                        continue

                    # b) Current roll, pitch from flange orientation
                    rvec_curr, _ = cv2.Rodrigues(T_base_fl[:3, :3])
                    roll_c, pitch_c, _ = rvec_curr.ravel()

                    # c) Project grasp endpoints + centroid
                    L_px = max_width_px / 2.0
                    u1, v1 = gx + L_px*math.cos(theta), gy + L_px*math.sin(theta)
                    u2, v2 = gx - L_px*math.cos(theta), gy - L_px*math.sin(theta)

                    P1 = project_pixel_to_table(u1, v1, T_base_fl)
                    P2 = project_pixel_to_table(u2, v2, T_base_fl)
                    Pcent = project_pixel_to_table(gx, gy, T_base_fl)
                    if P1 is None or P2 is None or Pcent is None:
                        print("[WARN] Projection failed; skipping pick.")
                        continue

                    # ----- NEW PART --------------------------------------------------------
                    Xb, Yb, Zf = Pcent            # keep original Z from your old code
                    dx3, dy3   = P2[0]-P1[0], P2[1]-P1[1]
                    yaw_des    = math.atan2(dy3, dx3)

                    yaw_tcp = yaw_des - math.pi/2 + math.radians(180.0)

                    cz, sz = math.cos(yaw_tcp), math.sin(yaw_tcp)
                    cy, sy = math.cos(pitch_c), math.sin(pitch_c)
                    cx, sx = math.cos(roll_c),  math.sin(roll_c)
                    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
                    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
                    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
                    R_tcp_des = Rz @ Ry @ Rx

                    # desired TCP pose
                    T_base_tcp_des = np.eye(4)
                    T_base_tcp_des[:3, :3] = R_tcp_des
                    T_base_tcp_des[:3, 3]  = [Xb, Yb, Zf]

                    # convert to flange pose with the constant inverse
                    T_base_fl_des = T_base_tcp_des @ T_tcp_fl
                    R_fl_des      = T_base_fl_des[:3, :3]
                    t_fl_des      = T_base_fl_des[:3, 3]
                    rvec_des, _   = cv2.Rodrigues(R_fl_des)

                    final_pose = [
                        float(t_fl_des[0]), float(t_fl_des[1]), float(t_fl_des[2]),
                        float(rvec_des[0]),  float(rvec_des[1]), float(rvec_des[2]),
                    ]
                    print(f"[INFO] Pose → X={t_fl_des[0]:.3f} Y={t_fl_des[1]:.3f} "
                        f"Z={t_fl_des[2]:.3f}  yawTCP={math.degrees(yaw_tcp):.1f}°")

                    robot_thread = threading.Thread(target=robot_communication_thread,
                                                    args=(final_pose,))
                    robot_thread.start()
                    pose_sent = True

            else:
                # No detections → just show raw
                cv2.imshow("Detection Feed", display_frame)
                if depth_vis_color is not None:
                    cv2.imshow("Depth Camera Feed", depth_vis_color)

        else:
            # No “pick” intent → show raw
            cv2.imshow("Detection Feed", display_frame)
            if depth_vis_color is not None:
                cv2.imshow("Depth Camera Feed", depth_vis_color)

        # ─── Key handling ──────────────────────────────────────────
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("r"):
            show_detections = not show_detections
            print("Detections", "ON" if show_detections else "OFF")
        elif k == ord("d"):
            show_depth = not show_depth
            if show_depth:
                cv2.namedWindow("Depth Camera Feed", cv2.WINDOW_NORMAL)
                print("Depth view ON")
            else:
                cv2.destroyWindow("Depth Camera Feed")
                print("Depth view OFF")

    # Cleanup
    release_cameras()
    cv2.destroyAllWindows()
    if robot_thread and robot_thread.is_alive():
        robot_thread.join(3)
    print("[INFO] Pipeline finished.")


if __name__ == "__main__":
    main()
