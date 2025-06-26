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

from robot.send2robot import get_tcp_pose_persistent, start_pose_server, pose_queue
from audio.audio2text import transcribe_audio
from detection.detect      import detect_target_objects_realtime
from command_extraction.chats.chat_bot import extract_intent_and_object
from cameras.cameras import capture_color_frame, capture_depth_frame, release_cameras
from segmentation.utils import overlay_mask, find_best_grasp
from time_calculation.timings import time_block


start_pose_server()
# ── CALIBRATION ───────────────────────────────────────────────────
with time_block("load_calibration"):
    CALIB_DIR = "/home/lab/Desktop/TFG/calib_def/dataset_flange_hd/calib_results"
    K    = np.load(os.path.join(CALIB_DIR, "K.npy"))
    dist = np.load(os.path.join(CALIB_DIR, "distCoeffs.npy"))
    T_cam2tcp = np.load(os.path.join(CALIB_DIR, "cam_to_tcp.npy"))
    t_fl_tcp = np.array([-1.6, 3.1, 212.8]) * 0.001   # metres
    R_fl_tcp, _ = cv2.Rodrigues(np.deg2rad([0, 0, -139.93]).astype(np.float64))
    T_fl_tcp            = np.eye(4, dtype=np.float64)
    T_fl_tcp[:3, :3]    = R_fl_tcp
    T_fl_tcp[:3,  3]    = t_fl_tcp
    T_tcp_fl            = np.linalg.inv(T_fl_tcp)

    T_cam2fl = T_cam2tcp @ T_tcp_fl

    print("Cam origin in flange (m):", T_cam2fl[:3, 3])
    print("Cam optical axis in flange:", T_cam2fl[:3, 2])



    print("[INFO] Loaded camera intrinsics and T_cam2flange.")
    

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ── LOAD MODELS ────────────────────────────────────────────────────
# 1) Whisper - transcription
with time_block("load_whisper"):
    whisper_model = whisper.load_model("medium").to(device)
    print("[INFO] Whisper loaded.")

# 2) YOLO - detection
with time_block("load_yolo"):
    WEIGHTS = "/home/lab/Desktop/TFG/runs/yolov8_6class/weights/best.pt"
    yolo_model = YOLO(WEIGHTS)
    yolo_model.to(device)
    print("[INFO] YOLO loaded.")

# 3) SAM - segmentation
with time_block("load_sam"):
    sam_path  = "/home/lab/Desktop/TFG/sam_vit_b_01ec64.pth"
    sam_model = sam_model_registry["vit_b"](checkpoint=sam_path)
    sam_model.to(device)
    sam_predictor = SamPredictor(sam_model)
    print("[INFO] SAM loaded.")

# 4) SLM - command extraction
# with time_block("load_slm"):
#     ckpt = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
#     slm_tokenizer = AutoTokenizer.from_pretrained(ckpt)
#     slm_model     = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
#     print("[INFO] LLM loaded.")

with time_block("load_slm"):
    ckpt = "Qwen/Qwen3-0.6B"
    slm_tokenizer = AutoTokenizer.from_pretrained(ckpt)
    slm_model     = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
    print("[INFO] LLM loaded.")

# ── CONSTANTS ──────────────────────────────────────────────────────
UPLOAD_FOLDER = "/home/lab/Desktop/TFG/app/uploads"
AUDIO_EXT     = ".mp3"
pix_per_mm    = 3.2
max_width_mm  = 100
base_open_px  = int(max_width_mm * pix_per_mm)

# ── GLOBAL STATE ───────────────────────────────────────────────────
pose_sent             = False
robot_thread          = None
current_intent        = None
current_target_object = None
show_detections       = True
show_depth            = False
latest_audio_file     = None

def process_audio(path: str):
    """
    Transcribe the latest .mp3 via Whisper, extract intent+object using LLM,
    set global current_intent & current_target_object.
    """
    global current_intent, current_target_object, pose_sent, latest_audio_file

    with time_block("whisper_transcribe"):
        language_code = os.path.splitext(os.path.basename(path))[0]
        transcript = transcribe_audio(whisper_model, path, language_code)

    with time_block("SLM_extract_intent_and_object"):
        actions = extract_intent_and_object(transcript, slm_model, slm_tokenizer, device)
        for key, value in actions.items():
            if isinstance(value, list):
                actions[key] = [v.lower() for v in value]
            else:
                actions[key] = value.lower()
    
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

def project_pixel_to_table(u: float, v: float, T_base_tcp: np.ndarray):
    """
    Given a pixel (u, v), undistort → build ray in camera coords → transform to base → intersect with z=0
    Returns [X, Y, Zflange], or None if no valid intersection.
    """
    xn, yn = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, dist).reshape(-1)
    ray_cam = np.array([xn, yn, 1.0], dtype=np.float64)

    T_base_cam = T_base_tcp @ T_cam2tcp

    R_bc = T_base_cam[:3, :3]
    O_bc = T_base_cam[:3, 3]

    ray_base = R_bc @ ray_cam
    if abs(ray_base[2]) < 1e-6:
        return None
    s = -O_bc[2] / ray_base[2]
    if s < 0:
        return None

    table_z = 0.0
    P = O_bc + s * ray_base
    P[2] = table_z 
    return [float(P[0]), float(P[1]), float(P[2])]


def main():
    global pose_sent, robot_thread, show_detections, show_depth

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    threading.Thread(target=monitor_upload_folder, daemon=True).start()

    cv2.namedWindow("Detection Feed", cv2.WINDOW_NORMAL)
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

        if current_intent == "pick" and current_target_object:
            with time_block("yolo_detection"):
                detections = detect_target_objects_realtime(
                    yolo_model, display_frame, current_target_object, False
                )

            if detections:
                last_det = max(detections, key=lambda d: d["confidence"])
                bbox = last_det["bbox"]
                conf = last_det["confidence"]
                x1, y1, x2, y2 = map(int, bbox)

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

                with time_block("sam_segmentation"):
                    sam_predictor.set_image(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
                    masks, _, _ = sam_predictor.predict(
                        box=np.array([x1, y1, x2, y2], dtype=np.float32),
                        multimask_output=False,
                    )
                    mask = masks[0]

                with time_block("get_best_grasp"):
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

                    try:
                        (gx, gy), theta, centroid_rc, major_vec, minor_vec = find_best_grasp(mask, max_width_px)
                    except Exception as e:
                        print(f"[WARN] No valid grasp: {e}")
                        (gx, gy), theta, centroid_rc, major_vec, minor_vec = (None, None), None, None, None, None

                    vis = overlay_mask(display_frame, mask)
                    if centroid_rc is not None:
                        c_x, c_y = map(int, centroid_rc)
                        L = max_width_px

                        p_major = (int(c_x + L * major_vec[0]),  int(c_y + L * major_vec[1]))
                        p_minor = (int(c_x + L * minor_vec[0]),  int(c_y + L * minor_vec[1]))

                        cv2.arrowedLine(vis, (c_x, c_y), p_major,  (0, 255, 255), 2)
                        cv2.arrowedLine(vis, (c_x, c_y), p_minor, (0, 165, 255), 2)
                    if gx is not None and gy is not None:
                        length = max_width_px // 2
                        dx     = math.cos(theta) * length
                        dy     = math.sin(theta) * length
                        pA     = (int(gx + dx), int(gy + dy))
                        pB     = (int(gx - dx), int(gy - dy))
                        cv2.line  (vis, pA, pB, (0, 255, 0), 3)
                        cv2.circle(vis, (int(gx), int(gy)), 5, (0, 0, 255), -1)

                cv2.imshow("Detection Feed", vis)
                if show_depth and depth_vis_color is not None:
                    cv2.imshow("Depth Camera Feed", depth_vis_color)

                if not pose_sent and centroid_rc is not None:
                    try:
                        with time_block("get_curr_pose"):
                            T_base_tcp = get_tcp_pose_persistent()
                    except Exception as e:
                        print(f"[ERROR] Could not read RT pose: {e}")
                        continue

                    with time_block("compute_final_pose"):
                        rvec_curr, _ = cv2.Rodrigues(T_base_tcp[:3, :3])
                        roll_c, pitch_c, _ = rvec_curr.ravel()

                        L_px = max_width_px / 2.0
                        u1, v1 = gx + L_px*math.cos(theta), gy + L_px*math.sin(theta)
                        u2, v2 = gx - L_px*math.cos(theta), gy - L_px*math.sin(theta)

                        P1 = project_pixel_to_table(u1, v1, T_base_tcp)
                        P2 = project_pixel_to_table(u2, v2, T_base_tcp)
                        Pcent = project_pixel_to_table(gx, gy, T_base_tcp)
                        if P1 is None or P2 is None or Pcent is None:
                            print("[WARN] Projection failed; skipping pick.")
                            continue

                        Xb, Yb, Zf = Pcent
                        Zf += 0.05
                        dx3, dy3   = P2[0]-P1[0], P2[1]-P1[1]
                        yaw_des    = math.atan2(dy3, dx3)

                        ROLL_FIXED  = math.pi
                        PITCH_FIXED = 0.0

                        yaw_tcp = yaw_des - math.pi/2 + math.radians(180.0)

                        cz, sz = math.cos(yaw_tcp), math.sin(yaw_tcp)
                        cy, sy = math.cos(PITCH_FIXED), math.sin(PITCH_FIXED)
                        cx, sx = math.cos(ROLL_FIXED),  math.sin(ROLL_FIXED)
                        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
                        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
                        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
                        R_tcp_des = Rz @ Ry @ Rx

                        T_base_tcp_des = np.eye(4)
                        T_base_tcp_des[:3, :3] = R_tcp_des
                        T_base_tcp_des[:3, 3]  = [Xb, Yb, Zf]

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

                    pose_queue.put(final_pose)
                    pose_sent = True

            else:
                cv2.imshow("Detection Feed", display_frame)
                if depth_vis_color is not None:
                    cv2.imshow("Depth Camera Feed", depth_vis_color)

        else:
            cv2.imshow("Detection Feed", display_frame)
            if depth_vis_color is not None:
                cv2.imshow("Depth Camera Feed", depth_vis_color)

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

    release_cameras()
    cv2.destroyAllWindows()
    if robot_thread and robot_thread.is_alive():
        robot_thread.join(3)
    print("[INFO] Pipeline finished.")


if __name__ == "__main__":
    main()
