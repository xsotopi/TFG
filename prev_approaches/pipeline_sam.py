import whisper, cv2, os, time, threading, torch, numpy as np
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer
from segment_anything import SamPredictor, sam_model_registry
import math

from ai.audio.audio2text  import transcribe_audio
from ai.detection.detect       import detect_target_objects_realtime
from chats.chat_bot  import extract_intent_and_object
from send2robot      import send_pose_secondary_monitor, compute_pose, server_send_pose, compute_pose_from_pixel, pixel_to_3d_on_table, get_tcp_pose_4x4, rotm_to_euler_zyx, move_robot

# ── CÁMARA persistente ─────────────────────────────
from camera.cameras import (
    capture_color_frame, capture_depth_frame, release_cameras
)

# ---------- calibración ----------
CALIB_DIR = "/home/lab/Desktop/TFG/new_calibration"

K         = np.load(os.path.join(CALIB_DIR, "K.npy"))
dist      = np.load(os.path.join(CALIB_DIR, "distCoeffs.npy"))
T_cam2fl  = np.load(os.path.join(CALIB_DIR, "cam_to_flange.npy"))
R         = T_cam2fl[:3, :3]
angles    = cv2.Rodrigues(R)[0].ravel() * 180 / np.pi
print("Rodrigues (deg):", angles)
print("[INFO] Calibración OK")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ---------- modelos ----------
whisper_model = whisper.load_model("medium")
whisper_model.to(device)
print("Whisper OK")

WEIGHTS    = "/home/lab/Desktop/TFG/yolo_finetune/runs/yolov8_coco_plus_tools_en/weights/best.pt"
yolo_model = YOLO(WEIGHTS)
yolo_model.to(device)
print("YOLO OK")

sam_path   = "/home/lab/Desktop/TFG/sam_vit_b_01ec64.pth"
sam_model  = sam_model_registry["vit_b"](checkpoint=sam_path)
sam_model.to(device)
sam_predictor = SamPredictor(sam_model)
print("SAM OK")

ckpt          = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
slm_tokenizer = AutoTokenizer.from_pretrained(ckpt)
slm_model     = AutoModelForCausalLM.from_pretrained(ckpt).to(device)

# ---------- constantes ----------
UPLOAD_FOLDER = "app/uploads"
AUDIO_EXT     = ".mp3"

# ---------- estado global ----------
pose_sent            = False
robot_comm_thread    = None
current_intent       = None
current_target_object = None
last_detection       = None
show_detections      = False
latest_audio_file    = None


def overlay_mask(img: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.4):
    """Returns a copy of *img* with *mask* overlayed in *color* (BGR)."""
    out = img.copy()
    col = np.full_like(img, color, dtype=np.uint8)
    return np.where(mask[..., None], cv2.addWeighted(col, alpha, out, 1 - alpha, 0), out)


# ---------- función de grasp ----------
def find_best_grasp(mask: np.ndarray, max_width_px: int, step_px: int = 3):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        raise ValueError("Mask vacía – no se puede calcular grasp.")
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    centroid = np.array([cy, cx])  # (row, col)

    pts = np.column_stack(np.where(mask > 0))
    pts_center = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts_center, full_matrices=False)
    major_vec = vt[0] / np.linalg.norm(vt[0])
    minor_vec = vt[1] / np.linalg.norm(vt[1])
    grasp_theta = np.arctan2(minor_vec[1], minor_vec[0])

    proj = pts_center @ major_vec
    half_len = int(np.percentile(np.abs(proj), 95))

    best_score, best_center = -np.inf, None
    for offset in range(-half_len, half_len + 1, step_px):
        c = centroid + offset * major_vec
        p1 = c + (max_width_px / 2) * minor_vec
        p2 = c - (max_width_px / 2) * minor_vec
        line = np.linspace(p1, p2, max_width_px)
        rr = np.clip(line[:, 0].astype(int), 0, mask.shape[0] - 1)
        cc = np.clip(line[:, 1].astype(int), 0, mask.shape[1] - 1)
        if np.all(mask[rr, cc]):
            torque = np.linalg.norm(c - centroid)
            score = -torque
            if score > best_score:
                best_score, best_center = score, c
    if best_center is None:
        raise RuntimeError("No valid grasp – incremente max_width_px o revise el mask.")

    # Return (x, y) in image coords, theta (radians), centroid (row,col), major_vec, minor_vec
    return (float(best_center[1]), float(best_center[0])), float(grasp_theta), centroid, major_vec, minor_vec


# ---------- helper cámaras (sin cambios de lógica) ----------
def project_bbox_to_depth(depth_frame, color_shape, bbox):
    h_c, w_c = color_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    h_d, w_d = depth_frame.shape[:2]
    sx, sy   = w_d / w_c, h_d / h_c
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))


# ---------- robot thread ----------
def robot_communication_thread(pose):
    try:
        print("[↗] Enviando pose:", pose)
        send_pose_secondary_monitor(pose)
    except Exception as e:
        print("Error hilo robot:", e)


# ---------- audio ----------
def process_audio(path):
    global current_intent, current_target_object
    global pose_sent, last_detection, show_detections, latest_audio_file

    transcript = transcribe_audio(whisper_model, path, "es")
    actions    = extract_intent_and_object(transcript, slm_model, slm_tokenizer, device)
    print("Transcript:", transcript, "| Actions:", actions)

    current_intent       = None
    current_target_object = None
    if actions and 'pick' in actions and actions['pick']:
        current_intent       = "pick"
        current_target_object = actions['pick'][0]

    pose_sent     = False
    last_detection = None
    show_detections = True
    latest_audio_file = None
    os.remove(path)


def monitor_upload_folder():
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


# ---------- MAIN ----------
def main():
    global pose_sent, robot_comm_thread, last_detection, show_detections

    pix_per_mm    = 3.2
    max_width_mm  = 55
    base_open_px  = int(max_width_mm * pix_per_mm)

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    threading.Thread(target=monitor_upload_folder, daemon=True).start()

    cv2.namedWindow('Detection Feed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Camera Feed', cv2.WINDOW_NORMAL)
    print("Loop: q=quit, r=toggle detections")

    while True:
        color_frame = capture_color_frame()
        if color_frame is None:
            print("No color frame")
            break

        depth_frame = capture_depth_frame()
        display_frame = color_frame.copy()
        depth_vis_color = None

        if depth_frame is not None:
            depth_vis = cv2.convertScaleAbs(depth_frame, alpha=0.05)
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        if current_intent == "pick" and current_target_object:
            detections = detect_target_objects_realtime(
                yolo_model, display_frame, current_target_object, False
            )

            if detections:
                last_detection = max(detections, key=lambda d: d['confidence'])
                bbox = last_detection['bbox']
                conf = last_detection['confidence']
                x1, y1, x2, y2 = map(int, bbox)

                if show_detections:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        f"{current_target_object}:{conf:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                    )

                    if depth_frame is not None and depth_vis_color is not None:
                        db = project_bbox_to_depth(depth_frame, color_frame.shape, bbox)
                        cv2.rectangle(depth_vis_color, (db[0], db[1]), (db[2], db[3]), (0, 255, 0), 2)

                # Segmentar con SAM sobre el color_frame
                sam_predictor.set_image(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
                masks, _, _ = sam_predictor.predict(
                    box=np.array([x1, y1, x2, y2], dtype=np.float32),
                    multimask_output=False
                )
                mask = masks[0]  # máscara binaria

                # Calcular ancho en eje menor para grasp
                pts = np.column_stack(np.where(mask > 0))
                if pts.size > 0:
                    pts_center   = pts - pts.mean(axis=0)
                    _, _, vt     = np.linalg.svd(pts_center, full_matrices=False)
                    minor_vec    = vt[1] / np.linalg.norm(vt[1])
                    proj_minor   = pts_center @ minor_vec
                    width_minor_px = proj_minor.max() - proj_minor.min()
                    margin_px    = 4
                    safe_open_px = max(1, int(width_minor_px - margin_px))
                    max_width_px = min(base_open_px, safe_open_px)
                else:
                    max_width_px = base_open_px

                try:
                    (gx, gy), theta, centroid, major_vec, minor_vec = find_best_grasp(mask, max_width_px)
                except Exception as e:
                    print(f"[WARNING] Grasp no encontrado para bbox {(x1, y1, x2, y2)}: {e}")
                    (gx, gy), theta, centroid, major_vec, minor_vec = None, None, None, None, None

                # Aplicar overlay_mask sobre display_frame
                vis = overlay_mask(display_frame, mask)

                # Dibujar centroid y ejes
                if centroid is not None:
                    c_row, c_col = int(centroid[0]), int(centroid[1])
                    L = max_width_px
                    p1_major = (
                        int(c_col + L * major_vec[1]),
                        int(c_row + L * major_vec[0])
                    )
                    p1_minor = (
                        int(c_col + L * minor_vec[1]),
                        int(c_row + L * minor_vec[0])
                    )
                    cv2.arrowedLine(vis, (c_col, c_row), p1_major, (0, 255, 255), 2)
                    cv2.arrowedLine(vis, (c_col, c_row), p1_minor, (0, 165, 255), 2)

                # Dibujar grasp line y punto de grasp
                if gx is not None and gy is not None:
                    length = max_width_px // 2
                    dx = np.cos(theta) * length
                    dy = np.sin(theta) * length
                    pA = (int(gx + dx), int(gy + dy))
                    pB = (int(gx - dx), int(gy - dy))
                    cv2.line(vis, pA, pB, (0, 255, 0), 3)
                    cv2.circle(vis, (int(gx), int(gy)), 5, (0, 0, 255), -1)

                # Mostrar sólo el frame coloreado con la máscara
                cv2.imshow('Detection Feed', vis)

                # En depth_vis_color, SOLO mostrar el rectángulo del bbox (sin máscara)
                if depth_vis_color is not None:
                    cv2.imshow('Depth Camera Feed', depth_vis_color)

                # =============================================================================
                # Aquí hacemos sólo una llamada a get_tcp_pose_4x4() y reutilizamos la matriz
                # =============================================================================
                if not pose_sent and centroid is not None and depth_frame is not None:
                    # 1) Obtener la transformada base ← flange una sola vez
                    T_base_fl = get_tcp_pose_4x4()  # base ← flange
                    R_curr = T_base_fl[:3, :3]
                    roll_curr, pitch_curr, yaw_curr = rotm_to_euler_zyx(R_curr)

                    # 2) Definir helper local para proyectar píxel→3D sobre la mesa usando T_base_fl
                    def project_pixel_to_table(u_pixel, v_pixel):
                        # 2a) Desdistorsionar el punto
                        xn, yn = cv2.undistortPoints(
                            np.array([[[u_pixel, v_pixel]]], np.float64),
                            K, dist
                        ).reshape(-1)
                        # 2b) Rayo en coords de cámara
                        ray_cam = np.array([xn, yn, 1.0], dtype=np.float64)
                        # 2c) base ← cam
                        T_base_cam = T_base_fl @ T_cam2fl
                        R_bc = T_base_cam[:3, :3]
                        O_bc = T_base_cam[:3, 3]
                        # 2d) Rayo en coords base
                        ray_base = R_bc @ ray_cam
                        # 2e) Intersectar con z=0 (mesa)
                        if abs(ray_base[2]) < 1e-6:
                            return None
                        s = -O_bc[2] / ray_base[2]
                        if s < 0:
                            return None
                        P_base = O_bc + s * ray_base
                        # Mantener altura Z actual del flange
                        P_base[2] = T_base_fl[2, 3]
                        return [float(P_base[0]), float(P_base[1]), float(P_base[2])]

                    # 3) Calcular los extremos de la línea de grasp en la imagen
                    L  = max_width_px // 2
                    u1 = gx + L * math.cos(theta)
                    v1 = gy + L * math.sin(theta)
                    u2 = gx - L * math.cos(theta)
                    v2 = gy - L * math.sin(theta)

                    # 4) Proyectarlos al plano de la mesa (z=0)
                    P1         = project_pixel_to_table(u1, v1)
                    P2         = project_pixel_to_table(u2, v2)
                    P_centroid = project_pixel_to_table(gx, gy)

                    if P1 is None or P2 is None or P_centroid is None:
                        print("[WARN] No se pudo proyectar línea de grasp.")
                    else:
                        # 5) Calcular yaw deseado en el plano XY-base
                        dx      = P2[0] - P1[0]
                        dy      = P2[1] - P1[1]
                        yaw_des = math.atan2(dy, dx)  # ángulo en radianes

                        # 6) Reconstruir la matriz de rotación deseada
                        cz = math.cos(yaw_des);  sz = math.sin(yaw_des)
                        cy = math.cos(pitch_curr); sy = math.sin(pitch_curr)
                        cx = math.cos(roll_curr);  sx = math.sin(roll_curr)

                        Rz = np.array([[ cz, -sz, 0],
                                       [ sz,  cz, 0],
                                       [  0,   0, 1]])
                        Ry = np.array([[  cy, 0, sy],
                                       [   0, 1,  0],
                                       [ -sy, 0, cy]])
                        Rx = np.array([[1,   0,    0],
                                       [0,  cx, -sx],
                                       [0,  sx,  cx]])
                        R_des = Rz @ Ry @ Rx
                        rvec_des, _ = cv2.Rodrigues(R_des)
                        rx_des, ry_des, rz_des = rvec_des.ravel()

                        # 7) Para la posición, usamos P_centroid (con Z=altura del flange)
                        Xb, Yb, Zf = P_centroid
                        final_pose = [
                            float(Xb), float(Yb), float(Zf),
                            float(rx_des), float(ry_des), float(rz_des)
                        ]

                        # 8) Imprimir la pose final sólo una vez
                        print(
                            f"[INFO] Pose enviada: "
                            f"X={Xb:.3f}, Y={Yb:.3f}, Z={Zf:.3f}, "
                            f"roll={math.degrees(roll_curr):.1f}°, "
                            f"pitch={math.degrees(pitch_curr):.1f}°, "
                            f"yaw_des={math.degrees(yaw_des):.1f}°"
                        )

                        # 9) Enviar al robot
                        robot_comm_thread = threading.Thread(
                            target=robot_communication_thread, args=(final_pose,)
                        )
                        robot_comm_thread.start()
                        pose_sent = True

            else:
                # Si no hay detecciones, mostrar el display_frame normal
                cv2.imshow('Detection Feed', display_frame)
                if depth_vis_color is not None:
                    cv2.imshow('Depth Camera Feed', depth_vis_color)

        else:
            # Si no hay intención 'pick' o no hay objeto, mostrar raw
            cv2.imshow('Detection Feed', display_frame)
            if depth_vis_color is not None:
                cv2.imshow('Depth Camera Feed', depth_vis_color)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('r'):
            show_detections = not show_detections
            print("Detections", "ON" if show_detections else "OFF")

    release_cameras()
    cv2.destroyAllWindows()
    if robot_comm_thread and robot_comm_thread.is_alive():
        robot_comm_thread.join(3)
    print("Pipeline finished.")


if __name__ == "__main__":
    main()
