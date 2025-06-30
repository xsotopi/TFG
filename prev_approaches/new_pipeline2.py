import whisper, cv2, os, time, threading, torch, numpy as np
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer


from ai.audio.audio2text  import transcribe_audio
from ai.detection.detect       import detect_target_objects_realtime
from chats.chat_bot  import extract_intent_and_object
# from send2robot2      import compute_pose, server_send_pose

# ── CÁMARA persistente ─────────────────────────────
from camera.cameras import (
    capture_color_frame, capture_depth_frame, release_cameras
)

# ---------- calibración ----------
CALIB_DIR = "/home/lab/Desktop/TFG/new_calibration"

K         = np.load(os.path.join(CALIB_DIR, "K.npy"))
dist      = np.load(os.path.join(CALIB_DIR, "distCoeffs.npy"))
T_cam2fl = np.load(os.path.join(CALIB_DIR, "cam_to_flange.npy"))
R = np.load(os.path.join(CALIB_DIR, "cam_to_flange.npy"))[:3, :3]
angles = cv2.Rodrigues(R)[0].ravel()*180/np.pi
print("Rodrigues (deg):", angles)


print("[INFO] Calibración OK")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")
# ---------- modelos ----------
whisper_model = whisper.load_model("small")
print("Whisper OK")
WEIGHTS = "/home/lab/Desktop/TFG/yolo_finetune/runs/yolov8_coco_plus_tools_en/weights/best.pt"
yolo_model    = YOLO(WEIGHTS);    
yolo_model.to(device)  # Mover modelo YOLO a GPU si está disponible
print("YOLO OK")



ckpt          = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
slm_tokenizer = AutoTokenizer.from_pretrained(ckpt)
slm_model     = AutoModelForCausalLM.from_pretrained(ckpt).to(device)

# ---------- constantes ----------
UPLOAD_FOLDER = "app/uploads"; 
AUDIO_EXT = ".mp3"

# ---------- estado global ----------
pose_sent = False
robot_comm_thread = None
current_intent = None
current_target_object = None
last_detection = None
show_detections = False
latest_audio_file = None

# ---------- helper cámaras (sin cambios de lógica) ----------
def project_bbox_to_depth(depth_frame, color_shape, bbox):
    h_c,w_c = color_shape[:2]; x1,y1,x2,y2 = map(int,bbox)
    h_d,w_d = depth_frame.shape[:2]
    sx,sy   = w_d/w_c, h_d/h_c
    return (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))

# ---------- robot thread ----------
# def robot_communication_thread(pose):
#     try:
#         print("[↗] Enviando pose:", pose)
#         server_send_pose(pose)
#     except Exception as e:
#         print("Error hilo robot:", e)

# ---------- audio ----------
def process_audio(path):
    global current_intent, current_target_object
    global pose_sent, last_detection, show_detections, latest_audio_file

    transcript = transcribe_audio(whisper_model, path, "es")
    actions    = extract_intent_and_object(transcript, slm_model, slm_tokenizer, device)
    print("Transcript:", transcript, "| Actions:", actions)

    current_intent = current_target_object = None
    if actions and 'pick' in actions and actions['pick']:
        current_intent = "pick"
        current_target_object = actions['pick'][0]

    pose_sent = False
    last_detection = None
    show_detections = True
    latest_audio_file = None
    os.remove(path)

def monitor_upload_folder():
    global latest_audio_file
    while True:
        time.sleep(1)
        files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(AUDIO_EXT)]
        if not files: continue
        newest = max(files, key=lambda f: os.path.getctime(os.path.join(UPLOAD_FOLDER,f)))
        if newest != latest_audio_file:
            latest_audio_file = newest
            process_audio(os.path.join(UPLOAD_FOLDER,newest))

# ---------- MAIN ----------
def main():
    global pose_sent, robot_comm_thread, last_detection, show_detections
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    threading.Thread(target=monitor_upload_folder, daemon=True).start()
    cv2.namedWindow('Detection Feed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Camera Feed', cv2.WINDOW_NORMAL)
    print("Loop: q=quit, r=toggle detections")

    while True:
        color_frame = capture_color_frame()
        if color_frame is None:
            print("Sin frame color"); break

        depth_frame = capture_depth_frame()

        display_frame = color_frame.copy()
        depth_vis_color = None
        if depth_frame is not None:
            depth_vis = cv2.convertScaleAbs(depth_frame, alpha=0.05)
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        if current_intent == "pick" and current_target_object and show_detections:
            detections = detect_target_objects_realtime(
                yolo_model, display_frame, current_target_object, False)
            
            # detections = [{'class': 'apple', 'confidence': 0.8567644357681274, 'bbox': [193, 123, 237, 169]}]

            if detections:
                last_detection = max(detections, key=lambda d:d['confidence'])
                bbox = last_detection['bbox']
                conf = last_detection['confidence']
                x1,y1,x2,y2 = map(int,bbox)
                cv2.rectangle(display_frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(display_frame,
                            f"{current_target_object}:{conf:.2f}",
                            (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

                if depth_frame is not None and depth_vis_color is not None:
                    db = project_bbox_to_depth(depth_frame, color_frame.shape, bbox)
                    cv2.rectangle(depth_vis_color,(db[0],db[1]),(db[2],db[3]),(0,255,0),2)

                # if not pose_sent:
                #     pose = compute_pose(bbox, K, dist, T_cam2fl)
                #     if pose:
                #         if not robot_comm_thread or not robot_comm_thread.is_alive():
                #             robot_comm_thread = threading.Thread(
                #                 target=robot_communication_thread, args=(pose,))
                #             robot_comm_thread.start()
                #             pose_sent = True

        cv2.imshow('Detection Feed', display_frame)
        if depth_vis_color is not None:
            cv2.imshow('Depth Camera Feed', depth_vis_color)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
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
