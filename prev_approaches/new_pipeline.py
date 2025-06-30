import whisper
import cv2
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from ai.audio.audio2text import transcribe_audio
from ai.command_extraction.extract_command_sent_trans import classify_intent, extract_object
from ai.detection.detect import detect_target_objects_realtime
from send2robot import compute_pose, server_send_pose
from camera.cameras import capture_color_frame, capture_depth_frame
from chats.chat_bot import extract_intent_and_object

import math
import time
import threading
import os
import torch
import numpy as np

CALIB_DIR = "camera"                       # carpeta donde están los .npy/.npz

K_dist         = np.load(os.path.join(CALIB_DIR, "intrinsics.npz"))
K              = K_dist["K"]               # matriz intrínseca 3×3
dist           = K_dist["dist"]            # distorsiones
T_cam2tcp      = np.load(os.path.join(CALIB_DIR, "T_cam2tcp.npy"))  # 4×4

fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]

print("[INFO] Calibración cargada OK")

# -------------------------
# Load models
# -------------------------
try:
    whisper_model = whisper.load_model("small")
    print("Whisper model loaded successfully!")
    # nlp_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
    # print("Sentence Transformer model loaded successfully!")
    yolo_model = YOLO("yolov8n.pt")
    print("YOLO model loaded successfully!")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    # checkpoint = "Qwen/Qwen3-0.6B" # For Qwen model
    slm_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
    slm_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    # slm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map=device, torch_dtype=torch.float32)
    # slm_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    
    
    
    print("SLM model loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

COLOR_CAMERA_INDEX = 6
DEPTH_CAMERA_INDEX = 0
UPLOAD_FOLDER = "app/uploads/"
AUDIO_FILE_EXTENSION = ".mp3"

# Global flags and variables
pose_sent = False
robot_comm_thread = None
current_intent = None
current_target_object = None
last_detection = None
new_audio_available = False
show_detections = False
latest_audio_file = None

# Lock to protect access to shared variables
audio_event = threading.Event()

def robot_communication_thread(pose_data):
    """Handles robot communication in a separate thread."""
    try:
        print(f"Starting robot communication thread for pose: {pose_data}")
        server_send_pose(pose_data)
        print("Robot communication thread finished.")
    except Exception as e:
        print(f"Error in robot communication thread: {e}")

def project_bbox_to_depth(depth_frame, color_frame_shape, bbox):
    h, w = color_frame_shape[:2]
    x_min, y_min, x_max, y_max = map(int, bbox)
    depth_h, depth_w = depth_frame.shape[:2] if depth_frame.ndim == 2 else depth_frame.shape[:2]
    scale_x = depth_w / w
    scale_y = depth_h / h
    depth_x_min = int(x_min * scale_x)
    depth_y_min = int(y_min * scale_y)
    depth_x_max = int(x_max * scale_x)
    depth_y_max = int(y_max * scale_y)
    return (depth_x_min, depth_y_min, depth_x_max, depth_y_max)

def process_audio(audio_path):
    """Processes the audio file for intent and object using the SLM and then deletes it."""
    global current_intent
    global current_target_object
    global pose_sent
    global last_detection
    global show_detections
    global latest_audio_file

    language_code = os.path.splitext(os.path.basename(audio_path))[0]

    print(f"Processing audio file: {audio_path} with language code: {language_code}")
    try:
        transcript = transcribe_audio(whisper_model, audio_path, language_code)
        transcript = "pick the apple"
        print("Transcript:", transcript)
        actions = extract_intent_and_object(transcript, slm_model, slm_tokenizer, device)
        print("Extracted Actions:", actions)

        if actions:
            if 'pick' in actions and actions['pick']:
                current_intent = "pick"
                current_target_object = actions['pick'][0] # Take the first object to pick
            elif 'place' in actions and actions['place']:
                current_intent = "place" # You might want to handle the object for placing differently
                current_target_object = actions['place'][0] # Assuming the object to place is mentioned
            else:
                current_intent = None
                current_target_object = None
        else:
            current_intent = None
            current_target_object = None

    except Exception as e:
        print(f"Error during transcription or command extraction: {e}")
        return

    pose_sent = False
    last_detection = None
    show_detections = True
    latest_audio_file = None

    # Delete the processed audio file
    try:
        os.remove(audio_path)
        print(f"Deleted processed audio file: {audio_path}")
    except Exception as e:
        print(f"Error deleting audio file {audio_path}: {e}")

def monitor_upload_folder():
    """Monitors the upload folder for new audio files."""
    global latest_audio_file
    while True:
        time.sleep(1)  # Check every 1 second
        try:
            files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(AUDIO_FILE_EXTENSION)]
            if files:
                latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(UPLOAD_FOLDER, f)))
                full_path = os.path.join(UPLOAD_FOLDER, latest_file)
                # Process only if it's a new file
                if latest_file != latest_audio_file:
                    print(f"New audio file detected: {latest_file}")
                    latest_audio_file = latest_file
                    process_audio(full_path)
        except Exception as e:
            print(f"Error monitoring upload folder: {e}")

def main():
    global pose_sent
    global robot_comm_thread
    global current_intent
    global current_target_object
    global last_detection

    show_detections = True

    # Create the upload folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    cap_color = cv2.VideoCapture(COLOR_CAMERA_INDEX)
    if not cap_color.isOpened():
        print(f"Error: Could not open color camera {COLOR_CAMERA_INDEX}. Check index and permissions.")
        return

    cv2.namedWindow('Detection Feed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Camera Feed', cv2.WINDOW_NORMAL)

    # Start the folder monitoring thread
    monitor_thread = threading.Thread(target=monitor_upload_folder, daemon=True)
    monitor_thread.start()

    print(f"Entering main camera loop. Press 'q' to quit, 'r' to toggle detections.")

    while True:
        ret_color, color_frame = cap_color.read()
        if not ret_color:
            print(f"Error reading frame from color camera {COLOR_CAMERA_INDEX}. End of stream or camera error?")
            break

        display_frame = color_frame.copy()
        depth_frame = capture_depth_frame(device_path=f"/dev/video{DEPTH_CAMERA_INDEX}")

        depth_vis_color = None
        if depth_frame is not None:
            depth_vis = cv2.convertScaleAbs(depth_frame, alpha= 0.05)
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        if current_intent == "pick" and current_target_object and show_detections:
            detections = detect_target_objects_realtime(yolo_model, display_frame, current_target_object, show_image=False)

            if detections:
                last_detection = max(detections, key=lambda x: x['confidence'])
                bbox = last_detection["bbox"]
                confidence = last_detection["confidence"]

                int_bbox = [int(b) for b in bbox]
                cv2.rectangle(display_frame, (int_bbox[0], int_bbox[1]), (int_bbox[2], int_bbox[3]), (0, 255, 0), 2)
                label = f'{current_target_object}: {confidence:.2f}'
                cv2.putText(display_frame, label, (int_bbox[0], int_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if depth_frame is not None:
                    if depth_vis_color is not None:
                        depth_bbox = project_bbox_to_depth(depth_frame, color_frame.shape[:2], bbox)
                        cv2.rectangle(depth_vis_color, (depth_bbox[0], depth_bbox[1]), (depth_bbox[2], depth_bbox[3]), (0, 255, 0), 2)

                    
                    if not pose_sent:
                        pose = compute_pose(bbox, depth_frame, color_frame.shape[:2], K, dist, T_cam2tcp)
                        if pose:
                            print("Computed pose:", pose)
                            if robot_comm_thread is None or not robot_comm_thread.is_alive():
                                robot_comm_thread = threading.Thread(target=robot_communication_thread, args=(pose,))
                                robot_comm_thread.start()
                                print("Pose sending initiated to robot.")
                                pose_sent = True
                            else:
                                print("Robot communication thread already active. Skipping new pose sending.")

        # Always show the camera feeds
        cv2.imshow('Detection Feed', display_frame)
        if depth_vis_color is not None:
            cv2.imshow('Depth Camera Feed', depth_vis_color)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('r'):
            show_detections = not show_detections
            print(f"Detections {'enabled' if show_detections else 'disabled'}.")

    cap_color.release()
    cv2.destroyAllWindows()

    if robot_comm_thread and robot_comm_thread.is_alive():
        print("Waiting for robot communication thread to complete...")
        robot_comm_thread.join(timeout=5)
        if robot_comm_thread.is_alive():
            print("Robot communication thread did not complete in time.")

    print("Pipeline finished.")

if __name__ == "__main__":
    main()