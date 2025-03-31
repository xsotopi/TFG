import whisper
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, util
from audio2text import transcribe_audio
from extract_command import classify_intent, extract_object
from detect import detect_target_objects
from send2robot import compute_pose, send_pose_to_robot

# -------------------------
# Load models
# -------------------------
# Whisper for transcription & translation
whisper_model = whisper.load_model("small")
print("Whisper model loaded successfully!")
# Sentence Transformer for intent classification and object extraction
nlp_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
print("Sentence Transformer model loaded successfully!")
# YOLOv8 for object detection (using the nano version for speed)
yolo_model = YOLO("yolov8n.pt")
print("YOLO model loaded successfully!")



# -------------------------
# Main pipeline integration
# -------------------------
def main():
    print("Starting main")
    # Define input paths (update these paths as needed)
    audio_path = r'C:\Users\xavim\Desktop\TFG\code\uploads\recording.mp3'  # Path to your audio file (e.g., a recording of "Pick up the apple")
    image_path = "fruta.jpg"   # Path to your image file (containing the objects)

    # 1. Transcribe and translate the audio to English
    print("Transcribing audio...")
    transcript = transcribe_audio(whisper_model, audio_path)
    print("Transcript:", transcript)
    # transcript = "Take the apple"
    
    # 2. Classify the command intent from the transcript
    intent = classify_intent(nlp_model, transcript)
    print("Classified Intent:", intent)
    
    # 3. Extract the target object using the same model
    target_object = extract_object(nlp_model, transcript)
    print("Extracted Object:", target_object)
    
    # 4. If the command is to pick and a target object is extracted, run YOLO detection
    if intent == "pick" and target_object:
        print(f"Command indicates picking up the {target_object}. Running YOLO detection...")
        detections = detect_target_objects(yolo_model, image_path, target_object)
        if detections:
            print(f"Detected {len(detections)} '{target_object}' objects:")
            for obj in detections:
                print(f"- {obj['class']} at {obj['bbox']} with {obj['confidence']*100:.2f}% confidence")
                # Send the first detected object's bounding box center to the robot
            pose = compute_pose(detections[0]["bbox"])
            send_pose_to_robot(pose)

            print("Pose sent to the robot!!")
        else:
            print(f"No '{target_object}' found above {0.5*100:.0f}% confidence!")
    else:
        print("No pick command detected or no target object extracted.")

if __name__ == "__main__":
    main()
