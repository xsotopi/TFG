import cv2, socket, struct, datetime, json, time, math
from pathlib import Path
import re

UR_IP   = "192.168.0.102"
RT_PORT = 30003
CAM_ID  = 6
CHESS_SIZE = (8, 7)
SAVE_DIR = Path("dataset_1")
ADD_IMAGES = 20
REQ_W, REQ_H = 1280, 800
# ───────────────────────────────────────────────

IMG_DIR  = SAVE_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
POSE_DIR = SAVE_DIR / "poses"
POSE_DIR.mkdir(parents=True, exist_ok=True)

# Check existing images and set starting index
already = sorted(IMG_DIR.glob("img_*.png"))
start_idx = 0
if already:
    m = re.search(r"img_(\d+)_", already[-1].stem)
    if m:
        start_idx = int(m.group(1)) + 1
print(f"→ Ya existen {len(already)} imágenes. "
      f"Comenzaré en índice {start_idx:02d}")

POSE_SLICE = slice(440, 488)

def get_tcp_pose():
    """Get the current TCP pose from the UR controller."""
    with socket.create_connection((UR_IP, RT_PORT), timeout=0.5) as s:
        hdr = s.recv(4)
        (length,) = struct.unpack(">I", hdr)
        data = s.recv(length-4, socket.MSG_WAITALL)
    return struct.unpack(">6d", data[POSE_SLICE])

cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  REQ_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_H)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

print("SPACE = capturar | ESC = salir\n")

captured = 0
while captured < ADD_IMAGES:
    ok, frame = cap.read()
    if not ok:
        time.sleep(0.02)
        continue

    # Find chessboard corners in current frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, c = cv2.findChessboardCorners(gray, CHESS_SIZE,
              cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

    # Display the frame with detection status
    view = frame.copy()
    cv2.putText(view, "Detected" if found else "Not detected",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0) if found else (0,0,255), 2)
    if found:
        cv2.drawChessboardCorners(view, CHESS_SIZE, c, found)
    cv2.imshow("Capture", view)

    key = cv2.waitKey(5) & 0xFF
    if key == 27: # ESC
        break
    # Capture image and pose
    if key == 32 and found: # SPACE
        idx = start_idx + captured
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path  = IMG_DIR  / f"img_{idx:02d}_{ts}.png"
        pose_path = POSE_DIR / f"pose_{idx:02d}_{ts}.json"
        cv2.imwrite(str(img_path), frame)

        pose = get_tcp_pose()
        rx, ry, rz = pose[3:6]
        ang_deg = math.degrees((rx*rx + ry*ry + rz*rz) ** 0.5)
        print(f"[{idx:02d}] rotación total = {ang_deg:5.1f}°   → {img_path.name}")

        with open(pose_path, "w") as f:
            json.dump({"pose_tcp": pose, "timestamp": ts}, f, indent=2)
        captured += 1

cap.release()
cv2.destroyAllWindows()
print(f"Captured {captured} new images. Total now: {start_idx + captured}")
