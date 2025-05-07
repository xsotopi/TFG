import cv2
import numpy as np
import subprocess

WIDTH_COLOR = 640
HEIGHT_COLOR = 480
WIDTH_DEPTH = 480
HEIGHT_DEPTH = 270

def capture_color_frame(camera_index):
    """Captures a single frame from the color camera."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open color camera {camera_index}.")
        return None
    ret, frame = cap.read()
    cap.release()
    return frame

def capture_depth_frame(device_path="/dev/video0", width=WIDTH_DEPTH, height=HEIGHT_DEPTH):
    """Captures a single depth frame using FFmpeg."""
    cmd = [
        'ffmpeg',
        '-f', 'v4l2',
        '-input_format', 'gray16le',  # Z16 is 16-bit grayscale
        '-video_size', f'{width}x{height}',
        '-i', device_path,
        '-pix_fmt', 'gray16le',
        '-f', 'rawvideo',
        '-'
    ]
    try:
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_frame = pipe.stdout.read(width * height * 2)
        pipe.terminate()
        if not raw_frame:
            return None
        frame = np.frombuffer(raw_frame, dtype=np.uint16).reshape((height, width))
        return frame
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install it.")
        return None
    except Exception as e:
        print(f"Error capturing depth frame: {e}")
        return None

if __name__ == "__main__":
    # Example usage (optional)
    color_frame = capture_color_frame(6)
    if color_frame is not None:
        cv2.imshow("Color Camera Test", color_frame)
    else:
        print("Could not capture color frame for test.")

    depth_frame = capture_depth_frame()
    if depth_frame is not None:
        frame_vis = cv2.convertScaleAbs(depth_frame, alpha=0.03)
        cv2.imshow("Depth Camera Test", frame_vis)
    else:
        print("Could not capture depth frame for test.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()