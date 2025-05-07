import cv2
import multiprocessing as mp
import time
import sys
import numpy as np
import subprocess

WIDTH_COLOR = 640
HEIGHT_COLOR = 480
WIDTH_DEPTH = 480
HEIGHT_DEPTH = 270

def display_color_camera_feed(camera_index):
    """Displays the real-time video feed from the color camera (index 6)."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open color camera {camera_index}. Error: {sys.exc_info()}", file=sys.stderr)
        return

    cv2.namedWindow(f'Camera {camera_index}', cv2.WINDOW_NORMAL)
    print(f"Displaying color camera feed from camera {camera_index}. Press 'q' to quit all.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame from color camera {camera_index}. Error: {sys.exc_info()}", file=sys.stderr)
            break

        cv2.imshow(f'Camera {camera_index}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

    cap.release()
    cv2.destroyWindow(f'Camera {camera_index}')
    print(f"Color camera {camera_index} feed stopped.")

def display_depth_camera_feed():
    """Captures and displays the real-time depth feed from camera 0 using FFmpeg."""
    cmd = [
        'ffmpeg',
        '-f', 'v4l2',
        '-input_format', 'gray16le',  # Z16 is 16-bit grayscale
        '-video_size', f'{WIDTH_DEPTH}x{HEIGHT_DEPTH}',
        '-i', '/dev/video0',
        '-pix_fmt', 'gray16le',
        '-f', 'rawvideo',
        '-'
    ]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    cv2.namedWindow("Camera 0 (Depth)", cv2.WINDOW_NORMAL)
    print("Displaying depth camera feed from camera 0. Press 'q' to quit all.")

    while True:
        raw_frame = pipe.stdout.read(WIDTH_DEPTH * HEIGHT_DEPTH * 2)  # 2 bytes por píxel (16-bit)
        if not raw_frame:
            break

        frame = np.frombuffer(raw_frame, dtype=np.uint16).reshape((HEIGHT_DEPTH, WIDTH_DEPTH))
        frame_vis = cv2.convertScaleAbs(frame, alpha=0.03)  # Adjust alpha as needed

        cv2.imshow("Camera 0 (Depth)", frame_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipe.terminate()
    cv2.destroyWindow("Camera 0 (Depth)")
    print("Depth camera 0 feed stopped.")

if __name__ == "__main__":
    # Create processes for the color and depth cameras
    color_process = mp.Process(target=display_color_camera_feed, args=(6,))
    depth_process = mp.Process(target=display_depth_camera_feed)

    # Start the processes
    color_process.start()
    depth_process.start()

    # Wait for both processes to finish (when 'q' is pressed in either window)
    color_process.join()
    depth_process.join()

    cv2.destroyAllWindows()
    print("All camera feeds stopped.")