import cv2, subprocess, numpy as np, atexit, fcntl, os

# --- Constants ---

#RGB resolution
W_COL, H_COL = 1280, 800

#Depth resolution
W_DEP, H_DEP = 480, 270

# Index of color and depth cameras
COLOR_INDEX  = 6
DEPTH_DEV    = "/dev/video0"

_color = cv2.VideoCapture(COLOR_INDEX)

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
_color.set(cv2.CAP_PROP_FOURCC, fourcc)
_color.set(cv2.CAP_PROP_FRAME_WIDTH,  W_COL)
_color.set(cv2.CAP_PROP_FRAME_HEIGHT, H_COL)

def capture_color_frame():
    """Caputres color frame as numpy array"""
    ok, frm = _color.read()
    return frm if ok else None

def _open_depth_pipe():
    """
    Opens a subprocess to capture depth frames from the depth camera.
    """
    cmd = [
        "ffmpeg", "-loglevel", "quiet",
        "-f", "v4l2", "-input_format", "gray16le",
        "-video_size", f"{W_DEP}x{H_DEP}",
        "-i", DEPTH_DEV,
        "-pix_fmt", "gray16le", "-f", "rawvideo", "-"
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    fl = fcntl.fcntl(p.stdout, fcntl.F_GETFL)
    fcntl.fcntl(p.stdout, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    return p

_depth_proc = _open_depth_pipe()
DEP_BYTES   = W_DEP * H_DEP * 2

def capture_depth_frame():
    """
    Captures depth frame as numpy array
    """
    try:
        raw = _depth_proc.stdout.read(DEP_BYTES)
        if raw is None or len(raw) < DEP_BYTES:
            return None
        return np.frombuffer(raw, np.uint16).reshape(H_DEP, W_DEP)
    except Exception:
        return None

def release_cameras():
    if _color.isOpened():
        _color.release()
    _depth_proc.terminate()

atexit.register(release_cameras)
