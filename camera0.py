import cv2
import numpy as np
import subprocess

width = 480
height = 270

# Comando FFmpeg para capturar del dispositivo Z16 y enviar rawvideo por pipe
cmd = [
    'ffmpeg',
    '-f', 'v4l2',
    '-input_format', 'gray16le',  # Z16 es 16-bit grayscale
    '-video_size', f'{width}x{height}',
    '-i', '/dev/video0',
    '-pix_fmt', 'gray16le',
    '-f', 'rawvideo',
    '-'
]

# Inicia el proceso de ffmpeg
pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

while True:
    raw_frame = pipe.stdout.read(width * height * 2)  # 2 bytes por píxel (16-bit)
    if not raw_frame:
        break

    # Convierte a np.array
    frame = np.frombuffer(raw_frame, dtype=np.uint16).reshape((height, width))

    # Normaliza para visualizar (conversión a 8-bit)
    frame_vis = cv2.convertScaleAbs(frame, alpha=0.03)  # Ajusta alpha según profundidad

    cv2.imshow("Camera 0", frame_vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipe.terminate()
cv2.destroyAllWindows()
