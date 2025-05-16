import socket
import math
import numpy as np
import time

def server_send_pose(pose, host="0.0.0.0", port=30002):
    """Esperar que el robot se conecte y enviar la pose"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    server_socket.settimeout(5)  # Timeout para aceptar la conexión
    print("[🖥️] Esperando conexión del robot...")
    
    conn = None
    try:
        conn, addr = server_socket.accept()
        print(f"[🤖] Conectado robot desde {addr}")

        # Formatear pose como texto
        pose_str = " ".join(f"{x:.5f}" for x in pose)
        conn.sendall(pose_str.encode())

        print(f"[✅] Pose enviada: {pose_str}")
    except socket.timeout:
        print(f"[❌] Error: No se pudo establecer conexión con el robot dentro de {5} segundos.")
    except Exception as e:
        print(f"[❌] Error durante la comunicación con el robot: {e}")
    finally:
        if conn:
            conn.close()
        server_socket.close()



def move_robot(pose, robot_ip="192.168.0.102", speed=0.25, accel=1.2):
    cmd = (
        f"movel(p[{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}, "
        f"{pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f}], "
        f"a={accel:.2f}, v={speed:.2f})\n"
    )
    with socket.create_connection((robot_ip, 30001), timeout=5) as sock:
        sock.sendall(cmd.encode())
    print("[✅] Sent:", cmd.strip())



def compute_pose(bbox, depth_frame, color_shape, K, dist, T_tcp_cam):
    x1, y1, x2, y2 = bbox
    h_c, w_c = color_shape[:2]
    u, v = (x1 + x2) / 2, (y1 + y2) / 2     # píxel en color

    # ---- profundidad (media 5×5) ----
    h_d, w_d = depth_frame.shape[:2]
    uu = int(u * w_d / w_c)
    vv = int(v * h_d / h_c)
    patch = depth_frame[max(0, vv-2):min(h_d, vv+3),
                        max(0, uu-2):min(w_d, uu+3)]
    nz = patch[patch > 0]
    if nz.size == 0:
        return None
    z = np.mean(nz) / 1000.0                 # mm → m

    # ---- punto 3-D en coords CÁMARA (Realsense: x-derecha, y-abajo, z-adelante) ----
    pts = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), K, dist)
    xn, yn = pts.reshape(-1)
    Xc, Yc, Zc = xn * z, yn * z, z          # cámara

    # ---- remapeo de ejes a convención UR (x→adelante, y→izquierda, z→arriba) ----
    # Camera  (Realsense):  X→derecha, Y→abajo, Z→adelante
    # Queremos UR-base previa a la hand-eye: X_camUR→ Zc, Y_camUR→ -Xc, Z_camUR→ -Yc
    Pc_camUR = np.array([Zc, -Xc, -Yc, 1.0])

    # ---- cámara → TCP (usando la INVERTIDA) ----
    Pc_tcp = T_tcp_cam @ Pc_camUR

    # ---- TCP → base (pose actual vía puerto 30003) ----
    T_base_tcp = get_tcp_pose_4x4()
    Pw_base = T_base_tcp @ Pc_tcp

    # orientación = orientación actual del TCP
    rvec, _ = cv2.Rodrigues(T_base_tcp[:3, :3])
    rx, ry, rz = rvec.ravel()

    return [float(Pw_base[0]), float(Pw_base[1]), float(Pw_base[2]),
            rx, ry, rz]

import socket, struct, cv2
def get_tcp_pose_4x4(ip="192.168.0.102", port=30003):
    """Lee 1 paquete del stream primario (30003) y devuelve matriz 4×4 TCP→base."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.4)
    s.connect((ip, port))
    hdr = s.recv(4)
    (pkt_len,) = struct.unpack(">I", hdr)
    data = s.recv(pkt_len-4, socket.MSG_WAITALL)
    s.close()

    pose6 = struct.unpack(">6d", data[440:488])  # 444-4 = 440
    x,y,z, rx,ry,rz = pose6
    angle = (rx**2+ry**2+rz**2)**0.5
    R = np.eye(3) if angle < 1e-6 else cv2.Rodrigues(np.array([rx,ry,rz]))[0]
    T = np.eye(4); T[:3,:3], T[:3,3] = R, [x,y,z]
    return T



