# send2robot2.py   (rename the file or the import so they match!)

import socket
import struct
import cv2
import numpy as np
import threading
import queue

UR_IP     = "192.168.0.102"
RT_PORT   = 30003           # real-time interface (only ONE client allowed)
POSE_PORT = 50000           # one single port the robot will connect to


# ── REAL-TIME SOCKET (persistent) ────────────────────────────────────────────
try:
    rt_socket = socket.create_connection((UR_IP, RT_PORT), timeout=0.4)
    print(f"[INFO] Connected to UR RT port {RT_PORT}.")
except Exception as e:
    print(f"[ERROR] Could not connect RT port: {e}")
    rt_socket = None


def get_tcp_pose_persistent():
    """
    Read one packet from the already-open rt_socket and return T_base_fl (4×4).
    """
    if rt_socket is None:
        raise RuntimeError("RT socket not connected.")

    # 1) packet length prefix (4 bytes, big-endian uint32)
    ln_bytes = rt_socket.recv(4)
    if len(ln_bytes) < 4:
        raise RuntimeError("RT stream closed.")
    ln = struct.unpack(">I", ln_bytes)[0]

    # 2) actual packet
    buf = rt_socket.recv(ln - 4, socket.MSG_WAITALL)
    if len(buf) < ln - 4:
        raise RuntimeError("Incomplete RT packet.")

    # 3) cartesian tool pose is at bytes 440–487 (6 doubles, big-endian)
    x, y, z, rx, ry, rz = struct.unpack(">6d", buf[440:488])

    R, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def send_pose_secondary_monitor(pose, port: int = POSE_PORT):
    """
    Wait for the URScript to connect on *port* and send one line with the pose.
    """
    # 1) open a tiny TCP server that the URScript will connect to
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    srv.settimeout(10.0)                     # ← don’t block forever
    print(f"[POSE SERVER] Listening on :{port} …")

    try:
        conn, addr = srv.accept()            # blocks max 10 s
        print(f"[POSE SERVER] UR connected from {addr}")

        line = "(" + ", ".join(f"{v:.6f}" for v in pose) + ")\n"
        conn.sendall(line.encode("utf8"))
        print(f"[POSE SERVER] Sent: {line.strip()}")

    except socket.timeout:
        print("[POSE SERVER] No URScript connection within 10 s.")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        srv.close()


pose_queue = queue.Queue()

def _pose_server(host="0.0.0.0", port=POSE_PORT):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(1)
    print(f"[POSE SERVER] Waiting on :{port} …")
    conn, addr = listener.accept()
    print(f"[POSE SERVER] Robot connected from {addr}")

    while True:
        pose = pose_queue.get()
        if pose is None:
            break

        line = "(" + ", ".join(f"{v:.6f}" for v in pose) + ")\n"
        data = line.encode("utf-8")

        while True:                              # retry loop
            try:
                conn.sendall(data)
                print(f"[POSE SERVER] Sent: {line.strip()}")
                break                            # success → next pose
            except (BrokenPipeError, ConnectionResetError):
                # robot closed link – wait for it to open a new one
                try:
                    conn.close()
                except Exception:
                    pass
                print("[POSE SERVER] Link lost – waiting for reconnect …")
                conn, addr = listener.accept()
                print(f"[POSE SERVER] Robot re-connected from {addr}")

    conn.close()
    listener.close()


def start_pose_server():
    threading.Thread(target=_pose_server, daemon=True).start()
