# tcp_pose_server.py
# ------------------

import socket
from urx.ursecmon import SecondaryMonitor

UR_IP = "192.168.0.102"   # Robot’s IP
PORT_SERVER = 30004       # Must match URScript

def main():
    # 1) Attach to the UR’s secondary monitor (port 30003)
    secmon = SecondaryMonitor(UR_IP)
    secmon.wait()

    # 2) Create a TCP server on port 30004
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", PORT_SERVER))
    sock.listen(1)
    print(f"[POSE SERVER] Listening on port {PORT_SERVER}...")

    while True:
        conn, addr = sock.accept()
        print(f"[POSE SERVER] Connection from {addr}")

        # 3) Read current pose
        pose = secmon.get_cartesian_info(wait=True)
        current_pose = [
            pose["X"],
            pose["Y"],
            pose["Z"],
            pose["Rx"],
            pose["Ry"],
            pose["Rz"],
        ]
        print("  Current pose:", current_pose)

        # 4) Add +0.05 m to Z
        new_pose = current_pose.copy()
        new_pose[2] += 0.05
        print("  New pose (Z+5 cm):", new_pose)

        # 5) Send them in the format "(x, y, z, Rx, Ry, Rz)\n"
        payload = "(" + ", ".join(f"{v:.6f}" for v in new_pose) + ")\n"
        conn.sendall(payload.encode("utf8"))
        print(f"[POSE SERVER] Sent: {payload.strip()}")

        # 6) Close and loop
        conn.close()
        print("[POSE SERVER] Closed connection. Waiting for next request.\n")

if __name__ == "__main__":
    main()
