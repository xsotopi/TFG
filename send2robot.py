import socket

def send_pose_to_robot(pose, robot_ip="192.168.0.100", port=30003):
    """
    Sends a pose [x, y, z, rx, ry, rz] to the robot's listening socket.
    """
    try:
        with socket.create_connection((robot_ip, port), timeout=5) as sock:
            data = " ".join(map(str, pose)) + "\n"
            sock.sendall(data.encode())
            print("[✅] Pose sent to robot:", pose)
    except Exception as e:
        print("[❌] Failed to send pose:", e)

def compute_pose (bbox):
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2

    world_x = 0.3  # dummy
    world_y = 0.2  # dummy
    world_z = 0.1  # dummy
    rx, ry, rz = 0, 3.14, 0

    pose = [world_x, world_y, world_z, rx, ry, rz]