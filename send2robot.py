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



def compute_pose(bbox, depth_frame, color_frame_shape):
    """
    Computes the pose (x_center_color, y_center_color, z_depth_meters, rx, ry, rz)
    of the detected object using bounding box and depth frame.
    Assumes the depth frame values are in millimeters.

    Args:
        bbox (list): Bounding box coordinates [x1, y1, x2, y2] in the color image.
        depth_frame (numpy.ndarray): The depth frame (numpy array, values in mm).
        color_frame_shape (tuple): The shape of the color frame (height, width).

    Returns:
        list: Pose [x_center_color, y_center_color, z_depth_meters, rx, ry, rz] or None on error.
              Rotations (rx, ry, rz) are placeholders (0,0,0).
    """
    try:
        x1, y1, x2, y2 = bbox
        color_height, color_width = color_frame_shape

        # Calculate the center coordinates of the bounding box in the color image
        x_center_color = (x1 + x2) / 2
        y_center_color = (y1 + y2) / 2

        # Get the depth value from the depth frame at the center coordinates.
        # Ensure the coordinates are within the depth frame bounds.
        # This requires that the depth frame is registered/aligned with the color frame.
        # If they are not aligned, or have different FoV/resolutions without calibration,
        # this mapping will be inaccurate.
        depth_height, depth_width = depth_frame.shape

        # Simple scaling if resolutions differ. For accuracy, use camera intrinsics for projection.
        x_center_depth = int(x_center_color * (depth_width / color_width))
        y_center_depth = int(y_center_color * (depth_height / color_height))

        # Clamp coordinates to be within depth frame bounds
        x_center_depth = max(0, min(x_center_depth, depth_width - 1))
        y_center_depth = max(0, min(y_center_depth, depth_height - 1))

        # Get depth value (e.g., in millimeters)
        z_depth_mm = depth_frame[y_center_depth, x_center_depth]

        if z_depth_mm == 0: # Often 0 means no valid depth data
            # print(f"Warning: Depth value at ({y_center_depth}, {x_center_depth}) is 0. Invalid depth. Bbox center ({x_center_color},{y_center_color})")
            # It might be better to average depth over a small patch around the center
            # or use a more robust method if 0 is a common 'no return' value.
            patch_size = 5 # pixels
            y_start, y_end = max(0, y_center_depth - patch_size), min(depth_height, y_center_depth + patch_size + 1)
            x_start, x_end = max(0, x_center_depth - patch_size), min(depth_width, x_center_depth + patch_size + 1)
            depth_patch = depth_frame[y_start:y_end, x_start:x_end]
            non_zero_depths = depth_patch[depth_patch > 0]
            if non_zero_depths.size > 0:
                z_depth_mm = np.mean(non_zero_depths)
                # print(f"Using mean of non-zero patch: {z_depth_mm:.2f} mm")
            else:
                print(f"Warning: Still no valid depth in patch for bbox center ({x_center_color},{y_center_color}). Pose computation failed.")
                return None


        # Convert depth to meters
        z_depth_meters = z_depth_mm / 1000.0

        # Placeholder for rotations
        rx, ry, rz = 0.0, 0.0, 0.0 # Placeholder for actual rotation calculation

        # The X and Y returned here are pixel coordinates in the color image.
        # If the robot needs world coordinates, further transformation using camera intrinsics is needed.
        # For now, assuming (x_center_color, y_center_color) and z_depth_meters are what's expected.
        return [x_center_color, y_center_color, z_depth_meters, rx, ry, rz]

    except IndexError:
        print(f"Error: Depth coordinates ({y_center_depth}, {x_center_depth}) likely out of bounds for depth_frame shape {depth_frame.shape}. Bbox: {bbox}, Color shape: {color_frame_shape}")
        return None
    except Exception as e:
        print(f"Error computing pose: {e}")
        return None



