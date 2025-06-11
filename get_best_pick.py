"""
YOLO ➜ SAM ➜ find_best_grasp (debug/visual version)
---------------------------------------------------
Shows every intermediate step using a fixed input image path:
* Bounding-boxes from YOLO (coloured rectangles)
* SAM mask overlay (transparent red)
* Centroid, principal axes, and final grasp line
Prints key data to stdout so you can verify numbers.

Run inside your (venv) with:
    python yolo_sam_grasp_debug.py
Press any key in the OpenCV window to advance; ESC or close window to exit.
"""

from pathlib import Path
import sys
import cv2
import numpy as np
from ultralytics import YOLO                               # YOLOv8 detect
from segment_anything import SamPredictor, sam_model_registry
import torch

# ----------------------------- UTILITIES -----------------------------------

def overlay_mask(img: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.4):
    """Returns a copy of *img* with *mask* overlayed in *color* (BGR)."""
    out = img.copy()
    col = np.full_like(img, color, dtype=np.uint8)
    return np.where(mask[..., None], cv2.addWeighted(col, alpha, out, 1 - alpha, 0), out)

# --------------------------- GRASP FUNCTION --------------------------------

def find_best_grasp(mask: np.ndarray, max_width_px: int, step_px: int = 3):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        raise ValueError("Mask is empty – cannot compute grasp.")
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    centroid = np.array([cy, cx])  # (row, col)

    pts = np.column_stack(np.where(mask > 0))
    pts_center = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts_center, full_matrices=False)
    major_vec = vt[0] / np.linalg.norm(vt[0])
    minor_vec = vt[1] / np.linalg.norm(vt[1])
    grasp_theta = np.arctan2(minor_vec[1], minor_vec[0])

    proj = pts_center @ major_vec
    half_len = int(np.percentile(np.abs(proj), 95))

    best_score, best_center = -np.inf, None
    for offset in range(-half_len, half_len + 1, step_px):
        c = centroid + offset * major_vec
        p1 = c + (max_width_px / 2) * minor_vec
        p2 = c - (max_width_px / 2) * minor_vec
        line = np.linspace(p1, p2, max_width_px)
        rr = np.clip(line[:, 0].astype(int), 0, mask.shape[0] - 1)
        cc = np.clip(line[:, 1].astype(int), 0, mask.shape[1] - 1)
        if np.all(mask[rr, cc]):
            torque = np.linalg.norm(c - centroid)
            score = -torque
            if score > best_score:
                best_score, best_center = score, c
    if best_center is None:
        raise RuntimeError("No valid grasp found – increase max_width_px or check mask.")

    return (float(best_center[1]), float(best_center[0])), float(grasp_theta), centroid, major_vec, minor_vec

# ------------------------------ MAIN ---------------------------------------

def main():
    # -------- fixed image path --------
    img_path = Path("/home/lab/Desktop/TFG/destornillador2.png")
    if not img_path.is_file():
        print(f"Image not found: {img_path}")
        sys.exit(1)

    # -------- parameters --------
    pix_per_mm = 3.2
    max_width_mm = 55
    max_width_px = int(max_width_mm * pix_per_mm)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # -------- models ------------
    yolo = YOLO("/home/lab/Desktop/TFG/yolo_finetune/runs/yolov8_coco_plus_tools_en/weights/best.pt")
    yolo.to(device)  # Move YOLO model to GPU if available
    sam_path = "/home/lab/Desktop/TFG/sam_vit_b_01ec64.pth"
    sam = sam_model_registry["vit_b"](checkpoint=sam_path)
    sam.to(device)  # Move SAM model to GPU if available
    predictor = SamPredictor(sam)

    # -------- load image --------
    im_bgr = cv2.imread(str(img_path))
    if im_bgr is None:
        print("Failed to load image.")
        sys.exit(1)

    # -------- YOLO detect --------
    det = yolo.predict(im_bgr, conf=0.5, verbose=False)[0]
    num = len(det.boxes)
    print(f"\nDetected {num} objects\n")
    if num == 0:
        cv2.imshow("No detections", im_bgr)
        print("No objects detected - showing raw image.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Draw initial bounding-boxes
    bgr_boxes = im_bgr.copy()
    for i, box in enumerate(det.boxes.xyxy.cpu().numpy(), 1):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(bgr_boxes, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(bgr_boxes, f"{i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("YOLO Bounding Boxes", bgr_boxes)
    cv2.waitKey(0)
    cv2.destroyWindow("YOLO Bounding Boxes")

    predictor.set_image(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))
    # -------- per-object debug --------
    for idx, box in enumerate(det.boxes.xyxy.cpu().numpy(), start=1):
        x1, y1, x2, y2 = map(int, box)
        print(f"Object {idx}: bbox=({x1},{y1})-({x2},{y2})")

        masks, _, _ = predictor.predict(
            box=np.array([x1, y1, x2, y2], np.float32),
            multimask_output=False
        )
        mask = masks[0]
        print(f"  Mask pixels: {mask.sum()} (non-zero)")
        
        # --- compute safe opening along minor axis instead of image width ---
        pts = np.column_stack(np.where(mask > 0))
        pts_center = pts - pts.mean(axis=0)
        # Reuse minor_vec from find_best_grasp or recompute here
        _, _, vt = np.linalg.svd(pts_center, full_matrices=False)
        minor_vec = vt[1] / np.linalg.norm(vt[1])
        proj_minor = pts_center @ minor_vec
        width_minor_px = proj_minor.max() - proj_minor.min()
        margin_px = 4
        safe_open_px = max(1, int(width_minor_px - margin_px))
        base_open_px = int(max_width_mm * pix_per_mm)
        max_width_px = min(base_open_px, safe_open_px)
        print(f"  width_minor_px={width_minor_px:.1f}, using max_width_px={max_width_px}")
        try:
            (u, v), theta, centroid, major_vec, minor_vec = find_best_grasp(mask, max_width_px)
            print(f"  Grasp center: ({u:.1f},{v:.1f}), angle: {np.degrees(theta):.1f}°")
        except Exception as e:
            print(f"  Grasp failed: {e}")
            continue

        vis = overlay_mask(im_bgr, mask)
        c_int = (int(centroid[1]), int(centroid[0]))
        L = max_width_px
        p1_major = (int(c_int[0] + L * major_vec[1]), int(c_int[1] + L * major_vec[0]))
        p1_minor = (int(c_int[0] + L * minor_vec[1]), int(c_int[1] + L * minor_vec[0]))
        cv2.arrowedLine(vis, c_int, p1_major, (0,255,255), 2)
        cv2.arrowedLine(vis, c_int, p1_minor, (0,165,255), 2)
        length = max_width_px // 2
        dx, dy = np.cos(theta)*length, np.sin(theta)*length
        pA = (int(u+dx), int(v+dy)); pB = (int(u-dx), int(v-dy))
        cv2.line(vis, pA, pB, (0,255,0), 3)
        cv2.circle(vis, (int(u),int(v)), 5, (0,0,255), -1)

        window = f"Object {idx} Debug"
        cv2.imshow(window, vis)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit early
            cv2.destroyWindow(window)
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
