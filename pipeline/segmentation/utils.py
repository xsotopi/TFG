import numpy as np
import cv2

def overlay_mask(img: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.4):
    """
    Overlay a binary mask onto img in the given BGR color.
    Returns:
        np.ndarray: Image with the mask overlayed.
    """
    out = img.copy()
    col = np.full_like(img, color, dtype=np.uint8)
    return np.where(mask[:, :, None], cv2.addWeighted(col, alpha, out, 1 - alpha, 0), out)


def find_best_grasp(mask: np.ndarray, max_width_px: int, step_px: int = 3):
    """
    Analyzes segmentation mask and finds the best grasp position and angle.
    1. Finds the orientation of the object using PCA.
    2. Searchs for a stable grasp along major axis.
    3. Best grasp is centered on the object and close to its center of mass.
    Return:
        (gx, gy) : grasp centre (x, y) in px
        grasp_theta : grasp angle
        centroid : global centroid (x, y)
        major_vec, minor_vec : unit PCA axes (x, y)
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise ValueError("Mask is empty – cannot compute grasp.")

    # 1. PCA to find the orientation of the object
    pts_xy   = np.column_stack((xs, ys)).astype(np.float32)
    centroid = pts_xy.mean(axis=0)
    pts_c    = pts_xy - centroid
    _, _, vt = np.linalg.svd(pts_c, full_matrices=False)
    major_vec, minor_vec = vt[0], vt[1]
    grasp_theta = np.arctan2(minor_vec[1], minor_vec[0])

    # 2. Search for best grasp point along the major axis
    proj     = pts_c @ major_vec
    half_len = int(np.percentile(np.abs(proj), 95))

    best_score, best_center = -np.inf, None
    for offset in range(-half_len, half_len + 1, step_px):
        c  = centroid + offset * major_vec
        p1 = c + (max_width_px / 2) * minor_vec
        p2 = c - (max_width_px / 2) * minor_vec

        line = np.linspace(p1, p2, max_width_px)
        xs_l = np.clip(line[:, 0].astype(int), 0, mask.shape[1] - 1)
        ys_l = np.clip(line[:, 1].astype(int), 0, mask.shape[0] - 1)
        in_mask_ratio = mask[ys_l, xs_l].mean()
        # If over 90% of grasp line is within mask, consider it valid
        if in_mask_ratio > 0.9:
            torque = np.linalg.norm(c - centroid)
            score  = -torque
            if score > best_score:
                best_score, best_center = score, c

    if best_center is None:
        raise RuntimeError(
            "No valid grasp found – increase max_width_px or check mask."
        )

    gx, gy = best_center
    return (float(gx), float(gy)), float(grasp_theta), centroid, major_vec, minor_vec
    