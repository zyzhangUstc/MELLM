import numpy as np
import cv2

def extract_eye_region_rois(
    image,
    landmarks,
    indices_map=None,
    roi_size=5,
    offset_pixels=None,
    pad_mode=cv2.BORDER_REFLECT_101
):
    if indices_map is None:
        indices_map = {
            'left_eye_lower' : [68, 67, 66],
            'right_eye_lower': [76, 75, 74],
            'left_eye_upper' : [62, 63, 64],
            'right_eye_upper': [70, 71, 72],
        }

    if offset_pixels is None:
        dx = roi_size
        dy = roi_size
    else:
        dx, dy = offset_pixels

    half = roi_size // 2

    img = image
    if img.ndim == 2:
        H, W = img.shape
        C = 1
    else:
        H, W, C = img.shape

    pad = int(max(dx, dy) + half + 2)
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, pad_mode)

    results = {}
    lm = np.asarray(landmarks)  # (98,2)
    if lm.shape[0] < 98:
        raise ValueError("landmarks 应至少包含 98 个点（形状应为 (98,2)）。")

    for region_name, indices in indices_map.items():
        region_list = []
        n = len(indices)
        for i, idx1 in enumerate(indices):
            idx0 = int(idx1) - 1
            x, y = lm[idx0]  
            x_shift = (-dx if i == 0 else (dx if i == n-1 else 0))
            if 'lower' in region_name:
                y_shift = dy   
            elif 'upper' in region_name:
                y_shift = -dy  
            else:
                y_shift = 0

            cx = float(x + x_shift)
            cy = float(y + y_shift)

            cx_pad = int(round(cx)) + pad
            cy_pad = int(round(cy)) + pad

            tx = cx_pad - half
            ty = cy_pad - half

            patch = padded[ty:ty + roi_size, tx:tx + roi_size].copy()
            if C == 1 and patch.ndim == 3:
                patch = patch[:, :, 0]

            region_list.append({
                'index': idx1,
                'center': (cx, cy),
                'topleft_in_padded': (tx, ty),
                'topleft_in_original': (tx - pad, ty - pad),  
                'roi': patch
            })

        results[region_name] = region_list

    return results

def draw_rois_on_image(image, rois, roi_size=5, line_thickness=1, text_scale=0.4):

    img = image.copy()
    H, W = img.shape[:2]

    # 颜色与样式
    rect_color = (0, 255, 0)  
    center_color = (0, 0, 255) 
    text_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    roi_dic = []
    for region_name, items in rois.items():
        for item in items:
            idx = item['index']  
            tx, ty = item.get('topleft_in_original', (None, None))
            if tx is None:
                cx, cy = item['center']
                tx = int(round(cx)) - roi_size // 2
                ty = int(round(cy)) - roi_size // 2

            x1 = int(round(tx))
            y1 = int(round(ty))
            x2 = x1 + roi_size - 1
            y2 = y1 + roi_size - 1

            x1_clip = max(0, min(W - 1, x1))
            y1_clip = max(0, min(H - 1, y1))
            x2_clip = max(0, min(W - 1, x2))
            y2_clip = max(0, min(H - 1, y2))
            temp_dic = {
                '62': "left_eye_upper_left",
                '63': 'left_eye_upper_center',
                '64': 'left_eye_upper_right', 

                '68': "left_eye_down_left",
                '67': 'left_eye_down_center',
                '66': 'left_eye_down_right', 

                '70': "right_eye_upper_left",
                '71': 'right_eye_upper_center',
                '72': 'right_eye_upper_right', 

                '76': "right_eye_down_left",
                '75': 'right_eye_down_center',
                '74': 'right_eye_down_right', 
            }
            if x2_clip >= x1_clip and y2_clip >= y1_clip:
                cv2.rectangle(img, (x1_clip, y1_clip), (x2_clip, y2_clip), rect_color, thickness=line_thickness)
                text = str(idx)
                roi_dic.append({
                    'pos_name': temp_dic[text],
                    'position': [x1_clip, y1_clip, x2_clip, y2_clip]
                })

    return roi_dic



def get_center(landmarks, indices):
    pts = landmarks[np.array(indices) - 1] 
    center = pts.mean(axis=0)
    return tuple(center.tolist())

def draw_roi(roi_list, center, roi_size=10, color=(0,255,0), thickness=1, label=None):
    cx, cy = int(round(center[0])), int(round(center[1]))
    half = roi_size // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    roi_list.append({
        'pos_name': label,
        'position': [x1, y1, x2, y2]
    })
    return roi_list

def draw_all_rois(landmarks, roi_size=15):
    roi_list = []

    roi_defs = {
        "left_outer_brow": [35, 42, 34],
        "left_inner_brow": [37, 38, 40, 39],
        "right_outer_brow": [46, 48, 47],
        "right_inner_brow": [43, 44, 51, 50],
    }

    for name, indices in roi_defs.items():
        center = get_center(landmarks, indices)
        roi_list = draw_roi(roi_list, center, roi_size=roi_size, color=(0,255,0), label=name)

    left_wing = landmarks[56-1] 
    left_wing_center = (left_wing[0] - roi_size // 2, left_wing[1] - roi_size // 2)
    roi_list = draw_roi(roi_list, left_wing_center, roi_size=roi_size + 5, color=(255,0,0), label="left_wing")

    right_wing = landmarks[60-1]
    right_wing_center = (right_wing[0] + roi_size // 2, right_wing[1] - roi_size // 2)
    roi_list = draw_roi(roi_list, right_wing_center, roi_size=roi_size + 5, color=(0,0,255), label="right_wing")

    return roi_list



def get_center_of_indices(landmarks, indices):
    """
    indices: list of 1-based indices
    returns: (cx, cy) float
    """
    pts = landmarks[np.array(indices, dtype=int) - 1]  # convert to 0-based
    return tuple(pts.mean(axis=0).tolist())

def make_box_from_center(center, roi_size):
    """
    center: (x,y)
    roi_size: int
    returns x1,y1,x2,y2 (integers, clipped to form exactly roi_size x roi_size)
    """
    cx = int(round(center[0]))
    cy = int(round(center[1]))
    half = roi_size // 2
    x1 = cx - half
    y1 = cy - half
    x2 = x1 + roi_size - 1
    y2 = y1 + roi_size - 1
    return x1, y1, x2, y2

def clip_box_to_image(box, img_w, img_h):
    x1,y1,x2,y2 = box
    # Note: after clipping width/height may be < roi_size, that's OK for drawing
    x1c = max(0, min(img_w-1, x1))
    y1c = max(0, min(img_h-1, y1))
    x2c = max(0, min(img_w-1, x2))
    y2c = max(0, min(img_h-1, y2))
    return x1c, y1c, x2c, y2c

def draw_mouth_rois_on_image(image, landmarks, roi_size=10, corner_shift=None):
    """
    image: BGR numpy array
    landmarks: (98,2) numpy array, (x,y), 1-based indices expected in helper calls
    roi_size: side length of square boxes (default 10)
    corner_shift: number of pixels to shift mouth corners horizontally (default roi_size)
    returns: image_with_rois, dict_of_boxes (name -> (x1,y1,x2,y2))
    """
    if corner_shift is None:
        corner_shift = roi_size

    h, w = image.shape[:2]
    boxes = {}

    # --- Mouth corner ROIs ---
    # left corner: base index 77, shift left
    left_corner = landmarks[77-1]  # 1-based -> 0-based
    left_center = (left_corner[0] - corner_shift, left_corner[1])
    box = make_box_from_center(left_center, roi_size)
    boxes['left_corner'] = clip_box_to_image(box, w, h)

    # right corner: base index 83, shift right
    right_corner = landmarks[83-1]
    right_center = (right_corner[0] + corner_shift, right_corner[1])
    box = make_box_from_center(right_center, roi_size)
    boxes['right_corner'] = clip_box_to_image(box, w, h)

    # --- Upper lip : three ROIs ---
    # 1) mid of 78 and 90
    c1 = get_center_of_indices(landmarks, [78])
    box = make_box_from_center(c1, roi_size)
    boxes['upper_78_90'] = clip_box_to_image(box, w, h)

    # 2) mid of 80 and 91
    c2 = get_center_of_indices(landmarks, [80])
    box = make_box_from_center(c2, roi_size)
    boxes['upper_80_91'] = clip_box_to_image(box, w, h)

    # 3) mid of 82 and 92
    c3 = get_center_of_indices(landmarks, [82])
    box = make_box_from_center(c3, roi_size)
    boxes['upper_82_92'] = clip_box_to_image(box, w, h)

    # --- Lower lip : three ROIs ---
    # 1) midpoint of 96, 87, 88 (three points)
    c4 = get_center_of_indices(landmarks, [87, 88])
    box = make_box_from_center(c4, roi_size)
    boxes['lower_96_87_88'] = clip_box_to_image(box, w, h)

    # 2) midpoint of 94, 85, 84
    c5 = get_center_of_indices(landmarks, [85, 84])
    box = make_box_from_center(c5, roi_size)
    boxes['lower_94_85_84'] = clip_box_to_image(box, w, h)

    # 3) midpoint of 95 and 86
    c6 = get_center_of_indices(landmarks, [86])
    box = make_box_from_center(c6, roi_size)
    boxes['lower_95_86'] = clip_box_to_image(box, w, h)

    # draw all boxes on a copy of image
    out = image.copy()
    # Colors (BGR)
    color_corner = (0, 0, 255)   # red
    color_upper  = (255, 0, 0)   # blue
    color_lower  = (0, 255, 0)   # green
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    text_th = 1
    roi_mouth = []
    for name, (x1,y1,x2,y2) in boxes.items():
        # choose color by name
        if name.startswith('left') or name.startswith('right') or 'corner' in name:
            col = color_corner
        elif name.startswith('upper'):
            col = color_upper
        else:
            col = color_lower

        # draw rectangle (cv2.rectangle uses inclusive coords)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, thickness)
        roi_mouth.append({
            'pos_name': name,
            'position': [x1, y1, x2, y2]
        })
    return roi_mouth


def face_scale_from_landmarks(landmarks):
    xs = landmarks[:,0]
    return float(xs.max() - xs.min())



def crop_cheek_rois(image, landmarks, face_bbox=None,
                    side_length_ratio=0.12,   
                    shit_ratio=0.05 
                   ):

    h_img, w_img = image.shape[:2]
    lm = np.array(landmarks, dtype=np.float32)

    L_eye_outer = lm[61-1]   # (x,y)
    R_eye_outer = lm[73-1]
    L_mouth = lm[77-1]
    R_mouth = lm[83-1]

    if face_bbox is not None:
        _, _, _, face_h = face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]
        H = float(face_h)
    else:
        y_coords = lm[:,1]
        H = float(y_coords.max() - y_coords.min()) 

    S = max(4, int(round(side_length_ratio * H)))      
    shit_dis = shit_ratio * H
    def make_square(center_x, center_y, side):
        half = side // 2
        x1 = int(round(center_x - half))
        y1 = int(round(center_y - half))
        x2 = x1 + side
        y2 = y1 + side
        return x1, y1, x2, y2

    def clip_bbox(x1,y1,x2,y2):
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(w_img, x2)
        y2c = min(h_img, y2)
        return x1c, y1c, x2c, y2c

    results = {}

    center_L = (L_eye_outer + L_mouth) / 2.0
    cx_L, cy_L = float(center_L[0]) - shit_dis, float(center_L[1]) - shit_dis

    x1L, y1L, x2L, y2L = make_square(cx_L, cy_L, S)

    x1Lc, y1Lc, x2Lc, y2Lc = clip_bbox(x1L, y1L, x2L, y2L)
    results['left'] = (x1Lc, y1Lc, x2Lc, y2Lc)

    center_R = (R_eye_outer + R_mouth) / 2.0
    cx_R, cy_R = float(center_R[0]) + shit_dis, float(center_R[1]) - shit_dis

    x1R, y1R, x2R, y2R = make_square(cx_R, cy_R, S)
    min_top_allowed_R = int(round(R_eye_outer[1]))
    if y1R < min_top_allowed_R:
        shift_down = min_top_allowed_R - y1R
        cy_R += shift_down
        x1R, y1R, x2R, y2R = make_square(cx_R, cy_R, S)

    x1Rc, y1Rc, x2Rc, y2Rc = clip_bbox(x1R, y1R, x2R, y2R)
    results['right'] = (x1Rc, y1Rc, x2Rc, y2Rc)

    return results['left'], results['right']


def draw_mouth_and_cheek_rois_on_image(image, landmarks,
                                      mouth_roi_size=10,
                                      chin_roi_size=25,
                                      cheek_roi_size=None,
                                      corner_shift=None,
                                      cheek_upper_ratio=0.02):
    """
    image: BGR image
    landmarks: (98,2) numpy array, (x,y)
    mouth_roi_size: side length for mouth ROIs (your specified 10)
    chin_roi_size: side length for chin ROI (15)
    cheek_roi_size: if None, auto from face width
    corner_shift: horizontal shift for mouth corners; if None use mouth_roi_size
    cheek_upper_ratio: how much (fraction of face_width) to shift cheek center upward
    returns: vis_image, boxes_dict
    """
    h, w = image.shape[:2]
    fw = face_scale_from_landmarks(landmarks)
    if corner_shift is None:
        corner_shift = mouth_roi_size
    boxes = {}
    # ---- chin ROI: center at landmark 17, size 15 ----
    chin_center = tuple(landmarks[17-1].tolist())
    if h - landmarks[17-1][1] < chin_roi_size / 2:
        chin_center = (
            chin_center[0], 
            (landmarks[17-1][1] + landmarks[86-1][1]) // 2
        )

    boxes['chin_17'] = clip_box_to_image(make_box_from_center(chin_center, chin_roi_size), w, h)

    boxes['left_cheek'], boxes['right_cheek'] = crop_cheek_rois(image, landmarks)

    roi_dic = []
    for name, (x1,y1,x2,y2) in boxes.items():
        roi_dic.append({
            'pos_name': name,
            'position': [x1, y1, x2, y2]    
        })

    return roi_dic

import os

def get_roi_flow(flow, landmarks):
    rois = extract_eye_region_rois(flow, landmarks, roi_size=10, offset_pixels=(5,5))
    rois_eyes = draw_rois_on_image(flow, rois, roi_size=10, line_thickness=1, text_scale=0.4)
    rois_brows_nose = draw_all_rois(landmarks, roi_size=15)

    roi_size = 15         
    corner_shift = roi_size // 2  

    roi_mouth = draw_mouth_rois_on_image(flow, landmarks, roi_size=roi_size, corner_shift=corner_shift)


    roi_cheek_chin = draw_mouth_and_cheek_rois_on_image(flow, landmarks,
                                                   mouth_roi_size=10,
                                                   chin_roi_size=25,
                                                   cheek_roi_size=None,  # auto from face width
                                                   corner_shift=10,
                                                   cheek_upper_ratio=0.02)
    roi_dic_total = rois_eyes + rois_brows_nose + roi_mouth + roi_cheek_chin
    flow_dic = {}
    h, w = flow.shape[:2]
    for sample in roi_dic_total:
        x1,y1,x2,y2 = sample['position']
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        roi_name = sample['pos_name']
        flow_dic[roi_name] = flow[y1:y2, x1:x2]

    return flow_dic, roi_dic_total


def describe_flow_direction(
    flow: np.ndarray,
    mean_mag_threshold: float = 1e-4,
    concentration_threshold: float = 0.15,
) -> str:
    """
    Describe the dominant motion direction in an ROI flow field.

    Args:
      flow: np.ndarray of shape (h, w, 2), dtype float. flow[...,0]=u (right+), flow[...,1]=v (down+).
      mean_mag_threshold: if mean magnitude < this, report "no significant motion".
      concentration_threshold: if resultant concentration R < this, report "no clear dominant direction".

    Returns:
      str: human-readable English description, e.g.
        "Motion is mainly right (dominant direction ≈ 351°, concentration=0.82, mean_mag=0.0034)."
    """
    EPS = 1e-12

    if flow is None:
        return "No flow data provided."

    arr = np.asarray(flow, dtype=np.float64)
    if arr.size == 0 or arr.shape[-1] != 2:
        return "Invalid flow shape."

    # Flatten vectors
    vecs = arr.reshape(-1, 2)  # (N,2)
    u = vecs[:, 0]
    v_img = vecs[:, 1]  # image coordinate: down is positive

    mags = np.linalg.norm(vecs, axis=1)
    mean_mag = float(mags.mean()) if mags.size > 0 else 0.0
    total_mag = float(mags.sum())

    if mean_mag < mean_mag_threshold:
        return f"No significant motion detected (mean magnitude ≈ {mean_mag:.6g})."

    # Convert to mathematical coordinates: v_math = -v_img (so up is +)
    vecs_math = np.stack([u, -v_img], axis=1)  # (N,2)

    # Resultant vector (sum of vectors) = magnitude-weighted direction
    resultant = vecs_math.sum(axis=0)  # [Rx, Ry]
    R_norm = np.linalg.norm(resultant)
    # concentration R = |resultant| / sum(magnitudes)  (in [0,1])
    R = float(R_norm / (total_mag + EPS))

    # compute angle in degrees, CCW from +x (right)
    angle_rad = np.arctan2(resultant[1], resultant[0])  # [-pi, pi]
    angle_deg = float(np.degrees(angle_rad) % 360.0)  # [0,360)

    # Determine nearest octant direction (centers at 0,45,90,...)
    octant_centers = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)
    octant_labels = [
        "right",
        "up-right",
        "up",
        "up-left",
        "left",
        "down-left",
        "down",
        "down-right",
    ]
    # angular distance (circular)
    ang_diffs = np.abs(((angle_deg - octant_centers + 180) % 360) - 180)
    idx = int(np.argmin(ang_diffs))
    label = octant_labels[idx]

    # If concentration very low, state no clear dominant direction
    if R < concentration_threshold:
        return (
            f"No clear dominant direction (motion is more isotropic or noisy; "
            f"mean magnitude ≈ {mean_mag:.6g}, concentration ≈ {R:.3f})."
        )

    # Round angle to integer degrees for human readability, keep one decimal for mean mag
    angle_rounded = int(round(angle_deg)) % 360
    return f"{label}({angle_rounded}°))"

roi_name_map = {
    "left_eye_down_left": "Left Lower Eyelid (Left Side)",
    "left_eye_down_center": "Left Lower Eyelid (Center)",
    "left_eye_down_right": "Left Lower Eyelid (Right Side)",
    "right_eye_down_left": "Right Lower Eyelid (Left Side)",
    "right_eye_down_center": "Right Lower Eyelid (Center)",
    "right_eye_down_right": "Right Lower Eyelid (Right Side)",
    "left_eye_upper_left": "Left Upper Eyelid (Left Side)",
    "left_eye_upper_center": "Left Upper Eyelid (Center)",
    "left_eye_upper_right": "Left Upper Eyelid (Right Side)",
    "right_eye_upper_left": "Right Upper Eyelid (Left Side)",
    "right_eye_upper_center": "Right Upper Eyelid (Center)",
    "right_eye_upper_right": "Right Upper Eyelid (Right Side)",
    "left_outer_brow": "Left Outer Eyebrow",
    "left_inner_brow": "Left Inner Eyebrow",
    "right_outer_brow": "Right Outer Eyebrow",
    "right_inner_brow": "Right Inner Eyebrow",
    "left_wing": "Left Nostril Wing",
    "right_wing": "Right Nostril Wing",
    "left_corner": "Left Mouth Corner",
    "right_corner": "Right Mouth Corner",
    "upper_78_90": "Upper Lip (Left Side)",
    "upper_80_91": "Upper Lip (Center)",
    "upper_82_92": "Upper Lip (Right Side)",
    "lower_96_87_88": "Lower Lip (Left Side)",
    "lower_94_85_84": "Lower Lip (Right Side)",
    "lower_95_86": "Lower Lip (Center)",
    "chin_17": "Chin",
    "left_cheek": "Left Cheek",
    "right_cheek": "Right Cheek",
}
