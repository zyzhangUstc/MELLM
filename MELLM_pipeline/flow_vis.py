import numpy as np
import cv2


def flow_uv_to_colors_hue_mix(u, v, convert_to_bgr=False, invert_y=False):

    assert u.shape == v.shape
    h0, w0 = u.shape
    eps = 1e-9

    if invert_y:
        v_proc = -v
    else:
        v_proc = v

    ang = np.arctan2(v_proc, u)  # -pi .. pi
    deg = np.degrees(ang)  # -180 .. 180
    hue_deg = (deg + 360.0) % 360.0  # 0 .. 360

    rad = np.sqrt(u * u + v * v) 


    H = np.floor(hue_deg / 2.0).astype(np.uint8)  # 0..179
    S = np.full_like(H, 255, dtype=np.uint8)
    V = np.full_like(H, 255, dtype=np.uint8)
    hsv = np.stack([H, S, V], axis=-1)  # uint8 HSV image

    rgb_full = (
        cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    )  # float in [0,1]

    col_final = np.empty_like(rgb_full, dtype=np.float32)  # [H,W,3]

    mask_in = rad <= 1.0
    if np.any(mask_in):
        rad_in = rad[mask_in].reshape(-1, 1)
        color_in = rgb_full[mask_in]  # (N,3)
        col_final[mask_in] = 1.0 - rad_in * (1.0 - color_in)

    mask_out = ~mask_in
    if np.any(mask_out):
        col_final[mask_out] = rgb_full[mask_out] * 0.75

    col_final = np.clip(col_final, 0.0, 1.0)

    if convert_to_bgr:
        col_final = col_final[..., ::-1]  # RGB -> BGR
    img_uint8 = np.floor(col_final * 255.0).astype(np.uint8)
    return img_uint8


def flow_to_image(
    flow_uv, clip_flow=None, convert_to_bgr=False, max_magnitude=None, invert_y=False
):

    assert (
        flow_uv.ndim == 3 and flow_uv.shape[2] == 2
    ), "flow_uv must have shape [H,W,2]"
    u = flow_uv[..., 0].astype(np.float32)
    v = flow_uv[..., 1].astype(np.float32)

    if clip_flow is not None:
        u = np.clip(u, -clip_flow, clip_flow)
        v = np.clip(v, -clip_flow, clip_flow)

    rad = np.sqrt(u * u + v * v)
    eps = 1e-9
    if max_magnitude is None:
        rad_max = np.max(rad)
    else:
        rad_max = float(max_magnitude)

    if rad_max <= 0 or not np.isfinite(rad_max):
        rad_max = 1.0
    # rad_max = 1.0

    u_norm = u / (rad_max + eps)
    v_norm = v / (rad_max + eps)

    img = flow_uv_to_colors_hue_mix(
        u_norm, v_norm, convert_to_bgr=convert_to_bgr, invert_y=invert_y
    )
    return img


import math


def generate_angle_color_wheel(save_path="angle_colors.png"):
    angles = list(range(0, 360, 30))  # 0,30,...330
    block_size = 100  
    margin = 20
    font = cv2.FONT_HERSHEY_SIMPLEX

    rows = 3
    cols = 4
    canvas_h = rows * (block_size + margin) + margin
    canvas_w = cols * (block_size + margin) + margin
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for idx, deg in enumerate(angles):
        theta = math.radians(deg)
        u = np.cos(theta) * np.ones((block_size, block_size), dtype=np.float32)
        v = np.sin(theta) * np.ones((block_size, block_size), dtype=np.float32)
        flow = np.stack([u, v], axis=-1)
        color_block = flow_to_image(flow, convert_to_bgr=True, invert_y=True)

        r = idx // cols
        c = idx % cols
        y0 = margin + r * (block_size + margin)
        x0 = margin + c * (block_size + margin)
        canvas[y0 : y0 + block_size, x0 : x0 + block_size] = color_block

        text = f"{deg}deg"
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 1)
        tx = x0 + (block_size - tw) // 2
        ty = y0 + block_size + th + 5
        cv2.putText(canvas, text, (tx, ty), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, canvas)
