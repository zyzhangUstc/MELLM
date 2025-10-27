import numpy as np
import cv2


def flow_uv_to_colors_hue_mix(u, v, convert_to_bgr=False, invert_y=False):
    """
    将 u,v 映射为颜色：
      - 方向 -> HSV 色相 (0° 向右为红，角度逆时针增加)
      - 强度 rad = sqrt(u^2+v^2) 作为从白到纯色的线性混合系数
      - 超出 rad>1 的纯色则整体乘 0.75 变暗
    参数:
      u, v: ndarray [H,W] (float)，可以是已归一化的（通常在 flow_to_image 中处理归一化）
      convert_to_bgr: bool，True 则输出 BGR uint8（适合 cv2），否则输出 RGB uint8
      invert_y: bool，若 True 则对 v 取负（常用于图像坐标系 v 向下为正）
    返回:
      img: uint8 ndarray [H,W,3]，RGB 或 BGR（取决于 convert_to_bgr）
    """
    assert u.shape == v.shape
    h0, w0 = u.shape
    eps = 1e-9

    # 根据图像坐标系是否需要反向 v
    if invert_y:
        v_proc = -v
    else:
        v_proc = v

    # 角度：atan2(y,x) -> 0 对应 +x（向右），逆时针为正
    ang = np.arctan2(v_proc, u)  # -pi .. pi
    deg = np.degrees(ang)  # -180 .. 180
    hue_deg = (deg + 360.0) % 360.0  # 0 .. 360

    # 归一化幅值（注意：这里假定传入的 u,v 可能已经被外部归一化；此处仅计算 rad）
    rad = np.sqrt(u * u + v * v)  # 可能 >1

    # 生成纯色 (HSV: H -> hue, S=255, V=255)
    # OpenCV H 范围是 0..179 对应 0..360°
    H = np.floor(hue_deg / 2.0).astype(np.uint8)  # 0..179
    S = np.full_like(H, 255, dtype=np.uint8)
    V = np.full_like(H, 255, dtype=np.uint8)
    hsv = np.stack([H, S, V], axis=-1)  # uint8 HSV image

    # 用 OpenCV 把 HSV->RGB（我们用 RGB 进行插值；最后再决定是否返回 BGR）
    rgb_full = (
        cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    )  # float in [0,1]

    # 准备输出并按照 rad 做插值：col_final = 1 - rad*(1 - color)
    col_final = np.empty_like(rgb_full, dtype=np.float32)  # [H,W,3]

    # 对于 rad <= 1：col_final = (1-rad)*1 + rad*color = 1 - rad*(1-color)
    mask_in = rad <= 1.0
    if np.any(mask_in):
        # 广播 rad 到三个通道
        rad_in = rad[mask_in].reshape(-1, 1)
        color_in = rgb_full[mask_in]  # (N,3)
        col_final[mask_in] = 1.0 - rad_in * (1.0 - color_in)

    # 对于 rad > 1：col_final = color * 0.75
    mask_out = ~mask_in
    if np.any(mask_out):
        col_final[mask_out] = rgb_full[mask_out] * 0.75

    # 数值稳健性：裁剪 0..1
    col_final = np.clip(col_final, 0.0, 1.0)

    # 最终转 uint8；注意 convert_to_bgr 决定通道顺序
    if convert_to_bgr:
        col_final = col_final[..., ::-1]  # RGB -> BGR
    img_uint8 = np.floor(col_final * 255.0).astype(np.uint8)
    return img_uint8


def flow_to_image(
    flow_uv, clip_flow=None, convert_to_bgr=False, max_magnitude=None, invert_y=False
):
    """
    Drop-in 风格函数：接收 flow_uv [H,W,2]，返回可视化图像 uint8 [H,W,3]
    这里采用的视觉规则正是你要求的：方向->HSV色相，强度->在白色与纯色之间线性混合，超出范围缩暗。
    参数:
      flow_uv: ndarray [H,W,2]
      clip_flow: 若不为 None，则先对每个分量裁到 [-clip_flow, clip_flow]
      convert_to_bgr: 最终返回 BGR（True）或 RGB（False）
      max_magnitude: 若不为 None，用它来归一化幅值（跨帧对比时传固定值），否则用当前帧最大值
      invert_y: 是否反转 v 分量（图像坐标通常为 True）
    返回:
      img: uint8 [H,W,3]
    """
    assert (
        flow_uv.ndim == 3 and flow_uv.shape[2] == 2
    ), "flow_uv must have shape [H,W,2]"
    u = flow_uv[..., 0].astype(np.float32)
    v = flow_uv[..., 1].astype(np.float32)

    if clip_flow is not None:
        u = np.clip(u, -clip_flow, clip_flow)
        v = np.clip(v, -clip_flow, clip_flow)

    # 归一化：与原实现类似，先以当前帧的最大幅值作为除数（或使用用户给定的 max_magnitude）
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

    # 交给 hue-mix 的渲染器
    img = flow_uv_to_colors_hue_mix(
        u_norm, v_norm, convert_to_bgr=convert_to_bgr, invert_y=invert_y
    )
    return img


import math


def generate_angle_color_wheel(save_path="angle_colors.png"):
    angles = list(range(0, 360, 30))  # 0,30,...330
    block_size = 100  # 每个小色块尺寸
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

        # 在方块正中下方写角度
        text = f"{deg}deg"
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 1)
        tx = x0 + (block_size - tw) // 2
        ty = y0 + block_size + th + 5
        cv2.putText(canvas, text, (tx, ty), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, canvas)
    print(f"保存完毕: {save_path}")
