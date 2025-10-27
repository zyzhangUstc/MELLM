import numpy as np
import random
import math
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
import torch.nn.functional as F
from torchvision.transforms import ColorJitter
from scipy.interpolate import griddata

def interpolate_holes_numpy(image, valid_mask):
    """
    Interpolate black holes in a NumPy image using linear interpolation.
    
    Args:
        image (np.ndarray): 2D or 3D NumPy array representing the image.
        valid_mask (np.ndarray): 2D binary mask, 1 = valid, 0 = invalid.
    
    Returns:
        np.ndarray: Image with holes interpolated.
    """
    # Ensure image is float
    image = image.astype(np.float32)
    valid_mask = valid_mask.astype(bool)
    
    # Create mesh grid of coordinates
    grid_y, grid_x = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    
    # Get valid coordinates and corresponding values
    valid_coords = np.stack((grid_y[valid_mask], grid_x[valid_mask]), axis=-1)
    valid_values = image[valid_mask]
    
    # Get coordinates of invalid pixels
    invalid_coords = np.stack((grid_y[~valid_mask], grid_x[~valid_mask]), axis=-1)
    
    # Perform interpolation
    interpolated_values = griddata(
        valid_coords, valid_values, invalid_coords, method='linear'
    )
    
    # Fill the invalid pixels in the image
    interpolated_image = image.copy()
    interpolated_image[~valid_mask] = interpolated_values
    interpolated_image[np.isnan(interpolated_image)] = 0
    return interpolated_image

class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, args=None):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2
        
    def color_transform(self, img1, img2):
        """ Photometric augmentation """
        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def spatial_transform(self, img1, img2, flow, valid):
        pad_t = 0
        pad_b = 0
        pad_l = 0
        pad_r = 0
        if self.crop_size[0] > img1.shape[0]:
            pad_b = self.crop_size[0] - img1.shape[0]
        if self.crop_size[1] > img1.shape[1]:
            pad_r = self.crop_size[1] - img1.shape[1]
            
        if pad_b != 0 or pad_r != 0:
            img1 = np.pad(img1, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            img2 = np.pad(img2, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            flow = np.pad(flow, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            valid = np.pad(valid, ((pad_t, pad_b), (pad_l, pad_r)), 'constant', constant_values=((0, 0), (0, 0)))
        
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)  

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)
        
        valid = (valid.astype(np.float32) > 0.5).astype(bool)
        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow[~valid] = 0           
            valid = valid.astype(np.float32)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            valid = cv2.resize(valid, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y] / (valid + 1e-5)[:, :, None]
            valid = (valid.astype(np.float32) > 0.5).astype(bool)
            flow[~valid] = 0

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        if img1.shape[0] == self.crop_size[0]:
            y0 = 0
        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            
        if img1.shape[1] == self.crop_size[1]:
            x0 = 0
        else:
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        return img1, img2, flow, valid


    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)
        return img1, img2, flow, valid



class FlowAugmentor_v2:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, args=None):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2
        
    def color_transform(self, img1, img2):
        """ Photometric augmentation """
        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def augment_face_pair(self, img1, img2, facial_flow, head_flow, valid, landmarks1, landmarks2, 
                        crop_size=(480,400),
                        face_min_ratio=0.8,   # 人脸最小占比（相对于 min(crop_h,crop_w)）
                        face_max_ratio=0.95,   # 人脸最大占比（除非强制占满）
                        prob_face_fill=0.15,  # 有多少概率强制占满（占比=1.0）
                        allow_horizontal_flip=True,
                        h_flip_prob=0.5,
                        stretch_prob=0.25,
                        max_stretch=0.1,
                        spatial_aug_prob=0.8,
                        min_log_scale=-0.5,   # 额外的随机 log2 缩放范围（可弱扰动）
                        max_log_scale=0.5
                        ):
        """
        返回增强后的 img1,img2,flow,valid,landmarks1,landmarks2
        """
        # --- 1. 统一输入类型与 basic vars ---
        img1 = img1.copy()
        img2 = img2.copy()
        head_flow = head_flow.copy()
        facial_flow = facial_flow.copy()

        valid = valid.copy()
        landmarks1 = landmarks1.copy()
        landmarks2 = landmarks2.copy()

        crop_h, crop_w = crop_size
        H, W = img1.shape[:2]

        # --- 2. pad 如果需要，保证后续能 crop ---
        pad_t = pad_b = pad_l = pad_r = 0
        if crop_h > H:
            pad_b = crop_h - H
        if crop_w > W:
            pad_r = crop_w - W
        if pad_b != 0 or pad_r != 0:
            img1 = np.pad(img1, ((pad_t, pad_b), (pad_l, pad_r), (0,0)), 'constant', constant_values=0)
            img2 = np.pad(img2, ((pad_t, pad_b), (pad_l, pad_r), (0,0)), 'constant', constant_values=0)
            facial_flow = np.pad(facial_flow, ((pad_t, pad_b), (pad_l, pad_r), (0,0)), 'constant', constant_values=0)
            head_flow = np.pad(head_flow, ((pad_t, pad_b), (pad_l, pad_r), (0,0)), 'constant', constant_values=0)           
            valid = np.pad(valid, ((pad_t, pad_b), (pad_l, pad_r)), 'constant', constant_values=0)
            landmarks1 += np.array([pad_l, pad_t])  # 更新 landmark 坐标
            landmarks2 += np.array([pad_l, pad_t])
            H, W = img1.shape[:2]

        # --- 3. 从 landmarks 估计人脸 bbox & 当前大小 ---
        # 使用 landmarks1（也可取两帧的 union）
        if landmarks1.shape[0] == 0:
            # 若没有 landmarks，回退到随机缩放策略（不基于脸）
            face_w = face_h = max(H, W) * 0.5
            face_cx = W / 2.0
            face_cy = H / 2.0
        else:
            min_xy = landmarks1.min(axis=0)  # (x_min, y_min)
            max_xy = landmarks1.max(axis=0)
            face_w = max_xy[0] - min_xy[0]
            face_h = max_xy[1] - min_xy[1]
            face_cx = (min_xy[0] + max_xy[0]) / 2.0
            face_cy = (min_xy[1] + max_xy[1]) / 2.0

        current_face_size = max(face_w, face_h, 1.0)  # 避免零除

        # --- 4. 采样目标 face 占比（有概率强制占满） ---
        if np.random.rand() < prob_face_fill:
            target_ratio = 1.0  # 人脸需要尽量占满（相对于 min(crop_h,crop_w)）
        else:
            target_ratio = np.random.uniform(face_min_ratio, face_max_ratio)

        # desired face size in pixels (relative to crop short side)
        desired_face_pixels = target_ratio * min(crop_h, crop_w)

        # --- 5. 计算缩放因子（以使 face 大小接近 desired_face_pixels） ---
        base_scale = desired_face_pixels / current_face_size

        # 在 base_scale 上添加少量随机 log-scale 干扰（让不完全总是精确）
        jitter = 2 ** np.random.uniform(min_log_scale, max_log_scale)
        scale_x = base_scale * jitter
        scale_y = base_scale * jitter

        # 可随机拉伸（小幅）
        if np.random.rand() < stretch_prob:
            scale_x *= 2 ** np.random.uniform(-max_stretch, max_stretch)
            scale_y *= 2 ** np.random.uniform(-max_stretch, max_stretch)

        # 保证 scale 不太小（以免后面 crop 不足）——这里保证经过缩放图像至少能 crop 出 crop_size
        min_scale_height = (crop_h + 1) / float(H)
        min_scale_width  = (crop_w + 1) / float(W)
        min_scale = max(min_scale_height, min_scale_width)
        scale_x = max(scale_x, min_scale)
        scale_y = max(scale_y, min_scale)

        # --- 6. 按概率执行空间增强（resize）并处理 flow/valid（与原代码思路一致） ---
        valid_bool = (valid.astype(np.float32) > 0.5).astype(bool)
        if np.random.rand() < spatial_aug_prob:
            # 先将 invalid 的 flow 置 0，避免插值污染
            facial_flow[~valid_bool] = 0.0
            head_flow[~valid_bool] = 0.0           
            # resize imgs
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            # resize flow 和 valid
            facial_flow = cv2.resize(facial_flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            head_flow = cv2.resize(head_flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            valid_f = valid_bool.astype(np.float32)
            valid_f = cv2.resize(valid_f, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            # 修正 flow 的尺度（像素移动量要按缩放放大）
            # 注意：valid_f 在插值后是 [0,1] 的权重，用来做除法补偿
            facial_flow = facial_flow * np.array([scale_x, scale_y])[None, None, :] / (valid_f[:, :, None] + 1e-5)
            head_flow = head_flow * np.array([scale_x, scale_y])[None, None, :] / (valid_f[:, :, None] + 1e-5)

            # 恢复 valid 布尔
            valid_bool = (valid_f > 0.5)
            facial_flow[~valid_bool] = 0.0
            head_flow[~valid_bool] = 0.0

            valid = valid_bool.astype(np.uint8)

            # landmarks 同步缩放
            landmarks1 = landmarks1 * np.array([scale_x, scale_y])
            landmarks2 = landmarks2 * np.array([scale_x, scale_y])

            H, W = img1.shape[:2]

        # --- 7. 以 face center 为中心做随机裁剪（确保最终 crop 包含 face） ---
        # jitter 控制 face 在 crop 中不是完全居中
        max_center_jitter = 0.2  # 相对于 crop 大小的最大抖动比例，可调
        jitter_x = int(np.round((np.random.uniform(-max_center_jitter, max_center_jitter)) * crop_w))
        jitter_y = int(np.round((np.random.uniform(-max_center_jitter, max_center_jitter)) * crop_h))

        cx = int(round(face_cx * scale_x)) if landmarks1.shape[0] > 0 else int(W / 2)
        cy = int(round(face_cy * scale_y)) if landmarks1.shape[0] > 0 else int(H / 2)

        # 设定 crop 左上角，使 face center 大致在 crop 中心 + jitter
        x0 = cx - crop_w // 2 + jitter_x
        y0 = cy - crop_h // 2 + jitter_y

        # 如果希望“占满”效果更强，可以把 x0,y0 推到使 bbox 与边界接触
        if target_ratio == 1.0 and landmarks1.shape[0] > 0:
            # 让 face bbox 尽量贴边（随机选择水平方向或垂直方向贴边）
            min_xy = landmarks1.min(axis=0)
            max_xy = landmarks1.max(axis=0)
            bbox_w = (max_xy[0] - min_xy[0])
            bbox_h = (max_xy[1] - min_xy[1])
            # 计算使 bbox 左对齐或右对齐的 x0 值（以增加“占满”的效果）
            if np.random.rand() < 0.5:
                # 左对齐
                x0 = int(round(min_xy[0] * scale_x))
            else:
                # 右对齐（bbox 右边落在 crop 右边）
                x0 = int(round(max_xy[0] * scale_x)) - crop_w + 1
            if np.random.rand() < 0.5:
                # 上对齐
                y0 = int(round(min_xy[1] * scale_y))
            else:
                y0 = int(round(max_xy[1] * scale_y)) - crop_h + 1

        # clamp x0,y0 到合法范围
        x0 = max(0, min(x0, W - crop_w)) if W > crop_w else 0
        y0 = max(0, min(y0, H - crop_h)) if H > crop_h else 0

        # 执行裁剪
        img1 = img1[y0:y0+crop_h, x0:x0+crop_w]
        img2 = img2[y0:y0+crop_h, x0:x0+crop_w]
        head_flow = head_flow[y0:y0+crop_h, x0:x0+crop_w]
        facial_flow = facial_flow[y0:y0+crop_h, x0:x0+crop_w]        
        valid = valid[y0:y0+crop_h, x0:x0+crop_w]

        # landmark 坐标也需要减去 crop 左上角
        landmarks1 = landmarks1 - np.array([x0, y0])
        landmarks2 = landmarks2 - np.array([x0, y0])

        # --- 8. 水平翻转（允许）并修正 flow 与 landmarks ---
        if allow_horizontal_flip and np.random.rand() < h_flip_prob:
            # 翻转图像与 flow（x 分量取反）
            img1 = img1[:, ::-1].copy()
            img2 = img2[:, ::-1].copy()
            head_flow = head_flow[:, ::-1].copy()
            facial_flow = facial_flow[:, ::-1].copy()
            head_flow[..., 0] *= -1.0  # x 分量取反
            facial_flow[..., 0] *= -1.0  # x 分量取反
            # valid 翻转
            valid = valid[:, ::-1].copy()

            # landmarks x = (width-1) - x
            landmarks1[:, 0] = (crop_w - 1) - landmarks1[:, 0]
            landmarks2[:, 0] = (crop_w - 1) - landmarks2[:, 0]

            # 可选：如果 landmarks 包含左右语义 (如 left_eye / right_eye)，需要交换对应索引
            # e.g. mirror_indices = [ ... ]  # 长度 N，每个 i 指向在镜像后对应的新索引
            # landmarks1 = landmarks1[mirror_indices]
            # landmarks2 = landmarks2[mirror_indices]

        # --- 9. 最后确保 valid 为布尔，flow 在 invalid 位置为0 ---
        valid = (valid.astype(np.float32) > 0.5).astype(np.uint8)
        facial_flow[valid == 0] = 0.0
        head_flow[valid == 0] = 0.0

        return img1, img2, facial_flow, head_flow, valid, landmarks1, landmarks2

    def __call__(self, img1, img2, facial_flow, head_flow, valid, lm_img1, lm_img2):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, facial_flow, head_flow, valid, lm_img1, lm_img2 = self.augment_face_pair(img1, img2, facial_flow, head_flow, valid, lm_img1, lm_img2)
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        facial_flow = np.ascontiguousarray(facial_flow)
        head_flow = np.ascontiguousarray(head_flow)
        valid = np.ascontiguousarray(valid)
        return img1, img2, facial_flow, head_flow, valid, lm_img1, lm_img2
