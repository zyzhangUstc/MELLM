# import numpy as np
# import random
# import math
# from PIL import Image

# import cv2
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

# import torch
# from torchvision.transforms import ColorJitter
# import torch.nn.functional as F


# class FlowAugmentor:
#     def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
#         # spatial augmentation params
#         self.crop_size = crop_size
#         self.min_scale = min_scale
#         self.max_scale = max_scale
#         self.spatial_aug_prob = 0.8
#         self.stretch_prob = 0.8
#         self.max_stretch = 0.2

#         # flip augmentation params
#         self.do_flip = do_flip
#         self.h_flip_prob = 0.5
#         self.v_flip_prob = 0.1

#         # photometric augmentation params
#         self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
#         self.asymmetric_color_aug_prob = 0.2
#         self.eraser_aug_prob = 0.5

#     def color_transform(self, img1, img2):
#         """ Photometric augmentation """

#         # asymmetric
#         if np.random.rand() < self.asymmetric_color_aug_prob:
#             img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
#             img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

#         # symmetric
#         else:
#             image_stack = np.concatenate([img1, img2], axis=0)
#             image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
#             img1, img2 = np.split(image_stack, 2, axis=0)

#         return img1, img2

#     def eraser_transform(self, img1, img2, bounds=[50, 100]):
#         """ Occlusion augmentation """

#         ht, wd = img1.shape[:2]
#         if np.random.rand() < self.eraser_aug_prob:
#             mean_color = np.mean(img2.reshape(-1, 3), axis=0)
#             for _ in range(np.random.randint(1, 3)):
#                 x0 = np.random.randint(0, wd)
#                 y0 = np.random.randint(0, ht)
#                 dx = np.random.randint(bounds[0], bounds[1])
#                 dy = np.random.randint(bounds[0], bounds[1])
#                 img2[y0:y0+dy, x0:x0+dx, :] = mean_color

#         return img1, img2

#     def spatial_transform(self, img1, img2, flow_e, flow_h):
#         # randomly sample scale
#         ht, wd = img1.shape[:2]
#         min_scale = np.maximum(
#             (self.crop_size[0] + 8) / float(ht), 
#             (self.crop_size[1] + 8) / float(wd))

#         scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
#         scale_x = scale
#         scale_y = scale
#         if np.random.rand() < self.stretch_prob:
#             scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
#             scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
#         scale_x = np.clip(scale_x, min_scale, None)
#         scale_y = np.clip(scale_y, min_scale, None)

#         if np.random.rand() < self.spatial_aug_prob:
#             # rescale the images
#             img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
#             img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

#             flow_e = cv2.resize(flow_e, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
#             flow_e = flow_e * [scale_x, scale_y]

#             flow_h = cv2.resize(flow_h, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
#             flow_h = flow_h * [scale_x, scale_y]

#         if self.do_flip:
#             if np.random.rand() < self.h_flip_prob: # h-flip
#                 img1 = img1[:, ::-1]
#                 img2 = img2[:, ::-1]
#                 flow_e = flow_e[:, ::-1] * [-1.0, 1.0]
#                 flow_h = flow_h[:, ::-1] * [-1.0, 1.0]

#             if np.random.rand() < self.v_flip_prob: # v-flip
#                 img1 = img1[::-1, :]
#                 img2 = img2[::-1, :]
#                 flow_e = flow_e[::-1, :] * [1.0, -1.0]
#                 flow_h = flow_h[::-1, :] * [1.0, -1.0]

#         y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
#         x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
#         img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
#         img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
#         flow_e = flow_e[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
#         flow_h = flow_h[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

#         return img1, img2, flow_e, flow_h

#     def __call__(self, img1, img2, flow_e, flow_h):
#         img1, img2 = self.color_transform(img1, img2)
#         img1, img2 = self.eraser_transform(img1, img2)
#         img1, img2, flow_e, flow_h = self.spatial_transform(img1, img2, flow_e, flow_h)

#         img1 = np.ascontiguousarray(img1)
#         img2 = np.ascontiguousarray(img2)
#         flow_e = np.ascontiguousarray(flow_e)
#         flow_h = np.ascontiguousarray(flow_h)

#         return img1, img2, flow_e, flow_h


# class SparseFlowAugmentor:
#     def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
#         # spatial augmentation params
#         self.crop_size = crop_size
#         self.min_scale = min_scale
#         self.max_scale = max_scale
#         self.spatial_aug_prob = 0.8
#         self.stretch_prob = 0.8
#         self.max_stretch = 0.2

#         # flip augmentation params
#         self.do_flip = do_flip
#         self.h_flip_prob = 0.5
#         self.v_flip_prob = 0.1

#         # photometric augmentation params
#         self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
#         self.asymmetric_color_aug_prob = 0.2
#         self.eraser_aug_prob = 0.5
        
#     def color_transform(self, img1, img2):
#         image_stack = np.concatenate([img1, img2], axis=0)
#         image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
#         img1, img2 = np.split(image_stack, 2, axis=0)
#         return img1, img2

#     def eraser_transform(self, img1, img2):
#         ht, wd = img1.shape[:2]
#         if np.random.rand() < self.eraser_aug_prob:
#             mean_color = np.mean(img2.reshape(-1, 3), axis=0)
#             for _ in range(np.random.randint(1, 3)):
#                 x0 = np.random.randint(0, wd)
#                 y0 = np.random.randint(0, ht)
#                 dx = np.random.randint(50, 100)
#                 dy = np.random.randint(50, 100)
#                 img2[y0:y0+dy, x0:x0+dx, :] = mean_color

#         return img1, img2

#     def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
#         ht, wd = flow.shape[:2]
#         coords = np.meshgrid(np.arange(wd), np.arange(ht))
#         coords = np.stack(coords, axis=-1)

#         coords = coords.reshape(-1, 2).astype(np.float32)
#         flow = flow.reshape(-1, 2).astype(np.float32)
#         valid = valid.reshape(-1).astype(np.float32)

#         coords0 = coords[valid>=1]
#         flow0 = flow[valid>=1]

#         ht1 = int(round(ht * fy))
#         wd1 = int(round(wd * fx))

#         coords1 = coords0 * [fx, fy]
#         flow1 = flow0 * [fx, fy]

#         xx = np.round(coords1[:,0]).astype(np.int32)
#         yy = np.round(coords1[:,1]).astype(np.int32)

#         v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
#         xx = xx[v]
#         yy = yy[v]
#         flow1 = flow1[v]

#         flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
#         valid_img = np.zeros([ht1, wd1], dtype=np.int32)

#         flow_img[yy, xx] = flow1
#         valid_img[yy, xx] = 1

#         return flow_img, valid_img

#     def spatial_transform(self, img1, img2, flow_e,flow_h, valid):
#         # randomly sample scale

#         ht, wd = img1.shape[:2]
#         min_scale = np.maximum(
#             (self.crop_size[0] + 1) / float(ht), 
#             (self.crop_size[1] + 1) / float(wd))

#         scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
#         scale_x = np.clip(scale, min_scale, None)
#         scale_y = np.clip(scale, min_scale, None)

#         if np.random.rand() < self.spatial_aug_prob:
#             # rescale the images
#             img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
#             img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
#             flow_e, valid = self.resize_sparse_flow_map(flow_e, valid, fx=scale_x, fy=scale_y)
#             flow_h, valid = self.resize_sparse_flow_map(flow_h, valid, fx=scale_x, fy=scale_y)
#         if self.do_flip:
#             if np.random.rand() < 0.5: # h-flip
#                 img1 = img1[:, ::-1]
#                 img2 = img2[:, ::-1]
#                 flow = flow[:, ::-1] * [-1.0, 1.0]
#                 valid = valid[:, ::-1]

#         margin_y = 20
#         margin_x = 50

#         y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
#         x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

#         y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
#         x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

#         img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
#         img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
#         flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
#         valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
#         return img1, img2, flow, valid

#     def __call__(self, img1, img2, flow_e, flow_h, valid):
#         img1, img2 = self.color_transform(img1, img2)
#         img1, img2 = self.eraser_transform(img1, img2)
#         img1, img2, flow_e, flow_h, valid = self.spatial_transform(img1, img2, flow_e, flow_h, valid)

#         img1 = np.ascontiguousarray(img1)
#         img2 = np.ascontiguousarray(img2)
#         flow = np.ascontiguousarray(flow)
#         valid = np.ascontiguousarray(valid)

#         return img1, img2, flow, valid



import io
import random
import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
from torchvision.utils import save_image
# ---------- Utilities ----------
def to_tensor_uint8(img):
    """Accept PIL or numpy uint8 HWC, return torch.float32 CxHxW in [0,1]."""
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            raise ValueError("numpy input expected uint8")
        img = Image.fromarray(img)
    # now img is PIL
    t = TF.to_tensor(img)  # float in [0,1], CxHxW
    return t

def to_pil(img_tensor):
    """tensor CxHxW in [0,1] -> PIL"""
    return TF.to_pil_image(img_tensor.clamp(0,1))

def ensure_square_crop_bbox(x0, y0, x1, y1, img_w, img_h, square_expand=True):
    """Given bbox coords (pixel), return square bbox within image bounds."""
    w = x1 - x0
    h = y1 - y0
    if square_expand:
        s = max(w, h)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        x0_new = cx - s/2.0
        x1_new = cx + s/2.0
        y0_new = cy - s/2.0
        y1_new = cy + s/2.0
    else:
        x0_new, y0_new, x1_new, y1_new = x0, y0, x1, y1
    # clamp to image
    x0c = max(0, math.floor(x0_new))
    y0c = max(0, math.floor(y0_new))
    x1c = min(img_w, math.ceil(x1_new))
    y1c = min(img_h, math.ceil(y1_new))
    return x0c, y0c, x1c, y1c

def crop_image_tensor(img_tensor, x0, y0, x1, y1):
    """img_tensor: [C,H,W] float. Returns cropped [C,h,w]."""
    _, H, W = img_tensor.shape
    x0i, y0i = int(x0), int(y0)
    x1i, y1i = int(x1), int(y1)
    return img_tensor[:, y0i:y1i, x0i:x1i].contiguous()

def crop_flow_tensor(flow_tensor, x0, y0, x1, y1):
    """flow_tensor: [2,H,W], crop spatially. Note flow vectors are unchanged by crop (coords shift externally)."""
    _, H, W = flow_tensor.shape
    x0i, y0i = int(x0), int(y0)
    x1i, y1i = int(x1), int(y1)
    return flow_tensor[:, y0i:y1i, x0i:x1i].contiguous()

def crop_landmarks(lm, x0, y0):
    """lm: (N,2) array/tensor in pixel coords. Subtract crop offset."""
    lm2 = lm.copy() if isinstance(lm, np.ndarray) else lm.clone()
    lm2[..., 0] = lm2[..., 0] - x0
    lm2[..., 1] = lm2[..., 1] - y0
    return lm2

def resize_image_tensor(img_tensor, out_h, out_w, interp_mode='bilinear'):
    """img_tensor: [C,H,W]. returns [C,out_h,out_w]"""
    t = img_tensor.unsqueeze(0)
    t2 = F.interpolate(t, size=(out_h, out_w), mode=interp_mode, align_corners=False if interp_mode=='bilinear' else None)
    return t2.squeeze(0)

def resize_flow_tensor(flow, out_h, out_w):
    """flow: [2,H,W] in pixels. Resize spatially and scale vector components accordingly."""
    _, H, W = flow.shape
    # resize each channel
    f = flow.unsqueeze(0)  # [1,2,H,W]
    f_res = F.interpolate(f, size=(out_h, out_w), mode='bilinear', align_corners=False)
    f_res = f_res.squeeze(0)
    scale_x = float(out_w) / float(W)
    scale_y = float(out_h) / float(H)
    f_res = f_res.clone()
    f_res[0, :, :] = f_res[0, :, :] * scale_x
    f_res[1, :, :] = f_res[1, :, :] * scale_y
    return f_res

def flip_horizontal_image_tensor(img_tensor):
    """flip spatially"""
    return torch.flip(img_tensor, dims=[2])  # flip width dim

def flip_horizontal_flow(flow):
    """flow: [2,H,W] ; spatially flip and flip x-channel sign"""
    f = torch.flip(flow, dims=[2])
    f = f.clone()
    f[0] = -f[0]  # invert x component
    return f

def flip_landmarks_x(lm, img_w):
    """lm: numpy or tensor N x 2. img_w: width after crop (pixel)"""
    lm2 = lm.copy() if isinstance(lm, np.ndarray) else lm.clone()
    lm2[..., 0] = (img_w - 1) - lm2[..., 0]
    return lm2


# ---------- Photometric augmentations (images only) ----------
def random_color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.15, p=0.8):
    if random.random() > p:
        return img
    # img: tensor CxHxW in [0,1]
    # using torchvision transforms functional
    b = 1.0 + random.uniform(-brightness, brightness)
    c = 1.0 + random.uniform(-contrast, contrast)
    s = 1.0 + random.uniform(-saturation, saturation)
    img = TF.adjust_brightness(img, b)
    img = TF.adjust_contrast(img, c)
    img = TF.adjust_saturation(img, s)
    return img

from PIL import ImageFilter

def random_gaussian_blur(img, kernel_max=5, p=0.5):
    if random.random() > p:
        return img
    pil = to_pil(img)  # 转 PIL
    r = random.uniform(0.1, 1.5)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=r))  # 注意这里改成 ImageFilter
    return TF.to_tensor(pil)

def random_noise(img, std_max=0.02, p=0.5):
    if random.random() > p:
        return img
    std = random.uniform(0.0, std_max)
    noise = torch.randn_like(img) * std
    return (img + noise).clamp(0,1)

def random_jpeg_compress(img, p=0.3, qmin=60, qmax=95):
    """simulate jpeg compression by roundtrip to PIL JPEG in-memory"""
    if random.random() > p:
        return img
    pil = to_pil(img)
    buf = io.BytesIO()
    q = random.randint(qmin, qmax)
    pil.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    pil2 = Image.open(buf).convert('RGB')
    return TF.to_tensor(pil2)

def random_occlusion(img, max_h_ratio=0.12, max_w_ratio=0.25, p=0.4):
    """place a random rectangle inside the face crop area to simulate occlusion"""
    if random.random() > p:
        return img
    C, H, W = img.shape
    h = int(random.uniform(0.05, max_h_ratio) * H)
    w = int(random.uniform(0.05, max_w_ratio) * W)
    x0 = random.randint(int(0.05 * W), max(1, W - w - 1))
    y0 = random.randint(int(0.05 * H), max(1, H - h - 1))
    img[:, y0:y0 + h, x0:x0 + w] = torch.rand(C,1,1, device=img.device)  # random patch
    return img


# ---------- Main preprocessing function ----------
def preprocess_pair(
    I_n, I_e, lm_n, lm_e, facial_flow, head_flow,
    out_size=384,
    face_expand=1.25,
    augment=True,
    device='cpu'
):
    """
    I_n, I_e: PIL.Image or numpy uint8 HxWxC or torch tensor CxHxW in [0,1]
    lm_n, lm_e: numpy arrays shape (N,2) in pixel coords relative to original full images
    flow_facial: torch tensor [2,H,W] full-res flow in pixels (matching original full image dims)
    facial_flow: torch tensor [2,H,W] full-res flow in pixels (matching original full image dims)
    out_size: desired square output size (int)
    face_expand: bbox margin multiplier (1.0 means tight box; >1 enlarges)
    augment: whether to apply photometric/geometric augment
    Returns: tensors on 'device':
      I_n_t, I_e_t: float tensors [3, out_size, out_size], in [0,1]
      lm_n_t, lm_e_t: float tensors [N,2] in pixel coords relative to cropped/resized images
      facial_flow_t: float tensor [2, out_size, out_size] in pixels (flow vectors scaled)
      meta: dict with original crop and scale info (useful for diagnostics)
    """
    # ---------------- normalize inputs to tensors ----------------
    # convert images to float tensors CxHxW in [0,1]
    if isinstance(I_n, torch.Tensor):
        img_n = I_n.clone()
        if img_n.max() > 2.0:  # maybe uint8 in tensor
            img_n = img_n.float() / 255.0
    else:
        img_n = to_tensor_uint8(I_n)
    if isinstance(I_e, torch.Tensor):
        img_e = I_e.clone()
        if img_e.max() > 2.0:
            img_e = img_e.float() / 255.0
    else:
        img_e = to_tensor_uint8(I_e)

    # facial_flow should be torch tensor [2,H,W]
    if not isinstance(facial_flow, torch.Tensor):
        facial_flow = torch.from_numpy(facial_flow).float()
    else:
        facial_flow = facial_flow.clone().float()

    if not isinstance(head_flow, torch.Tensor):
        head_flow = torch.from_numpy(head_flow).float()
    else:
        head_flow = head_flow.clone().float()

    # landmarks to numpy for easy math if necessary
    lm_n_np = lm_n.copy() if isinstance(lm_n, np.ndarray) else lm_n.detach().cpu().numpy()
    lm_e_np = lm_e.copy() if isinstance(lm_e, np.ndarray) else lm_e.detach().cpu().numpy()

    _, H_full, W_full = img_n.shape

    # ---------------- compute face bbox from union of landmarks (use both frames to be safe) ----------------
    all_lms = np.concatenate([lm_n_np, lm_e_np], axis=0) if lm_e_np is not None else lm_n_np
    x_min = float(np.min(all_lms[:, 0]))
    y_min = float(np.min(all_lms[:, 1]))
    x_max = float(np.max(all_lms[:, 0]))
    y_max = float(np.max(all_lms[:, 1]))

    # expand bbox
    w = x_max - x_min
    h = y_max - y_min
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    s = max(w, h) * face_expand
    x0 = cx - s/2.0
    y0 = cy - s/2.0
    x1 = cx + s/2.0
    y1 = cy + s/2.0

    # clamp to image bounds and ensure int
    x0c, y0c, x1c, y1c = ensure_square_crop_bbox(x0, y0, x1, y1, W_full, H_full, square_expand=False)
    crop_w = x1c - x0c
    crop_h = y1c - y0c
    # sometimes out-of-bound due to expand, pad if needed (here we simply clamp; you may pad with background if preferred)

    # -------------- crop images, flow, landmarks ----------------
    img_n_crop = crop_image_tensor(img_n, x0c, y0c, x1c, y1c)
    img_e_crop = crop_image_tensor(img_e, x0c, y0c, x1c, y1c)
    facial_flow_crop = crop_flow_tensor(facial_flow, x0c, y0c, x1c, y1c)
    head_flow_crop = crop_flow_tensor(head_flow, x0c, y0c, x1c, y1c)
    lm_n_crop = crop_landmarks(lm_n_np, x0c, y0c)
    lm_e_crop = crop_landmarks(lm_e_np, x0c, y0c)

    # -------------- optional augmentation BEFORE resize (geometric: random translate crop) --------------
    # We'll allow a small random shift inside the crop (translation) to simulate framing jitter.
    if augment:
        # max translate fraction relative to crop size (e.g., +/- 8%)
        max_tx = int(0.08 * crop_w)
        max_ty = int(0.08 * crop_h)
        tx = random.randint(-max_tx, max_tx) if max_tx > 0 else 0
        ty = random.randint(-max_ty, max_ty) if max_ty > 0 else 0
        if tx != 0 or ty != 0:
            # compute new crop coords, keep within original cropped patch bounds
            # note: we are cropping inside the already-cropped image to simulate shift
            C, Hc, Wc = img_n_crop.shape
            # compute intersection bounds
            x0s = max(0, tx)  # if tx positive, shift right means crop moves right inside patch
            y0s = max(0, ty)
            x1s = x0s + Wc - abs(tx)
            y1s = y0s + Hc - abs(ty)
            # apply spatial crop on crop (effectively shift content)
            # simpler: pad_and_crop approach: create padded patch then crop
            # but easier: shift landmarks and use F.pad if needed; to keep simple, only perform shifts that keep full patch inside
            # ensure stays valid
            if 0 <= x0s < x1s <= Wc and 0 <= y0s < y1s <= Hc:
                img_n_crop = img_n_crop[:, y0s:y1s, x0s:x1s]
                img_e_crop = img_e_crop[:, y0s:y1s, x0s:x1s]
                facial_flow_crop = facial_flow_crop[:, y0s:y1s, x0s:x1s]
                head_flow_crop = head_flow_crop[:, y0s:y1s, x0s:x1s]
                lm_n_crop[..., 0] = lm_n_crop[..., 0] - x0s
                lm_n_crop[..., 1] = lm_n_crop[..., 1] - y0s
                lm_e_crop[..., 0] = lm_e_crop[..., 0] - x0s
                lm_e_crop[..., 1] = lm_e_crop[..., 1] - y0s
                # update crop dims
                crop_h = y1s - y0s
                crop_w = x1s - x0s

    # -------------- resize to out_size and scale flow accordingly ----------------
    img_n_rs = resize_image_tensor(img_n_crop, out_size, out_size, interp_mode='bilinear')
    img_e_rs = resize_image_tensor(img_e_crop, out_size, out_size, interp_mode='bilinear')
    facial_flow_rs = resize_flow_tensor(facial_flow_crop, out_size, out_size)
    head_flow_rs = resize_flow_tensor(head_flow_crop, out_size, out_size)
    # scale landmarks accordingly (float)
    scale_x = out_size / float(crop_w)
    scale_y = out_size / float(crop_h)
    lm_n_rs = lm_n_crop.astype(np.float32)
    lm_e_rs = lm_e_crop.astype(np.float32)
    lm_n_rs[..., 0] = lm_n_rs[..., 0] * scale_x
    lm_n_rs[..., 1] = lm_n_rs[..., 1] * scale_y
    lm_e_rs[..., 0] = lm_e_rs[..., 0] * scale_x
    lm_e_rs[..., 1] = lm_e_rs[..., 1] * scale_y

    # -------------- random horizontal flip ----------------
    if augment and random.random() < 0.5:
        img_n_rs = flip_horizontal_image_tensor(img_n_rs)
        img_e_rs = flip_horizontal_image_tensor(img_e_rs)
        facial_flow_rs = flip_horizontal_flow(facial_flow_rs)
        head_flow_rs = flip_horizontal_flow(head_flow_rs)
        lm_n_rs = flip_landmarks_x(lm_n_rs, out_size)
        lm_e_rs = flip_landmarks_x(lm_e_rs, out_size)

    # -------------- photometric augmentations (images only) ----------------
    if augment:
        # apply same photometric jitter to both images? usually slightly different per-frame helps robustness to lighting change.
        # Here we apply *independent* small photometric transforms to each frame (closer to real capture).
        img_n_rs = random_color_jitter(img_n_rs, p=0.9)
        img_e_rs = random_color_jitter(img_e_rs, p=0.9)
        # blur / noise / jpeg (apply independently)
        img_n_rs = random_gaussian_blur(img_n_rs, p=0.3)
        img_e_rs = random_gaussian_blur(img_e_rs, p=0.3)
        img_n_rs = random_noise(img_n_rs, p=0.4)
        img_e_rs = random_noise(img_e_rs, p=0.4)
        img_n_rs = random_jpeg_compress(img_n_rs, p=0.25)
        img_e_rs = random_jpeg_compress(img_e_rs, p=0.25)
        # random occlusion inside face crop
        img_n_rs = random_occlusion(img_n_rs, p=0.25)
        img_e_rs = random_occlusion(img_e_rs, p=0.25)

    # clamp images
    img_n_rs = img_n_rs.clamp(0.0, 1.0)
    img_e_rs = img_e_rs.clamp(0.0, 1.0)

    # ---------------- prepare outputs and meta ----------------
    # convert landmarks to torch tensors (float), keep pixel coords relative to out_size
    lm_n_t = torch.from_numpy(lm_n_rs).float()
    lm_e_t = torch.from_numpy(lm_e_rs).float()
    return img_n_rs.to(device)*255.0, img_e_rs.to(device)*255.0, lm_n_t.to(device), lm_e_t.to(device), facial_flow_rs.to(device), head_flow_rs.to(device)




# # ---------- Main preprocessing function ----------
# def preprocess_pair(
#     I_n, I_e, lm_n, lm_e, exp_flow, facial_flow, head_flow,
#     out_size=384,
#     face_expand=1.25,
#     augment=True,
#     device='cpu'
# ):
#     """
#     I_n, I_e: PIL.Image or numpy uint8 HxWxC or torch tensor CxHxW in [0,1]
#     lm_n, lm_e: numpy arrays shape (N,2) in pixel coords relative to original full images
#     flow_facial: torch tensor [2,H,W] full-res flow in pixels (matching original full image dims)
#     facial_flow: torch tensor [2,H,W] full-res flow in pixels (matching original full image dims)
#     out_size: desired square output size (int)
#     face_expand: bbox margin multiplier (1.0 means tight box; >1 enlarges)
#     augment: whether to apply photometric/geometric augment
#     Returns: tensors on 'device':
#       I_n_t, I_e_t: float tensors [3, out_size, out_size], in [0,1]
#       lm_n_t, lm_e_t: float tensors [N,2] in pixel coords relative to cropped/resized images
#       facial_flow_t: float tensor [2, out_size, out_size] in pixels (flow vectors scaled)
#       meta: dict with original crop and scale info (useful for diagnostics)
#     """
#     # ---------------- normalize inputs to tensors ----------------
#     # convert images to float tensors CxHxW in [0,1]
#     if isinstance(I_n, torch.Tensor):
#         img_n = I_n.clone()
#         if img_n.max() > 2.0:  # maybe uint8 in tensor
#             img_n = img_n.float() / 255.0
#     else:
#         img_n = to_tensor_uint8(I_n)
#     if isinstance(I_e, torch.Tensor):
#         img_e = I_e.clone()
#         if img_e.max() > 2.0:
#             img_e = img_e.float() / 255.0
#     else:
#         img_e = to_tensor_uint8(I_e)

#     # facial_flow should be torch tensor [2,H,W]
#     if not isinstance(facial_flow, torch.Tensor):
#         facial_flow = torch.from_numpy(facial_flow).float()
#     else:
#         facial_flow = facial_flow.clone().float()

#     if not isinstance(exp_flow, torch.Tensor):
#         exp_flow = torch.from_numpy(exp_flow).float()
#     else:
#         exp_flow = exp_flow.clone().float()

#     if not isinstance(head_flow, torch.Tensor):
#         head_flow = torch.from_numpy(head_flow).float()
#     else:
#         head_flow = head_flow.clone().float()

#     # landmarks to numpy for easy math if necessary
#     lm_n_np = lm_n.copy() if isinstance(lm_n, np.ndarray) else lm_n.detach().cpu().numpy()
#     lm_e_np = lm_e.copy() if isinstance(lm_e, np.ndarray) else lm_e.detach().cpu().numpy()

#     _, H_full, W_full = img_n.shape

#     # ---------------- compute face bbox from union of landmarks (use both frames to be safe) ----------------
#     all_lms = np.concatenate([lm_n_np, lm_e_np], axis=0) if lm_e_np is not None else lm_n_np
#     x_min = float(np.min(all_lms[:, 0]))
#     y_min = float(np.min(all_lms[:, 1]))
#     x_max = float(np.max(all_lms[:, 0]))
#     y_max = float(np.max(all_lms[:, 1]))

#     # expand bbox
#     w = x_max - x_min
#     h = y_max - y_min
#     cx = (x_min + x_max) / 2.0
#     cy = (y_min + y_max) / 2.0
#     s = max(w, h) * face_expand
#     x0 = cx - s/2.0
#     y0 = cy - s/2.0
#     x1 = cx + s/2.0
#     y1 = cy + s/2.0

#     # clamp to image bounds and ensure int
#     x0c, y0c, x1c, y1c = ensure_square_crop_bbox(x0, y0, x1, y1, W_full, H_full, square_expand=False)
#     crop_w = x1c - x0c
#     crop_h = y1c - y0c
#     # sometimes out-of-bound due to expand, pad if needed (here we simply clamp; you may pad with background if preferred)

#     # -------------- crop images, flow, landmarks ----------------
#     img_n_crop = crop_image_tensor(img_n, x0c, y0c, x1c, y1c)
#     img_e_crop = crop_image_tensor(img_e, x0c, y0c, x1c, y1c)
#     exp_flow_crop = crop_flow_tensor(exp_flow, x0c, y0c, x1c, y1c)
#     facial_flow_crop = crop_flow_tensor(facial_flow, x0c, y0c, x1c, y1c)
#     head_flow_crop = crop_flow_tensor(head_flow, x0c, y0c, x1c, y1c)
#     lm_n_crop = crop_landmarks(lm_n_np, x0c, y0c)
#     lm_e_crop = crop_landmarks(lm_e_np, x0c, y0c)

#     # -------------- optional augmentation BEFORE resize (geometric: random translate crop) --------------
#     # We'll allow a small random shift inside the crop (translation) to simulate framing jitter.
#     if augment:
#         # max translate fraction relative to crop size (e.g., +/- 8%)
#         max_tx = int(0.08 * crop_w)
#         max_ty = int(0.08 * crop_h)
#         tx = random.randint(-max_tx, max_tx) if max_tx > 0 else 0
#         ty = random.randint(-max_ty, max_ty) if max_ty > 0 else 0
#         if tx != 0 or ty != 0:
#             # compute new crop coords, keep within original cropped patch bounds
#             # note: we are cropping inside the already-cropped image to simulate shift
#             C, Hc, Wc = img_n_crop.shape
#             # compute intersection bounds
#             x0s = max(0, tx)  # if tx positive, shift right means crop moves right inside patch
#             y0s = max(0, ty)
#             x1s = x0s + Wc - abs(tx)
#             y1s = y0s + Hc - abs(ty)
#             # apply spatial crop on crop (effectively shift content)
#             # simpler: pad_and_crop approach: create padded patch then crop
#             # but easier: shift landmarks and use F.pad if needed; to keep simple, only perform shifts that keep full patch inside
#             # ensure stays valid
#             if 0 <= x0s < x1s <= Wc and 0 <= y0s < y1s <= Hc:
#                 img_n_crop = img_n_crop[:, y0s:y1s, x0s:x1s]
#                 img_e_crop = img_e_crop[:, y0s:y1s, x0s:x1s]
#                 exp_flow_crop = exp_flow_crop[:, y0s:y1s, x0s:x1s]
#                 facial_flow_crop = facial_flow_crop[:, y0s:y1s, x0s:x1s]
#                 head_flow_crop = head_flow_crop[:, y0s:y1s, x0s:x1s]
#                 lm_n_crop[..., 0] = lm_n_crop[..., 0] - x0s
#                 lm_n_crop[..., 1] = lm_n_crop[..., 1] - y0s
#                 lm_e_crop[..., 0] = lm_e_crop[..., 0] - x0s
#                 lm_e_crop[..., 1] = lm_e_crop[..., 1] - y0s
#                 # update crop dims
#                 crop_h = y1s - y0s
#                 crop_w = x1s - x0s

#     # -------------- resize to out_size and scale flow accordingly ----------------
#     img_n_rs = resize_image_tensor(img_n_crop, out_size, out_size, interp_mode='bilinear')
#     img_e_rs = resize_image_tensor(img_e_crop, out_size, out_size, interp_mode='bilinear')
#     exp_flow_rs = resize_flow_tensor(exp_flow_crop, out_size, out_size)
#     facial_flow_rs = resize_flow_tensor(facial_flow_crop, out_size, out_size)
#     head_flow_rs = resize_flow_tensor(head_flow_crop, out_size, out_size)
#     # scale landmarks accordingly (float)
#     scale_x = out_size / float(crop_w)
#     scale_y = out_size / float(crop_h)
#     lm_n_rs = lm_n_crop.astype(np.float32)
#     lm_e_rs = lm_e_crop.astype(np.float32)
#     lm_n_rs[..., 0] = lm_n_rs[..., 0] * scale_x
#     lm_n_rs[..., 1] = lm_n_rs[..., 1] * scale_y
#     lm_e_rs[..., 0] = lm_e_rs[..., 0] * scale_x
#     lm_e_rs[..., 1] = lm_e_rs[..., 1] * scale_y

#     # -------------- random horizontal flip ----------------
#     if augment and random.random() < 0.5:
#         img_n_rs = flip_horizontal_image_tensor(img_n_rs)
#         img_e_rs = flip_horizontal_image_tensor(img_e_rs)
#         exp_flow_rs = flip_horizontal_flow(exp_flow_rs)
#         facial_flow_rs = flip_horizontal_flow(facial_flow_rs)
#         head_flow_rs = flip_horizontal_flow(head_flow_rs)
#         lm_n_rs = flip_landmarks_x(lm_n_rs, out_size)
#         lm_e_rs = flip_landmarks_x(lm_e_rs, out_size)

#     # -------------- photometric augmentations (images only) ----------------
#     if augment:
#         # apply same photometric jitter to both images? usually slightly different per-frame helps robustness to lighting change.
#         # Here we apply *independent* small photometric transforms to each frame (closer to real capture).
#         img_n_rs = random_color_jitter(img_n_rs, p=0.9)
#         img_e_rs = random_color_jitter(img_e_rs, p=0.9)
#         # blur / noise / jpeg (apply independently)
#         img_n_rs = random_gaussian_blur(img_n_rs, p=0.3)
#         img_e_rs = random_gaussian_blur(img_e_rs, p=0.3)
#         img_n_rs = random_noise(img_n_rs, p=0.4)
#         img_e_rs = random_noise(img_e_rs, p=0.4)
#         img_n_rs = random_jpeg_compress(img_n_rs, p=0.25)
#         img_e_rs = random_jpeg_compress(img_e_rs, p=0.25)
#         # random occlusion inside face crop
#         img_n_rs = random_occlusion(img_n_rs, p=0.25)
#         img_e_rs = random_occlusion(img_e_rs, p=0.25)

#     # clamp images
#     img_n_rs = img_n_rs.clamp(0.0, 1.0)
#     img_e_rs = img_e_rs.clamp(0.0, 1.0)

#     # ---------------- prepare outputs and meta ----------------
#     # convert landmarks to torch tensors (float), keep pixel coords relative to out_size
#     lm_n_t = torch.from_numpy(lm_n_rs).float()
#     lm_e_t = torch.from_numpy(lm_e_rs).float()
#     return img_n_rs.to(device)*255.0, img_e_rs.to(device)*255.0, lm_n_t.to(device), lm_e_t.to(device), exp_flow_rs.to(device), facial_flow_rs.to(device), head_flow_rs.to(device)


# if __name__ == "__main__":
#     # synthetic small test
#     import numpy as np
#     H,W = 480, 640
#     # create dummy images
#     I_n = (np.random.rand(H,W,3)*255).astype(np.uint8)
#     I_e = (np.random.rand(H,W,3)*255).astype(np.uint8)
#     # dummy landmarks (eyes,nose,mouth corners)
#     lmn = np.array([[200,150],[240,150],[220,200],[210,250],[230,250]], dtype=np.float32)
#     lme = lmn + np.random.randn(*lmn.shape)*1.0
#     # dummy head flow (small translation + rotation approx)
#     flow_facial = np.zeros((2,H,W), dtype=np.float32)
#     flow_facial[0,:,:] = 1.2  # small right shift everywhere
#     flow_facial[1,:,:] = 0.3
#     facial_flow = np.zeros((2,H,W), dtype=np.float32)
#     facial_flow[0,:,:] = 1.2  # small right shift everywhere
#     facial_flow[1,:,:] = 0.3
#     out = preprocess_pair(I_n, I_e, lmn, lme, flow_facial, facial_flow, out_size=384, augment=True, device='cpu')
#     print(torch.max(out[0]))
#     # print("outputs:", {k: (v.shape if isinstance(v, torch.Tensor) else v) for k,v in out.items() if k!='meta'})
#     # print("meta:", out['meta'])