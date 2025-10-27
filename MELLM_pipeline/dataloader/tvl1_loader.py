import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
from tqdm import tqdm
from glob import glob
import os.path as osp
from utils import frame_utils
import json

def load_list_from_json(filename):
    """从JSON文件读取包含字典的列表"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)
def load_list_from_jsonl(filename):
    """从 JSONL 文件读取包含字典的列表"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))
    return data


import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
# from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from utils.augmentor import *
import torchvision.transforms.functional as TF

def random_grayscale_pair(img1, img2, p=0.2):
    """以概率 p 同时把两张图转为黑白 (3通道灰度)"""
    if random.random() < p:
        img1 = TF.rgb_to_grayscale(img1, num_output_channels=3)
        img2 = TF.rgb_to_grayscale(img2, num_output_channels=3)
    return img1, img2


# # ---------- Main preprocessing function ----------
def preprocess_pair_v2(
    I_n, I_e, lm_n, lm_e, exp_flow, facial_flow, head_flow,
    out_size=(400, 480),
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

    if not isinstance(exp_flow, torch.Tensor):
        exp_flow = torch.from_numpy(exp_flow).float()
    else:
        exp_flow = exp_flow.clone().float()

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
    x0 = cx - w/2.0
    y0 = cy - s/2.0
    x1 = cx + w/2.0
    y1 = cy + s/2.0

    # clamp to image bounds and ensure int
    x0c, y0c, x1c, y1c = ensure_square_crop_bbox(x0, y0, x1, y1, W_full, H_full, square_expand=False)
    crop_w = x1c - x0c
    crop_h = y1c - y0c
    # sometimes out-of-bound due to expand, pad if needed (here we simply clamp; you may pad with background if preferred)

    # -------------- crop images, flow, landmarks ----------------
    img_n_crop = crop_image_tensor(img_n, x0c, y0c, x1c, y1c)
    img_e_crop = crop_image_tensor(img_e, x0c, y0c, x1c, y1c)
    exp_flow_crop = crop_flow_tensor(exp_flow, x0c, y0c, x1c, y1c)
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
                exp_flow_crop = exp_flow_crop[:, y0s:y1s, x0s:x1s]
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
    img_n_rs = resize_image_tensor(img_n_crop, out_size[1], out_size[0], interp_mode='bilinear')
    img_e_rs = resize_image_tensor(img_e_crop, out_size[1], out_size[0], interp_mode='bilinear')
    exp_flow_rs = resize_flow_tensor(exp_flow_crop, out_size[1], out_size[0])
    facial_flow_rs = resize_flow_tensor(facial_flow_crop, out_size[1], out_size[0])
    head_flow_rs = resize_flow_tensor(head_flow_crop, out_size[1], out_size[0])
    # scale landmarks accordingly (float)
    scale_x = out_size[0] / float(crop_w)
    scale_y = out_size[1] / float(crop_h)
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
        exp_flow_rs = flip_horizontal_flow(exp_flow_rs)
        facial_flow_rs = flip_horizontal_flow(facial_flow_rs)
        head_flow_rs = flip_horizontal_flow(head_flow_rs)
        lm_n_rs = flip_landmarks_x(lm_n_rs, out_size[0])
        lm_e_rs = flip_landmarks_x(lm_e_rs, out_size[0])

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

        img_n_rs, img_e_rs = random_grayscale_pair(img_n_rs, img_e_rs, p=0.2)

    # clamp images
    img_n_rs = img_n_rs.clamp(0.0, 1.0)
    img_e_rs = img_e_rs.clamp(0.0, 1.0)

    # ---------------- prepare outputs and meta ----------------
    # convert landmarks to torch tensors (float), keep pixel coords relative to out_size
    lm_n_t = torch.from_numpy(lm_n_rs).float()
    lm_e_t = torch.from_numpy(lm_e_rs).float()
    return img_n_rs.to(device)*255.0, img_e_rs.to(device)*255.0, lm_n_t.to(device), lm_e_t.to(device), exp_flow_rs.to(device), facial_flow_rs.to(device), head_flow_rs.to(device)




import cv2

class FlowDataset_v2(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, split='train'):
        self.augmentor = None
        self.sparse = sparse
        self.split = split

        self.is_test = False
        self.init_seed = False

        self.facial_flow_list = []
        self.head_flow_list = []
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.mask_list = []
        self.occ_list = None
        self.seg_list = None
        self.seg_inv_list = None
    

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        head_flow = frame_utils.read_gen(self.head_flow_list[index])
        facial_flow = frame_utils.read_gen(self.facial_flow_list[index])
        tvl1_flow = np.load(f'{os.path.dirname(self.image_list[index][0])}/tvl1_flow.npy')
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        
        head_flow = np.array(head_flow).astype(np.float32)
        facial_flow = np.array(facial_flow).astype(np.float32)
        tvl1_flow = np.array(tvl1_flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        img1 = img1[..., :3]
        img2 = img2[..., :3]

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        head_flow = torch.from_numpy(head_flow).permute(2, 0, 1).float()
        facial_flow = torch.from_numpy(facial_flow).permute(2, 0, 1).float()
        tvl1_flow = torch.from_numpy(tvl1_flow).permute(2, 0, 1).float()
        lmn = np.load(f'{os.path.dirname(self.image_list[index][0])}/lm0.npy')
        lme = np.load(f'{os.path.dirname(self.image_list[index][0])}/lm1.npy')
        
        exp_flow = facial_flow - head_flow
        img1, img2, lmn, lme, exp_flow, tvl1_flow, head_flow  = preprocess_pair_v2(img1, img2, lmn, lme, exp_flow, tvl1_flow, head_flow, out_size=(400, 480), device='cpu')

        valid_exp = (exp_flow[0].abs() < 1000) & (exp_flow[1].abs() < 1000)
        valid_tvl1 = (tvl1_flow[0].abs() < 1000) & (tvl1_flow[1].abs() < 1000)
        valid_head = (head_flow[0].abs() < 1000) & (head_flow[1].abs() < 1000)
        return img1, img2, lmn, lme, exp_flow, tvl1_flow, head_flow, valid_exp.float(), valid_tvl1.float(), valid_head.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)



class FacialFlow_v2(FlowDataset_v2):
    def __init__(self, aug_params=None, split = "train", dataset_root = "/data/zyzhang/dataset/facialFlowNet"):
        super(FacialFlow_v2, self).__init__(aug_params, split = split)

        # if split == 'train':
        selected = load_list_from_jsonl('/data/zyzhang/flow_code/WAFT/labels/train_dic_list_v7.jsonl')
        # else:
        #     selected = load_list_from_json('/data/zyzhang/flow_code/WAFT/labels/test_dic_list_v4.json')

        for sample in tqdm(selected):
            # if 'Mixed_Static_code' not in sample['frame_1_path']:
            #     continue
            # if not os.path.exists(f'{os.path.dirname(sample['frame_1_path'])}/tvl1_flow.npy'):
            #     continue
            # if not os.path.exists(f'{os.path.dirname(sample['frame_1_path'])}/lm1.npy'):
            #     continue         
            self.image_list += [ [sample['frame_1_path'], sample['frame_2_path']] ]
            self.facial_flow_list.append(sample['flow_exp_pose_path'])
            self.head_flow_list.append(sample['flow_pose_path'])