# import numpy as np
# import torch
# import torch.utils.data as data
# import torch.nn.functional as F

# import os
# import math
# import random
# import h5py
# import cv2
# from tqdm import tqdm
# from glob import glob
# import os.path as osp

# from utils import frame_utils
# from dataloader.augmentor import FlowAugmentor

# class FlowDataset(data.Dataset):
#     def __init__(self, aug_params=None, sparse=False):
#         self.augmentor = None
#         self.sparse = sparse
#         # self.subsample_groundtruth = False
#         if aug_params is not None:
#             self.augmentor = FlowAugmentor(**aug_params)

#         self.is_test = False
#         self.init_seed = False
#         self.flow_list = []
#         self.image_list = []
#         self.mask_list = []
#         self.extra_info = []

#     def __getitem__(self, index):
#         while True:
#             try:
#                 return self.fetch(index)
#             except Exception as e:
#                 index = random.randint(0, len(self) - 1)
#             return self.fetch(index)

#     def read_flow(self, index):
#         raise NotImplementedError

#     def fetch(self, index):
#         if self.is_test:
#             img1 = frame_utils.read_gen(self.image_list[index][0])
#             img2 = frame_utils.read_gen(self.image_list[index][1])
#             img1 = np.array(img1).astype(np.uint8)[..., :3]
#             img2 = np.array(img2).astype(np.uint8)[..., :3]
#             img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
#             img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
#             return img1, img2, self.extra_info[index]

#         index = index % len(self.image_list)
#         flow, valid = self.read_flow(index)
#         img1 = frame_utils.read_gen(self.image_list[index][0])
#         img2 = frame_utils.read_gen(self.image_list[index][1])
#         flow = np.array(flow).astype(np.float32)
#         img1 = np.array(img1).astype(np.uint8)
#         img2 = np.array(img2).astype(np.uint8)
#         # grayscale images
#         if len(img1.shape) == 2:
#             img1 = np.tile(img1[...,None], (1, 1, 3))
#             img2 = np.tile(img2[...,None], (1, 1, 3))
#         else:
#             img1 = img1[..., :3]
#             img2 = img2[..., :3]

#         if self.augmentor is not None:
#             img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)

#         img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
#         img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
#         flow = torch.from_numpy(flow).permute(2, 0, 1).float()
#         valid = torch.from_numpy(valid)
#         valid = (valid >= 0.5) & ((~torch.isnan(flow)).all(dim=0)) & ((~torch.isinf(flow)).all(dim=0))
#         flow[torch.isinf(flow)] = 0
#         flow[torch.isnan(flow)] = 0
#         return img1, img2, flow, valid.float()


#     def __rmul__(self, v):
#         self.flow_list = v * self.flow_list
#         self.image_list = v * self.image_list
#         return self
        
#     def __len__(self):
#         return len(self.image_list)






# class FlowDataset(data.Dataset):
#     def __init__(self, aug_params=None, sparse=False, split='train'):
#         self.augmentor = None
#         self.sparse = sparse
#         self.split = split
#         if aug_params is not None:
#             if sparse:
#                 self.augmentor = SparseFlowAugmentor(**aug_params)
#             else:
#                 self.augmentor = FlowAugmentor(**aug_params)

#         self.is_test = False
#         self.init_seed = False

#         self.exp_flow_list = []
#         self.head_flow_list = []
#         self.flow_list = []
#         self.image_list = []
#         self.extra_info = []
#         self.mask_list = []
#         self.occ_list = None
#         self.seg_list = None
#         self.seg_inv_list = None

#     def __getitem__(self, index):

#         if self.is_test:
#             img1 = frame_utils.read_gen(self.image_list[index][0])
#             img2 = frame_utils.read_gen(self.image_list[index][1])
#             img1 = np.array(img1).astype(np.uint8)[..., :3]
#             img2 = np.array(img2).astype(np.uint8)[..., :3]
#             img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
#             img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
#             return img1, img2, self.extra_info[index]

#         if not self.init_seed:
#             worker_info = torch.utils.data.get_worker_info()
#             if worker_info is not None:
#                 torch.manual_seed(worker_info.id)
#                 np.random.seed(worker_info.id)
#                 random.seed(worker_info.id)
#                 self.init_seed = True

#         index = index % len(self.image_list)
#         valid = None
#         if self.sparse:
#             flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
#         else:
#             exp_flow = frame_utils.read_gen(self.exp_flow_list[index])
#             head_flow = frame_utils.read_gen(self.head_flow_list[index])

#         if self.occ_list is not None:
#             occ = frame_utils.read_gen(self.occ_list[index])
#             occ = np.array(occ).astype(np.uint8)
#             occ = torch.from_numpy(occ // 255).bool()

#         if self.seg_list is not None:
#             f_in = np.array(frame_utils.read_gen(self.seg_list[index]))
#             seg_r = f_in[:, :, 0].astype('int32')
#             seg_g = f_in[:, :, 1].astype('int32')
#             seg_b = f_in[:, :, 2].astype('int32')
#             seg_map = (seg_r * 256 + seg_g) * 256 + seg_b
#             seg_map = torch.from_numpy(seg_map)

#         if self.seg_inv_list is not None:
#             seg_inv = frame_utils.read_gen(self.seg_inv_list[index])
#             seg_inv = np.array(seg_inv).astype(np.uint8)
#             seg_inv = torch.from_numpy(seg_inv // 255).bool()

#         img1 = frame_utils.read_gen(self.image_list[index][0])
#         img2 = frame_utils.read_gen(self.image_list[index][1])

#         exp_flow = np.array(exp_flow).astype(np.float32)
#         head_flow = np.array(head_flow).astype(np.float32)
#         img1 = np.array(img1).astype(np.uint8)
#         img2 = np.array(img2).astype(np.uint8)
#         # if self.split == 'test':
#         #     mask1 = np.load(self.mask_list[index][0]).astype(np.uint8)
#         #     mask = torch.from_numpy(mask1).float()

#         if len(img1.shape) == 2:
#             img1 = np.tile(img1[...,None], (1, 1, 3))
#             img2 = np.tile(img2[...,None], (1, 1, 3))
#         else:
#             img1 = img1[..., :3]
#             img2 = img2[..., :3]



#         if self.augmentor is not None:
#             if self.sparse:
#                 img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
#             else:
#                 img1, img2, exp_flow, head_flow = self.augmentor(img1, img2, exp_flow, head_flow)

#         img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
#         img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
#         exp_flow = torch.from_numpy(exp_flow).permute(2, 0, 1).float()
#         head_flow = torch.from_numpy(head_flow).permute(2, 0, 1).float()
      
#         if valid is not None:
#             valid = torch.from_numpy(valid)
#         else:
#             valid_e = (exp_flow[0].abs() < 1000) & (exp_flow[1].abs() < 1000)
#             valid_h = (head_flow[0].abs() < 1000) & (head_flow[1].abs() < 1000)

#         if self.occ_list is not None:
#             return img1, img2, flow, valid.float(), occ, self.occ_list[index]
#         elif self.seg_list is not None and self.seg_inv_list is not None:
#             return img1, img2, flow, valid.float(), seg_map, seg_inv
#         else:
#             # if self.split == 'test':
#             #     return img1, img2, exp_flow, head_flow, valid_e.float(), valid_h.float(), mask 
#             return img1, img2, exp_flow, head_flow, valid_e.float(), valid_h.float()

#     def __rmul__(self, v):
#         self.flow_list = v * self.flow_list
#         self.image_list = v * self.image_list
#         return self
        
#     def __len__(self):
#         return len(self.image_list)

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
from dataloader.augmentor import FlowAugmentor_v2



class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, split='train'):
        self.augmentor = FlowAugmentor_v2(aug_params)
        self.sparse = sparse
        self.split = split

        self.is_test = False
        self.init_seed = False
        self.landmark_list = []
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
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        
        head_flow = np.array(head_flow).astype(np.float32)
        facial_flow = np.array(facial_flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        img1 = img1[..., :3]
        img2 = img2[..., :3]


        lmn = np.load(self.landmark_list[index][0])
        lme = np.load(self.landmark_list[index][1])
        valid = (np.abs(head_flow[..., 0]) < 1000) & (np.abs(head_flow[..., 1]) < 1000) & (np.abs(facial_flow[..., 0]) < 1000) & (np.abs(facial_flow[..., 1]) < 1000)

        if self.split == 'train':
            img1, img2, facial_flow, head_flow, valid, lmn, lme  = self.augmentor(img1, img2, facial_flow, head_flow, valid, lmn, lme)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        head_flow = torch.from_numpy(head_flow).permute(2, 0, 1).float()
        facial_flow = torch.from_numpy(facial_flow).permute(2, 0, 1).float()
        valid = torch.from_numpy(valid)
        valid = (valid >= 0.5) & ((~torch.isnan(facial_flow)).all(dim=0)) & ((~torch.isinf(facial_flow)).all(dim=0)) & ((~torch.isnan(head_flow)).all(dim=0)) & ((~torch.isinf(head_flow)).all(dim=0))
        return img1, img2, facial_flow, head_flow, valid.float(), lmn, lme

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)


# import cv2

# class FlowDataset(data.Dataset):
#     def __init__(self, aug_params=None, sparse=False, split='train'):
#         self.augmentor = None
#         self.sparse = sparse
#         self.split = split

#         self.is_test = False
#         self.init_seed = False

#         self.facial_flow_list = []
#         self.head_flow_list = []
#         self.flow_list = []
#         self.image_list = []
#         self.extra_info = []
#         self.mask_list = []
#         self.occ_list = None
#         self.seg_list = None
#         self.seg_inv_list = None
    

#     def __getitem__(self, index):
#         if not self.init_seed:
#             worker_info = torch.utils.data.get_worker_info()
#             if worker_info is not None:
#                 torch.manual_seed(worker_info.id)
#                 np.random.seed(worker_info.id)
#                 random.seed(worker_info.id)
#                 self.init_seed = True

#         index = index % len(self.image_list)
#         head_flow = frame_utils.read_gen(self.head_flow_list[index])
#         facial_flow = frame_utils.read_gen(self.facial_flow_list[index])
#         tvl1_flow = np.load(f'{os.path.dirname(self.image_list[index][0])}/tvl1_flow.npy')
#         img1 = frame_utils.read_gen(self.image_list[index][0])
#         img2 = frame_utils.read_gen(self.image_list[index][1])
        
#         head_flow = np.array(head_flow).astype(np.float32)
#         facial_flow = np.array(facial_flow).astype(np.float32)
#         tvl1_flow = np.array(tvl1_flow).astype(np.float32)
#         img1 = np.array(img1).astype(np.uint8)
#         img2 = np.array(img2).astype(np.uint8)
#         img1 = img1[..., :3]
#         img2 = img2[..., :3]

#         img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
#         img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
#         head_flow = torch.from_numpy(head_flow).permute(2, 0, 1).float()
#         facial_flow = torch.from_numpy(facial_flow).permute(2, 0, 1).float()
#         tvl1_flow = torch.from_numpy(tvl1_flow).permute(2, 0, 1).float()
#         lmn = np.load(f'{os.path.dirname(self.image_list[index][0])}/lm0.npy')
#         lme = np.load(f'{os.path.dirname(self.image_list[index][0])}/lm1.npy')
        
#         exp_flow = facial_flow - head_flow
#         img1, img2, lmn, lme, exp_flow, tvl1_flow, head_flow  = preprocess_pair(img1, img2, lmn, lme, exp_flow, tvl1_flow, head_flow, out_size=480, device='cpu')

#         valid_exp = (exp_flow[0].abs() < 1000) & (exp_flow[1].abs() < 1000)
#         valid_tvl1 = (tvl1_flow[0].abs() < 1000) & (tvl1_flow[1].abs() < 1000)
#         valid_head = (head_flow[0].abs() < 1000) & (head_flow[1].abs() < 1000)
#         return img1, img2, lmn, lme, exp_flow, tvl1_flow, head_flow, valid_exp.float(), valid_tvl1.float(), valid_head.float()

#     def __rmul__(self, v):
#         self.flow_list = v * self.flow_list
#         self.image_list = v * self.image_list
#         return self
        
#     def __len__(self):
#         return len(self.image_list)