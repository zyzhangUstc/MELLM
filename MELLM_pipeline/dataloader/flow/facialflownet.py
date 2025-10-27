import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
from tqdm import tqdm
from glob import glob
import os.path as osp

from utils import frame_utils
from dataloader.template import FlowDataset
import json

def write_jsonl(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")    
            
def load_list_from_jsonl(filename):
    """从 JSONL 文件读取包含字典的列表"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))
    return data
# class FacialFlow(FlowDataset):
#     def __init__(self, aug_params=None, split = "train", dataset_root = "/data/zyzhang/dataset/facialFlowNet"):
#         super(FacialFlow, self).__init__(aug_params, split = split)


#         exp_image_root = osp.join(dataset_root, "image", "facial", split)
#         exp_flow_root = osp.join(dataset_root, "flow", "facial",  split)


#         head_image_root = osp.join(dataset_root, "image", "head", split)
#         head_flow_root = osp.join(dataset_root, "flow", "head",  split)
#         if split == 'test':
#             mask_root = osp.join(dataset_root, "new_mask", split)


#         for emotion in os.listdir(exp_flow_root):
#             sub_list = os.listdir(osp.join(exp_flow_root, emotion))
#             for sub in sub_list:

#                 image_list = sorted(glob(osp.join(exp_image_root, emotion, sub, "*.jpg")))
#                 if split == 'test':
#                     mask_list = sorted(glob(osp.join(mask_root, emotion, sub, "*.npy")))
#                 for i in range(len(image_list) - 1):
#                     self.image_list += [ [image_list[i], image_list[i+1]] ]
#                     if split == 'test':
#                         self.mask_list += [ [mask_list[i]] ]
#                     self.extra_info += [ ("{}_{}".format(emotion, sub), i) ]


#                 self.exp_flow_list += sorted(glob(osp.join(exp_flow_root, emotion, sub, "*.flo")))
#                 self.head_flow_list += sorted(glob(osp.join(head_flow_root, emotion, sub, "*.flo")))
def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))
    return data

class FacialFlow(FlowDataset):
    def __init__(self, aug_params=None, split = "train", dataset_root = "/data/zyzhang/dataset/facialFlowNet"):
        super(FacialFlow, self).__init__(aug_params, split = split)
        selected = []
        if split == 'train':
            with open('/data/zyzhang/flow_code/WAFT/labels/train_data.txt', 'r') as f:
                for line in f:
                    selected.append(line.strip())
            extend_train_list = read_jsonl('/data/zyzhang/code/FacialFlowNet/facialflownet_files.jsonl')
            for sample in tqdm(extend_train_list):
                self.image_list += [ [sample['frame_1'], sample['frame_2']] ]
                self.facial_flow_list.append(sample['facial_flow'])
                self.head_flow_list.append(sample['head_flow'])
                self.landmark_list.append( [sample['landmarks_onset'], sample['landmarks_apex']] )
        else:
            with open('/data/zyzhang/flow_code/WAFT/labels/test_data.txt', 'r') as f:
                for line in f:
                    selected.append(line.strip())            

        for sample in tqdm(selected):    
            self.image_list += [ [f'{sample}render_output_exp_pose/frame_0001.png', f'{sample}render_output_exp_pose/frame_0002.png'] ]
            self.facial_flow_list.append(f'{sample}render_output_exp_pose/0001.npz')
            self.head_flow_list.append(f'{sample}render_output_pose/0001.npz')
            self.landmark_list.append( [f'{sample}render_output_exp_pose/lm0.npy', f'{sample}render_output_exp_pose/lm1.npy'] )




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



class FlowDataset_test(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, split='train'):
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
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        
        head_flow = np.array(head_flow).astype(np.float32)
        facial_flow = np.array(facial_flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        img1 = img1[..., :3]
        img2 = img2[..., :3]
        mask1 = np.load(self.mask_list[index]).astype(np.uint8)
        mask = torch.from_numpy(mask1).float()

        valid = (np.abs(head_flow[..., 0]) < 1000) & (np.abs(head_flow[..., 1]) < 1000) & (np.abs(facial_flow[..., 0]) < 1000) & (np.abs(facial_flow[..., 1]) < 1000)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        head_flow = torch.from_numpy(head_flow).permute(2, 0, 1).float()
        facial_flow = torch.from_numpy(facial_flow).permute(2, 0, 1).float()
        valid = torch.from_numpy(valid)
        valid = (valid >= 0.5) & ((~torch.isnan(facial_flow)).all(dim=0)) & ((~torch.isinf(facial_flow)).all(dim=0)) & ((~torch.isnan(head_flow)).all(dim=0)) & ((~torch.isinf(head_flow)).all(dim=0))
        return img1, img2, facial_flow, head_flow, valid.float(), mask

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)









class FacialFlow_test(FlowDataset_test):
    def __init__(self, aug_params=None, split = "train", dataset_root = "/data/zyzhang/dataset/facialFlowNet"):
        super(FacialFlow_test, self).__init__(aug_params, split = split)
        extend_train_list = read_jsonl('/data/zyzhang/code/FacialFlowNet/facialflownet_files_test.jsonl')
        for sample in tqdm(extend_train_list):
            self.image_list += [ [sample['frame_1'], sample['frame_2']] ]
            self.facial_flow_list.append(sample['facial_flow'])
            self.head_flow_list.append(sample['head_flow'])
            self.mask_list.append(sample['mask'])
            