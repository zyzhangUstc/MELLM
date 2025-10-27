import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
import h5py
import cv2
from tqdm import tqdm
from glob import glob
import os.path as osp

import sys
sys.path.append(os.getcwd())
from utils import frame_utils
from dataloader.template import FlowDataset
from utils.utils import check_cycle_consistency, induced_flow
from utils.flow_viz import flow_to_image

K = np.array([[320, 0, 320],
              [0, 320, 240],
              [0, 0, 1]], dtype=np.float32)

"""
Code from https://github.com/huyaoyu/ImageFlow/blob/master/ImageFlow.py
"""
def from_quaternion_to_rotation_matrix(q):
    """
    q: A numpy vector, 4x1.
    """
    qi2 = q[0, 0]**2
    qj2 = q[1, 0]**2
    qk2 = q[2, 0]**2
    qij = q[0, 0] * q[1, 0]
    qjk = q[1, 0] * q[2, 0]
    qki = q[2, 0] * q[0, 0]
    qri = q[3, 0] * q[0, 0]
    qrj = q[3, 0] * q[1, 0]
    qrk = q[3, 0] * q[2, 0]
    s = 1.0 / ( q[3, 0]**2 + qi2 + qj2 + qk2 )
    ss = 2 * s
    R = [\
        [ 1.0 - ss * (qj2 + qk2), ss * (qij - qrk), ss * (qki + qrj) ],\
        [ ss * (qij + qrk), 1.0 - ss * (qi2 + qk2), ss * (qjk - qri) ],\
        [ ss * (qki - qrj), ss * (qjk + qri), 1.0 - ss * (qi2 + qj2) ],\
    ]
    R = np.array(R, dtype = np.float32)
    return R

class TartanAir(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/TartanAir'):
        super(TartanAir, self).__init__(aug_params)
        self.root = root
        self.worldT = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.build_dataset_adjacent()

    def build_dataset_adjacent(self):
        scenes = glob(osp.join(self.root, '*/*/*/'))
        for scene in sorted(scenes):
            images = sorted(glob(osp.join(scene, 'image_left/*.png')))
            for idx in range(len(images) - 1):
                frame0 = str(idx).zfill(6)
                frame1 = str(idx + 1).zfill(6)
                self.image_list.append([images[idx], images[idx + 1]])
                self.flow_list.append(osp.join(scene, 'flow', f"{frame0}_{frame1}_flow.npy"))
                self.mask_list.append(osp.join(scene, 'flow', f"{frame0}_{frame1}_mask.npy"))
    
    def process_tartanair_pose(self, data):
        data = data.reshape((-1, 1))
        t = data[:3, 0].reshape((-1, 1))
        q = data[3:, 0].reshape((-1, 1))
        R = from_quaternion_to_rotation_matrix(q)
        T = np.eye(4)
        T[:3, :3] = R.transpose()
        T[:3, 3] = -R.transpose().dot(t).reshape((-1,))
        T = self.worldT @ T
        return T

    def build_dataset_all_pair(self):
        self.depth_list = []
        self.cam_list = []
        scenes = glob(osp.join(self.root, '*/*/*/'))
        for scene in sorted(scenes):
            for view in ['left']:
                images = sorted(glob(osp.join(scene, f"image_{view}/*.png")))
                depths = sorted(glob(osp.join(scene, f"depth_{view}/*.npy")))
                tartanair_pose_data = np.loadtxt(osp.join(scene, f"pose_{view}.txt"))
                poses = [self.process_tartanair_pose(data) for data in tartanair_pose_data]
                for i in range(len(images) - 1):
                    for j in range(i+1, min(len(images), i+self.n_frames+1)):
                        self.image_list.append([images[i], images[j]])
                        self.depth_list.append([depths[i], depths[j]])
                        self.cam_list.append([poses[i], poses[j]])

    def read_flow_adjacent(self, index):
        flow = np.load(self.flow_list[index])
        valid = np.load(self.mask_list[index])
        # rescale the valid mask to [0, 1]
        valid = 1 - valid / 100
        return flow, valid
    
    def read_flow_all_pair(self, index):
        T0 = self.cam_list[index][0]
        T1 = self.cam_list[index][1]
        depth0 = np.load(self.depth_list[index][0])
        depth1 = np.load(self.depth_list[index][1])
        cam_data = {'T0': T0, 'T1': T1, 'K0': K, 'K1': K}
        flow_01, flow_10 = induced_flow(depth0, depth1, cam_data)
        valid_01 = check_cycle_consistency(flow_01, flow_10)
        flow_01[valid_01 == 0] = 0
        return flow_01, valid_01
    
    def read_flow(self, index):
        return self.read_flow_adjacent(index)

if __name__ == "__main__":
    dataset = TartanAir(root='datasets/TartanAir')
    print(len(dataset))
    image1, image2, flow, valid = dataset[1200]
    print(image1.shape, image2.shape, flow.shape, valid.shape)
    image1 = image1.cpu().numpy().transpose(1, 2, 0)
    image2 = image2.cpu().numpy().transpose(1, 2, 0)
    flow = flow.cpu().numpy().transpose(1, 2, 0)
    print(flow.max(), flow.min(), flow.mean())
    valid = valid.cpu().numpy()
    flow_vis = flow_to_image(flow, convert_to_bgr=True)
    path = './demo/TartanAir/all_pair/'
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(path + 'flow.png', flow_vis)
    cv2.imwrite(path + 'image1.png', image1)
    H, W = image1.shape[:2]
    new_coords = flow + np.stack(
        np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), axis=-1
    )
    warped_image1 = cv2.remap(
        image2, new_coords.astype(np.float32), None, interpolation=cv2.INTER_LINEAR
    )
    cv2.imwrite(path + 'warped_image1.png', warped_image1)
    cv2.imwrite(path + 'image2.png', image2)
    cv2.imwrite(path + 'valid.png', (valid*255).astype(np.uint8))