import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
import h5py
from tqdm import tqdm
from glob import glob
import os.path as osp

from utils.ddp_utils import *

from dataloader.flow.chairs import FlyingChairs
from dataloader.flow.things import FlyingThings3D
from dataloader.flow.sintel import MpiSintel
from dataloader.flow.kitti import KITTI
from dataloader.flow.spring import Spring
from dataloader.flow.hd1k import HD1K
from dataloader.flow.facialflownet import FacialFlow
from dataloader.stereo.tartanair import TartanAir
from dataloader.tvl1_loader import FacialFlow_v2

def fetch_dataloader(args, rank=0, world_size=1, use_ddp=False):
    """ Create the data loader for the corresponding trainign set """
    print(args.dataset)
    if args.dataset == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.1, 'max_scale': args.scale + 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.dataset == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.4, 'max_scale': args.scale + 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.dataset == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.6, 'do_flip': True}
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        train_dataset = sintel_clean + sintel_final

    elif args.dataset == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')
    
    elif args.dataset == 'spring':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale, 'max_scale': args.scale + 0.2, 'do_flip': True}
        train_dataset = Spring(aug_params, split='train') + Spring(aug_params, split='val')

    elif args.dataset == 'tartanair':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.2, 'do_flip': True}
        train_dataset = TartanAir(aug_params)

    elif args.dataset == 'TSKH':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.4, 'max_scale': args.scale + 0.8, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass') + FlyingThings3D(aug_params, dstype='frames_finalpass')
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.6, 'do_flip': True}
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        kitti = KITTI({'crop_size': args.image_size, 'min_scale': args.scale - 0.3, 'max_scale': args.scale + 0.5, 'do_flip': True})
        hd1k = HD1K({'crop_size': args.image_size, 'min_scale': args.scale - 0.5, 'max_scale': args.scale + 0.2, 'do_flip': True})
        train_dataset = 20 * sintel_clean + 20 * sintel_final + 80 * kitti + 30 * hd1k + things
    elif args.dataset == "facialflow":
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        facialflow = FacialFlow(aug_params, split="train")
        train_dataset = facialflow
    elif args.dataset == "facialflow_v2":
        print(args.dataset)
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        facialflow_v2 = FacialFlow_v2(aug_params, split="train")
        train_dataset = facialflow_v2 

    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank)
        num_gpu = torch.cuda.device_count()
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
            shuffle=(train_sampler is None), num_workers=10, sampler=train_sampler)
    else:
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
            pin_memory=False, shuffle=True, num_workers=10, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

