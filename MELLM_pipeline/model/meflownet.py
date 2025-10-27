# This code is modified from the original work by Princeton:
# https://github.com/princeton-vl/WAFT

# The original work is licensed under the BSD 3-Clause "New" or "Revised" License.
# A copy of the license can be found in the LICENSE file in the root of this repository.

import numpy as np
import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.depthanythingv2 import DepthAnythingFeature
from model.backbone.vit import VisionTransformer, MODEL_CONFIGS

from utils.utils import coords_grid, Padder, bilinear_sampler

import timm


class resconv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1):
        super(resconv, self).__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=k // 2, bias=True),
            nn.GELU(),
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
        )
        if inp != oup or s != 1:
            self.skip_conv = nn.Conv2d(
                inp, oup, kernel_size=1, stride=s, padding=0, bias=True
            )
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)


class ResNet18Deconv(nn.Module):
    def __init__(self, inp, oup):
        super(ResNet18Deconv, self).__init__()
        self.feature_dims = [64, 128, 256, 512]
        self.ds1 = resconv(inp, 64, k=7, s=2)
        self.conv1 = timm.create_model(
            "resnet18.a3_in1k", pretrained=True, features_only=True
        ).layer1
        self.conv2 = timm.create_model(
            "resnet18.a3_in1k", pretrained=True, features_only=True
        ).layer2
        self.conv3 = timm.create_model(
            "resnet18.a3_in1k", pretrained=True, features_only=True
        ).layer3
        self.conv4 = timm.create_model(
            "resnet18.a3_in1k", pretrained=True, features_only=True
        ).layer4
        self.up_4 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2, padding=0, bias=True
        )
        self.proj_3 = resconv(256, 256, k=3, s=1)
        self.up_3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2, padding=0, bias=True
        )
        self.proj_2 = resconv(128, 128, k=3, s=1)
        self.up_2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2, padding=0, bias=True
        )
        self.proj_1 = resconv(64, oup, k=3, s=1)

    def forward(self, x):
        out_1 = self.ds1(x)
        out_1 = self.conv1(out_1)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        out_3 = self.proj_3(out_3 + self.up_4(out_4))
        out_2 = self.proj_2(out_2 + self.up_3(out_3))
        out_1 = self.proj_1(out_1 + self.up_2(out_2))
        return [out_1, out_2, out_3, out_4]


class MEFlowNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.da_feature = DepthAnythingFeature(encoder=args.dav2_backbone)
        self.pretrain_dim = self.da_feature.model_configs[args.dav2_backbone][
            "features"
        ]
        self.network_dim = MODEL_CONFIGS[args.network_backbone]["features"]
        self.fnet = ResNet18Deconv(self.pretrain_dim // 2 + 3, 64)

        self.fmap_conv = nn.Conv2d(
            self.pretrain_dim // 2 + 64,
            self.network_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # ------------------ facial flow decoder ------------------
        self.facial_refine_net = VisionTransformer(
            args.network_backbone, self.network_dim, patch_size=8
        )
        self.facial_hidden_conv = nn.Conv2d(
            self.network_dim * 2,
            self.network_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.facial_warp_linear = nn.Conv2d(
            3 * self.network_dim + 2, self.network_dim, 1, 1, 0, bias=True
        )
        self.facial_refine_transform = nn.Conv2d(
            self.network_dim // 2 * 3, self.network_dim, 1, 1, 0, bias=True
        )
        self.facial_upsample_weight = nn.Sequential(
            nn.Conv2d(self.network_dim, 2 * self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.network_dim, 4 * 9, 1, padding=0, bias=True),
        )
        self.facial_flow_head = nn.Sequential(
            nn.Conv2d(self.network_dim, 2 * self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.network_dim, 6, 1, padding=0, bias=True),
        )

        # ------------------ head flow decoder ------------------
        self.head_refine_net = VisionTransformer(
            args.network_backbone, self.network_dim, patch_size=8
        )
        self.head_hidden_conv = nn.Conv2d(
            self.network_dim * 2,
            self.network_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.head_warp_linear = nn.Conv2d(
            3 * self.network_dim + 2, self.network_dim, 1, 1, 0, bias=True
        )
        self.head_refine_transform = nn.Conv2d(
            self.network_dim // 2 * 3, self.network_dim, 1, 1, 0, bias=True
        )
        self.head_upsample_weight = nn.Sequential(
            nn.Conv2d(self.network_dim, 2 * self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.network_dim, 4 * 9, 1, padding=0, bias=True),
        )
        self.head_flow_head = nn.Sequential(
            nn.Conv2d(self.network_dim, 2 * self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.network_dim, 6, 1, padding=0, bias=True),
        )

    def upsample_data(self, flow, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 2 * H, 2 * W), up_info.reshape(N, C, 2 * H, 2 * W)

    def normalize_image(self, img):
        """
        @img: (B,C,H,W) in range 0-255, RGB order
        """
        tf = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False
        )
        return tf(img / 255.0).contiguous()

    def forward(self, image1, image2, iters=None, head_flow_gt=None):
        if iters is None:
            iters = self.args.iters
        image1 = self.normalize_image(image1)
        image2 = self.normalize_image(image2)
        padder = Padder(image1.shape, factor=112)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        facial_flow_predictions = []
        facial_info_predictions = []
        head_flow_predictions = []
        head_info_predictions = []
        N, _, H, W = image1.shape
        # initial feature
        da_feature1 = self.da_feature(image1)
        da_feature2 = self.da_feature(image2)
        fmap1_feats = self.fnet(torch.cat([da_feature1["out"], image1], dim=1))
        fmap2_feats = self.fnet(torch.cat([da_feature2["out"], image2], dim=1))
        da_feature1_2x = F.interpolate(
            da_feature1["out"], scale_factor=0.5, mode="bilinear", align_corners=True
        )
        da_feature2_2x = F.interpolate(
            da_feature2["out"], scale_factor=0.5, mode="bilinear", align_corners=True
        )
        fmap1_2x = self.fmap_conv(torch.cat([fmap1_feats[0], da_feature1_2x], dim=1))
        fmap2_2x = self.fmap_conv(torch.cat([fmap2_feats[0], da_feature2_2x], dim=1))

        # ---------------- facial decoder ----------------
        net = self.facial_hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))
        flow_2x = torch.zeros(N, 2, H // 2, W // 2).to(image1.device)
        for itr in range(iters):
            flow_2x = flow_2x.detach()
            coords2 = (
                coords_grid(N, H // 2, W // 2, device=image1.device) + flow_2x
            ).detach()
            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))
            refine_inp = self.facial_warp_linear(
                torch.cat([fmap1_2x, warp_2x, net, flow_2x], dim=1)
            )
            refine_outs = self.facial_refine_net(refine_inp)
            net = self.facial_refine_transform(
                torch.cat([refine_outs["out"], net], dim=1)
            )
            flow_update = self.facial_flow_head(net)
            weight_update = 0.25 * self.facial_upsample_weight(net)
            flow_2x = flow_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_2x, info_2x, weight_update)
            facial_flow_predictions.append(flow_up)
            facial_info_predictions.append(info_up)

        # unpad facial outputs
        for i in range(len(facial_info_predictions)):
            facial_flow_predictions[i] = padder.unpad(facial_flow_predictions[i])
            facial_info_predictions[i] = padder.unpad(facial_info_predictions[i])

        # ---------------- head decoder ----------------
        # Initialize head net from separate head_hidden_conv
        net_h = self.head_hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))
        flow_h_2x = torch.zeros(N, 2, H // 2, W // 2).to(image1.device)
        for itr in range(iters):
            flow_h_2x = flow_h_2x.detach()
            coords2_h = (
                coords_grid(N, H // 2, W // 2, device=image1.device) + flow_h_2x
            ).detach()
            warp_2x_h = bilinear_sampler(fmap2_2x, coords2_h.permute(0, 2, 3, 1))
            refine_inp_h = self.head_warp_linear(
                torch.cat([fmap1_2x, warp_2x_h, net_h, flow_h_2x], dim=1)
            )
            refine_outs_h = self.head_refine_net(refine_inp_h)
            net_h = self.head_refine_transform(
                torch.cat([refine_outs_h["out"], net_h], dim=1)
            )
            flow_update_h = self.head_flow_head(net_h)
            weight_update_h = 0.25 * self.head_upsample_weight(net_h)
            flow_h_2x = flow_h_2x + flow_update_h[:, :2]
            info_2x_h = flow_update_h[:, 2:]
            # upsample predictions
            flow_up_h, info_up_h = self.upsample_data(
                flow_h_2x, info_2x_h, weight_update_h
            )
            head_flow_predictions.append(flow_up_h)
            head_info_predictions.append(info_up_h)

        # unpad head outputs
        for i in range(len(head_info_predictions)):
            head_flow_predictions[i] = padder.unpad(head_flow_predictions[i])
            head_info_predictions[i] = padder.unpad(head_info_predictions[i])

        # ensure lists have same length (they should)
        assert len(facial_flow_predictions) == len(head_flow_predictions)

        # compute expression flow = facial - head for each iteration
        expression_flow_predictions = []
        for i in range(len(facial_flow_predictions)):
            expression_flow_predictions.append(
                facial_flow_predictions[i] - head_flow_predictions[i]
            )

        output = {
            "head_flow": head_flow_predictions,
            "expression_flow": expression_flow_predictions,
            "facial_flow": facial_flow_predictions,
        }
        return output