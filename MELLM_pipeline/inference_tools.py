import sys
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


def generate_gaussian(size, sigma):
    """
    Generate a 2D Gaussian pattern based on the distance to the center.

    Args:
        size (tuple): (height, width) of the output pattern.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        torch.Tensor: 2D Gaussian pattern.
    """
    height, width = size
    # Create a coordinate grid
    y = torch.arange(0, height).view(-1, 1) / height
    x = torch.arange(0, width).view(1, -1) / width

    # Compute the center coordinates
    center_y, center_x = 0.5, 0.5

    # Compute the squared distance from each point to the center
    distance_squared = (x - center_x) ** 2 + (y - center_y) ** 2

    # Apply the Gaussian function
    gaussian = torch.exp(-distance_squared / (2 * sigma**2))
    return gaussian


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val += val
        self.count += n
        self.avg = self.val / self.count


class InferenceWrapper(object):
    def __init__(
        self, model, scale=0, train_size=None, pad_to_train_size=False, tiling=False
    ):
        self.model = model
        self.train_size = train_size
        self.scale = scale
        self.pad_to_train_size = pad_to_train_size
        self.tiling = tiling

    def inference_padding(self, image):
        h, w = self.train_size
        H, W = image.shape[2:]
        pad_h = max(h - H, 0)
        pad_w = max(w - W, 0)
        padded_image = F.pad(
            image,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            mode="constant",
            value=0,
        )
        return padded_image, pad_h, pad_w

    def patch_inference(self, image1, image2, patches, tile_h, tile_w):
        output = None
        n, _, h, w = image1.shape
        valid = torch.zeros((n, h, w), device=image1.device)
        for h_ij, w_ij in patches:
            hl, hr = h_ij, h_ij + tile_h
            wl, wr = w_ij, w_ij + tile_w
            weight = (
                generate_gaussian((hr - hl, wr - wl), 1).to(image1.device).unsqueeze(0)
            )
            image1_ij = image1[:, :, hl:hr, wl:wr]
            image2_ij = image2[:, :, hl:hr, wl:wr]
            output_ij = self.model(image1_ij, image2_ij)
            valid[:, hl:hr, wl:wr] += weight
            if output is None:
                output = {}
                for key in output_ij.keys():
                    if "head_flow" in key or "expression_flow" in key:
                        output[key] = [
                            torch.zeros((n, 2, h, w), device=image1.device)
                            for _ in range(len(output_ij[key]))
                        ]
                    elif "info" in key:
                        output[key] = [
                            torch.zeros((n, 4, h, w), device=image1.device)
                            for _ in range(len(output_ij[key]))
                        ]
                    else:
                        output[key] = [
                            torch.zeros((n, 2, h, w), device=image1.device)
                            for _ in range(len(output_ij[key]))
                        ]

            for i in range(len(output_ij["head_flow"])):
                for key in output.keys():
                    output[key][i][:, :, hl:hr, wl:wr] += weight * output_ij[key][i]

        for i in range(len(output["head_flow"])):
            for key in output.keys():
                output[key][i] /= valid.unsqueeze(1)

        return output

    def forward_flow(self, image1, image2):
        H, W = image1.shape[2:]
        if self.pad_to_train_size:
            image1, inf_pad_h, inf_pad_w = self.inference_padding(image1)
            image2, inf_pad_h, inf_pad_w = self.inference_padding(image2)
        else:
            inf_pad_h, inf_pad_w = 0, 0

        if self.tiling and self.pad_to_train_size:
            h, w = image1.shape[2:]
            tile_h, tile_w = self.train_size
            step_h, step_w = tile_h // 4 * 3, tile_w // 4 * 3
            patches = []
            for i in range(0, h, step_h):
                for j in range(0, w, step_w):
                    h_ij = max(min(i, h - tile_h), 0)
                    w_ij = max(min(j, w - tile_w), 0)
                    patches.append((h_ij, w_ij))

            # remove duplicates
            patches = list(set(patches))
        else:
            h, w = image1.shape[2:]
            tile_h, tile_w = h, w
            patches = [(0, 0)]

        output = self.patch_inference(image1, image2, patches, tile_h, tile_w)

        for i in range(len(output["head_flow"])):
            for key in output.keys():
                output[key][i] = output[key][i][
                    :,
                    :,
                    inf_pad_h // 2 : inf_pad_h // 2 + H,
                    inf_pad_w // 2 : inf_pad_w // 2 + W,
                ]

        return output

    def calc_flow(self, image1, image2):
        img1 = F.interpolate(
            image1, scale_factor=2**self.scale, mode="bilinear", align_corners=True
        )
        img2 = F.interpolate(
            image2, scale_factor=2**self.scale, mode="bilinear", align_corners=True
        )
        H, W = img1.shape[2:]
        output = self.forward_flow(img1, img2)
        for i in range(len(output["head_flow"])):
            for key in output.keys():
                if "head_flow" in key or "expression_flow" in key:
                    output[key][i] = F.interpolate(
                        output[key][i],
                        scale_factor=0.5**self.scale,
                        mode="bilinear",
                        align_corners=True,
                    ) * (0.5**self.scale)
                else:
                    output[key][i] = F.interpolate(
                        output[key][i],
                        scale_factor=0.5**self.scale,
                        mode="bilinear",
                        align_corners=True,
                    )
        return output
