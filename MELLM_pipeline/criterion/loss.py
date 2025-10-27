import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EPELoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): 'mean' | 'sum' | 'none'
        """
        super(EPELoss, self).__init__()
        self.reduction = reduction

    def forward(self, exp_flow, exp_flow_gt):
        """
        Args:
            exp_flow (Tensor): 预测的光流, shape = [B, 2, H, W]
            exp_flow_gt (Tensor): GT光流, shape = [B, 2, H, W]
        Returns:
            Tensor: EPE loss
        """
        # 差值
        diff = exp_flow - exp_flow_gt  # [B, 2, H, W]
        # 计算欧式距离
        epe = torch.norm(diff, p=2, dim=1)  # [B, H, W]
        
        if self.reduction == 'mean':
            return epe.mean()
        elif self.reduction == 'sum':
            return epe.sum()
        elif self.reduction == 'none':
            return epe
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


MAX_FLOW = 40000
def sequence_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = output['nf'][i]
        mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
        if mask.sum() == 0:
            flow_loss += 0 * loss_i.sum()
        else:
            flow_loss += i_weight * ((mask * loss_i).sum()) / mask.sum()

    return flow_loss


def sequence_loss_v2(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(output['head_flow'])
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = output['nf'][i]
        mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
        if mask.sum() == 0:
            flow_loss += 0 * loss_i.sum()
        else:
            flow_loss += i_weight * ((mask * loss_i).sum()) / mask.sum()

    return flow_loss



import torch
import torch.nn.functional as F
from typing import List, Dict, Optional

def clamp_roi(x1, y1, x2, y2, W, H):
    x1c = max(0, min(W-1, int(round(x1))))
    y1c = max(0, min(H-1, int(round(y1))))
    x2c = max(0, min(W, int(round(x2))))
    y2c = max(0, min(H, int(round(y2))))
    # ensure at least 1x1
    if x2c <= x1c:
        x2c = min(W, x1c + 1)
    if y2c <= y1c:
        y2c = min(H, y1c + 1)
    return x1c, y1c, x2c, y2c

def compute_roi_flow_loss(
    pred_flow: torch.Tensor,     # [B,2,H,W]
    gt_flow: torch.Tensor,       # [B,2,H,W]
    batch_rois: List[List[Dict]],# length B, each element list of dicts {'pos_name':str,'position':[x1,y1,x2,y2]}
    per_roi_weight_map: Optional[Dict[str, float]] = None,
    reduction: str = 'mean',     # 'mean' or 'sum' over ROIs
    normalization: Optional[str] = 'per_roi', # None | 'per_roi' | 'per_sample' | 'global'
    eps: float = 1e-3,
    alpha: float = 0.0,         # optional offset inside denom: 1/(mean_mag + alpha)
    min_scale: float = 0.1,
    max_scale: float = 10.0,
    device: Optional[torch.device] = None
):
    """
    Compute ROI-based optical flow loss across batch with optional normalization.

    Returns:
      loss (scalar tensor), stats (dict): contains per-ROI diagnostics
    """
    B, C, H, W = pred_flow.shape
    assert C == 2
    if device is None:
        device = pred_flow.device

    total_weighted = 0.0
    total_count = 0
    stats = {'per_roi': []}  # list of dicts for each ROI

    # Precompute global baseline if needed
    if normalization == 'global':
        # compute mean magnitude across all GT (avoid zero)
        gt_mag_all = torch.norm(gt_flow, dim=1)  # [B,H,W]
        global_mean = gt_mag_all.mean().item()
        global_scale = 1.0 / (global_mean + alpha + eps)
        global_scale = max(min_scale, min(max_scale, global_scale))
    else:
        global_scale = None

    for b in range(B):
        rois = batch_rois[b]
        if rois is None or len(rois) == 0:
            continue
        pred_b = pred_flow[b:b+1]  # [1,2,H,W]
        gt_b = gt_flow[b:b+1]
        # sample-level mean magnitude if normalization == 'per_sample'
        if normalization == 'per_sample':
            sample_mean = torch.norm(gt_b, dim=1).mean().item()
            sample_scale = 1.0 / (sample_mean + alpha + eps)
            sample_scale = max(min_scale, min(max_scale, sample_scale))
        else:
            sample_scale = None

        for roi in rois:
            name = roi.get('pos_name', 'roi')
            x1,y1,x2,y2 = roi['position']
            # clamp ROI coords to valid ints
            x1c,y1c,x2c,y2c = clamp_roi(x1,y1,x2,y2,W,H)
            # crop
            # indices: flow is [1,2,H,W], slice: :, :, y1:y2, x1:x2
            pred_crop = pred_b[..., y1c:y2c, x1c:x2c]  # [1,2,hh,ww]
            gt_crop = gt_b[..., y1c:y2c, x1c:x2c]
            # if empty skip
            if pred_crop.numel() == 0:
                continue
            # per-pixel EPE
            diff = pred_crop - gt_crop
            epe_map = torch.sqrt((diff**2).sum(dim=1) + 1e-12)  # [1,hh,ww]
            epe_mean = float(epe_map.mean().item())

            # compute normalization scale
            if normalization is None:
                scale = 1.0
            elif normalization == 'per_roi':
                mean_mag_roi = float(torch.norm(gt_crop, dim=1).mean().item())
                scale = 1.0 / (mean_mag_roi + alpha + eps)
                scale = max(min_scale, min(max_scale, scale))
            elif normalization == 'per_sample':
                scale = sample_scale
            elif normalization == 'global':
                scale = global_scale
            else:
                raise ValueError("unknown normalization mode")

            # optional per-ROI name weight multiplier
            name_w = 1.0
            if per_roi_weight_map and name in per_roi_weight_map:
                name_w = float(per_roi_weight_map[name])

            weighted_loss = name_w * (scale * epe_mean)

            total_weighted += weighted_loss
            total_count += 1

            stats['per_roi'].append({
                'batch_idx': b,
                'name': name,
                'coords': (x1c,y1c,x2c,y2c),
                'epe_mean': epe_mean,
                'scale': scale,
                'name_weight': name_w,
                'weighted_loss': weighted_loss
            })

    if total_count == 0:
        # no ROI in the batch -> zero loss
        loss_val = torch.tensor(0.0, device=device, requires_grad=True)
        return loss_val, stats

    if reduction == 'mean':
        loss_scalar = total_weighted / float(total_count)
    else:
        loss_scalar = total_weighted

    # wrap scalar into tensor so it can be used in backward (we must return tensor)
    loss = torch.tensor(loss_scalar, device=device, dtype=pred_flow.dtype, requires_grad=True)
    # Note: loss is a leaf tensor; to allow proper autograd with preds, better compute differentiable
    # alternative: compute aggregated differentiable tensor instead of scalar python ops.
    # For simplicity and correctness in autograd, re-implement with tensors below:

    # --- differentiable path: compute the same but with tensors (vectorized loop) ---
    # We'll compute a differentiable loss_tensor by summing EPE_map*scale*name_w and dividing by count.
    loss_items = []
    count_items = 0
    for s in stats['per_roi']:
        b_idx = s['batch_idx']
        x1c,y1c,x2c,y2c = s['coords']
        pred_crop = pred_flow[b_idx:b_idx+1, :, y1c:y2c, x1c:x2c]
        gt_crop = gt_flow[b_idx:b_idx+1, :, y1c:y2c, x1c:x2c]
        if pred_crop.numel() == 0:
            continue
        diff = pred_crop - gt_crop
        epe_map = torch.sqrt((diff**2).sum(dim=1) + 1e-12)  # [1,hh,ww]
        epe_mean_t = epe_map.mean()
        # reconstruct scale and name_w as tensors
        scale_t = torch.tensor(s['scale'], device=device, dtype=pred_flow.dtype)
        name_w_t = torch.tensor(s['name_weight'], device=device, dtype=pred_flow.dtype)
        loss_items.append(name_w_t * scale_t * epe_mean_t)
        count_items += 1
    if count_items == 0:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        if reduction == 'mean':
            loss = torch.stack(loss_items).sum() / float(len(loss_items))
        else:  # 'sum'
            loss = torch.stack(loss_items).sum()

    return loss, stats


from roi_get import get_roi_flow
def ROI_loss(output, flow_gt, lm98, valid, gamma=0.8):
    facial_flow = output['flow']
    B = facial_flow[-1].shape[0]
    roi_dic_list = []
    for idx in range(B):
        roi_dic_list.append(get_roi_flow(np.array(facial_flow[-1][idx].cpu().detach().numpy()), np.array(lm98[idx].cpu())))
    roi_loss, _ = compute_roi_flow_loss(facial_flow[-1], flow_gt, roi_dic_list, normalization=None)
    return roi_loss




def exp_head_loss(output, exp_flow_gt, head_flow_gt, exp_valid, head_valid, gamma=0.8, max_flow=MAX_FLOW):
    n_predictions = len(output['expression_flow'])
    exp_flow_loss = 0.0
    head_flow_loss = 0.0
    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(exp_flow_gt**2, dim=1).sqrt()
    exp_valid = (exp_valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = output['expression_flow'][i]
        # compute EPE
        epe = torch.norm(flow_pred - exp_flow_gt, dim=1, keepdim=True)
        # mask invalid/nan/inf
        mask = (~torch.isnan(epe.detach())) & (~torch.isinf(epe.detach())) & exp_valid[:, None]
        if mask.sum() == 0:
            exp_flow_loss += 0 * epe.sum()
        else:
            exp_flow_loss += i_weight * ((mask * epe).sum()) / mask.sum()    


    mag = torch.sum(head_flow_gt**2, dim=1).sqrt()
    head_valid = (head_valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = output['nf'][i]
        mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & head_valid[:, None]
        if mask.sum() == 0:
            head_flow_loss += 0 * loss_i.sum()
        else:
            head_flow_loss += i_weight * ((mask * loss_i).sum()) / mask.sum()
    return exp_flow_loss + head_flow_loss, head_flow_loss.item(), exp_flow_loss.item()



def epe_loss_function(output, exp_flow_gt, exp_valid, gamma=0.8, max_flow=MAX_FLOW):
    """ EPE-based sequence loss """
    n_predictions = len(output['expression_flow'])
    exp_flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(exp_flow_gt**2, dim=1).sqrt()
    exp_valid = (exp_valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = output['expression_flow'][i]
        # compute EPE
        epe = torch.norm(flow_pred - exp_flow_gt, dim=1, keepdim=True)
        # mask invalid/nan/inf
        mask = (~torch.isnan(epe.detach())) & (~torch.isinf(epe.detach())) & exp_valid[:, None]
        if mask.sum() == 0:
            exp_flow_loss += 0 * epe.sum()
        else:
            exp_flow_loss += i_weight * ((mask * epe).sum()) / mask.sum()
    return exp_flow_loss


def tvl1_head_loss(output, exp_flow_gt, tvl1_flow, head_flow_gt, exp_valid, head_valid, gamma=0.8, max_flow=MAX_FLOW):
    """ EPE-based sequence loss """
    n_predictions = len(output['flow'])
    exp_flow_loss = 0.0
    head_flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(exp_flow_gt**2, dim=1).sqrt()
    exp_valid = (exp_valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = tvl1_flow - output['flow'][i]
        # compute EPE
        epe = torch.norm(flow_pred - exp_flow_gt, dim=1, keepdim=True)
        # mask invalid/nan/inf
        mask = (~torch.isnan(epe.detach())) & (~torch.isinf(epe.detach())) & exp_valid[:, None]
        if mask.sum() == 0:
            exp_flow_loss += 0 * epe.sum()
        else:
            exp_flow_loss += i_weight * ((mask * epe).sum()) / mask.sum()

    return exp_flow_loss