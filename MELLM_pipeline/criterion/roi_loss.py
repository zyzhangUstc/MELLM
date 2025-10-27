import torch
import torch.nn.functional as F
from criterion.roi_mask_get import get_roi_mask
def _ensure4d(t):
    # accept BxH xW or Bx1xHxW or BxCxHxW
    if t.dim() == 3:
        return t.unsqueeze(1)  # Bx1xHxW
    return t

def _resize_flow_and_roi(gt_flow, roi, out_h, out_w, align_corners=True):
    """
    Resize gt_flow (B,2,H,W) to (B,2,out_h,out_w) and scale flow values properly.
    roi: (B,1,H,W) or (B,H,W)
    """
    B, C, H, W = gt_flow.shape
    # resize flows by bilinear
    gt_resized = F.interpolate(gt_flow, size=(out_h, out_w), mode='bilinear', align_corners=align_corners)
    # scale channels: x (u) by width ratio, y (v) by height ratio
    scale_x = out_w / float(W)
    scale_y = out_h / float(H)
    gt_resized[:, 0:1, :, :] = gt_resized[:, 0:1, :, :] * scale_x
    gt_resized[:, 1:2, :, :] = gt_resized[:, 1:2, :, :] * scale_y

    roi4 = _ensure4d(roi)
    roi_resized = F.interpolate(roi4, size=(out_h, out_w), mode='bilinear', align_corners=align_corners)

    return gt_resized, roi_resized

def charbonnier_epe(pred, gt, eps=1e-3):
    # pred,gt: Bx2xHxW
    diff = pred - gt
    epe = torch.norm(diff, p=2, dim=1)  # BxHxW
    return torch.sqrt(epe * epe + eps * eps)  # BxHxW

def angular_loss(pred, gt, ang_eps=1e-3):
    """
    pred, gt: Bx2xHxW
    returns angular loss map: 1 - cos(theta) in [0,2]
    """
    # flatten channel dim for dot product
    # dot = u1*u2 + v1*v2
    dot = (pred * gt).sum(dim=1)  # BxHxW
    pred_norm = torch.norm(pred, p=2, dim=1)  # BxHxW
    gt_norm = torch.norm(gt, p=2, dim=1)      # BxHxW
    denom = (pred_norm * gt_norm).clamp(min=ang_eps)
    cos = (dot / denom).clamp(-1.0, 1.0)
    ang = 1.0 - cos  # 0 when perfect alignment, up to 2 when opposite
    return ang

def roi_multiscale_iterative_loss(pred_flows,
                                   gt_flow,
                                   roi_mask,
                                   alpha_char=1.0,
                                   beta_ang=0.5,
                                   gamma=0.8,
                                   char_eps=1e-3,
                                   ang_eps=1e-3,
                                   align_corners=True):
    """
    Multi-scale iterative ROI loss with Charbonnier EPE + angular error.

    Args:
        pred_flows: list of tensors [pred_0, pred_1, ..., pred_K-1]
                    each pred_i is B x 2 x H_i x W_i (flow in pixel units)
                    len(pred_flows) == number of refine steps/scales
        gt_flow: B x 2 x H x W  (ground-truth flow in pixel units at full resolution)
        roi_mask: B x 1 x H x W  or B x H x W  (soft weights in [0,1])
        alpha_char: weight for Charbonnier EPE
        beta_ang: weight for angular loss
        scale_weights: list/iterable of length K giving per-scale weight.
                       If None, defaults to linearly increasing weights from 0.5..1.0
        char_eps: small eps for Charbonnier
        ang_eps: small eps for angular denom stability
        align_corners: for interpolate
    Returns:
        loss (scalar), dict of components (per-scale and totals)
    """
    assert isinstance(pred_flows, (list, tuple)) and len(pred_flows) > 0
    K = len(pred_flows)
    device = gt_flow.device

    B, _, H, W = gt_flow.shape
    roi4 = _ensure4d(roi_mask).to(device)  # Bx1xHxW

    total_loss = 0.0
    summary = {'char_per_scale': [], 'ang_per_scale': []}

    # iterate through scales (assume pred_flows[0] is coarsest or earliest refine)
    for k, pred in enumerate(pred_flows):
        assert pred.dim() == 4 and pred.shape[1] == 2, "pred must be Bx2xHxW"
        _, _, Hk, Wk = pred.shape

        # resize gt and roi to current pred size and properly scale gt flow vectors
        gt_k, roi_k = _resize_flow_and_roi(gt_flow, roi4, Hk, Wk, align_corners=align_corners)

        # compute char EPE map and angular map
        char_map = charbonnier_epe(pred, gt_k, eps=char_eps)  # BxHxW
        ang_map = angular_loss(pred, gt_k, ang_eps)           # BxHxW

        # weighted sums over ROI (normalize by roi sum per-sample to keep stable)
        # roi_k: Bx1xHk xWk -> squeeze to BxHxW
        roi_s = roi_k.squeeze(1)
        denom = roi_s.view(B, -1).sum(dim=1).clamp_min(1e-6)  # B

        char_num = (roi_s * char_map).view(B, -1).sum(dim=1)  # B
        ang_num  = (roi_s * ang_map ).view(B, -1).sum(dim=1)  # B

        L_char_k = (char_num / denom).mean()  # scalar
        L_ang_k  = (ang_num  / denom).mean()  # scalar

        wk = gamma ** (K - k - 1)
        loss_k = wk * (alpha_char * L_char_k + beta_ang * L_ang_k)

        total_loss = total_loss + loss_k

        summary['char_per_scale'].append(L_char_k.item())
        summary['ang_per_scale'].append(L_ang_k.item())

    summary['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else float(total_loss)
    return total_loss, summary


def get_roi_loss(pred_flows, gt_flow, lm01):
    roi_mask = get_roi_mask(gt_flow, lm01)
    total_loss, _ = roi_multiscale_iterative_loss(pred_flows, gt_flow, roi_mask.to(gt_flow.device))
    return total_loss




if __name__ == '__main__':
    torch.manual_seed(0)

    B, H, W = 2, 64, 64   # batch=2, 全分辨率
    # 生成 ground truth flow (像素单位)，小范围随机位移
    gt_flow = torch.randn(B, 2, H, W) * 0.5  

    # ROI mask: 以中心为高斯分布，越靠近中心权重越大
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    yy = yy.float(); xx = xx.float()
    center_y, center_x = H/2, W/2
    sigma = H/6
    roi_mask = torch.exp(-(((yy-center_y)**2+(xx-center_x)**2)/(2*sigma**2)))
    roi_mask = roi_mask.unsqueeze(0).repeat(B,1,1)  # BxHxW

    # 模拟网络预测的多尺度 flows
    # 三个尺度，尺寸分别 16x16, 32x32, 64x64
    pred_flow_1 = gt_flow[:, :, ::4, ::4] + 0.1*torch.randn(B,2,H//4,W//4)
    pred_flow_2 = gt_flow[:, :, ::2, ::2] + 0.1*torch.randn(B,2,H//2,W//2)
    pred_flow_3 = gt_flow + 0.1*torch.randn(B,2,H,W)

    pred_flows = [pred_flow_1, pred_flow_2, pred_flow_3]

    # 调用损失函数
    from math import isfinite
    loss, info = roi_multiscale_iterative_loss(
        pred_flows, gt_flow, roi_mask,
        alpha_char=1.0, beta_ang=0.3
    )

    print("Total loss:", loss.item())
    print("Per-scale Char EPE:", info['char_per_scale'])
    print("Per-scale Angular:", info['ang_per_scale'])
    print("Scale weights:", info['weight_per_scale'])
