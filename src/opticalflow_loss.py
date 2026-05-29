# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def flow_to_norm(flow, H, W):
    flow_x = 2.0 * flow[:, 0:1] / max(W - 1, 1)
    flow_y = 2.0 * flow[:, 1:2] / max(H - 1, 1)
    return torch.cat([flow_x, flow_y], dim=1)


def make_identity_grid(N, H, W, device):
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(N, 1, 1, 1)
    return base_grid


def warp_backward(src, flow_bwd):
    """Warp src at t-1 into t coordinates using backward flow F_{t->t-1}."""
    N, C, H, W = src.shape
    norm_flow = flow_to_norm(flow_bwd, H, W)
    base_grid = make_identity_grid(N, H, W, src.device)
    grid = base_grid + norm_flow.permute(0, 2, 3, 1)
    warped = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid_x = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0)
    valid_y = (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    valid = (valid_x & valid_y).float().unsqueeze(1)
    return warped, valid


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, diff):
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


class TorchvisionRAFT:
    def __init__(self, device, use_small=False):
        if use_small:
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

            self.model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        else:
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

            self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        self.model.to(device).eval()

    @torch.no_grad()
    def backward(self, curr, prev):
        out = self.model(curr, prev)
        if isinstance(out, (list, tuple)):
            flow = out[-1]
        else:
            flow = out
        return flow


def get_temporal_loss_model(accelerator, weight_dtype, use_small=False):
    raft = TorchvisionRAFT(device=accelerator.device, use_small=use_small)
    raft.model.requires_grad_(False)
    raft.model.to(accelerator.device, dtype=weight_dtype)
    raft.model = accelerator.prepare(raft.model)

    temp_loss = CharbonnierLoss(eps=1e-3)
    return raft, temp_loss


def get_temporal_loss(x_tgt_pred, x_tgt, raft, temp_loss, visual=False):
    x_tgt = rearrange(x_tgt, "b c v h w -> b v c h w")
    x_tgt_pred = rearrange(x_tgt_pred, "b c v h w -> b v c h w")

    B, T, C, H, W = x_tgt.shape

    if visual and T == 1:
        loss_tmp, visual_warp = 0, x_tgt_pred
        return loss_tmp, visual_warp

    with torch.no_grad():
        flows_bwd = []
        for t in range(1, T):
            F_bwd_t = raft.backward(x_tgt[:, t], x_tgt[:, 0])
            flows_bwd.append(F_bwd_t)
        flow_bwd = torch.stack(flows_bwd, dim=1)

    tmp_losses = []
    if visual:
        visual_warp = torch.zeros(x_tgt.shape, device=x_tgt.device, dtype=x_tgt.dtype)
        visual_warp[:, 0] = x_tgt_pred[:, 0]

    for t in range(1, T):
        F_bwd_t = flow_bwd[:, t - 1]
        warped_t, valid = warp_backward(x_tgt_pred[:, 0], F_bwd_t)
        tmp_losses.append(temp_loss(valid * (x_tgt[:, t] - warped_t)))

        if visual:
            visual_warp[:, t] = warped_t

    loss_tmp = torch.stack(tmp_losses).mean()

    if visual:
        visual_warp = rearrange(visual_warp, "b v c h w -> b c v h w")
        return loss_tmp, visual_warp
    else:
        return loss_tmp
