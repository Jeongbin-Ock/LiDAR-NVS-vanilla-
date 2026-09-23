"""Stateless torch pose and box geometry.

``apply_pose`` moves Dynamic 2DGS tokens into the window reference frame;
``point_in_box`` labels points with nuScenes boxes for the dataloader and the
evaluation viewers.
"""
from __future__ import annotations

import torch
from torch import Tensor


def apply_pose(points: Tensor, pose: Tensor) -> Tensor:
    """Apply a 4x4 homogeneous pose to (N, 3) points.

    points : (N, 3)
    pose   : (4, 4)  e.g. frame_i -> frame_0
    return : (N, 3) = (pose @ [points; 1])[:, :3]
    """
    N = points.shape[0]
    ones = torch.ones(N, 1, device=points.device, dtype=points.dtype)
    pts_h = torch.cat([points, ones], dim=-1)   # (N, 4)
    return (pose @ pts_h.T).T[:, :3]            # (N, 3)


def point_in_box(points: Tensor, boxes: Tensor) -> Tensor:
    """Assign each of (N, 3) points to the box it falls inside (or -1).

    points : (N, 3)
    boxes  : (B_f, 7) x,y,z,w,l,h,yaw, same frame as points.
    return : (N,) long, -1 = background, 0..B_f-1 = box index. Ties (point
             inside more than one box) resolve to the lowest box index.

    Size convention: columns 3:6 are nuScenes `size` = (width, length, height)
    and yaw rotates the box's LENGTH axis onto local +x (nuScenes `Box.corners`
    builds x from l and y from w). So local x is tested against l/2 and local y
    against w/2 -- testing x against w/2 rotates every non-square box by 90 deg
    and was measured to capture 2.2x fewer points on seq_1250_1300.
    """
    N, B_f = points.shape[0], boxes.shape[0]
    if B_f == 0:
        return torch.full((N,), -1, dtype=torch.long, device=points.device)

    cx, cy, cz = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    w, l, h    = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    yaw        = boxes[:, 6]

    dx = points[:, 0].unsqueeze(1) - cx.unsqueeze(0)  # (N, B_f)
    dy = points[:, 1].unsqueeze(1) - cy.unsqueeze(0)
    dz = points[:, 2].unsqueeze(1) - cz.unsqueeze(0)

    cos_y  = torch.cos(-yaw).unsqueeze(0)
    sin_y  = torch.sin(-yaw).unsqueeze(0)
    local_x = cos_y * dx - sin_y * dy
    local_y = sin_y * dx + cos_y * dy

    inside = (
        (local_x.abs() <= l.unsqueeze(0) / 2) &   # local +x spans the LENGTH
        (local_y.abs() <= w.unsqueeze(0) / 2) &   # local +y spans the WIDTH
        (dz.abs()      <= h.unsqueeze(0) / 2)
    )  # (N, B_f)

    # ``argmax`` returns the first maximum, preserving the existing lowest-box
    # tie break without launching one masked-assignment kernel per box.  Rows
    # with no match need an explicit guard because argmax would return 0 there.
    has_match = inside.any(dim=1)
    first_match = inside.to(dtype=torch.uint8).argmax(dim=1).long()
    return torch.where(has_match, first_match, torch.full_like(first_match, -1))
