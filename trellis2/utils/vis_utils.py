"""稀疏张量可视化：把稀疏 latent 的特征用 PCA 压成 RGB，渲成环拍视频。

vis_sparse_tensor 只支持 batch=1 的 (b,x,y,z) 稀疏张量；体素边长取 1/2^ceil(log2(空间分辨率))，
即把整个立方体归一化到 [-0.5,0.5]³ 后再交给渲染器。
"""
from typing import *
import numpy as np
import torch
from ..modules import sparse as sp
from ..representations import Voxel
from .render_utils import render_video


def pca_color(feats: torch.Tensor, channels: Tuple[int, int, int] = (0, 1, 2)) -> torch.Tensor:
    """
    Apply PCA to the features and return the first three principal components.

    用左奇异向量的指定 3 列当 RGB，再逐列 min-max 归一化到 [0,1]。
    """
    feats = feats.detach()
    u, s, v = torch.svd(feats)
    color = u[:, channels]
    color = (color - color.min(dim=0, keepdim=True)[0]) / (color.max(dim=0, keepdim=True)[0] - color.min(dim=0, keepdim=True)[0])
    return color
    

def vis_sparse_tensor(
    x: sp.SparseTensor,
    num_frames: int = 300,
):
    """把稀疏张量特征 PCA 成颜色后渲成环拍视频，返回帧列表（仅支持 batch=1）。"""
    assert x.shape[0] == 1, "Only support batch size 1"
    assert x.coords.shape[1] == 4, "Only support 3D coordinates"
    
    coords = x.coords.cuda().detach()[:, 1:]
    feats = x.feats.cuda().detach()
    color = pca_color(feats)
    
    resolution = max(list(x.spatial_shape))
    resolution = int(2**np.ceil(np.log2(resolution)))
    
    rep = Voxel(
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1/resolution,
        coords=coords,
        attrs=color,
        layout={
            'color': slice(0, 3),
        }
    )

    return render_video(rep, colors_overwrite=color, num_frames=num_frames)['color']
