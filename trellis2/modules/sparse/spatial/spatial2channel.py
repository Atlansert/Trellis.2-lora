"""空间↔通道的 2 通道变换（不改变信息量，只重排稀疏布局）。

SparseSpatial2Channel：把 factor^DIM 个相邻子体素折叠进通道 → 点数减少 factor^DIM 倍，
通道数乘以 factor^DIM，坐标 = 父坐标（子体素坐标 = 父坐标*factor + child（factor=2 时即 *2+child））。
SparseChannel2Spatial：其逆操作，通道拆回空间 → 点数增加 factor^DIM 倍，通道数除以 factor^DIM。
两者通过 _spatial_cache 里的映射（new_coords, idx, subidx）配对；若无缓存，上采样需要
外部传入 subdivision（标记每个父体素实际存在哪些子体素）来恢复精确的稀疏结构。
"""

from typing import *
import torch
import torch.nn as nn
from .. import SparseTensor


class SparseSpatial2Channel(nn.Module):
    """
    Downsample a sparse tensor by a factor of `factor`.
    Implemented as rearranging its features from spatial to channel.
    """
    # 输出 feats: (N/factor^DIM, C*factor^DIM)
    def __init__(self, factor: int = 2):
        super(SparseSpatial2Channel, self).__init__()
        self.factor = factor

    def forward(self, x: SparseTensor) -> SparseTensor:
        DIM = x.coords.shape[-1] - 1  # 空间维数（coords 去掉 batch 列）
        cache = x.get_spatial_cache(f'spatial2channel_{self.factor}')
        if cache is None:
            coord = list(x.coords.unbind(dim=-1))
            for i in range(DIM):
                coord[i+1] = coord[i+1] // self.factor  # 父坐标 = 子坐标 // factor
            subidx = x.coords[:, 1:] % self.factor  # 子体素在父块内的偏移 (0..factor-1)
            subidx = sum([subidx[..., i] * self.factor ** i for i in range(DIM)])  # 压成一个线性下标

            # 把 (batch, x, y, z) 编码成单一整数 code 后 unique，得到父体素列表与映射 idx
            MAX = [(s + self.factor - 1) // self.factor for s in x.spatial_shape]
            OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
            code = sum([c * o for c, o in zip(coord, OFFSET)])
            code, idx = code.unique(return_inverse=True)

            new_coords = torch.stack(
                [code // OFFSET[0]] +
                [(code // OFFSET[i+1]) % MAX[i] for i in range(DIM)],
                dim=-1
            )
        else:
            new_coords, idx, subidx = cache
            
        # 按 (父下标, 子偏移) 放置特征，再 reshape 把 factor^DIM 个子体素并入通道维
        new_feats = torch.zeros(new_coords.shape[0] * self.factor ** DIM, x.feats.shape[1], device=x.feats.device, dtype=x.feats.dtype)
        new_feats[idx * self.factor ** DIM + subidx] = x.feats

        out = SparseTensor(new_feats.reshape(new_coords.shape[0], -1), new_coords, None if x._shape is None else torch.Size([x._shape[0], x._shape[1] * self.factor ** DIM]))
        out._scale = tuple([s * self.factor for s in x._scale])  # 尺度变大（体素变粗）
        out._spatial_cache = x._spatial_cache
        
        if cache is None:
            # 缓存映射，供后续 Channel2Spatial 精确还原
            x.register_spatial_cache(f'spatial2channel_{self.factor}', (new_coords, idx, subidx))
            out.register_spatial_cache(f'channel2spatial_{self.factor}', (x.coords, idx, subidx))
            out.register_spatial_cache(f'shape', torch.Size(MAX))
            if self.training:
                # 训练时缓存 subdivision：每个父体素实际被占用的子体素掩码 (N, factor^DIM)
                subdivision = torch.zeros((new_coords.shape[0], self.factor ** DIM), device=x.device, dtype=torch.bool)
                subdivision[idx, subidx] = True
                out.register_spatial_cache(f'subdivision', subdivision)
                
        return out


class SparseChannel2Spatial(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as rearranging its features from channel to spatial.
    """
    # 输入 feats: (N, C*factor^DIM) -> 输出 (N*factor^DIM 个真实子体素, C)
    def __init__(self, factor: int = 2):
        super(SparseChannel2Spatial, self).__init__()
        self.factor = factor

    def forward(self, x: SparseTensor, subdivision: Optional[SparseTensor] = None) -> SparseTensor:
        DIM = x.coords.shape[-1] - 1

        cache = x.get_spatial_cache(f'channel2spatial_{self.factor}')
        if cache is None:
            if subdivision is None:
                raise ValueError('Cache not found. Provide subdivision tensor or pair SparseChannel2Spatial with SparseSpatial2Channel.')
            else:
                # 无缓存时靠 subdivision 推断子体素：sub 为 (N, factor^DIM) 的占用掩码
                sub = subdivision.feats         # [N, self.factor ** DIM]
                N_leaf = sub.sum(dim=-1)        # [N] 每个父体素下的子体素数
                subidx = sub.nonzero()[:, -1]   # 被占用的子偏移线性下标
                new_coords = x.coords.clone().detach()
                new_coords[:, 1:] *= self.factor  # 父坐标 * factor
                new_coords = torch.repeat_interleave(new_coords, N_leaf, dim=0, output_size=subidx.shape[0])
                for i in range(DIM):
                    new_coords[:, i+1] += subidx // self.factor ** i % self.factor  # 再 + child
                idx = torch.repeat_interleave(torch.arange(x.coords.shape[0], device=x.device), N_leaf, dim=0, output_size=subidx.shape[0])
        else:
            new_coords, idx, subidx = cache

        # 通道维按 factor^DIM 拆开，按 (父下标, 子偏移) 取回各自的特征
        x_feats = x.feats.reshape(x.feats.shape[0] * self.factor ** DIM, -1)
        new_feats = x_feats[idx * self.factor ** DIM + subidx]
        out = SparseTensor(new_feats, new_coords, None if x._shape is None else torch.Size([x._shape[0], x._shape[1] // self.factor ** DIM]))
        out._scale = tuple([s / self.factor for s in x._scale])  # 尺度变小（体素变细）
        if cache is not None:           # only keep cache when subdiv following it
            out._spatial_cache = x._spatial_cache
        return out
