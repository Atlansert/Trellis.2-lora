"""体素(Voxel)表示：稀疏体素坐标 + 每体素的属性张量，可读写磁盘。

- coords: (N, 3) 整数体素坐标；attrs: (N, C) 属性，按 layout 里各字段的 slice 解释。
- 世界坐标 = (coords + 0.5) * voxel_size + origin，即 origin 在体素中心而非角点。
- save/load 走 o_voxel 的 IO，load 时按字段顺序重建 layout。
"""
from typing import Dict
import torch


class Voxel:
    def __init__(
            self, 
            origin: list,
            voxel_size: float,
            coords: torch.Tensor = None,
            attrs: torch.Tensor = None,
            layout: Dict = {},
            device: torch.device = 'cuda'
        ):
        self.origin = torch.tensor(origin, dtype=torch.float32, device=device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.layout = layout
        self.device = device
        
    @property
    def position(self):
        # 体素中心的世界坐标：(coords + 0.5) * voxel_size + origin
        return (self.coords + 0.5) * self.voxel_size + self.origin[None, :]
    
    def split_attrs(self):
        # 按 layout 把 attrs 拆成 {字段名: 子张量}
        return {
            k: self.attrs[:, self.layout[k]]
            for k in self.layout
        }
        
    def save(self, path):
        # lazy import
        if 'o_voxel' not in globals():
            import o_voxel
        o_voxel.io.write(
            path,
            self.coords,
            self.split_attrs(),
        )
        
    def load(self, path):
        # lazy import
        if 'o_voxel' not in globals():
            import o_voxel
        coord, attrs = o_voxel.io.read(path)
        self.coords = coord.int().to(self.device)
        self.attrs = torch.cat([attrs[k] for k in attrs], dim=1).to(self.device)
        # build layout
        start = 0
        self.layout = {}
        for k in attrs:
            self.layout[k] = slice(start, start + attrs[k].shape[1])
            start += attrs[k].shape[1]
