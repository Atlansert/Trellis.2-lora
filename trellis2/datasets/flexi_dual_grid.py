"""双网格（Flexi Dual Grid）数据集：同时提供网格与稀疏体素两套监督。

职责：从 mesh_dump 的 pickle 读 GT 网格（合并多物体、居中归一化到 [-0.5,0.5]），
从 dual_grid 的 vxz 读活跃体素属性（顶点偏移/相交标志），一起返回给形状 VAE 训练。
输入：roots（含 mesh_dump 与 dual_grid 两个子目录）；输出：{'mesh', 'vertices', 'intersected'}。
契约：vertices 是稀疏张量（coords 已补 batch 维 0），intersected 为同坐标的 3 位 bool 标志；
collate_fn 用 sparse_cat 沿 batch 维合并稀疏张量，并按活跃体素数均衡分组。
"""

import os
import numpy as np
import pickle
import torch
import utils3d
from .components import StandardDatasetBase
from ..modules import sparse as sp
from ..renderers import MeshRenderer
from ..representations import Mesh
from ..utils.data_utils import load_balanced_group_indices
import pandas as pd
import o_voxel


class FlexiDualGridVisMixin:
    """双网格可视化 mixin：用随机环绕相机渲染 GT 网格的法线图。"""

    @torch.no_grad()
    def visualize_sample(self, x: dict):
        """渲染样本里的 GT 网格：4 个随机环绕视角的法线图拼成 2x2（[N,3,H,W]）。"""
        mesh = x['mesh']
        
        renderer = MeshRenderer({'near': 1, 'far': 3})
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.ssaa = 4
        
        # Build camera
        # 4 个方位角（间隔 90°，整体随机偏移）+ 随机俯仰；距离 2、fov 30°、up 轴 +z
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)
        
        # Build each representation
        images = []
        for m in mesh:
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = \
                    renderer.render(m.cuda(), ext, intr)['normal']
            images.append(image)
        images = torch.stack(images)
        
        return images
    

class FlexiDualGridDataset(FlexiDualGridVisMixin, StandardDatasetBase):
    """
    Flexible Dual Grid Dataset
    
    双网格数据集：一个样本同时含 GT 网格（渲染用）与稀疏体素（几何/相交监督）。
    
    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the voxel grid
        min_aesthetic_score (float): minimum aesthetic score of the instances to be included in the dataset
    """

    def __init__(
        self,
        roots,
        resolution: int = 1024,
        max_active_voxels: int = 1000000,
        max_num_faces: int = None,
        min_aesthetic_score: float = 5.0,
    ):
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.max_active_voxels = max_active_voxels
        self.max_num_faces = max_num_faces
        self.value_range = (0, 1)

        super().__init__(roots)
        
        # 记录每个实例的活跃体素数，供均衡分组/统计；索引重复时 loc 返回 Series，用 iloc[0] 兜底
        _loads_col = self.metadata['dual_grid_size']
        self.loads = [int(_loads_col.loc[sha256].iloc[0]) if isinstance(_loads_col.loc[sha256], pd.Series) else int(_loads_col.loc[sha256]) for _, sha256 in self.instances]
        
    def __str__(self):
        lines = [
            super().__str__(),
            f'  - Resolution: {self.resolution}',
        ]
        return '\n'.join(lines)
        
    def filter_metadata(self, metadata):
        # 过滤：已转双网格 -> 美学分达标（可选）-> 活跃体素数上限 -> 面数上限（可选）
        stats = {}
        metadata = metadata[metadata[f'dual_grid_converted'] == True]
        stats['Dual Grid Converted'] = len(metadata)
        if self.min_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
            stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata[f'dual_grid_size'] <= self.max_active_voxels]
        stats[f'Active Voxels <= {self.max_active_voxels}'] = len(metadata)
        if self.max_num_faces is not None:
            metadata = metadata[metadata['num_faces'] <= self.max_num_faces]
            stats[f'Faces <= {self.max_num_faces}'] = len(metadata)
        return metadata, stats
    
    def read_mesh(self, root, instance):
        # 读取 pickle 里的多物体网格：跳过空物体，面索引按累计顶点数做偏移后合并
        with open(os.path.join(root, f'{instance}.pickle'), 'rb') as f:
            dump = pickle.load(f)
        start = 0
        vertices = []
        faces = []
        for obj in dump['objects']:
            if obj['vertices'].size == 0 or obj['faces'].size == 0:
                continue
            vertices.append(obj['vertices'])
            faces.append(obj['faces'] + start)
            start += len(obj['vertices'])
        vertices = torch.from_numpy(np.concatenate(vertices, axis=0)).float()
        faces = torch.from_numpy(np.concatenate(faces, axis=0)).long()
        # 居中并缩放到 [-0.5, 0.5]：scale 取最长边刚好略小于 1，保证落在单位立方体内
        vertices_min = vertices.min(dim=0)[0]
        vertices_max = vertices.max(dim=0)[0]
        center = (vertices_min + vertices_max) / 2
        scale = 0.99999 / (vertices_max - vertices_min).max()
        vertices = (vertices - center) * scale
        assert torch.all(vertices >= -0.5) and torch.all(vertices <= 0.5), 'vertices out of range'
        return {'mesh': [Mesh(vertices=vertices, faces=faces)]}
    
    def read_dual_grid(self, root, instance):
        # 读 vxz：顶点属性除以 255 归一化到 [0,1]；coords 前补全 0 作为 batch 维（单样本）
        coords, attr = o_voxel.io.read_vxz(os.path.join(root, f'{instance}.vxz'), num_threads=4)
        vertices = sp.SparseTensor(
            (attr['vertices'] / 255.0).float(),
            torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
        )
        # intersected 用一个整数的 3 个 bit 编码（%2 / //2%2 / //4%2），拆成 3 通道 bool
        intersected = vertices.replace(torch.cat([
            attr['intersected'] % 2,
            attr['intersected'] // 2 % 2,
            attr['intersected'] // 4 % 2,
        ], dim=-1).bool())
        return {'vertices': vertices, 'intersected': intersected}

    def get_instance(self, root, instance):
        # 一个样本 = GT 网格（渲染监督）+ 双网格体素（几何监督）
        mesh = self.read_mesh(root['mesh_dump'], instance)
        dual_grid = self.read_dual_grid(root['dual_grid'], instance)
        return {**mesh, **dual_grid}
    
    @staticmethod
    def collate_fn(batch, split_size=None):
        """按活跃体素数均衡分组后拼批：Tensor 用 stack、SparseTensor 用 sparse_cat、list 展平。"""
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['vertices'].feats.shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}

            keys = [k for k in sub_batch[0].keys()]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], sp.SparseTensor):
                    pack[k] = sp.sparse_cat([b[k] for b in sub_batch], dim=0)
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
            
            packs.append(pack)
        
        if split_size is None:
            return packs[0]
        return packs
    