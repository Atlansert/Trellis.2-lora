"""稀疏体素 PBR 数据集：提供材质属性体素（以及可选的带贴图 GT 网格）。

职责：从 pbr_dump 的 pickle 还原带 PBR 材质的网格（贴图/材质参数/UV），从 pbr_voxel 的
vxz 读活跃体素的属性（base_color/metallic/roughness/emissive/alpha），属性拼成 feats。
输入：roots（含 pbr_dump 与 pbr_voxel 子目录）；输出：{'x': 属性稀疏张量} 或再加上 'mesh'。
契约：feats 用 [-1,1] 归一化（读入时由 [0,255] 线性映射而来），layout 切片定义属性语义；
with_mesh 控制是否附带 GT 网格（渲染监督需要）；collate_fn 按属性体素数均衡分组。
"""

import os
import io
from typing import Union
import numpy as np
import pickle
import torch
from PIL import Image
import o_voxel
import utils3d
from .components import StandardDatasetBase
from ..modules import sparse as sp
from ..renderers import VoxelRenderer
from ..representations import Voxel
from ..representations.mesh import MeshWithPbrMaterial, TextureFilterMode, TextureWrapMode, AlphaMode, PbrMaterial, Texture

from ..utils.data_utils import load_balanced_group_indices


def is_power_of_two(n: int) -> bool:
    """判断 n 是否为 2 的幂（贴图尺寸要求）。"""
    return n > 0 and (n & (n - 1)) == 0


def nearest_power_of_two(n: int) -> int:
    """返回离 n 最近的 2 的幂（距离相同时取较大者）。"""
    if n < 1:
        raise ValueError("n must be >= 1")
    if is_power_of_two(n):
        return n
    lower = 2 ** (n.bit_length() - 1)
    upper = 2 ** n.bit_length()
    if n - lower < upper - n:
        return lower
    else:
        return upper
    

class SparseVoxelPbrVisMixin:
    """PBR 体素可视化 mixin：把每种属性单独当作颜色渲染 4 视角，便于逐属性排查。"""

    @torch.no_grad()
    def visualize_sample(self, x: Union[sp.SparseTensor, dict]):
        """按 layout 里每个属性分别渲染 4 视角图，返回 {属性名: [N,3,H,W]}。"""
        x = x if isinstance(x, sp.SparseTensor) else x['x']
        
        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.ssaa = 4
        
        # Build camera
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

        images = {k: [] for k in self.layout}
        
        # 每种属性单独出一张 2x2 拼图；属性通道数不足 3 时用 expand 复制成 3 通道当伪彩色
        x = x.cuda()
        for i in range(x.shape[0]):
            rep = Voxel(
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1/self.resolution,
                coords=x[i].coords[:, 1:].contiguous(),
                attrs=None,
                layout={
                    'color': slice(0, 3),
                }
            )
            for k in self.layout:
                image = torch.zeros(3, 1024, 1024).cuda()
                tile = [2, 2]
                for j, (ext, intr) in enumerate(zip(exts, ints)):
                    attr = x[i].feats[:, self.layout[k]].expand(-1, 3)
                    res = renderer.render(rep, ext, intr, colors_overwrite=attr)
                    image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
                images[k].append(image)
        
        for k in self.layout:
            images[k] = torch.stack(images[k])
            
        return images


class SparseVoxelPbrDataset(SparseVoxelPbrVisMixin, StandardDatasetBase):
    """
    Sparse Voxel PBR dataset.
    
    稀疏体素 PBR 数据集：属性体素（可选附带带贴图 GT 网格），feats 位于 [-1,1]。
    
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
        attrs: list[str] = ['base_color', 'metallic', 'roughness', 'emissive', 'alpha'],
        with_mesh: bool = True,
    ):
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.max_active_voxels = max_active_voxels
        self.max_num_faces = max_num_faces
        self.with_mesh = with_mesh
        self.value_range = (-1, 1)   # 属性 feats 的取值范围（可视化时用）
        # 各属性通道数 -> layout 给出它们在 feats 里的切片
        self.channels = {
            'base_color': 3,
            'metallic': 1,
            'roughness': 1,
            'emissive': 3,
            'alpha': 1,
        }
        self.layout = {}
        start = 0
        for attr in attrs:
            self.layout[attr] = slice(start, start + self.channels[attr])
            start += self.channels[attr]

        super().__init__(roots)
        
        self.loads = [self.metadata.loc[sha256, f'num_pbr_voxels'] for _, sha256 in self.instances]
        
    def __str__(self):
        lines = [
            super().__str__(),
            f'  - Resolution: {self.resolution}',
            f'  - Attributes: {list(self.layout.keys())}',
        ]
        return '\n'.join(lines)
        
    def filter_metadata(self, metadata):
        # 过滤：已做 PBR 体素化 -> 美学分达标（可选）-> 体素数上限 -> 面数上限（可选）
        stats = {}
        metadata = metadata[metadata['pbr_voxelized'] == True]
        stats['PBR Voxelized'] = len(metadata)
        if self.min_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
            stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['num_pbr_voxels'] <= self.max_active_voxels]
        stats[f'Active voxels <= {self.max_active_voxels}'] = len(metadata)
        if self.max_num_faces is not None:
            metadata = metadata[metadata['num_faces'] <= self.max_num_faces]
            stats[f'Faces <= {self.max_num_faces}'] = len(metadata)
        return metadata, stats

    @staticmethod
    def _texture_from_dump(pack) -> Texture:
        """把 dump 里的贴图字节解码成 Texture：非正方形/非 2 次幂时缩放到最近 2 次幂并映射插值/包裹模式。"""
        png_bytes = pack['image']
        image = Image.open(io.BytesIO(png_bytes))
        if image.width != image.height or not is_power_of_two(image.width):
            size = nearest_power_of_two(max(image.width, image.height))
            image = image.resize((size, size), Image.LANCZOS)
        texture = torch.tensor(np.array(image) / 255.0, dtype=torch.float32).reshape(image.height, image.width, -1)
        filter_mode = {
            'Linear': TextureFilterMode.LINEAR,
            'Closest': TextureFilterMode.CLOSEST,
            'Cubic': TextureFilterMode.LINEAR,
            'Smart': TextureFilterMode.LINEAR,
        }[pack['interpolation']]
        wrap_mode = {
            'REPEAT': TextureWrapMode.REPEAT,
            'EXTEND': TextureWrapMode.CLAMP_TO_EDGE,
            'CLIP': TextureWrapMode.CLAMP_TO_EDGE,
            'MIRROR': TextureWrapMode.MIRRORED_REPEAT,
        }[pack['extension']]
        return Texture(texture, filter_mode=filter_mode, wrap_mode=wrap_mode)

    def read_mesh_with_texture(self, root, instance):
        with open(os.path.join(root, f'{instance}.pickle'), 'rb') as f:
            dump = pickle.load(f)
            
        # 兼容修正：有 alpha 贴图却标成 OPAQUE 的材质，改为 BLEND，否则 alpha 不生效
        for mat in dump['materials']:
            if mat['alphaTexture'] is not None and mat['alphaMode'] == 'OPAQUE':
                mat['alphaMode'] = 'BLEND'

        # process material
        materials = []
        for mat in dump['materials']:
            materials.append(PbrMaterial(
                base_color_texture=self._texture_from_dump(mat['baseColorTexture']) if mat['baseColorTexture'] is not None else None,
                base_color_factor=mat['baseColorFactor'],
                metallic_texture=self._texture_from_dump(mat['metallicTexture']) if mat['metallicTexture'] is not None else None,
                metallic_factor=mat['metallicFactor'],
                roughness_texture=self._texture_from_dump(mat['roughnessTexture']) if mat['roughnessTexture'] is not None else None,
                roughness_factor=mat['roughnessFactor'],
                alpha_texture=self._texture_from_dump(mat['alphaTexture']) if mat['alphaTexture'] is not None else None,
                alpha_factor=mat['alphaFactor'],
                alpha_mode={
                    'OPAQUE': AlphaMode.OPAQUE,
                    'MASK': AlphaMode.MASK,
                    'BLEND': AlphaMode.BLEND,
                }[mat['alphaMode']],
                alpha_cutoff=mat['alphaCutoff'],
            ))
        materials.append(PbrMaterial(
            base_color_factor=[0.8, 0.8, 0.8],
            alpha_factor=1.0,
            metallic_factor=0.0,
            roughness_factor=0.5,
            alpha_mode=AlphaMode.OPAQUE,
            alpha_cutoff=0.5,
        ))  # append default material

        # process mesh
        start = 0
        vertices = []
        faces = []
        material_ids = []
        uv_coords = []
        for obj in dump['objects']:
            if obj['vertices'].size == 0 or obj['faces'].size == 0:
                continue
            vertices.append(obj['vertices'])
            faces.append(obj['faces'] + start)
            obj['mat_ids'][obj['mat_ids'] == -1] = len(materials) - 1
            material_ids.append(obj['mat_ids'])
            uv_coords.append(obj['uvs'] if obj['uvs'] is not None else np.zeros((obj['faces'].shape[0], 3, 2), dtype=np.float32))
            start += len(obj['vertices'])
        
        vertices = torch.from_numpy(np.concatenate(vertices, axis=0)).float()
        faces = torch.from_numpy(np.concatenate(faces, axis=0)).long()
        material_ids = torch.from_numpy(np.concatenate(material_ids, axis=0)).long()
        uv_coords = torch.from_numpy(np.concatenate(uv_coords, axis=0)).float()
        
        # Normalize vertices
        # 居中并缩放到 [-0.5, 0.5]（与 dual grid 网格保持同一坐标约定）
        vertices_min = vertices.min(dim=0)[0]
        vertices_max = vertices.max(dim=0)[0]
        center = (vertices_min + vertices_max) / 2
        scale = 0.99999 / (vertices_max - vertices_min).max()
        vertices = (vertices - center) * scale
        assert torch.all(vertices >= -0.5) and torch.all(vertices <= 0.5), 'vertices out of range'
        
        return {'mesh': [MeshWithPbrMaterial(
            vertices=vertices,
            faces=faces,
            material_ids=material_ids,
            uv_coords=uv_coords,
            materials=materials,
        )]}

    def read_pbr_voxel(self, root, instance):
        # 读 vxz 属性并按 layout 顺序拼成 feats；[0,255] -> [-1,1]；coords 前补 batch 维 0
        coords, attr = o_voxel.io.read_vxz(os.path.join(root, f'{instance}.vxz'), num_threads=4)
        feats = torch.concat([attr[k] for k in self.layout], dim=-1) / 255.0 * 2 - 1
        x = sp.SparseTensor(
            feats.float(),
            torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
        )
        return {'x': x}
    
    def get_instance(self, root, instance):
        if self.with_mesh:
            mesh = self.read_mesh_with_texture(root['pbr_dump'], instance)
            pbr_voxel = self.read_pbr_voxel(root['pbr_voxel'], instance)
            return {**mesh, **pbr_voxel}
        else:
            return self.read_pbr_voxel(root['pbr_voxel'], instance)
    
    @staticmethod
    def collate_fn(batch, split_size=None):
        """按属性体素数均衡分组拼批：Tensor stack、SparseTensor sparse_cat、list 展平。"""
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['x'].feats.shape[0] for b in batch], split_size)
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
