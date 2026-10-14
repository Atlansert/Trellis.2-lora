"""PBR 属性 VAE 训练器（稀疏体素 + 材质属性）。

职责：训练 PBR VAE——encoder 把稀疏体素的材质属性（base_color/metallic/roughness/alpha）
编码为潜变量，decoder 重建属性；损失为 feats 上的直接回归（l1/l2）+ 把重建结果包回
带材质网格后用随机相机渲染的 SSIM/LPIPS 监督 + KL 正则。
输入：x（属性稀疏张量）与 mesh（带 PBR 材质的 GT 网格列表）；输出：分项损失与渲染快照。
契约：渲染监督要求同一组相机同时渲染 GT 与重建；feats 属于 [-1,1] 归一化空间，
渲染前需用 feats*0.5+0.5 还原到 [0,1]；layout 定义各通道切片的语义。
"""

from typing import *
import os
import copy
import functools
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils3d
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...modules import sparse as sp
from ...renderers import MeshRenderer
from ...representations import Mesh, MeshWithPbrMaterial, MeshWithVoxel
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler
from ...utils.loss_utils import l1_loss, l2_loss, ssim, lpips


class PbrVaeTrainer(BasicTrainer):
    """
    Trainer for PBR attributes VAE
    
    PBR 材质属性 VAE 训练器：属性重建 + 可微渲染（base_color / MRA）监督 + KL。
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
        
        loss_type (str): Loss type.
        lambda_kl (float): KL loss weight.
        lambda_ssim (float): SSIM loss weight.
        lambda_lpips (float): LPIPS loss weight.
    """
    
    def __init__(
        self,
        *args,
        loss_type: str = 'l1',
        lambda_kl: float = 1e-6,
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.2,
        lambda_render: float = 1.0,
        render_resolution: float = 1024,
        camera_randomization_config: dict = {
            'radius_range': [2, 100],
        },
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_render = lambda_render
        self.camera_randomization_config = camera_randomization_config
        
        self.renderer = MeshRenderer({'near': 1, 'far': 3, 'resolution': render_resolution}, device=self.device)
        
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        均衡可恢复采样 + 多进程 DataLoader；collate 阶段按 batch_split 切分。
        """
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)
        
    def _randomize_camera(self, num_samples: int):
        # 随机相机：半径按 1/r^2 均匀采样，fov = 2·asin(0.5/r) 保证物体占满视野；
        # 相机看向原点、up 轴 +z；near 在 [r-1, r] 内随机
        # sample radius and fov
        r_min, r_max = self.camera_randomization_config['radius_range']
        k_min = 1 / r_max**2
        k_max = 1 / r_min**2
        ks = torch.rand(num_samples, device=self.device) * (k_max - k_min) + k_min
        radius = 1 / torch.sqrt(ks)
        fov = 2 * torch.arcsin(0.5 / radius)
        origin = radius.unsqueeze(-1) * F.normalize(torch.randn(num_samples, 3, device=self.device), dim=-1)
        
        # build camera
        extrinsics = utils3d.torch.extrinsics_look_at(origin, torch.zeros_like(origin), torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device))
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        near = [np.random.uniform(r - 1, r) for r in radius.tolist()]
        
        return {
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'near': near,
        }
        
    def _render_batch(self, reps: List[Mesh], extrinsics: torch.Tensor, intrinsics: torch.Tensor, near: List,
        ) -> Dict[str, torch.Tensor]:
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
            
        Returns: 
            a dict with
                base_color : [N x 3 x H x W] tensor of base color.
                metallic : [N x 1 x H x W] tensor of metallic.
                roughness : [N x 1 x H x W] tensor of roughness.
                alpha : [N x 1 x H x W] tensor of alpha.
        返回按通道拆开的属性渲染结果：base_color 3 通道，metallic/roughness/alpha 各 1 通道。
        """
        ret = {k : [] for k in ['base_color', 'metallic', 'roughness', 'alpha']}
        for i, rep in enumerate(reps):
            # 逐样本设 near/far（far = near + 2），返回材质属性图
            self.renderer.rendering_options['near'] = near[i]
            self.renderer.rendering_options['far'] = near[i] + 2
            out_dict = self.renderer.render(rep, extrinsics[i], intrinsics[i], return_types=['attr'])
            for k in out_dict:
                ret[k].append(out_dict[k])
        for k in ret:
            ret[k] = torch.stack(ret[k])
        return ret
    
    def training_losses(
        self,
        x: sp.SparseTensor,
        mesh: List[MeshWithPbrMaterial] = None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            x (SparseTensor): Input sparse tensor for pbr materials.
            mesh (List[MeshWithPbrMaterial]): The list of meshes with PBR materials.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.

        """
        # 编码：重参数化采样并返回 mean/logvar；解码得到重建的属性稀疏张量
        z, mean, logvar = self.training_models['encoder'](x, sample_posterior=True, return_raw=True)
        y = self.training_models['decoder'](z)
        
        terms = edict(loss = 0.0)
        
        # direct regression
        # 直接回归：只在稀疏张量的 feats 上算 l1/l2
        if self.loss_type == 'l1':
            terms["l1"] = l1_loss(x.feats, y.feats)
            terms["loss"] = terms["loss"] + terms["l1"]
        elif self.loss_type == 'l2':
            terms["l2"] = l2_loss(x.feats, y.feats)
            terms["loss"] = terms["loss"] + terms["l2"]
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')
        
        # rendering loss
        # 渲染监督：把预测 feats 反归一化（*0.5+0.5）并按 layout 语义包成带材质网格，
        # 用同一组随机相机分别渲染 GT 与重建，比较 base_color 与 MRA（SSIM+LPIPS）
        if self.lambda_render != 0.0:
            # layout 指定 feats 各通道的语义：0:3 基色、3 metallic、4 roughness、5 alpha
            recon = [MeshWithVoxel(
                m.vertices,
                m.faces,
                [-0.5, -0.5, -0.5],
                1 / self.dataset.resolution,
                v.coords[:, 1:],
                v.feats * 0.5 + 0.5,
                torch.Size([*v.shape, *v.spatial_shape]),
                layout={
                    'base_color': slice(0, 3),
                    'metallic': slice(3, 4),
                    'roughness': slice(4, 5),
                    'alpha': slice(5, 6),
                }
            ) for m, v in zip(mesh, y)]
            cameras = self._randomize_camera(len(mesh))
            gt_renders = self._render_batch(mesh, **cameras)
            pred_renders = self._render_batch(recon, **cameras)
            gt_base_color = gt_renders['base_color']
            pred_base_color = pred_renders['base_color']
            gt_mra = torch.cat([gt_renders['metallic'], gt_renders['roughness'], gt_renders['alpha']], dim=1)
            pred_mra = torch.cat([pred_renders['metallic'], pred_renders['roughness'], pred_renders['alpha']], dim=1)
            terms['render/base_color/ssim'] = 1 - ssim(pred_base_color, gt_base_color)
            terms['render/base_color/lpips'] = lpips(pred_base_color, gt_base_color)
            terms['render/mra/ssim'] = 1 - ssim(pred_mra, gt_mra)
            terms['render/mra/lpips'] = lpips(pred_mra, gt_mra)
            terms['loss'] = terms['loss'] + \
                            self.lambda_render * (self.lambda_ssim * terms['render/base_color/ssim'] + self.lambda_lpips * terms['render/base_color/lpips'] + \
                                                self.lambda_ssim * terms['render/mra/ssim'] + self.lambda_lpips * terms['render/mra/lpips'])
            
        # KL regularization
        # KL 正则：约束后验接近标准正态
        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]
            
        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        dataloader.dataset.with_mesh = True   # 快照需要 GT 网格来渲染对比

        # inference
        # eval 模式下编码（取后验均值）-> 解码 -> 包回带材质网格，随后渲染对比
        gts = []
        recons = []
        self.models['encoder'].eval()
        self.models['decoder'].eval()
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch] for k, v in data.items()}
            args = recursive_to_device(args, self.device)
            z = self.models['encoder'](args['x'])
            y = self.models['decoder'](z)
            gts.extend(args['mesh'])
            recons.extend([MeshWithVoxel(
                m.vertices,
                m.faces,
                [-0.5, -0.5, -0.5],
                1 / self.dataset.resolution,
                v.coords[:, 1:],
                v.feats * 0.5 + 0.5,
                torch.Size([*v.shape, *v.spatial_shape]),
                layout={
                    'base_color': slice(0, 3),
                    'metallic': slice(3, 4),
                    'roughness': slice(4, 5),
                    'alpha': slice(5, 6),
                }
            ) for m, v in zip(args['mesh'], y)])
        self.models['encoder'].train()
        self.models['decoder'].train()
        
        # 渲染 GT / 重建的 base_color 与 MRA；*2-1 把 [0,1] 映射回 [-1,1] 便于统一可视化
        cameras = self._randomize_camera(num_samples)
        gt_renders = self._render_batch(gts, **cameras)
        pred_renders = self._render_batch(recons, **cameras)

        sample_dict = {
            'gt_base_color': {'value': gt_renders['base_color'] * 2 - 1, 'type': 'image'},
            'pred_base_color': {'value': pred_renders['base_color'] * 2 - 1, 'type': 'image'},
            'gt_mra': {'value': torch.cat([gt_renders['metallic'], gt_renders['roughness'], gt_renders['alpha']], dim=1) * 2 - 1, 'type': 'image'},
            'pred_mra': {'value': torch.cat([pred_renders['metallic'], pred_renders['roughness'], pred_renders['alpha']], dim=1) * 2 - 1, 'type': 'image'},
        }
            
        return sample_dict
